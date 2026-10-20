"""GNet: background-guided differential segmentation network (paper Section 3.1-3.2).

Architecture (per the paper, DeepLab-style):
  - shared ResNet101 encoder, output stride 8
  - high-resolution feature (layer1, stride 4, 256ch) -> differential feature d_l
  - deep feature (layer4, stride 8, 2048ch) -> depthwise-separable ASPP
    -> multi-scale differential features d_i -> stacked -> TSAM
  - decoder fuses upsampled TSAM output with the shallow differential feature
    and recovers full resolution.

Difference operator: D = F_cur - F_bg (paper Eq.(1),(2)). The released
`models/GNet.py` concatenates the foreground feature as well, so `fusion`
controls `sub` (default, paper) or `cat`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.tsam import TSAM


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with depthwise separable atrous convs."""

    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, 3, padding=r, dilation=r,
                              groups=in_ch, bias=False),
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        g = F.interpolate(self.global_pool(x), size=size,
                          mode="bilinear", align_corners=False)
        outs.append(g)
        return outs


class ResNet101OS8(nn.Module):
    """ResNet101 with output stride 8; returns layer1 (stride4) and layer4 (stride8)."""

    def __init__(self, pretrained=True, replace_stride_with_dilation=(False, True, True)):
        super().__init__()
        if pretrained:
            weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2
        else:
            weights = None
        net = torchvision.models.resnet101(weights=weights,
                                           replace_stride_with_dilation=replace_stride_with_dilation)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)   # stride 4, 256
        c2 = self.layer2(c1)  # stride 8, 512
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)  # stride 8 (dilated), 2048
        return c1, c4


class Decoder(nn.Module):
    def __init__(self, low_ch, deep_ch, mid_ch=256, num_classes=1):
        super().__init__()
        self.deep_proj = nn.Sequential(
            nn.Conv2d(deep_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True))
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch + 48, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(mid_ch, num_classes, 1)

    def forward(self, deep, low, out_size):
        x = self.deep_proj(deep)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_proj(low)
        x = self.fuse(torch.cat([x, low], dim=1))
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return self.head(x)


class GNet(nn.Module):
    def __init__(self, num_classes=1, embed_dim=256, tsam_depth=12, tsam_heads=8,
                 tsam_patch=5, aspp_out=256, fusion="cat", low_ch=256,
                 pretrained=True, rel_gate=False, boundary=False):
        super().__init__()
        self.fusion = fusion
        self.rel_gate = rel_gate
        self.boundary = boundary
        if boundary:
            # M3: boundary-aware differential attention. An edge map from the
            # shallow current-frame feature amplifies the differential response
            # near object boundaries; strength is learned (softplus >= 0).
            self.edge_conv = nn.Sequential(
                nn.Conv2d(low_ch, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1))
            self.edge_gain = nn.Parameter(torch.tensor(1.0))
        if rel_gate:
            # Learn a monotone gate from the reliability map: a scalar
            # temperature/bias plus a small spatial residual (M1, learnable).
            self.gate_scale = nn.Parameter(torch.tensor(4.0))
            self.gate_bias = nn.Parameter(torch.tensor(-2.0))
            self.gate_res = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1))
            nn.init.zeros_(self.gate_res[-1].weight)
            nn.init.zeros_(self.gate_res[-1].bias)
        self.backbone = ResNet101OS8(pretrained=pretrained)
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_ch, low_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(low_ch), nn.ReLU(inplace=True))
        self.aspp = ASPP(2048, aspp_out)
        n_scale = 5  # 4 rates + global
        merged_ch = aspp_out * (2 * n_scale) if fusion == "cat" else aspp_out * n_scale
        self.deep_merge = nn.Sequential(
            nn.Conv2d(merged_ch, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.tsam = TSAM(embed_dim, embed_dim=embed_dim, patch=tsam_patch,
                         depth=tsam_depth, heads=tsam_heads)
        # shallow differential feature channels
        low_out_ch = low_ch * (2 if fusion == "cat" else 1)
        self.decoder = Decoder(low_out_ch, embed_dim, mid_ch=256,
                               num_classes=num_classes)

    def _diff_or_cat(self, cur, bg, rel=None):
        diff = cur - bg
        if rel is not None:
            diff = diff * rel
        if self.fusion == "sub":
            return diff
        return torch.cat([diff, cur], dim=1)

    def _gate_map(self, rel):
        """Turn a (B,1,H,W) reliability map into a gate map in [0,1].

        Training-free path (no rel_gate): gate == reliability.
        Learnable path (rel_gate): sigmoid(scale*rel + bias + small_conv(rel)).
        """
        if rel is None:
            return None
        if self.rel_gate:
            logit = self.gate_scale * rel + self.gate_bias + self.gate_res(rel)
            return torch.sigmoid(logit)
        return rel

    @staticmethod
    def _to_feat(gate, feat):
        """Resize a (B,1,H,W) gate map to a feature map's resolution."""
        if gate is None:
            return None
        return F.interpolate(gate, size=feat.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, img, bg_img, rel=None):
        c1_i, c4_i = self.backbone(img)
        c1_b, c4_b = self.backbone(bg_img)

        gfull = self._gate_map(rel)
        low_i = self.low_conv(c1_i)
        low_b = self.low_conv(c1_b)

        # M3 edge attention (from the shallow current-frame feature)
        edge = None
        if self.boundary:
            edge = torch.sigmoid(self.edge_conv(low_i))
            edge = 1.0 + F.softplus(self.edge_gain) * edge  # >= 1, amplifies boundaries

        def comb(feat):
            g = self._to_feat(gfull, feat)
            if edge is not None:
                e = F.interpolate(edge, size=feat.shape[-2:], mode="bilinear",
                                  align_corners=False)
                g = e if g is None else g * e
            return g

        low = self._diff_or_cat(low_i, low_b, comb(low_i))

        as_i = self.aspp(c4_i)
        as_b = self.aspp(c4_b)
        g = comb(c4_i)
        diffs = [self._diff_or_cat(a, b, g) for a, b in zip(as_i, as_b)]
        deep = torch.cat(diffs, dim=1)
        deep = self.deep_merge(deep)

        deep = self.tsam(deep)
        out = self.decoder(deep, low, out_size=img.shape[-2:])
        return out


if __name__ == "__main__":
    net = GNet(pretrained=False, tsam_depth=2)
    y = net(torch.randn(1, 3, 257, 257), torch.randn(1, 3, 257, 257))
    print("GNet out", tuple(y.shape), "params(M)",
          sum(p.numel() for p in net.parameters()) / 1e6)
