"""Transformer of Spatial Aggregation Module (TSAM).

Faithful re-implementation of Section 3.2 of the paper:
the deep feature map is split into PxP patches, linearly embedded to D dims,
a learned positional encoding (ViT-style, ref [36]) is added, then the sequence
is processed by an L-layer Transformer encoder (multi-head self-attention +
MLP) and reshaped back to a spatial feature map.
"""
import torch
import torch.nn as nn


class TSAM(nn.Module):
    def __init__(self, in_ch, embed_dim=256, patch=5, depth=12, heads=8,
                 mlp_ratio=4.0, dropout=0.0, max_tokens=4096):
        super().__init__()
        self.patch = patch
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.depth = depth

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.out_proj = nn.Conv2d(embed_dim, in_ch, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch
        hh, ww = h // p, w // p
        x = x[:, :, :hh * p, :ww * p]
        z = self.proj(x)                      # (B, D, hh, ww)
        z = z.flatten(2).transpose(1, 2)      # (B, N, D)
        n = z.shape[1]
        z = self.pos_drop(self.norm(z) + self.pos_embed[:, :n])
        z = self.encoder(z)                   # (B, N, D)
        z = z.transpose(1, 2).reshape(b, self.embed_dim, hh, ww)
        return self.out_proj(z)               # (B, C, hh, ww)


if __name__ == "__main__":
    m = TSAM(64, embed_dim=128, patch=5, depth=4, heads=8)
    y = m(torch.randn(2, 64, 65, 65))
    print("TSAM out", tuple(y.shape))
