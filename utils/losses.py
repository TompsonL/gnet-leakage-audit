import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer("pos_weight",
                             torch.tensor([pos_weight]) if pos_weight else None)

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight)
        prob = torch.sigmoid(logits)
        num = 2 * (prob * target).sum(dim=(1, 2, 3)) + 1e-6
        den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
        dice = 1 - (num / den).mean()
        return self.bce_weight * bce + self.dice_weight * dice


def _boundary_band(target, band=3):
    """Soft boundary band from a {0,1} mask via max/avg pooling."""
    k = 2 * band + 1
    dil = F.max_pool2d(target, k, stride=1, padding=band)
    ero = -F.max_pool2d(-target, k, stride=1, padding=band)
    return (dil - ero).clamp(0, 1)


class BoundaryAwareLoss(nn.Module):
    """BCE-Dice on the mask + extra BCE weighted toward the boundary band (M3)."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0, boundary_weight=1.0,
                 band=3, pos_weight=None):
        super().__init__()
        self.base = BCEDiceLoss(bce_weight, dice_weight, pos_weight)
        self.boundary_weight = boundary_weight
        self.band = band

    def forward(self, logits, target):
        loss = self.base(logits, target)
        if self.boundary_weight <= 0:
            return loss
        band_mask = _boundary_band(target, self.band)
        w = 1.0 + 4.0 * band_mask
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = loss + self.boundary_weight * (bce * w).sum() / w.sum()
        return loss
