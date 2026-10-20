"""Real CDnet training set using SuBSENSE masks as pseudo-labels.

Yields (current frame, reconstructed background, binary mask) for a set of
videos. Used to test whether target-domain (pseudo-labelled) training is
required to reach the accuracy reported in the paper.
"""
import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.background import estimate_video_background


def _sorted(d):
    return sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))


class RealCDnetTrain(Dataset):
    """Video-agnostic supervised training on CDnet.

    label="subsense": SuBSENSE masks as pseudo-labels (leaky teacher).
    label="gt":       CDnet ground-truth masks (255 = foreground).
    bg="running":     SuBSENSE-guided running background (leaky).
    bg="causal_med":  mask-free causal median background (leakage-free).
    """

    def __init__(self, root, videos, size=513, stride=5, bg_method="running",
                 max_per_video=200, label="subsense", bg="causal_med"):
        self.size = size
        self.bg = bg
        self.label = label
        self.items = []  # (img_path, bg_image, mask_path, fg_is_gt)
        for cat, v in videos:
            vdir = os.path.join(root, cat, v)
            inp = _sorted(os.path.join(vdir, "input"))
            if label == "gt":
                masks = _sorted(os.path.join(vdir, "groundtruth"))
            else:
                masks = _sorted(os.path.join(vdir, "bgs_subsense"))
            if not inp or not masks:
                continue
            if bg == "causal_med":
                from data.background import estimate_causal_background_median
                r = estimate_causal_background_median(vdir)
                bg_by_index = {i: e[0] for i, e in r[1].items()} if r else {}
            else:
                r = estimate_video_background(vdir, method=bg_method)
                bg_by_index = r[1] if r else {}
            if not bg_by_index:
                continue
            idxs = sorted(bg_by_index.keys())[::stride][:max_per_video]
            for i in idxs:
                if i < len(masks):
                    self.items.append((inp[i], bg_by_index[i], masks[i], label == "gt"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        ip, bg, mp, is_gt = self.items[k]
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros(img.shape[:2], np.uint8)
        img_r = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        bg_r = cv2.resize(bg, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        m_r = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        bg_r = cv2.cvtColor(bg_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if is_gt:
            lab = (m_r == 255).astype(np.float32)[None]  # CDnet GT: 255 = foreground
        else:
            lab = (m_r > 127).astype(np.float32)[None]
        return (torch.from_numpy(img_r.transpose(2, 0, 1)),
                torch.from_numpy(bg_r.transpose(2, 0, 1)),
                torch.from_numpy(lab))
