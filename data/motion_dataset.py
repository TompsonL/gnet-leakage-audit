"""T4: label-free training dataset from raw video using the causal teacher.

Yields the same (img, bg, mask[, rel]) tuples as the synthetic dataset, but the
target mask comes from the non-learning teacher in `data.motion_labels` and the
background from the causal estimator in `data.background`. No human annotation
and no pretrained segmentation model are used.
"""
import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.motion_labels import teacher_labels_for_video
from data.background import estimate_causal_background, estimate_causal_background_rel


def _frames(d):
    return sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))


def _troi(p):
    if not os.path.isfile(p):
        return 1, 10 ** 9
    n = [int(x) for x in open(p).read().split()]
    return n[0], n[-1]


class MotionLabelDataset(Dataset):
    def __init__(self, root, categories=None, split_file=None, size=513,
                 K=30, thr=35, gmc="translation", motion_gate=False,
                 frame_stride=1, label_stride=5, max_videos=None, with_rel=False,
                 rel_tau=100.0, cache_dir="logs/motion_cache"):
        self.size = size
        self.K = K
        self.thr = thr
        self.gmc = gmc
        self.motion_gate = motion_gate
        self.with_rel = with_rel
        self.rel_tau = rel_tau
        self.label_stride = label_stride
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        cats = categories or sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        vids = []
        for c in cats:
            cdir = os.path.join(root, c)
            if not os.path.isdir(cdir):
                continue
            for v in sorted(os.listdir(cdir)):
                if os.path.isdir(os.path.join(cdir, v)):
                    vids.append(os.path.join(cdir, v))
        if split_file and os.path.isfile(split_file):
            keep = set(open(split_file).read().split())
            vids = [p for p in vids
                    if f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}" in keep]
        if max_videos:
            vids = vids[:max_videos]
        self.vdirs = vids
        self.samples = []
        for vdir in vids:
            inp = _frames(os.path.join(vdir, "input"))
            f0, f1 = _troi(os.path.join(vdir, "temporalROI.txt"))
            for i in range(max(0, f0 - 1), min(f1, len(inp)), frame_stride):
                self.samples.append((vdir, i))
        self._cache = {}

    def _video(self, vdir):
        if vdir in self._cache:
            return self._cache[vdir]
        key = f"{os.path.basename(os.path.dirname(vdir))}_{os.path.basename(vdir)}" \
              f"_K{self.K}_t{self.thr}_{self.gmc}_g{int(self.motion_gate)}_r{int(self.with_rel)}"
        cache_path = os.path.join(self.cache_dir, key + ".npz") if self.cache_dir else None
        if cache_path and os.path.isfile(cache_path):
            z = np.load(cache_path)
            masks = {int(k[2:]): z[k] for k in z.files if k.startswith("m_")}
            bgs = {int(k[2:]): z[k] for k in z.files if k.startswith("b_")}
            rels = {int(k[2:]): z[k] for k in z.files if k.startswith("r_")}
            if self.with_rel:
                bgs = {i: (bgs[i], rels[i]) for i in bgs}
            self._cache[vdir] = (masks, bgs)
            return self._cache[vdir]

        masks = {}
        for i, m in teacher_labels_for_video(vdir, K=self.K, thr=self.thr,
                                             gmc=self.gmc, motion_gate=self.motion_gate,
                                             stride=self.label_stride):
            masks[i] = m
        if self.with_rel:
            r = estimate_causal_background_rel(vdir, tau=self.rel_tau)
            bgs = r[1] if r else {}
        else:
            r = estimate_causal_background(vdir)
            bgs = r[1] if r else {}

        if cache_path:
            arr = {}
            for i, m in masks.items():
                arr[f"m_{i}"] = m
            for i, e in bgs.items():
                if isinstance(e, tuple):
                    arr[f"b_{i}"] = e[0]; arr[f"r_{i}"] = e[1]
                else:
                    arr[f"b_{i}"] = e
            try:
                np.savez_compressed(cache_path, **arr)
            except Exception:
                pass
        self._cache[vdir] = (masks, bgs)
        return self._cache[vdir]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        vdir, i = self.samples[k]
        masks, bgs = self._video(vdir)
        inp = _frames(os.path.join(vdir, "input"))
        img = cv2.imread(inp[i], cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.size, self.size, 3), np.uint8)
        mask = masks.get(i)
        if mask is None:
            mask = np.zeros(img.shape[:2], np.uint8)
        entry = bgs.get(i)
        if entry is None:
            bg = img.copy()
            rel = np.ones(img.shape[:2], np.float32)
        elif isinstance(entry, tuple):
            bg, rel = entry
        else:
            bg, rel = entry, np.ones(img.shape[:2], np.float32)

        img_r = cv2.cvtColor(cv2.resize(img, (self.size, self.size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        bg_r = cv2.cvtColor(cv2.resize(bg, (self.size, self.size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        m_r = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        img_t = torch.from_numpy(img_r.transpose(2, 0, 1))
        bg_t = torch.from_numpy(bg_r.transpose(2, 0, 1))
        m_t = torch.from_numpy(m_r[None].astype(np.float32))
        if self.with_rel:
            rel_r = cv2.resize(rel, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            return img_t, bg_t, m_t, torch.from_numpy(rel_r[None].astype(np.float32))
        return img_t, bg_t, m_t
