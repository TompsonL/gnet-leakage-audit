"""T4: label-free pseudo labels for foreground (no human annotation).

A robust, non-learning *teacher* that turns raw video into per-frame foreground
masks, used as training supervision. It uses only the video itself:

  1. causal temporal-median background over the past K frames (robust to moving
     foreground, unlike a plain running average);
  2. optional global-motion compensation (translation via phase correlation, or
     affine via ECC) so that camera motion does not create false residuals;
  3. residual threshold + morphological clean-up + small-component removal;
  4. optional motion gate (frame difference) to reject stale background pixels.

No pretrained segmentation model is used, so no external bias is injected.
These masks supervise training; they are NOT used for evaluation.
"""
import glob
import os

import cv2
import numpy as np


def _g(d):
    return sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))


def causal_median_background(buf):
    """Median over a list of BGR frames (causal window). Returns uint8 BGR."""
    return np.median(np.stack(buf, 0), axis=0).astype(np.uint8)


def estimate_translation(prev_gray, cur_gray):
    """Global translation (dx, dy) such that cur ~= shift(prev)."""
    p = prev_gray.astype(np.float32)
    c = cur_gray.astype(np.float32)
    (dx, dy), resp = cv2.phaseCorrelate(p, c)
    return float(dx), float(dy), float(resp)


def shift_image(img, dx, dy, border=cv2.BORDER_REPLICATE):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=border)


def clean_mask(m, min_area=32, open_it=1, close_it=1, dilate=1):
    k = np.ones((3, 3), np.uint8)
    if open_it:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=open_it)
    if close_it:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_it)
    if dilate:
        m = cv2.dilate(m, k, iterations=dilate)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


def teacher_label(cur, bg, prev=None, thr=35, gmc="translation", motion_gate=False,
                  motion_thr=25, min_area=32, dilate=1):
    """Produce a foreground pseudo-mask for `cur` from a causal background `bg`."""
    gc = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

    if gmc == "translation" and prev is not None:
        gp = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        dx, dy, resp = estimate_translation(gp, gc)
        if resp > 0.05:  # reliable global motion
            bg = shift_image(bg, dx, dy)

    res = cv2.absdiff(cur, bg).max(axis=2)
    m = (res > thr).astype(np.uint8)

    if motion_gate and prev is not None:
        diff = cv2.absdiff(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), gc)
        gate = (diff > motion_thr).astype(np.uint8)
        gate = cv2.dilate(gate, np.ones((3, 3), np.uint8), iterations=2)
        m = np.minimum(m, gate)

    return clean_mask(m, min_area=min_area, dilate=dilate)


def teacher_labels_for_video(vdir, K=30, thr=35, gmc="translation",
                             motion_gate=False, min_area=32, stride=1,
                             start=None, end=None):
    # defaults tuned by scripts/tune_t4.py (K=30, thr=35, translation)
    """Yield (index, mask) over the video using a causal median teacher."""
    inp = _g(os.path.join(vdir, "input"))
    ims = [cv2.imread(p, cv2.IMREAD_COLOR) for p in inp]
    buf = []
    prev = None
    for t, cur in enumerate(ims):
        if cur is None:
            continue
        buf.append(cur)
        if len(buf) > K:
            buf.pop(0)
        if start is not None and t < start:
            prev = cur
            continue
        if end is not None and t > end:
            break
        if t % stride:
            prev = cur
            continue
        bg = causal_median_background(buf)
        m = teacher_label(cur, bg, prev=prev, thr=thr, gmc=gmc,
                          motion_gate=motion_gate, min_area=min_area)
        yield t, m
        prev = cur


# ---- standalone motion / residual helpers (kept for ablation) ----
def motion_pseudo_mask(prev, cur, nxt, thr=25, min_area=32, dilate=2):
    gp = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gc = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    gn = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
    m = (np.maximum(cv2.absdiff(gp, gc), cv2.absdiff(gc, gn)) > thr).astype(np.uint8)
    return clean_mask(m, min_area=min_area, dilate=dilate)


def residual_pseudo_mask(cur, bg, thr=25, min_area=32, dilate=1):
    res = cv2.absdiff(cur, bg).max(axis=2)
    return clean_mask((res > thr).astype(np.uint8), min_area=min_area, dilate=dilate)


if __name__ == "__main__":
    import sys
    v = sys.argv[1] if len(sys.argv) > 1 else "datasets/cdnet/CDnet14/baseline/highway"
    for i, m in teacher_labels_for_video(v, stride=20):
        print(f"idx={i} fg_frac={m.mean():.4f}")
