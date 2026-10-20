"""Background image estimation for inference.

The paper extracts a background image with SuBSENSE. The CDnet packaging ships
SuBSENSE foreground masks (`bgs_subsense/`); we reconstruct a SuBSENSE-style
background image causally by running-average updating the pixels that SuBSENSE
marks as background. A static per-video background (final estimate) is cached;
per-frame backgrounds can be produced on demand.
"""
import os
import glob
import cv2
import numpy as np


def _sorted(d):
    return sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))


def _read_temporal(path):
    if not os.path.isfile(path):
        return 1, 10 ** 9
    n = [int(x) for x in open(path).read().split()]
    return n[0], n[-1]


def running_background(frames, masks, alpha=0.05, post_alpha=1.0):
    """Return per-frame background list (float32 BGR) using SuBSENSE masks."""
    bg = frames[0].astype(np.float32).copy()
    outs = []
    for f, m in zip(frames, masks):
        f = f.astype(np.float32)
        bmask = (m < 128)
        if post_alpha >= 1.0:
            for c in range(bg.shape[2]):
                bg[..., c][bmask] = (1 - alpha) * bg[..., c][bmask] + alpha * f[..., c][bmask]
        else:
            for c in range(bg.shape[2]):
                upd = (1 - alpha) * bg[..., c] + alpha * f[..., c]
                bg[..., c] = np.where(bmask, upd, bg[..., c])
        outs.append(bg.astype(np.uint8).copy())
    return outs


def running_background_nomask(frames, alpha=0.05):
    """Pure causal running average over ALL pixels (no SuBSENSE guidance).

    Used as a leakage control: the reference at frame t depends only on
    frames <= t and contains no SuBSENSE decision.
    """
    bg = frames[0].astype(np.float32).copy()
    outs = []
    for f in frames:
        bg = (1 - alpha) * bg + alpha * f.astype(np.float32)
        outs.append(bg.astype(np.uint8).copy())
    return outs


def running_background_reliability(frames, alpha=0.05, tau=100.0):
    """Mask-free causal running average with a per-pixel reliability map (M1).

    Reliability is a monotone decreasing function of the exponential moving
    variance of the recent (causal) frames:

        rel = 1 / (1 + ema_var / tau)   in (0, 1]

    High where the pixel is temporally stable (the running mean is a trustworthy
    background), low where motion / illumination changes make it unreliable.
    No segmentation label or BGS mask is used, so the map is leakage-free.

    Returns (bg_list_uint8 BGR, rel_list_float32 (H,W)).
    """
    bg = frames[0].astype(np.float32).copy()
    var = np.zeros(bg.shape[:2], np.float32)
    outs_bg, outs_rel = [], []
    for f in frames:
        f = f.astype(np.float32)
        diff = f - bg
        bg = bg + alpha * diff
        d2 = (diff * diff).mean(axis=2)
        var = (1 - alpha) * var + alpha * d2
        rel = 1.0 / (1.0 + var / tau)
        outs_bg.append(bg.astype(np.uint8).copy())
        outs_rel.append(rel.astype(np.float32))
    return outs_bg, outs_rel


def estimate_causal_background_rel(vdir, alpha=0.05, tau=100.0):
    """Mask-free causal background + reliability. Returns (paths, {idx: (bg, rel)}, f0, f1)."""
    inp = _sorted(os.path.join(vdir, "input"))
    if not inp:
        return None
    f0, f1 = _read_temporal(os.path.join(vdir, "temporalROI.txt"))
    frames, idxs = [], []
    for i in range(len(inp)):
        if i + 1 < f0 or i + 1 > f1:
            continue
        im = cv2.imread(inp[i], cv2.IMREAD_COLOR)
        if im is None:
            continue
        frames.append(im); idxs.append(i)
    if not frames:
        return None
    bgs, rels = running_background_reliability(frames, alpha=alpha, tau=tau)
    return inp, {idx: (bg, rel) for idx, bg, rel in zip(idxs, bgs, rels)}, f0, f1


def running_background_median(frames, K=10, tau=100.0):
    """Causal temporal-median background + variance-based reliability.

    Median is more robust to foreground contamination than the EMA running
    average (L2a: residual-F1 0.619 vs 0.538 on 5 videos). Reliability uses the
    same formula as elsewhere: R = 1/(1 + V/tau), V = window variance.
    Returns (bg_list_uint8, rel_list_float32).
    """
    buf = []
    outs_bg, outs_rel = [], []
    for f in frames:
        buf.append(f.astype(np.float32))
        if len(buf) > K:
            buf.pop(0)
        stack = np.stack(buf, 0)
        med = np.median(stack, axis=0)
        var = (stack.var(axis=0)).mean(axis=2)
        rel = 1.0 / (1.0 + var / tau)
        outs_bg.append(np.clip(med, 0, 255).astype(np.uint8))
        outs_rel.append(rel.astype(np.float32))
    return outs_bg, outs_rel


def estimate_causal_background_median(vdir, K=10, tau=100.0):
    """Causal median background + reliability. Returns (paths, {idx:(bg,rel)}, f0, f1)."""
    inp = _sorted(os.path.join(vdir, "input"))
    if not inp:
        return None
    f0, f1 = _read_temporal(os.path.join(vdir, "temporalROI.txt"))
    frames, idxs = [], []
    for i in range(len(inp)):
        if i + 1 < f0 or i + 1 > f1:
            continue
        im = cv2.imread(inp[i], cv2.IMREAD_COLOR)
        if im is None:
            continue
        frames.append(im); idxs.append(i)
    if not frames:
        return None
    bgs, rels = running_background_median(frames, K=K, tau=tau)
    return inp, {idx: (bg, rel) for idx, bg, rel in zip(idxs, bgs, rels)}, f0, f1


def estimate_causal_background(vdir, alpha=0.05):
    """Mask-free causal background. Returns (paths, {orig_index: bg}, f0, f1)."""
    inp = _sorted(os.path.join(vdir, "input"))
    if not inp:
        return None
    f0, f1 = _read_temporal(os.path.join(vdir, "temporalROI.txt"))
    frames, idxs = [], []
    for i in range(len(inp)):
        if i + 1 < f0 or i + 1 > f1:
            continue
        im = cv2.imread(inp[i], cv2.IMREAD_COLOR)
        if im is None:
            continue
        frames.append(im); idxs.append(i)
    if not frames:
        return None
    bgs = running_background_nomask(frames, alpha=alpha)
    return inp, {idx: bg for idx, bg in zip(idxs, bgs)}, f0, f1


def estimate_video_background(vdir, alpha=0.05, method="running"):
    """Return (frames_paths, {orig_index: background_image}, f0, f1)."""
    inp = _sorted(os.path.join(vdir, "input"))
    mdir = os.path.join(vdir, "bgs_subsense")
    masks = _sorted(mdir) if os.path.isdir(mdir) else []
    if not inp or not masks:
        return None
    f0, f1 = _read_temporal(os.path.join(vdir, "temporalROI.txt"))
    frames, ms, idxs = [], [], []
    for i in range(len(inp)):
        if i + 1 < f0 or i + 1 > f1:
            continue
        m = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE) if i < len(masks) else None
        if m is None:
            continue
        im = cv2.imread(inp[i], cv2.IMREAD_COLOR)
        if im is None:
            continue
        if m.shape[:2] != im.shape[:2]:
            m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        frames.append(im); ms.append(m); idxs.append(i)
    if not frames:
        return None
    if method == "median":
        stack = np.stack(frames).astype(np.float32)
        mask_stack = np.stack(ms) < 128
        # temporal median over frames where the pixel is background
        med = np.zeros(stack.shape[1:], np.float32)
        for c in range(3):
            masked = np.where(mask_stack, stack[..., c], np.nan)
            with np.errstate(all="ignore"):
                medc = np.nanmedian(masked, axis=0)
            med[..., c] = np.where(np.isnan(medc), stack[..., c].mean(0), medc)
        med = np.clip(med, 0, 255).astype(np.uint8)
        bg_by_index = {idx: med for idx in idxs}
    else:
        bgs = running_background(frames, ms, alpha=alpha)
        bg_by_index = {idx: bg for idx, bg in zip(idxs, bgs)}
    return inp, bg_by_index, f0, f1


if __name__ == "__main__":
    import sys
    v = sys.argv[1]
    r = estimate_video_background(v)
    print("frames", len(r[0]) if r else 0)
