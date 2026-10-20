#!/usr/bin/env python3
"""Evaluate the label-free causal-median teacher on the four held-out categories.

Uses the tuned config (K=30, thr=35, translation GMC) and the frozen protocol
(shadow as background, ROI ignored). The default stride is 5 to keep the CPU cost
manageable; pass --stride 1 for the full protocol.
"""
import argparse
import glob
import os

import cv2
import numpy as np
from multiprocessing import Pool

from data.motion_labels import teacher_labels_for_video
from utils.metrics import confusion, f1_pwc

CATS = ["badWeather", "nightVideos", "turbulence", "thermal"]
ROOT = "datasets/cdnet/CDnet14"
STRIDE = 5


def troi(vdir):
    p = os.path.join(vdir, "temporalROI.txt")
    if not os.path.isfile(p):
        return 1, 10 ** 9
    n = [int(x) for x in open(p).read().split()]
    return n[0], n[-1]


def roi_of(vdir):
    for name in ("ROI.bmp", "ROI.jpg", "ROI.png"):
        p = os.path.join(vdir, name)
        if os.path.isfile(p):
            return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return None


def gts(vdir):
    return sorted(glob.glob(os.path.join(vdir, "groundtruth", "*.png")) +
                  glob.glob(os.path.join(vdir, "groundtruth", "*.jpg")))


def eval_video(vdir):
    cat, vid = vdir.rstrip("/").split("/")[-2:]
    f0, f1 = troi(vdir)
    roi = roi_of(vdir)
    gt = gts(vdir)
    tp = tn = fp = fn = 0
    for i, m in teacher_labels_for_video(vdir, K=30, thr=35, gmc="translation",
                                         stride=STRIDE, start=f0 - 1, end=f1 - 1):
        if i >= len(gt):
            continue
        g = cv2.imread(gt[i], cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if roi is not None:
            r = roi
            if r.shape[:2] != g.shape[:2]:
                r = cv2.resize(r, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
            g = g.copy()
            g[r < 128] = 85
        if m.shape[:2] != g.shape[:2]:
            m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        x = confusion(m.astype(np.uint8), g)
        tp += x[0]; tn += x[1]; fp += x[2]; fn += x[3]
    f, pwc, _, _ = f1_pwc(tp, tn, fp, fn)
    return cat, vid, f, pwc


def main():
    global STRIDE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    STRIDE = a.stride
    out = a.out or ("logs/teacher_holdout_stride%d.csv" % STRIDE)
    vdirs = []
    for c in CATS:
        vdirs += sorted(glob.glob(os.path.join(ROOT, c, "*")))
    rows = []
    with Pool(a.workers) as pool:
        for cat, vid, f, p in pool.imap_unordered(eval_video, vdirs):
            rows.append((cat, vid, f, p))
            print("%s/%s F1=%.4f PWC=%.4f" % (cat, vid, f, p), flush=True)
    rows.sort()
    by = {}
    for c, v, f, p in rows:
        by.setdefault(c, []).append(f)
    cats = [sum(by[c]) / len(by[c]) for c in CATS if by.get(c)]
    avg = sum(cats) / len(cats)
    with open(out, "w") as fh:
        fh.write("category,video,F1,PWC\n")
        for c, v, f, p in rows:
            fh.write("%s,%s,%.4f,%.4f\n" % (c, v, f, p))
        fh.write("AVG_category,%.4f,0.0000\n" % avg)
    print("AVG_category=%.4f (stride=%d) -> %s" % (avg, STRIDE, out))


if __name__ == "__main__":
    main()
