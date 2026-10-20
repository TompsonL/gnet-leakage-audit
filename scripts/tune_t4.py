#!/usr/bin/env python3
"""Tune the label-free teacher pseudo-labels against CDnet GT (F1).

Grid over window K, threshold, global-motion compensation and motion gate.
Prints F1 per video and the grid mean so the best config can be chosen.
"""
import argparse
import glob
import os

import cv2
import numpy as np

from data.motion_labels import teacher_labels_for_video
from utils.metrics import confusion, f1_pwc


def troi(p):
    if not os.path.isfile(p):
        return 1, 10 ** 9
    n = [int(x) for x in open(p).read().split()]
    return n[0], n[-1]


def load_roi(vdir):
    for rp in ("ROI.bmp", "ROI.jpg", "ROI.png"):
        p = os.path.join(vdir, rp)
        if os.path.isfile(p):
            return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return None


def gts(d):
    return sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))


def eval_video(vdir, stride, **kw):
    f0, f1 = troi(os.path.join(vdir, "temporalROI.txt"))
    roi = load_roi(vdir)
    gtf = gts(os.path.join(vdir, "groundtruth"))
    tp = tn = fp = fn = 0
    for i, m in teacher_labels_for_video(vdir, stride=stride, start=f0 - 1, end=f1 - 1, **kw):
        if i >= len(gtf):
            continue
        g = cv2.imread(gtf[i], cv2.IMREAD_GRAYSCALE)
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
    return f1_pwc(tp, tn, fp, fn)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/cdnet/CDnet14")
    ap.add_argument("--videos", default="baseline/PETS2006,baseline/highway,"
                    "shadow/busStation,cameraJitter/boulevard,PTZ/continuousPan,"
                    "thermal/corridor,dynamicBackground/fall,nightVideos/bridgeEntry")
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()

    vids = a.videos.split(",")
    configs = []
    for K in (7, 15, 30):
        for thr in (15, 25, 35):
            for gmc in ("none", "translation"):
                configs.append(dict(K=K, thr=thr, gmc=gmc, motion_gate=False))
    configs.append(dict(K=15, thr=25, gmc="translation", motion_gate=True))

    print(f"{'K':>3} {'thr':>3} {'gmc':>11} {'gate':>5} | " +
          " ".join(f"{v.split('/')[-1][:9]:>9}" for v in vids) + " |  mean")
    best = None
    for c in configs:
        f1s = []
        for v in vids:
            try:
                f1s.append(eval_video(os.path.join(a.root, v), a.stride, **c))
            except Exception as e:
                f1s.append(float("nan"))
        mean = float(np.nanmean(f1s))
        row = f"{c['K']:>3} {c['thr']:>3} {c['gmc']:>11} {str(c['motion_gate']):>5} | " + \
              " ".join(f"{x:9.4f}" for x in f1s) + f" | {mean:.4f}"
        print(row)
        if best is None or mean > best[0]:
            best = (mean, c)
    print("\nBEST:", best)


if __name__ == "__main__":
    main()
