#!/usr/bin/env python3
"""L2a: choose the causal background estimator by its resulting residual-F1.

For each candidate (running EMA with rate alpha, or causal temporal median with
window K), build the per-frame causal background and threshold the residual over
a tau sweep; report the best category-average F1. The best estimator is then
used consistently in training (`_causal_background`) and inference
(`running_background_nomask`).
"""
import argparse
import glob
import os
from collections import defaultdict

import cv2
import numpy as np

from utils.metrics import confusion, f1_pwc


def frames(d):
    return sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))


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


def causal_bgs(frames_list, method, param):
    """Return list of backgrounds (uint8) aligned with frames_list (causal)."""
    if method == "running":
        B = frames_list[0].astype(np.float32)
        out = []
        for f in frames_list:
            B = B + param * (f.astype(np.float32) - B)
            out.append(B.astype(np.uint8))
        return out
    # temporal median over a causal window
    K = param
    buf = []
    out = []
    stack = None
    for f in frames_list:
        buf.append(f)
        if len(buf) > K:
            buf.pop(0)
        out.append(np.median(np.stack(buf), axis=0).astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/cdnet/CDnet14")
    ap.add_argument("--videos", default="baseline/highway,baseline/PETS2006,PTZ/twoPositionPTZCam,"
                    "thermal/corridor,shadow/busStation,cameraJitter/boulevard,"
                    "dynamicBackground/fall,nightVideos/bridgeEntry")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--taus", default="5,10,15,20,25,30,40")
    a = ap.parse_args()
    taus = [int(x) for x in a.taus.split(",")]
    methods = [("running", x) for x in (0.02, 0.05, 0.1, 0.2)] + \
              [("median", k) for k in (10, 30)]

    # per method: per-category F1 at each tau
    agg = {m: {t: defaultdict(lambda: [0, 0, 0, 0]) for t in taus} for m in methods}
    for v in a.videos.split(","):
        vdir = os.path.join(a.root, v)
        cat = v.split("/")[0]
        inp = frames(os.path.join(vdir, "input"))
        gtf = frames(os.path.join(vdir, "groundtruth"))
        f0, f1 = troi(os.path.join(vdir, "temporalROI.txt"))
        roi = load_roi(vdir)
        fl = [cv2.imread(inp[i], cv2.IMREAD_COLOR)
              for i in range(0, len(inp), a.stride) if f0 - 1 <= i <= f1 - 1]
        if len(fl) < 3:
            continue
        for m in methods:
            bgs = causal_bgs(fl, m[0], m[1])
            idxs = [i for i in range(0, len(inp), a.stride) if f0 - 1 <= i <= f1 - 1]
            for k, i in enumerate(idxs):
                if i >= len(gtf):
                    continue
                g = cv2.imread(gtf[i], cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                if roi is not None:
                    r = roi
                    if r.shape[:2] != g.shape[:2]:
                        r = cv2.resize(r, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
                    g = g.copy(); g[r < 128] = 85
                res = np.abs(fl[k].astype(np.int16) - bgs[k].astype(np.int16)).max(axis=2)
                for t in taus:
                    pred = (res > t).astype(np.uint8)
                    x = confusion(pred, g)
                    s = agg[m][t][cat]
                    for j in range(4):
                        s[j] += x[j]
        print("done", v, flush=True)

    print(f"\n{'method':>14} {'tau*':>4} {'F1_cat':>8}")
    best = None
    for m in methods:
        bm = None
        for t in taus:
            cats = agg[m][t]
            if not cats:
                continue
            means = [f1_pwc(*cats[c])[0] for c in cats]
            f1c = float(np.mean(means))
            if bm is None or f1c > bm[1]:
                bm = (t, f1c)
        if bm:
            print(f"{str(m):>14} {bm[0]:>4} {bm[1]:>8.4f}")
            if best is None or bm[1] > best[1]:
                best = (m, bm[1], bm[0])
    print("\nBEST:", best)


if __name__ == "__main__":
    main()
