#!/usr/bin/env python3
"""T3 calibration: match the synthetic background-degradation statistics to the
REAL causal-background error statistics on CDnet.

Real error is measured as |causal running background - oracle background|
(the packaged `backgrounds/` = whole-video temporal median), plus the
per-pixel running variance used for reliability. Synthetic error is measured
the same way against the clean Cityscapes compositing background.

Prints matched statistics so the degradation parameters (ghost alpha,
illumination gain/bias, blur, noise, tau) can be tuned.
"""
import argparse
import glob
import os

import cv2
import numpy as np

from data.background import running_background_nomask


def _frames(d):
    return sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))


def troi(p):
    if not os.path.isfile(p):
        return 1, 10 ** 9
    n = [int(x) for x in open(p).read().split()]
    return n[0], n[-1]


def stats(err):
    err = err.ravel()
    return dict(mean=err.mean(), p50=np.percentile(err, 50),
                p90=np.percentile(err, 90), p99=np.percentile(err, 99),
                f10=(err > 10).mean(), f25=(err > 25).mean())


def real_stats(root, videos, stride=5, alpha=0.05):
    rows = []
    for v in videos:
        vdir = os.path.join(root, v)
        inp = _frames(os.path.join(vdir, "input"))
        f0, f1 = troi(os.path.join(vdir, "temporalROI.txt"))
        frames = []
        for i in range(0, len(inp), stride):
            if i + 1 < f0 or i + 1 > f1:
                continue
            im = cv2.imread(inp[i], cv2.IMREAD_COLOR)
            if im is not None:
                frames.append(im)
        if len(frames) < 5:
            continue
        # causal running background at the last frame
        bgs = running_background_nomask(frames, alpha=alpha)
        B_causal = bgs[-1].astype(np.float32)
        B_oracle = np.median(np.stack(frames).astype(np.float32), axis=0)
        err = np.abs(B_causal - B_oracle).mean(axis=2)
        st = stats(err)
        rows.append((v, st))
    return rows


def synth_stats(n=60, size=256, degrade=True, shadow=True, seed=0):
    from data.synth import SynthDataset, COCOObjects, CityscapesBackgrounds
    import random
    random.seed(seed); np.random.seed(seed)
    cs = CityscapesBackgrounds("datasets/raw/cityscapes_train.parquet")
    coco = COCOObjects("datasets/raw/annotations/instances_val2017.json",
                       "datasets/raw/val2017")
    ds = SynthDataset(cs, coco, size=size, length=n, with_rel=True,
                      degrade=degrade, shadow=shadow)
    errs, rels = [], []
    for _ in range(n):
        bg, scene, mask, placed = ds._compose()
        train_bg, rel = ds._pmb_background(bg, placed)
        if degrade:
            train_bg, rel = ds._degrade_background(train_bg, placed)
        e = np.abs(train_bg.astype(np.float32) - bg.astype(np.float32)).mean(axis=2)
        errs.append(e.ravel())
        rels.append(np.asarray(rel).ravel())
    err = np.concatenate(errs)
    rel = np.concatenate(rels)
    st = stats(err)
    st["rel_mean"] = float(rel.mean())
    st["rel_f50"] = float((rel < 0.5).mean())
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/cdnet/CDnet14")
    ap.add_argument("--videos", default="baseline/highway,PTZ/continuousPan,"
                    "thermal/corridor,shadow/busStation,intermittentObjectMotion/parking,"
                    "dynamicBackground/fall")
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()
    print("=== REAL causal-background error |B_causal - B_oracle| (0-255) ===")
    print(f"{'video':32s} {'mean':>6} {'p50':>6} {'p90':>6} {'p99':>6} {'>10':>6} {'>25':>6}")
    rows = real_stats(a.root, a.videos.split(","), stride=a.stride)
    means = []
    for v, st in rows:
        means.append(st["mean"])
        print(f"{v:32s} {st['mean']:6.1f} {st['p50']:6.1f} {st['p90']:6.1f} "
              f"{st['p99']:6.1f} {st['f10']:6.2f} {st['f25']:6.2f}")
    if means:
        print(f"{'MEAN':32s} {np.mean(means):6.1f}")
    print("\n=== SYNTHETIC background error |train_bg - clean_bg| ===")
    s = synth_stats(degrade=True)
    print(f"{'synthetic':32s} {s['mean']:6.1f} {s['p50']:6.1f} {s['p90']:6.1f} "
          f"{s['p99']:6.1f} {s['f10']:6.2f} {s['f25']:6.2f}  rel_mean={s['rel_mean']:.2f} rel<0.5={s['rel_f50']:.2f}")


if __name__ == "__main__":
    main()
