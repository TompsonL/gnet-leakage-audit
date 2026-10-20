"""SuBSENSE baseline (the "floor") under the CDnet14 protocol.

Evaluates the packaged SuBSENSE masks (bgs_subsense/) directly against the
ground truth, using exactly the same metric/aggregation as eval.py:
per-video -> category-average F1/PWC, ROI ignored, shadow counted as
background. This is the ceiling a background-guided refiner (e.g. GNet)
cannot fairly exceed by a large margin, since it consumes the same masks.
"""
import os
import csv
import glob
import json
import argparse
from collections import defaultdict

import numpy as np
import cv2

from utils.metrics import confusion, f1_pwc
from utils.protocol import protocol_record


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/cdnet/CDnet14")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--out", default="logs/subsense_baseline.csv")
    a = ap.parse_args()

    agg = defaultdict(lambda: [0, 0, 0, 0])
    for cat in sorted(os.listdir(a.root)):
        cdir = os.path.join(a.root, cat)
        if not os.path.isdir(cdir):
            continue
        for v in sorted(os.listdir(cdir)):
            vdir = os.path.join(cdir, v)
            gts = frames(os.path.join(vdir, "groundtruth"))
            bgs = frames(os.path.join(vdir, "bgs_subsense"))
            if not gts or not bgs:
                continue
            f0, f1 = troi(os.path.join(vdir, "temporalROI.txt"))
            roi = load_roi(vdir)
            tp = tn = fp = fn = 0
            for i in range(0, min(len(gts), len(bgs)), a.frame_stride):
                if i + 1 < f0 or i + 1 > f1:
                    continue
                g = cv2.imread(gts[i], cv2.IMREAD_GRAYSCALE)
                m = cv2.imread(bgs[i], cv2.IMREAD_GRAYSCALE)
                if g is None or m is None:
                    continue
                if m.shape[:2] != g.shape[:2]:
                    m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
                if roi is not None:
                    r = roi
                    if r.shape[:2] != g.shape[:2]:
                        r = cv2.resize(r, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
                    g = g.copy()
                    g[r < 128] = 85
                pred = (m > 127).astype(np.uint8)
                x = confusion(pred, g)
                tp += x[0]; tn += x[1]; fp += x[2]; fn += x[3]
            agg[f"{cat}/{v}"] = [tp, tn, fp, fn]
            print(f"{cat}/{v} done", flush=True)

    per_video = {v: f1_pwc(*x) for v, x in agg.items()}
    per_cat = defaultdict(list)
    for v, r in per_video.items():
        per_cat[v.split("/")[0]].append((r[0], r[1]))
    cat_mean = {c: (float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r])))
                for c, r in per_cat.items()}
    avg_vid = (float(np.mean([x[0] for x in per_video.values()])),
               float(np.mean([x[1] for x in per_video.values()])))
    avg_cat = (float(np.mean([x[0] for x in cat_mean.values()])),
               float(np.mean([x[1] for x in cat_mean.values()])))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["video", "F1", "PWC", "Precision", "Recall"])
        for v in sorted(per_video):
            r = per_video[v]
            wr.writerow([v, f"{r[0]:.4f}", f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}"])
        wr.writerow([])
        wr.writerow(["category", "F1", "PWC"])
        for c in sorted(cat_mean):
            wr.writerow([c, f"{cat_mean[c][0]:.4f}", f"{cat_mean[c][1]:.4f}"])
        wr.writerow([])
        wr.writerow(["AVG_video", f"{avg_vid[0]:.4f}", f"{avg_vid[1]:.4f}"])
        wr.writerow(["AVG_category", f"{avg_cat[0]:.4f}", f"{avg_cat[1]:.4f}"])
    print(f"AVG_category F1={avg_cat[0]:.4f} PWC={avg_cat[1]:.4f}", flush=True)
    record = protocol_record("subsense_mask", a.frame_stride, 0.5, False)
    record.update({"method": "SuBSENSE (packaged masks)",
                   "root": a.root, "n_videos": len(per_video),
                   "per_category": {c: {"F1": round(v[0], 4), "PWC": round(v[1], 4)}
                                    for c, v in sorted(cat_mean.items())},
                   "summary": {"F1_video_avg": round(avg_vid[0], 4),
                               "PWC_video_avg": round(avg_vid[1], 4),
                               "F1_category_avg": round(avg_cat[0], 4),
                               "PWC_category_avg": round(avg_cat[1], 4)}})
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
