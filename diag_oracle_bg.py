"""Diagnostic 2a: leakage / oracle-background ceiling test.

Question: can the reported ~0.96 F1 be produced by feeding an *oracle* background
(whole-video temporal median = future-frame information, i.e. the dataset's
`backgrounds/`) into a trivial residual thresholder?  If yes, the paper's
numbers are consistent with an oracle background rather than a fair causal
pipeline.

For every CDnet14 video we build the background three ways and threshold the
max-channel absolute residual |I_t - B_t| over a sweep of tau, then report the
best (per-category) category-average F1:
  oracle   : per-pixel median over ALL frames (future frames included)
  causal   : running average over frames <= t (no future info)
  first    : first evaluated frame (static, no temporal info)
"""
import os
import glob
import json
import argparse
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


def build_bgs(vdir, fps, f0, f1, stride, med_sub):
    """Return {orig_index: uint8 background} for oracle/causal/first."""
    ims, idxs = [], []
    for i in range(0, len(fps), stride):
        if i + 1 < f0 or i + 1 > f1:
            continue
        im = cv2.imread(fps[i], cv2.IMREAD_COLOR)
        if im is None:
            continue
        ims.append(im)
        idxs.append(i)
    if not ims:
        return None
    # oracle: median over a subsample of ALL evaluated frames (future included)
    sub = ims[::max(1, med_sub)]
    stack = np.stack(sub).astype(np.float32)
    oracle = np.median(stack, axis=0).astype(np.uint8)
    # causal running average over past frames (no future)
    curs = []
    run = ims[0].astype(np.float32)
    for im in ims:
        run = 0.95 * run + 0.05 * im.astype(np.float32)
        curs.append(run.astype(np.uint8))
    first = ims[0]
    return {i: (oracle, curs[k], first) for k, i in enumerate(idxs)}


def run(stride, taus, med_sub, out_csv, root="datasets/cdnet/CDnet14"):
    TAU = taus
    # per (source, cat, tau) accumulate video F1
    per_video = {}  # (src, tau) -> {video: (tp,tn,fp,fn)}
    for cat in sorted(os.listdir(root)):
        cdir = os.path.join(root, cat)
        if not os.path.isdir(cdir):
            continue
        for v in sorted(os.listdir(cdir)):
            vdir = os.path.join(cdir, v)
            fps = frames(os.path.join(vdir, "input"))
            gts = frames(os.path.join(vdir, "groundtruth"))
            if not fps or not gts:
                continue
            f0, f1 = troi(os.path.join(vdir, "temporalROI.txt"))
            roi = load_roi(vdir)
            bgs = build_bgs(vdir, fps, f0, f1, stride, med_sub)
            if bgs is None:
                continue
            vkey = f"{cat}/{v}"
            acc = {(s, t): [0, 0, 0, 0] for s in ("oracle", "causal", "first") for t in TAU}
            for i, (oracle, causal, first) in bgs.items():
                img = cv2.imread(fps[i], cv2.IMREAD_COLOR)
                g = cv2.imread(gts[i], cv2.IMREAD_GRAYSCALE) if i < len(gts) else None
                if img is None or g is None:
                    continue
                if roi is not None:
                    r = roi
                    if r.shape[:2] != g.shape[:2]:
                        r = cv2.resize(r, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
                    g = g.copy()
                    g[r < 128] = 85
                for s, bg in (("oracle", oracle), ("causal", causal), ("first", first)):
                    res = np.abs(img.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
                    for t in TAU:
                        pred = (res > t).astype(np.uint8)
                        a, b, c, d = confusion(pred, g)
                        acc[(s, t)][0] += a; acc[(s, t)][1] += b
                        acc[(s, t)][2] += c; acc[(s, t)][3] += d
            for key, val in acc.items():
                per_video.setdefault(key, {})[vkey] = val

    rows = []
    print(f"{'src':7s} {'tau':>4s} {'AVG_video':>9s} {'AVG_cat':>8s}")
    for s in ("oracle", "causal", "first"):
        best = None
        for t in TAU:
            pv = per_video.get((s, t), {})
            if not pv:
                continue
            vf = {v: f1_pwc(*vals)[0] for v, vals in pv.items()}
            catvals = {}
            for v, ff in vf.items():
                catvals.setdefault(v.split("/")[0], []).append(ff)
            catmean = {c: float(np.mean(x)) for c, x in catvals.items()}
            avgv = float(np.mean(list(vf.values())))
            avgc = float(np.mean(list(catmean.values())))
            rows.append((s, t, avgv, avgc, catmean))
            if best is None or avgc > best[3]:
                best = (s, t, avgv, avgc, catmean)
        if best:
            print(f"{best[0]:7s} {best[1]:>4d} {best[2]:9.4f} {best[3]:8.4f}  <-best tau")
            for c in sorted(best[4]):
                print(f"        {c:28s} {best[4][c]:.4f}")
    with open(out_csv, "w") as f:
        json.dump([{"src": s, "tau": t, "avg_video": av, "avg_category": ac, "per_cat": cm}
                   for (s, t, av, ac, cm) in rows], f, indent=2)
    print("wrote", out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--med-sub", type=int, default=20)
    ap.add_argument("--taus", default="2,5,8,10,15,20,25,30,40,50")
    ap.add_argument("--out", default="logs/diag_oracle_bg.json")
    ap.add_argument("--root", default="datasets/cdnet/CDnet14")
    a = ap.parse_args()
    TAU = [int(x) for x in a.taus.split(",")]
    run(a.stride, TAU, a.med_sub, a.out, root=a.root)
