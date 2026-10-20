"""Evaluate a trained GNet on CDnet 2014 / BMC 2012 (F1, PWC)."""
import argparse
import os
import csv
import json
from collections import defaultdict

import numpy as np
import cv2
import torch

from models.gnet import GNet
from data.cdnet import CDnetDataset
from utils.metrics import confusion, f1_pwc
from utils.protocol import check_bg_source, protocol_record


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--root", required=True, help="dir containing <category>/<video>/")
    p.add_argument("--size", type=int, default=513)
    p.add_argument("--bg-source", default="causal", choices=["provided","firstframe","reconstructed","running","causal","causal_rel","causal_relfb","causal_med","median","self"])
    p.add_argument("--fusion", default="cat")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--categories", default="")
    p.add_argument("--limit-videos", type=int, default=0)
    p.add_argument("--split-file", default="", help="keep only 'cat/video' lines")
    p.add_argument("--out", default="logs/eval.csv")
    p.add_argument("--boundary", action="store_true")
    p.add_argument("--allow-leaky", action="store_true", help="permit SuBSENSE-guided (leaky) background for the leakage-control experiment")
    p.add_argument("--allow-oracle", action="store_true", help="permit future-frame oracle background (diagnostics only)")
    # E8: controlled background perturbation (test-time robustness)
    p.add_argument("--bg-noise", type=float, default=0.0, help="gaussian noise sigma added to background (0-1 scale)")
    p.add_argument("--bg-ghost", type=float, default=0.0, help="blend current frame into background (static-foreground residue)")
    p.add_argument("--bg-illum", type=float, default=0.0, help="global illumination gain error on background (>0)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    a = get_args()
    bg_class = check_bg_source(a.bg_source, allow_leaky=a.allow_leaky,
                               allow_oracle=a.allow_oracle)
    if a.frame_stride != 1:
        print(f"[protocol] WARNING: frame_stride={a.frame_stride} != 1; "
              f"not the official CDnet frame set.", flush=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    print(json.dumps(vars(a)), flush=True)
    print(f"[protocol] class={bg_class} stride={a.frame_stride}", flush=True)
    ckpt = torch.load(a.weights, map_location="cpu")
    cargs = ckpt.get("args", {})
    fusion = cargs.get("fusion", a.fusion)
    tsam_depth = 1 if cargs.get("no_tsam", False) else cargs.get("tsam_depth", 12)
    rel_gate = cargs.get("rel_gate", False)
    boundary = cargs.get("boundary", False)
    model = GNet(num_classes=1, embed_dim=cargs.get("embed_dim", 256),
                 tsam_depth=tsam_depth,
                 tsam_heads=cargs.get("tsam_heads", 8),
                 tsam_patch=cargs.get("tsam_patch", 5),
                 fusion=fusion, pretrained=False, rel_gate=rel_gate,
                 boundary=boundary)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"loaded {a.weights} fusion={fusion}", flush=True)

    cats = a.categories.split(",") if a.categories else None
    ds = CDnetDataset(a.root, split_file=(a.split_file or None), size=a.size,
                      bg_source=a.bg_source, frame_stride=a.frame_stride,
                      max_frames=a.max_frames or None,
                      categories=cats, limit_videos=a.limit_videos or None)
    print(f"samples: {len(ds)}", flush=True)

    agg = defaultdict(lambda: [0, 0, 0, 0])  # video -> TP,TN,FP,FN
    bnd = defaultdict(list)  # video -> boundary F1 per frame
    from utils.boundary import boundary_f1
    with torch.no_grad():
        for k in range(len(ds)):
            img, bg, meta = ds[k]
            if a.bg_ghost > 0:
                bg = (1 - a.bg_ghost) * bg + a.bg_ghost * img
            if a.bg_illum > 0:
                g = 1.0 + a.bg_illum
                bg = torch.clamp(bg * g, 0, 1)
            if a.bg_noise > 0:
                bg = torch.clamp(bg + torch.randn_like(bg) * a.bg_noise, 0, 1)
            # causal_rel = training-free reliability gating (always apply);
            # causal_med = reliability gating only for gate-trained models,
            # otherwise it is just a (median) background source.
            use_rel = (a.bg_source == "causal_rel") or \
                      (a.bg_source == "causal_med" and rel_gate)
            rel = meta.get("rel") if use_rel else None
            if rel is not None:
                rel = rel[None].to(device)
            logit = model(img[None].to(device), bg[None].to(device), rel)
            prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
            pred = (prob >= a.threshold).astype(np.uint8)
            h, w = meta["h"], meta["w"]
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            gt = meta.get("gt")
            if gt is None:
                continue
            roi = meta.get("roi")
            if roi is not None:
                gt = gt.copy()
                gt[(roi < 128)] = 85  # outside ROI -> ignore
            tp, tn, fp, fn = confusion(pred, gt)
            v = f"{meta['cat']}/{meta['video']}"
            s = agg[v]; s[0] += tp; s[1] += tn; s[2] += fp; s[3] += fn
            if a.boundary:
                ign = (gt == 85) | (gt == 170)
                bf1, _, _ = boundary_f1(pred, gt, tol=2, ignore_mask=ign)
                bnd[v].append(bf1)
            if k % 200 == 0:
                print(f"  {k}/{len(ds)} {v} f={meta['idx']}", flush=True)

    per_video = {}
    per_cat = defaultdict(list)
    for v, (tp, tn, fp, fn) in agg.items():
        f1, pwc, prec, rec = f1_pwc(tp, tn, fp, fn)
        per_video[v] = (f1, pwc, prec, rec)
        cat = v.split("/")[0]
        per_cat[cat].append((f1, pwc))
    cat_mean = {c: (float(np.mean([x[0] for x in vals])),
                    float(np.mean([x[1] for x in vals]))) for c, vals in per_cat.items()}
    overall_f1 = float(np.mean([x[0] for x in per_video.values()])) if per_video else 0
    overall_pwc = float(np.mean([x[1] for x in per_video.values()])) if per_video else 0
    catofcat_f1 = float(np.mean([x[0] for x in cat_mean.values()])) if cat_mean else 0
    catofcat_pwc = float(np.mean([x[1] for x in cat_mean.values()])) if cat_mean else 0

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["video", "F1", "PWC", "Precision", "Recall"] + (["BF1"] if a.boundary else []))
        for v in sorted(per_video):
            f1, pwc, prec, rec = per_video[v]
            row = [v, f"{f1:.4f}", f"{pwc:.4f}", f"{prec:.4f}", f"{rec:.4f}"]
            if a.boundary:
                row.append(f"{np.mean(bnd[v]):.4f}" if bnd[v] else "0.0000")
            wr.writerow(row)
        wr.writerow([])
        wr.writerow(["category", "F1", "PWC"])
        for c in sorted(cat_mean):
            wr.writerow([c, f"{cat_mean[c][0]:.4f}", f"{cat_mean[c][1]:.4f}"])
        wr.writerow([])
        wr.writerow(["AVG_video", f"{overall_f1:.4f}", f"{overall_pwc:.4f}"])
        wr.writerow(["AVG_category", f"{catofcat_f1:.4f}", f"{catofcat_pwc:.4f}"])
    summary = {"F1_video_avg": round(overall_f1, 4),
               "PWC_video_avg": round(overall_pwc, 4),
               "F1_category_avg": round(catofcat_f1, 4),
               "PWC_category_avg": round(catofcat_pwc, 4)}
    record = protocol_record(a.bg_source, a.frame_stride, a.threshold, a.boundary)
    record.update({"weights": a.weights, "root": a.root, "size": a.size,
                   "fusion": fusion, "categories": a.categories or "all",
                   "n_videos": len(per_video),
                   "per_category": {c: {"F1": round(v[0], 4), "PWC": round(v[1], 4)}
                                    for c, v in sorted(cat_mean.items())},
                   "summary": summary})
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
