#!/usr/bin/env python3
"""Collect the final paper tables from per-video result files.

Reads the stride-1 leakage recheck (logs/leak1_*), the full causal evaluations
for the causal column, and the stride-1 E8 key points. Prints ready-to-paste
LaTeX rows and flags missing runs.
"""
import csv
import os

HOLDOUT = ["badWeather", "nightVideos", "thermal", "turbulence"]


def avg_category(path, cats=None):
    rows = [r for r in csv.DictReader(open(path)) if "/" in r["video"]]
    by = {}
    for r in rows:
        by.setdefault(r["video"].split("/")[0], []).append(float(r["F1"]))
    keys = cats if cats else sorted(by)
    vals = [sum(by[c]) / len(by[c]) for c in keys if c in by]
    return sum(vals) / len(vals)


def causal_holdout(path):
    return avg_category(path)


def main():
    print("== RQ1 stride-1 table (4 held-out categories) ==")
    sources = ["reconstructed", "causal", "firstframe", "provided"]
    for model, causal_csv in [
        ("gnet_main", "logs/fair_seed0_causal.csv"),
        ("gnet_real", "logs/fair_gnet_real_causal.csv"),
    ]:
        vals = {}
        for s in sources:
            if s == "causal":
                p = causal_csv
            else:
                p = "logs/leak1_%s_%s.csv" % (model, s)
            vals[s] = avg_category(p, HOLDOUT) if os.path.isfile(p) else None
        if all(v is not None for v in vals.values()):
            dl = vals["reconstructed"] - vals["causal"]
            do = vals["provided"] - vals["causal"]
            print("%s & %.4f & %.4f & %.4f & %.4f & %+.4f & %+.4f \\\\" % (
                model, vals["reconstructed"], vals["causal"], vals["firstframe"],
                vals["provided"], dl, do))
        else:
            print("%s: %s" % (model, {k: (None if v is None else round(v, 4))
                                     for k, v in vals.items()}))

    print("\n== E8 stride-1 key points (4 categories) ==")
    for model in ["gnet_main_seed0", "gnet_real"]:
        row = []
        for tag in ["base", "ghost0.30", "illum0.20", "noise0.10"]:
            p = "logs/e8s1_%s_%s.csv" % (model, tag)
            row.append(avg_category(p) if os.path.isfile(p) else None)
        print("%s & %s \\\\" % (model, " & ".join(
            "%.4f" % v if v is not None else "---" for v in row)))


if __name__ == "__main__":
    main()
