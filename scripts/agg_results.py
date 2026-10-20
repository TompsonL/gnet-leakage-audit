#!/usr/bin/env python3
"""Aggregate evaluation results into a single table + oracle-gap report.

Reads the new JSON sidecars (logs/*.json, produced by eval.py / eval_subsense.py)
and legacy CSVs (logs/eval_*.csv) and prints a markdown table.

Usage:
    python3 scripts/agg_results.py            # all results
    python3 scripts/agg_results.py --gaps     # oracle gap for matched runs
"""
import argparse
import csv
import glob
import json
import os
import re


def load_jsons(logs="logs"):
    rows = []
    for f in sorted(glob.glob(os.path.join(logs, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "summary" not in d:
            continue
        s = d["summary"]
        rows.append({
            "file": os.path.basename(f),
            "model": d.get("weights") or d.get("method") or "-",
            "bg": d.get("bg_source", "?"),
            "cls": d.get("bg_class", "?"),
            "stride": d.get("frame_stride", "?"),
            "f1": s.get("F1_category_avg"),
            "pwc": s.get("PWC_category_avg"),
            "nvid": d.get("n_videos"),
            "src": "json",
        })
    return rows


def load_legacy_csvs(logs="logs"):
    rows = []
    for f in sorted(glob.glob(os.path.join(logs, "eval_*.csv"))):
        f1 = pwc = None
        for line in open(f):
            if line.startswith("AVG_category,"):
                parts = line.strip().split(",")
                f1, pwc = float(parts[1]), float(parts[2])
        if f1 is None:
            continue
        name = os.path.basename(f)[len("eval_"):-len(".csv")]
        rows.append({"file": os.path.basename(f), "model": name,
                     "bg": "", "cls": "legacy", "stride": "", "f1": f1, "pwc": pwc,
                     "nvid": "", "src": "csv"})
    return rows


def short(m):
    if m in ("-", None):
        return "-"
    m = re.sub(r"checkpoints/", "", m)
    m = re.sub(r"/last\.pt", "", m)
    return m


def print_table(rows):
    rows = sorted(rows, key=lambda r: (str(r["stride"]), r["cls"], -(r["f1"] or 0)))
    print("\n| model | bg | class | stride | F1(cat) | PWC(cat) | nvid | src |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        f1 = f"{r['f1']:.4f}" if isinstance(r["f1"], float) else "-"
        pwc = f"{r['pwc']:.4f}" if isinstance(r["pwc"], float) else "-"
        print(f"| {short(r['model'])} | {r['bg']} | {r['cls']} | {r['stride']} | "
              f"{f1} | {pwc} | {r['nvid']} | {r['src']} |")


def print_gaps(rows):
    """oracle gap = F1(oracle) - F1(causal), matched by model file stem."""
    by_model = {}
    for r in rows:
        key = short(r["model"])
        by_model.setdefault(key, {})[r["bg"]] = r["f1"]
    print("\n## oracle gap (F1_causal - F1_oracle; smaller = more leak-resistant)\n")
    print("| model | causal | oracle | gap |")
    print("|---|---|---|---|")
    for m, d in sorted(by_model.items()):
        c = d.get("causal") or d.get("causal_rel") or d.get("self")
        o = d.get("provided") or d.get("median")
        if isinstance(c, float) and isinstance(o, float):
            print(f"| {m} | {c:.4f} | {o:.4f} | {c - o:+.4f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--gaps", action="store_true")
    a = ap.parse_args()
    rows = load_jsons(a.logs) + load_legacy_csvs(a.logs)
    print_table(rows)
    if a.gaps:
        print_gaps(rows)


if __name__ == "__main__":
    main()
