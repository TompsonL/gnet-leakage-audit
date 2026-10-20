#!/usr/bin/env python3
"""Consolidated results report for the frozen-protocol experiments.

Sections:
  E1   main comparison (local runs, from JSON sidecars / legacy CSVs)
  GAP  oracle gap = F1(causal) - F1(oracle), matched per model
  E3   ablation table (fair_<tag>_causal)
  E8   background-error robustness curves (ghost / illum / noise)

Usage: python3 scripts/report.py [--logs logs]
"""
import argparse
import glob
import json
import os
import re


def read_json(logs):
    out = {}
    for f in sorted(glob.glob(os.path.join(logs, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if isinstance(d, dict) and "summary" in d:
            out[os.path.basename(f)] = d
    return out


def read_csv(logs):
    res = {}
    for f in sorted(glob.glob(os.path.join(logs, "*.csv"))):
        f1 = None
        for line in open(f):
            if line.startswith("AVG_category,"):
                p = line.strip().split(",")
                f1 = float(p[1])
        if f1 is not None:
            res[os.path.basename(f)] = f1
    return res


def short(m):
    if not m:
        return "-"
    m = re.sub(r"checkpoints/", "", m)
    m = re.sub(r"/last\.pt", "", m)
    return m


def section(t):
    print(f"\n## {t}")


def e1(json_d, csv_d):
    section("E1  main comparison (local, frozen protocol)")
    print("| model | bg | class | stride | F1(cat) | PWC(cat) | src |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for name, d in json_d.items():
        s = d["summary"]
        rows.append((short(d.get("weights") or d.get("method", name)),
                     d.get("bg_source", "?"), d.get("bg_class", "?"),
                     d.get("frame_stride", "?"), s.get("F1_category_avg"),
                     s.get("PWC_category_avg"), "json"))
    for name, f1 in csv_d.items():
        if name.startswith("eval_"):
            model = name[5:-4]
            bg = "reconstructed" if "recon" in name else ""
            rows.append((model, bg, "legacy", "legacy", f1, None, "csv"))
    for m, bg, cls, st, f1, pwc, src in sorted(rows, key=lambda x: -(x[4] or 0)):
        f1s = f"{f1:.4f}" if isinstance(f1, float) else "-"
        pwcs = f"{pwc:.4f}" if isinstance(pwc, float) else "-"
        print(f"| {m} | {bg} | {cls} | {st} | {f1s} | {pwcs} | {src} |")


def gaps(json_d):
    section("GAP  oracle gap = F1(causal/self) - F1(oracle provided/median)")
    by = {}
    for name, d in json_d.items():
        if "cdnet" not in str(d.get("root", "")):
            continue
        key = short(d.get("weights") or d.get("method", name))
        by.setdefault(key, {})[d.get("bg_source")] = d["summary"].get("F1_category_avg")
    print("| model | causal | oracle | gap |")
    print("|---|---|---|---|")
    for m, d in sorted(by.items()):
        c = d.get("causal") or d.get("causal_rel") or d.get("self")
        o = d.get("provided") or d.get("median")
        if isinstance(c, float) and isinstance(o, float):
            print(f"| {m} | {c:.4f} | {o:.4f} | {c - o:+.4f} |")


def e3(json_d):
    section("E3  ablations (fair_<tag>_causal)")
    got = False
    for name, d in sorted(json_d.items()):
        if name.startswith("fair_") and d.get("bg_source") in ("causal", "causal_rel", "self"):
            tag = short(d.get("weights") or name)
            print(f"- {tag}: F1={d['summary']['F1_category_avg']:.4f} "
                  f"PWC={d['summary']['PWC_category_avg']:.4f}")
            got = True
    if not got:
        print("(pending)")


def mechanism(json_d):
    section("E3b  mechanism chain (net gain, same protocol)")
    want = [("仅当前帧", "gnet_cfo"), ("+背景", "gnet_main_seed0"),
            ("+可靠性(M1)", "gnet_m1"), ("+边界(M3)", "gnet_m3"),
            ("+T3退化(shadow)", "gnet_full")]
    by = {}
    for name, d in json_d.items():
        if "cdnet" not in str(d.get("root", "")):
            continue
        key = short(d.get("weights") or d.get("method", name))
        s = d["summary"]
        by[key] = (s.get("F1_category_avg"), s.get("PWC_category_avg"),
                   d.get("bg_source"), d.get("frame_stride"))
    print("| 阶段 | 模型 | bg | stride | F1(cat) | PWC(cat) |")
    print("|---|---|---|---|---|---|")
    for label, m in want:
        if m in by:
            f1, pwc, bg, st = by[m]
            print(f"| {label} | {m} | {bg} | {st} | {f1:.4f} | {pwc:.4f} |")
        else:
            print(f"| {label} | {m} | - | - | 待出 | - |")


def e8(csv_d):
    section("E8  background-error robustness (F1 category-avg)")
    pat = re.compile(r"e8_(?P<model>.+)_(?P<kind>ghost|illum|noise)(?P<lvl>[0-9.]+)\.csv$")
    rows = {}
    for name, f1 in csv_d.items():
        m = pat.match(name)
        if m:
            rows.setdefault((m["model"], m["kind"]), []).append((float(m["lvl"]), f1))
    if not rows:
        print("(pending)")
        return
    for (model, kind), vals in sorted(rows.items()):
        vals.sort()
        print(f"- **{model} {kind}**: " + "  ".join(f"{l}={f:.3f}" for l, f in vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    a = ap.parse_args()
    j = read_json(a.logs)
    c = read_csv(a.logs)
    e1(j, c)
    gaps(j)
    mechanism(j)
    e3(j)
    e8(c)


if __name__ == "__main__":
    main()
