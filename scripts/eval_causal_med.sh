#!/bin/bash
# DEPRECATED: per-frame bg+rel cache needs ~340GB on full CDnet (OOM).
# L2a is reported from scripts/tune_bg.py (5-video small-sample) instead.
# L2a: re-evaluate with the causal MEDIAN background (K=10) chosen by tune_bg,
# versus the running EMA. Reuses trained models; runs last.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
cd "$ROOT"
export PYTHONPATH=.
while ! [ -f logs/M3_DONE ]; do sleep 120; done
export CUDA_VISIBLE_DEVICES=0
for m in gnet_main_seed0 gnet_m1 gnet_full; do
  [ -f checkpoints/$m/last.pt ] || { echo "skip $m"; continue; }
  bg=causal_med
  echo "[$(date '+%F %T')] eval $m bg=causal_med (stride 1)"
  python3 eval.py --weights checkpoints/$m/last.pt --root datasets/cdnet/CDnet14 \
    --bg-source "$bg" --frame-stride 1 --boundary \
    --out "logs/fair_${m}_causalmed.csv" > "logs/fair_${m}_causalmed.log" 2>&1
done
echo "[$(date '+%F %T')] CAUSAL_MED DONE"
touch logs/CMED_DONE
