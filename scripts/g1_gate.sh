#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Route C, gate G1: does the mask-free causal background + reliability (M1)
# close the gap to the future-frame oracle without collapsing?
# Runs gnet_main_seed0 on the hard categories at official stride, four bg modes:
#   causal      : mask-free causal running avg (baseline)
#   causal_rel  : same bg, reliability map gates the differential branch (training-free M1)
#   causal_relfb: unreliable pixels fall back to the current frame
#   provided    : future-frame oracle (upper bound, diagnostics)
cd "$ROOT"
export PYTHONPATH=.
# wait for the fair-protocol jobs to release the GPUs
while ! grep -q "FAIR MAIN GPU0 DONE" logs/fair_main0_done.log 2>/dev/null; do sleep 60; done
while ! grep -q "FAIR MAIN GPU1 DONE" logs/fair_main1_done.log 2>/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=0
CATS=PTZ,nightVideos,intermittentObjectMotion,thermal
W=checkpoints/gnet_main_seed0/last.pt
for bg in causal causal_rel causal_relfb; do
  echo "[$(date '+%F %T')] G1 $bg"
  python3 eval.py --weights "$W" --root datasets/cdnet/CDnet14 --categories "$CATS" \
    --bg-source "$bg" --frame-stride 1 --boundary \
    --out "logs/g1_${bg}.csv" > "logs/g1_${bg}.log" 2>&1
done
echo "[$(date '+%F %T')] G1 oracle (diagnostic)"
python3 eval.py --weights "$W" --root datasets/cdnet/CDnet14 --categories "$CATS" \
  --bg-source provided --allow-oracle --frame-stride 1 --boundary \
  --out "logs/g1_provided.csv" > "logs/g1_provided.log" 2>&1
echo "[$(date '+%F %T')] G1 GATE DONE"
touch logs/G1_DONE
