#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# E8: background-error robustness curve. Test-time perturbation of the causal
# background (ghost / illumination / noise). A leak-resistant method degrades
# gracefully; a leak-prone one (using SuBSENSE-guided bg) collapses.
# Runs last, after the full-method training queue.
cd "$ROOT"
export PYTHONPATH=.
while ! [ -f logs/FULL_DONE ]; do sleep 60; done
export CUDA_VISIBLE_DEVICES=0
CATS=PTZ,nightVideos,intermittentObjectMotion,thermal
for model in gnet_main_seed0 gnet_real; do
  W=checkpoints/$model/last.pt
  [ -f "$W" ] || continue
  # ghost residue
  for v in 0.0 0.15 0.30; do
    echo "[$(date '+%F %T')] E8 $model ghost=$v"
    python3 eval.py --weights "$W" --root datasets/cdnet/CDnet14 --categories "$CATS" \
      --bg-source causal --frame-stride 5 --bg-ghost "$v" \
      --out "logs/e8_${model}_ghost${v}.csv" > "logs/e8_${model}_ghost${v}.log" 2>&1
  done
  # illumination
  for v in 0.0 0.10 0.20; do
    echo "[$(date '+%F %T')] E8 $model illum=$v"
    python3 eval.py --weights "$W" --root datasets/cdnet/CDnet14 --categories "$CATS" \
      --bg-source causal --frame-stride 5 --bg-illum "$v" \
      --out "logs/e8_${model}_illum${v}.csv" > "logs/e8_${model}_illum${v}.log" 2>&1
  done
  # noise
  for v in 0.0 0.05 0.10; do
    echo "[$(date '+%F %T')] E8 $model noise=$v"
    python3 eval.py --weights "$W" --root datasets/cdnet/CDnet14 --categories "$CATS" \
      --bg-source causal --frame-stride 5 --bg-noise "$v" \
      --out "logs/e8_${model}_noise${v}.csv" > "logs/e8_${model}_noise${v}.log" 2>&1
  done
done
echo "[$(date '+%F %T')] E8 DONE"
touch logs/E8_DONE
