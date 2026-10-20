#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# RQ1 stride=1 recheck of the leakage-control table (4 held-out categories).
# Same weights, only the background source changes:
#   reconstructed = leaky (SuBSENSE-guided causal running)
#   firstframe    = static first frame
#   provided      = dataset static background = whole-video median (future-frame oracle)
# The causal column is taken from the full causal evaluation (fair_*_causal.csv),
# so it is not re-run here.
# Runs on the idle GPU (default 1) while the causal-median evaluation occupies GPU0.
cd "$ROOT"
GPU=${1:-1}
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU
CATS=badWeather,nightVideos,turbulence,thermal
for model in gnet_main gnet_real; do
  [ -f "checkpoints/$model/last.pt" ] || { echo "missing $model"; continue; }
  for bg in reconstructed firstframe provided; do
    echo "[$(date '+%F %T')] leak1 $model bg=$bg"
    python3 eval.py --weights "checkpoints/$model/last.pt" --root datasets/cdnet/CDnet14 \
      --categories "$CATS" --bg-source "$bg" --frame-stride 1 \
      --allow-leaky --allow-oracle \
      --out "logs/leak1_${model}_${bg}.csv" > "logs/leak1_${model}_${bg}.log" 2>&1
  done
done
echo "[$(date '+%F %T')] LEAK1 DONE"
touch logs/LEAK1_DONE
