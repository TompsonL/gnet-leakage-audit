#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Leakage control: evaluate main + real models under different background sources.
# reconstructed = causal running avg updated only on SuBSENSE background pixels (info leak)
# causal        = mask-free causal running avg (no SuBSENSE info)
# firstframe    = static first frame (no temporal info)
# provided      = dataset static background = whole-video temporal median (future-frame oracle)
cd "$ROOT"
GPU=${1:-0}
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU
CATS=badWeather,nightVideos,turbulence,thermal
for model in gnet_main gnet_real; do
  [ -f "checkpoints/$model/last.pt" ] || { echo "missing $model"; continue; }
  for bg in reconstructed causal firstframe provided; do
    echo "[$(date '+%F %T')] eval $model bg=$bg"
    # leakage-control experiment: reconstructed = leaky, provided = oracle
    python3 eval.py --weights "checkpoints/$model/last.pt" --root datasets/cdnet/CDnet14 \
      --categories "$CATS" --bg-source "$bg" --frame-stride 3 \
      --allow-leaky --allow-oracle \
      --out "logs/leak_${model}_${bg}.csv" > "logs/leak_${model}_${bg}.log" 2>&1
  done
done
echo "[$(date '+%F %T')] LEAK ABLATION DONE"
touch logs/LEAK_DONE
