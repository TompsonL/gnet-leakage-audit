#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Full method: M1 (reliability gate) + M3 (boundary attention) + T3 (degradation
# simulation + shadow). Trains after the fair-ablation queue releases the GPUs.
cd "$ROOT"
export PYTHONPATH=.
while ! grep -q "ABLATION GPU0 DONE" logs/fair_abl_gpu0.log 2>/dev/null; do sleep 60; done
while ! grep -q "ABLATION GPU1 DONE" logs/fair_abl_gpu1.log 2>/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=0
echo "[$(date '+%F %T')] train FULL (M1+M3+T3)"
python3 train.py --rel-gate --boundary --degrade --shadow --seed 0 \
  --batch 8 --workers 6 --size 513 --epochs 8 --steps-per-epoch 600 --lr 0.01 --synth-bg causal \
  --out checkpoints/gnet_full > logs/train_full.log 2>&1
echo "[$(date '+%F %T')] eval FULL causal_rel (+boundary)"
python3 eval.py --weights checkpoints/gnet_full/last.pt --root datasets/cdnet/CDnet14 \
  --bg-source causal_med --frame-stride 1 --boundary \
  --out logs/fair_full_causal.csv > logs/fair_full_causal.log 2>&1
echo "[$(date '+%F %T')] FULL DONE"
touch logs/FULL_DONE
