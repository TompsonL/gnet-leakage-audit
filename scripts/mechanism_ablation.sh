#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# E3b: mechanism net-gain chain. Trains the missing single-factor variant
# (M1+M3: reliability gate + boundary, WITHOUT the T3 data engine) so the chain
#   no-bg -> +bg -> +reliability -> +boundary -> +T3
# is isolated. Reuses already-trained models; runs last.
cd "$ROOT"
export PYTHONPATH=.
while ! [ -f logs/MOTION_DONE ]; do sleep 120; done
export CUDA_VISIBLE_DEVICES=0

echo "[$(date '+%F %T')] train M3 (rel+boundary, no T3)"
python3 train.py --rel-gate --boundary --seed 0 \
  --batch 8 --workers 6 --size 513 --epochs 8 --steps-per-epoch 600 --lr 0.01 --synth-bg causal \
  --out checkpoints/gnet_m3 > logs/train_m3.log 2>&1

echo "[$(date '+%F %T')] eval M3 (causal_rel, stride 1)"
python3 eval.py --weights checkpoints/gnet_m3/last.pt --root datasets/cdnet/CDnet14 \
  --bg-source causal_med --frame-stride 1 --boundary \
  --out logs/fair_m3_causal.csv > logs/fair_m3_causal.log 2>&1

echo "[$(date '+%F %T')] MECHANISM DONE"
touch logs/M3_DONE
