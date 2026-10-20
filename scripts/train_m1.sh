#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# M1: train GNet with a learnable reliability gate (synthetic PMB reliability),
# then evaluate on full CDnet14 with the causal reliability map (official stride).
cd "$ROOT"
export PYTHONPATH=.
# wait for the fair-protocol GPU1 queue (cfo + seed2) to release the GPU
while ! grep -q "FAIR MAIN GPU1 DONE" logs/fair_main1_done.log 2>/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=1
echo "[$(date '+%F %T')] train M1 (rel-gate)"
python3 train.py --rel-gate --seed 0 --batch 8 --workers 6 --size 513 \
  --epochs 8 --steps-per-epoch 600 --lr 0.01 --synth-bg causal --out checkpoints/gnet_m1 \
  > logs/train_m1.log 2>&1
echo "[$(date '+%F %T')] eval M1 causal_rel (stride 1)"
python3 eval.py --weights checkpoints/gnet_m1/last.pt --root datasets/cdnet/CDnet14 \
  --bg-source causal_med --frame-stride 1 --boundary \
  --out logs/fair_m1_causal.csv > logs/fair_m1_causal.log 2>&1
echo "[$(date '+%F %T')] M1 DONE"
touch logs/M1_DONE
