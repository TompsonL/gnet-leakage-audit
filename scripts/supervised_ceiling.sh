#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Setting (2) ceiling: video-agnostic supervised GNet on CDnet GT labels, using
# the official BSUV-Net 4-fold split. Trains on 3 folds, tests on the held-out
# fold, frozen causal-median background, stride 1. Estimate how far our
# architecture can go under the supervised video-agnostic regime.
# Usage: bash scripts/supervised_ceiling.sh [fold]  (default 1)
cd "$ROOT"
export PYTHONPATH=.
FOLD=${1:-1}
while ! [ -f logs/CMED_DONE ]; do sleep 120; done
export CUDA_VISIBLE_DEVICES=0

echo "[$(date '+%F %T')] train supervised fold${FOLD} (GT labels, causal_med bg)"
python3 train_real.py --root datasets/cdnet/CDnet14 \
  --split-file splits/cv4_fold${FOLD}_train.txt --label gt --bg causal_med \
  --seed 0 --size 513 --epochs 8 --steps-per-epoch 600 --batch 8 --lr 0.01 \
  --out checkpoints/gnet_sup_fold${FOLD} > logs/train_sup_fold${FOLD}.log 2>&1

echo "[$(date '+%F %T')] eval supervised fold${FOLD} (test split, causal_med, stride 1)"
python3 eval.py --weights checkpoints/gnet_sup_fold${FOLD}/last.pt \
  --root datasets/cdnet/CDnet14 --split-file splits/cv4_fold${FOLD}_test.txt \
  --bg-source causal_med --frame-stride 1 --boundary \
  --out logs/sup_fold${FOLD}.csv > logs/sup_fold${FOLD}.log 2>&1

echo "[$(date '+%F %T')] SUPERVISED FOLD${FOLD} DONE"
touch logs/SUP_FOLD${FOLD}_DONE
