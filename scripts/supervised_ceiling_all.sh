#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Setting (2): full official 4-fold cross-validation, video-agnostic supervised
# GNet on CDnet GT labels, frozen causal-median background, stride 1.
# Runs after the main queue; trains/evaluates folds 1..4 sequentially on GPU0.
cd "$ROOT"
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
while ! [ -f logs/CMED_DONE ]; do sleep 120; done
for FOLD in 1 2 3 4; do
  echo "[$(date '+%F %T')] fold${FOLD} train (GT, causal_med bg)"
  python3 train_real.py --root datasets/cdnet/CDnet14 \
    --split-file splits/cv4_fold${FOLD}_train.txt --label gt --bg causal_med \
    --seed 0 --size 513 --epochs 8 --steps-per-epoch 600 --batch 8 --lr 0.01 \
    --out checkpoints/gnet_sup_fold${FOLD} > logs/train_sup_fold${FOLD}.log 2>&1
  echo "[$(date '+%F %T')] fold${FOLD} eval (test split, causal_med, stride 1)"
  python3 eval.py --weights checkpoints/gnet_sup_fold${FOLD}/last.pt \
    --root datasets/cdnet/CDnet14 --split-file splits/cv4_fold${FOLD}_test.txt \
    --bg-source causal_med --frame-stride 1 --boundary \
    --out logs/sup_fold${FOLD}.csv > logs/sup_fold${FOLD}.log 2>&1
done
echo "[$(date '+%F %T')] SUPERVISED CV DONE"
touch logs/SUP_CV_DONE
