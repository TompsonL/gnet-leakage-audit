#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Train one seed of the main model and evaluate on full CDnet 2014 (stride 5).
SEED=${1:?seed}
GPU=${2:-0}
cd "$ROOT"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU
OUT=checkpoints/gnet_main_seed${SEED}
C="--seed $SEED --batch 8 --workers 6 --size 513 --epochs 8 --steps-per-epoch 600 --lr 0.01"
echo "[$(date '+%F %T')] train seed $SEED"
python3 train.py $C --out "$OUT" > "logs/train_seed${SEED}.log" 2>&1
echo "[$(date '+%F %T')] eval seed $SEED (full CDnet)"
python3 eval.py --weights "$OUT/last.pt" --root datasets/cdnet/CDnet14 \
  --bg-source reconstructed --frame-stride 5 --out "logs/eval_seed${SEED}_full.csv" \
  > "logs/eval_seed${SEED}_full.log" 2>&1
echo "[$(date '+%F %T')] SEED $SEED DONE"
