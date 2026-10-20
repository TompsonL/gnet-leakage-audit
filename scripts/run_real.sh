#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# After GPU1 queue finishes, train GNet on real CDnet pseudo-labels and eval on held-out categories.
cd "$ROOT"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=1
while ! grep -q "GPU1 DONE" logs/run_gpu1.log 2>/dev/null; do sleep 180; done
echo "[$(date '+%F %T')] start real-domain training"
python3 train_real.py --root datasets/cdnet/CDnet14 \
  --holdout badWeather,nightVideos,turbulence,thermal \
  --epochs 8 --steps-per-epoch 600 --batch 8 --lr 0.01 \
  --out checkpoints/gnet_real > logs/train_real.log 2>&1
echo "[$(date '+%F %T')] real training done; eval on holdout + synthetic-trained main on same holdout"
python3 eval.py --weights checkpoints/gnet_real/last.pt --root datasets/cdnet/CDnet14 \
  --categories badWeather,nightVideos,turbulence,thermal --bg-source reconstructed \
  --frame-stride 3 --out logs/eval_real_holdout.csv > logs/eval_real_holdout.log 2>&1
python3 eval.py --weights checkpoints/gnet_main/last.pt --root datasets/cdnet/CDnet14 \
  --categories badWeather,nightVideos,turbulence,thermal --bg-source reconstructed \
  --frame-stride 3 --out logs/eval_main_holdout.csv > logs/eval_main_holdout.log 2>&1
echo "[$(date '+%F %T')] REAL EXPERIMENT DONE"
