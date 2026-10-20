#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# T4: label-free training from the causal teacher (no human annotation).
# Trains on the CDnet "realtrain" split (7 cats) and evaluates on the held-out
# 4 categories under the frozen causal protocol. Precomputes the teacher cache
# on CPU first (the slow part), then trains after the GPU queue is free.
cd "$ROOT"
export PYTHONPATH=.

# (1) precompute teacher masks + causal backgrounds for the training split (CPU)
echo "[$(date '+%F %T')] precompute teacher cache"
python3 - <<'PY' > logs/motion_precompute.log 2>&1
from data.motion_dataset import MotionLabelDataset
ds = MotionLabelDataset("datasets/cdnet/CDnet14", split_file="splits/cdnet14_realtrain.txt",
                        K=30, thr=35, gmc="translation", label_stride=5,
                        cache_dir="logs/motion_cache")
print("videos:", len(ds.vdirs))
for v in ds.vdirs:
    ds._video(v)
print("cache done")
PY
echo "[$(date '+%F %T')] teacher cache ready"

# (2) wait for the GPU queue to free
while ! [ -f logs/E8_DONE ]; do sleep 120; done
export CUDA_VISIBLE_DEVICES=0

echo "[$(date '+%F %T')] train label-free"
python3 train.py --motion-labels --motion-root datasets/cdnet/CDnet14 \
  --motion-split splits/cdnet14_realtrain.txt --motion-label-stride 5 \
  --seed 0 --batch 8 --workers 6 --size 513 --epochs 8 --steps-per-epoch 600 --lr 0.01 \
  --out checkpoints/gnet_motion > logs/train_motion.log 2>&1

echo "[$(date '+%F %T')] eval holdout (causal, stride 1)"
python3 eval.py --weights checkpoints/gnet_motion/last.pt --root datasets/cdnet/CDnet14 \
  --categories badWeather,nightVideos,turbulence,thermal --bg-source causal --frame-stride 1 \
  --out logs/fair_motion_holdout.csv > logs/fair_motion_holdout.log 2>&1
echo "[$(date '+%F %T')] MOTION DONE"
touch logs/MOTION_DONE
