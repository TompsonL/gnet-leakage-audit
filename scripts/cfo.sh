#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Fair-protocol additions:
#   (1) SuBSENSE "floor" under the CDnet14 protocol (CPU)
#   (2) current-frame-only (no background) model + eval (GPU1)
#   (3) zero-cost "self" (bg=current frame) degradation of the bg-trained model (GPU0)
cd "$ROOT"
export PYTHONPATH=.

# (1) floor baseline
nohup python3 eval_subsense.py --root datasets/cdnet/CDnet14 --frame-stride 1 \
  --out logs/subsense_baseline.csv > logs/subsense_baseline.log 2>&1 &
echo "subsense-baseline pid $!"

# (2) current-frame-only model
nohup bash -c '
  cd "$ROOT"
  export PYTHONPATH=. CUDA_VISIBLE_DEVICES=1
  python3 train.py --no-bg --seed 0 --batch 8 --workers 6 --size 513 \
    --epochs 8 --steps-per-epoch 600 --lr 0.01 --out checkpoints/gnet_cfo \
    > logs/train_cfo.log 2>&1
  python3 eval.py --weights checkpoints/gnet_cfo/last.pt --root datasets/cdnet/CDnet14 \
    --bg-source self --frame-stride 5 --out logs/eval_cfo.csv > logs/eval_cfo.log 2>&1
  echo "[$(date "+%F %T")] CFO DONE"
' > logs/cfo.log 2>&1 &
echo "cfo pid $!"

# (3) bg-trained model with background removed at test time (zero-cost ablation)
nohup bash -c '
  cd "$ROOT"
  export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
  python3 eval.py --weights checkpoints/gnet_main_seed0/last.pt --root datasets/cdnet/CDnet14 \
    --bg-source self --frame-stride 5 --out logs/eval_seed0_self.csv > logs/eval_seed0_self.log 2>&1
  echo "[$(date "+%F %T")] SELF-ABLATION DONE"
' > logs/self_ablation.log 2>&1 &
echo "self-ablation pid $!"
