#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# T6: cross-dataset generalization on SBI2015 (frozen causal protocol).
# Models were never trained on SBI -> measures true cross-dataset transfer.
cd "$ROOT"
export PYTHONPATH=.
GPU=${1:-1}
export CUDA_VISIBLE_DEVICES=$GPU
SBI_ROOT=datasets/sbi_root
for m in gnet_main_seed0 gnet_real gnet_full gnet_m1; do
  W=checkpoints/$m/last.pt
  [ -f "$W" ] || { echo "skip $m (no ckpt)"; continue; }
  bg=causal
  [ "$m" = "gnet_full" -o "$m" = "gnet_m1" ] && bg=causal_rel
  echo "[$(date '+%F %T')] SBI eval $m bg=$bg"
  python3 eval.py --weights "$W" --root "$SBI_ROOT" --categories sbi \
    --bg-source "$bg" --frame-stride 1 --out "logs/sbi_${m}.csv" > "logs/sbi_${m}.log" 2>&1
done
echo "[$(date '+%F %T')] SBI DONE"
touch logs/SBI_DONE
