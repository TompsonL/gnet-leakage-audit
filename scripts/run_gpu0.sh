#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# GPU0: ablation runs (TSAM depth 8/16, fusion=sub)
cd "$ROOT"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
C="--batch 8 --workers 6 --size 513 --epochs 8 --steps-per-epoch 600 --lr 0.01"
run(){ echo "[$(date '+%F %T')] RUN $1 $2"; python3 train.py $C $2 --out "checkpoints/$1"; }

run gnet_d8  "--tsam-depth 8"
run gnet_d16 "--tsam-depth 16"
run gnet_sub "--fusion sub"
echo "[$(date '+%F %T')] GPU0 DONE"
