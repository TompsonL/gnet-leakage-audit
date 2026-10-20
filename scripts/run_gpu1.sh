#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# GPU1: main run, then key ablations (no-TSAM / no-DDA / no-PMB)
cd "$ROOT"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=1
C="--batch 8 --workers 6 --size 513 --epochs 8 --steps-per-epoch 600 --lr 0.01"
run(){ echo "[$(date '+%F %T')] RUN $1 $2"; python3 train.py $C $2 --out "checkpoints/$1"; }

run gnet_main ""
run gnet_notsam "--no-tsam"
run gnet_nodda  "--no-dda"
run gnet_nopmb  "--no-pmb"
echo "[$(date '+%F %T')] GPU1 DONE"
