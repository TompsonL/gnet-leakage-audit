#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Sequential ablation training queue (run on a single free GPU).
cd "$ROOT"
export PYTHONPATH=.
COMMON="--batch 8 --workers 8 --size 513 --epochs 12 --steps-per-epoch 600 --lr 0.01"
run(){ echo "[$(date '+%F %T')] RUN $1"; python3 train.py $COMMON $2 --out "checkpoints/$1"; }

run gnet_notsam  "--no-tsam"
run gnet_d8      "--tsam-depth 8"
run gnet_d16     "--tsam-depth 16"
run gnet_nodda   "--no-dda"
run gnet_nopmb   "--no-pmb"
run gnet_sub     "--fusion sub"
run gnet_dda_nopmb "--no-pmb --no-dda"
echo "[$(date '+%F %T')] ABLATIONS DONE"
