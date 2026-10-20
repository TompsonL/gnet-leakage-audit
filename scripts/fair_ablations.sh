#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Fair ablation table (E3): re-evaluate all ablation variants under the frozen
# causal protocol (stride 1). Waits for the G1 and M1 queues so it never steals
# the GPUs from higher-priority runs.
cd "$ROOT"
export PYTHONPATH=.
while ! grep -q . logs/G1_DONE 2>/dev/null; do sleep 60; done
while ! grep -q . logs/M1_DONE 2>/dev/null; do sleep 60; done
sleep 30

run () {  # $1=gpu  $2..=model tags
  gpu=$1; shift
  for tag in "$@"; do
    ck=checkpoints/$tag/last.pt
    [ -f "$ck" ] || { echo "missing $tag"; continue; }
    bg=causal
    [ "$tag" = "gnet_sub" ] && bg=causal
    echo "[$(date '+%F %T')] fair eval $tag (bg=$bg, gpu=$gpu)"
    CUDA_VISIBLE_DEVICES=$gpu python3 eval.py --weights "$ck" \
      --root datasets/cdnet/CDnet14 --bg-source "$bg" --frame-stride 1 --boundary \
      --out "logs/fair_${tag}_causal.csv" > "logs/fair_${tag}_causal.log" 2>&1
  done
  echo "[$(date '+%F %T')] ABLATION GPU$gpu DONE"
}

nohup bash -c "$(declare -f run); cd "$ROOT"; export PYTHONPATH=.; run 0 gnet_notsam gnet_nodda gnet_nopmb gnet_d8" > logs/fair_abl_gpu0.log 2>&1 &
echo "ablation gpu0 pid $!"
nohup bash -c "$(declare -f run); cd "$ROOT"; export PYTHONPATH=.; run 1 gnet_d16 gnet_sub gnet_real" > logs/fair_abl_gpu1.log 2>&1 &
echo "ablation gpu1 pid $!"
echo "FAIR ABLATIONS QUEUED"
