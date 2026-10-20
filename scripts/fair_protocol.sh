#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Frozen CDnet14 official protocol (route C, P0).
#   - every frame inside temporalROI (frame-stride 1)
#   - causal background only (no SuBSENSE decision, no future frames)
#   - shadow=background, ROI ignored, per-video -> category-average
# Produces the fair baseline table:
#   (1) SuBSENSE floor (packaged masks)              -> CPU
#   (2) current-frame-only network (gnet_cfo, bg=self)
#   (3) GNet main, seeds 0/1/2, bg=causal
cd "$ROOT"
export PYTHONPATH=.

# (1) SuBSENSE floor: causal? uses packaged masks; official frame set.
nohup python3 eval_subsense.py --root datasets/cdnet/CDnet14 --frame-stride 1 \
  --out logs/fair_subsense.csv > logs/fair_subsense.log 2>&1 &
echo "subsense pid $!"

# (2) current-frame-only baseline (GPU1)
nohup bash -c '
  cd "$ROOT"; export PYTHONPATH=. CUDA_VISIBLE_DEVICES=1
  python3 eval.py --weights checkpoints/gnet_cfo/last.pt --root datasets/cdnet/CDnet14 \
    --bg-source self --frame-stride 1 --boundary \
    --out logs/fair_cfo.csv > logs/fair_cfo.log 2>&1
  echo "[$(date "+%F %T")] FAIR CFO DONE"
' > logs/fair_cfo_done.log 2>&1 &
echo "cfo pid $!"

# (3) GNet main seeds, bg=causal. GPU0: seeds 0,1 ; GPU1: seed 2 after cfo.
nohup bash -c '
  cd "$ROOT"; export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
  for s in 0 1; do
    python3 eval.py --weights checkpoints/gnet_main_seed${s}/last.pt \
      --root datasets/cdnet/CDnet14 --bg-source causal --frame-stride 1 --boundary \
      --out logs/fair_seed${s}_causal.csv > logs/fair_seed${s}_causal.log 2>&1
  done
  echo "[$(date "+%F %T")] FAIR MAIN GPU0 DONE"
' > logs/fair_main0_done.log 2>&1 &
echo "main gpu0 pid $!"

nohup bash -c '
  cd "$ROOT"
  while ! grep -q "FAIR CFO DONE" logs/fair_cfo_done.log 2>/dev/null; do sleep 60; done
  export PYTHONPATH=. CUDA_VISIBLE_DEVICES=1
  python3 eval.py --weights checkpoints/gnet_main_seed2/last.pt \
    --root datasets/cdnet/CDnet14 --bg-source causal --frame-stride 1 --boundary \
    --out logs/fair_seed2_causal.csv > logs/fair_seed2_causal.log 2>&1
  echo "[$(date "+%F %T")] FAIR MAIN GPU1 DONE"
' > logs/fair_main1_done.log 2>&1 &
echo "main gpu1 pid $!"
echo "ALL LAUNCHED"
