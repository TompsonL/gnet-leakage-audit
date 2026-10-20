#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# E8-stride1: protocol-aligned background-error robustness check.
# Key operating points only (baseline, ghost 0.30, illum 0.20, noise 0.10)
# for the synthetic and in-domain models. Runs on GPU0 while RQ1 stride=1
# leakage recheck uses GPU1.
cd "$ROOT"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
CATS=PTZ,nightVideos,intermittentObjectMotion,thermal
for model in gnet_main_seed0 gnet_real; do
  W=checkpoints/$model/last.pt
  [ -f "$W" ] || continue
  echo "[$(date '+%F %T')] E8s1 $model baseline"
  python3 eval.py --weights "$W" --root datasets/cdnet/CDnet14 --categories "$CATS" \
    --bg-source causal --frame-stride 1 --bg-ghost 0.0 \
    --out "logs/e8s1_${model}_base.csv" > "logs/e8s1_${model}_base.log" 2>&1
  echo "[$(date '+%F %T')] E8s1 $model ghost=0.30"
  python3 eval.py --weights "$W" --root datasets/cdnet/CDnet14 --categories "$CATS" \
    --bg-source causal --frame-stride 1 --bg-ghost 0.30 \
    --out "logs/e8s1_${model}_ghost0.30.csv" > "logs/e8s1_${model}_ghost0.30.log" 2>&1
  echo "[$(date '+%F %T')] E8s1 $model illum=0.20"
  python3 eval.py --weights "$W" --root datasets/cdnet/CDnet14 --categories "$CATS" \
    --bg-source causal --frame-stride 1 --bg-illum 0.20 \
    --out "logs/e8s1_${model}_illum0.20.csv" > "logs/e8s1_${model}_illum0.20.log" 2>&1
  echo "[$(date '+%F %T')] E8s1 $model noise=0.10"
  python3 eval.py --weights "$W" --root datasets/cdnet/CDnet14 --categories "$CATS" \
    --bg-source causal --frame-stride 1 --bg-noise 0.10 \
    --out "logs/e8s1_${model}_noise0.10.csv" > "logs/e8s1_${model}_noise0.10.log" 2>&1
done
echo "[$(date '+%F %T')] E8 STRIDE1 DONE"
touch logs/E8S1_DONE
