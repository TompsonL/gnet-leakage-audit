#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Wait for main training to finish, then evaluate on all available CDnet categories.
cd "$ROOT"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=1
while ! grep -q "^DONE" logs/train_main.log 2>/dev/null; do sleep 120; done
echo "[$(date '+%F %T')] training done, extracting any new categories"
for t in datasets/raw/*.done; do :; done
bash scripts/prepare_data.sh || true
echo "[$(date '+%F %T')] running eval"
python3 eval.py --weights checkpoints/gnet_main/last.pt \
  --root datasets/cdnet/CDnet14 --bg-source reconstructed --boundary \
  --out logs/eval_main.csv > logs/eval_main.log 2>&1
echo "[$(date '+%F %T')] EVAL DONE"
