#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# After all trainings finish, evaluate main + ablations on all CDnet categories.
cd "$ROOT"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=1
bash scripts/prepare_data.sh || true
for tag in gnet_main gnet_notsam gnet_nodda gnet_nopmb gnet_d8 gnet_d16 gnet_sub; do
  [ -f "checkpoints/$tag/last.pt" ] || { echo "missing $tag"; continue; }
  echo "[$(date '+%F %T')] eval $tag"
  python3 eval.py --weights "checkpoints/$tag/last.pt" \
    --root datasets/cdnet/CDnet14 --bg-source reconstructed --boundary \
    --frame-stride 3 --out "logs/eval_$tag.csv" > "logs/eval_$tag.log" 2>&1
done
echo "[$(date '+%F %T')] ALL EVAL DONE"
