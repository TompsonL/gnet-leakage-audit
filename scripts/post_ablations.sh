#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# After the ablation queue finishes, evaluate each variant on CDnet.
cd "$ROOT"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
while ! grep -q "ABLATIONS DONE" logs/train_ablations.log 2>/dev/null; do sleep 180; done
bash scripts/prepare_data.sh || true
for tag in gnet_notsam gnet_d8 gnet_d16 gnet_nodda gnet_nopmb gnet_sub gnet_dda_nopmb; do
  [ -f "checkpoints/$tag/last.pt" ] || continue
  echo "[$(date '+%F %T')] eval $tag"
  python3 eval.py --weights "checkpoints/$tag/last.pt" \
    --root datasets/cdnet/CDnet14 --bg-source reconstructed \
    --frame-stride 3 --out "logs/eval_$tag.csv" > "logs/eval_$tag.log" 2>&1
done
echo "[$(date '+%F %T')] ABLATION EVAL DONE"
