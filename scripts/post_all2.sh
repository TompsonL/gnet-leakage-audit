#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Final: wait for both training queues, then evaluate all ablation variants.
cd "$ROOT"
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
while ! grep -q "GPU0 DONE" logs/run_gpu0.log 2>/dev/null; do sleep 120; done
while ! grep -q "GPU1 DONE" logs/run_gpu1.log 2>/dev/null; do sleep 120; done
sleep 20
bash scripts/prepare_data.sh || true
for tag in gnet_notsam gnet_nodda gnet_nopmb gnet_d8 gnet_d16 gnet_sub; do
  [ -f "checkpoints/$tag/last.pt" ] || { echo "missing $tag"; continue; }
  echo "[$(date '+%F %T')] eval $tag"
  python3 eval.py --weights "checkpoints/$tag/last.pt" \
    --root datasets/cdnet/CDnet14 --bg-source reconstructed --boundary \
    --frame-stride 5 --out "logs/eval_$tag.csv" > "logs/eval_$tag.log" 2>&1
done
echo "[$(date '+%F %T')] ABLATION EVAL DONE"
