#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Task C orchestration: leakage ablation (GPU0) + multi-seed training (GPU1 + GPU0).
cd "$ROOT"
rm -f logs/LEAK_DONE
nohup bash scripts/leak_ablation.sh 0 > logs/leak_ablation.log 2>&1 &
echo "leak_ablation pid $!"
nohup bash -c 'bash scripts/seed_one.sh 0 1; bash scripts/seed_one.sh 1 1' > logs/seeds_gpu1.log 2>&1 &
echo "seeds 0,1 (gpu1) pid $!"
nohup bash -c 'while [ ! -f logs/LEAK_DONE ]; do sleep 30; done; bash scripts/seed_one.sh 2 0' > logs/seeds_gpu0.log 2>&1 &
echo "seed 2 (gpu0) pid $!"
