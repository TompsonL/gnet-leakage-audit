#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
set -e
cd "$ROOT"/datasets/raw
echo "[$(date)] start CDnet2014"
curl -L -C - -o dataset2014.zip "https://changedetection.net/static/dataset/dataset2014.zip"
echo "[$(date)] CDnet2014 done: $(ls -l dataset2014.zip)"
echo "[$(date)] start COCO val2017"
curl -L -C - -o val2017.zip "http://images.cocodataset.org/zips/val2017.zip"
echo "[$(date)] start COCO annotations"
curl -L -C - -o annotations_trainval2017.zip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
echo "[$(date)] COCO done"
echo "[$(date)] ALL DONE"
