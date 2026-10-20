#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
DATA=$ROOT/datasets
RAW=$DATA/raw
mkdir -p "$RAW"; cd "$RAW" || exit 1
log(){ echo "[$(date '+%F %T')] $*"; }
dl(){ local url="$1" out="$2"; [ -f "$out" ] && { log "skip $out"; return; }; log "GET $out"; curl -L -C - --retry 8 --retry-delay 5 --retry-all-errors -o "$out" "$url"; log "DONE $out ($(du -h "$out"|cut -f1))"; }

dl "https://hf-mirror.com/datasets/huggan/cityscapes/resolve/main/data/train-00000-of-00001.parquet" "cityscapes_train.parquet"
dl "https://hf-mirror.com/datasets/huggan/cityscapes/resolve/main/data/val-00000-of-00001.parquet" "cityscapes_val.parquet"
dl "https://hf-mirror.com/datasets/pcuenq/coco-2017-mirror/resolve/main/val2017.zip" "val2017.zip"
dl "https://hf-mirror.com/datasets/pcuenq/coco-2017-mirror/resolve/main/annotations_trainval2017.zip" "annotations_trainval2017.zip"
log "TRAIN DATA DOWNLOADS FINISHED"
