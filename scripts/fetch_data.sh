#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Fetch all datasets for the GNet reproduction.
DATA=$ROOT/datasets
RAW=$DATA/raw
mkdir -p "$RAW"
cd "$RAW" || exit 1
MIR=https://hf-mirror.com/datasets/tauseefakhund/D2-CDNet/resolve/main/raw_tar
log(){ echo "[$(date '+%F %T')] $*"; }

dl(){ # url outfile
  local url="$1" out="$2"
  if [ -s "$out" ] && [ "$(stat -c%s "$out")" -ge "${3:-1}" ]; then
    log "skip $out (exists)"; return
  fi
  log "GET $out"
  curl -L -C - --retry 8 --retry-delay 5 --retry-all-errors -o "$out" "$url"
  log "DONE $out ($(du -h "$out" | cut -f1))"
}

# ---- CDnet 2014 categories (smaller first) ----
CDNET="PTZ thermal baseline cameraJitter intermittentObjectMotion shadow lowFramerate nightVideos dynamicBackground badWeather turbulence"
for c in $CDNET; do
  dl "$MIR/$c.tar" "$c.tar"
done

# ---- Cityscapes (CycleGAN unlabeled street scenes) for training backgrounds ----
dl "https://hf-mirror.com/datasets/huggan/cityscapes/resolve/main/data/train-00000-of-00001.parquet" "cityscapes_train.parquet"
dl "https://hf-mirror.com/datasets/huggan/cityscapes/resolve/main/data/val-00000-of-00001.parquet" "cityscapes_val.parquet"

# ---- COCO 2017 (val images + annotations) for synthetic foreground objects ----
dl "http://images.cocodataset.org/zips/val2017.zip" "val2017.zip"
dl "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "annotations_trainval2017.zip"

log "ALL DOWNLOADS FINISHED"
