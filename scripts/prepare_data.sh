#!/bin/bash
# Extract CDnet tars and COCO zips (idempotent).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
RAW=$ROOT/datasets/raw
CD=$ROOT/datasets/cdnet
mkdir -p "$CD"
log(){ echo "[$(date '+%F %T')] $*"; }

for t in "$RAW"/*.tar; do
  [ -e "$t" ] || continue
  name=$(basename "$t" .tar)
  if [ -d "$CD/CDnet14/$name" ]; then log "skip $name"; continue; fi
  # a tar may still be downloading: only extract if present in DONE log
  if [ -f "$RAW/$name.done" ]; then
    log "extract $name"; tar -xf "$t" -C "$CD"
  else
    log "wait (not done): $name"
  fi
done

cd "$RAW"
if [ -f val2017.zip ] && [ ! -d val2017 ]; then log "unzip val2017"; unzip -q -o val2017.zip; fi
if [ -f annotations_trainval2017.zip ] && [ ! -d annotations ]; then
  log "unzip annotations"; unzip -q -o annotations_trainval2017.zip
fi
log "PREPARE DONE"
