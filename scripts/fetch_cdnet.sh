#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# Parallel (3-way) download of CDnet category tars from HF mirror.
DATA=$ROOT/datasets/raw
mkdir -p "$DATA"; cd "$DATA" || exit 1
MIR=https://hf-mirror.com/datasets/tauseefakhund/D2-CDNet/resolve/main/raw_tar
CATS="PTZ thermal baseline cameraJitter intermittentObjectMotion shadow lowFramerate nightVideos dynamicBackground badWeather turbulence"
MAXJ=3
log(){ echo "[$(date '+%F %T')] $*"; }

one(){
  local c="$1"
  [ -f "$c.done" ] && { log "done $c"; return; }
  log "start $c"
  curl -L -C - --retry 20 --retry-delay 5 --retry-all-errors --speed-time 60 --speed-limit 1000 -o "$c.tar" "$MIR/$c.tar"
  touch "$c.done"; log "DONE $c ($(du -h $c.tar|cut -f1))"
}

for c in $CATS; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 5; done
  one "$c" &
done
wait
log "CDNET ALL DONE"
