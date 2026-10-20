#!/bin/bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; export ROOT
# T0/T6 asset acquisition (robust): SBI cross-dataset + FgSegNet baseline repo.
# The SBI host is flaky, so each sequence is retried and zip-validated.
cd "$ROOT"
mkdir -p datasets/sbi/sequences third_party
BASE=http://sbmi2015.na.icar.cnr.it/MODLab/BckgInit

valid() { python3 - "$1" <<'PY'
import sys, zipfile
try:
    z = zipfile.ZipFile(sys.argv[1]); z.testzip()
except Exception:
    sys.exit(1)
PY
}

get() {  # $1=url  $2=out
  local url="$1" out="$2"
  [ -s "$out" ] && valid "$out" && return 0
  for a in 1 2 3 4 5; do
    rm -f "$out"
    curl -sL --connect-timeout 20 --max-time 400 --retry 2 --retry-delay 5 \
      -o "$out" "$url"
    if [ -s "$out" ] && valid "$out"; then return 0; fi
    echo "[$(date '+%F %T')] retry $a failed: $url"
    sleep 5
  done
  return 1
}

for s in Board Candela_m1.10 CAVIAR1 CAVIAR2 CaVignal Foliage HallAndMonitor \
         HighwayI HighwayII HumanBody2 IBMtest2 PeopleAndFoliage Snellen Toscana; do
  out=datasets/sbi/sequences/$s.zip
  if [ -s "$out" ] && valid "$out"; then echo "skip $s"; continue; fi
  echo "[$(date '+%F %T')] download $s"
  get "$BASE/Sequences/$s.zip" "$out" && echo "  OK $s" || echo "  FAILED $s"
  sleep 3
done
get "$BASE/GT/GT.zip" datasets/sbi/GT.zip && echo "OK GT" || echo "FAILED GT"
echo "[$(date '+%F %T')] SBI DONE"
touch datasets/sbi/DONE

[ -d third_party/FgSegNet ] || git clone --depth 1 https://github.com/lim-anggun/FgSegNet.git third_party/FgSegNet
echo "[$(date '+%F %T')] BASELINES DONE"
touch third_party/DONE
