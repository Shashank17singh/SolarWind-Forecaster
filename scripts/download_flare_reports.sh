#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/flare_reports
base="https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/"

start_year=${1:-1975}
end_year=${2:-2016}

for y in $(seq "$start_year" "$end_year"); do
  url="${base}goes-xrs-report_${y}.txt"
  out="data/flare_reports/goes-xrs-report_${y}.txt"
  if [ -f "$out" ]; then
    continue
  fi
  if wget -q -O "$out" "$url"; then
    echo "downloaded ${y}"
  else
    rm -f "$out"
  fi
done
