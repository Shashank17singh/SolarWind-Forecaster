#!/usr/bin/env bash
set -euo pipefail

# WARNING: This dataset is very large (multi-GB per year).

mkdir -p data/goes_xrs

start_year=${1:-2017}
end_year=${2:-2018}

for y in $(seq "$start_year" "$end_year"); do
  url="https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-avg1m_science/${y}/"
  echo "Downloading XRS 1-min data for ${y}"
  wget -r -np -nH --accept "*.nc" --reject "index.html*" -e robots=off --no-clobber \
    "$url" -P data/goes_xrs/
done
