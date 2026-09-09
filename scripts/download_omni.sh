#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/omni

start_year=${1:-1995}
end_year=${2:-2025}

for y in $(seq "$start_year" "$end_year"); do
  url="https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_min${y}.asc"
  echo "Downloading $url"
  wget -c "$url" -P data/omni/
done
