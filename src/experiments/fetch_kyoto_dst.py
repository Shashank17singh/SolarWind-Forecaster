import argparse
import csv
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

FINAL_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_final"
PROV_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional"
REAL_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime"


ROW_RE = re.compile(r"^\s*(\d{1,2})\s+(-?\d+)")


def _fetch(url: str, timeout: int) -> str:
    with urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _parse_month(text: str, year: int, month: int):
    rows = []
    for line in text.splitlines():
        if not ROW_RE.match(line):
            continue
        parts = line.strip().split()
        if len(parts) < 25:
            continue
        day = int(parts[0])
        values = parts[1:25]
        for hour, raw in enumerate(values):
            try:
                val = int(raw)
            except ValueError:
                continue
            if val >= 9999:
                continue
            ts = pd.Timestamp(year=year, month=month, day=day, hour=hour)
            rows.append((ts, val))
    return rows


def _month_url(base: str, year: int, month: int) -> str:
    return f"{base}/{year}{month:02d}/index.html"


def _try_month(base: str, year: int, month: int, timeout: int):
    url = _month_url(base, year, month)
    try:
        text = _fetch(url, timeout)
        rows = _parse_month(text, year, month)
        return rows, url
    except HTTPError:
        return [], url
    except URLError:
        return [], url


def _collect_series(
    start_year: int, end_year: int, base: str, timeout: int, label: str
):
    data = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            rows, url = _try_month(base, year, month, timeout)
            if rows:
                data.extend((ts, val, label) for ts, val in rows)

            time.sleep(0.05)
        print(f"[dst] {label}: fetched year {year}", flush=True)
    return data


def fetch_all(out_csv: str, timeout: int):

    final = _collect_series(1957, 2020, FINAL_BASE, timeout, "final")

    provisional = _collect_series(2021, 2025, PROV_BASE, timeout, "provisional")

    now = pd.Timestamp.utcnow()
    realtime = _collect_series(now.year - 1, now.year, REAL_BASE, timeout, "realtime")

    df = pd.DataFrame(final + provisional + realtime, columns=["time", "dst", "source"])
    if df.empty:
        raise RuntimeError("No Dst data fetched. Check connectivity or URLs.")

    priority = {"final": 3, "provisional": 2, "realtime": 1}
    df["priority"] = df["source"].map(priority).fillna(0)
    df = df.sort_values(["time", "priority"], ascending=[True, False])
    df = df.drop_duplicates(subset=["time"], keep="first")
    df = df.drop(columns=["priority"]).sort_values("time")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[dst] wrote {len(df)} rows -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Dst hourly index (final/provisional/realtime)"
    )
    parser.add_argument("--out-csv", default="data/indices/kyoto/dst_hourly.csv")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    fetch_all(args.out_csv, args.timeout)
