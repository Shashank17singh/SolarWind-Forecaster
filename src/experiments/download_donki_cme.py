import argparse
import json
import time
from datetime import datetime, timedelta
from urllib.request import Request, urlopen

BASE_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME"


def _fetch_range(start_date: str, end_date: str, timeout: int = 30):
    url = f"{BASE_URL}?startDate={start_date}&endDate={end_date}"
    req = Request(url, headers={"User-Agent": "space-weather-sentinel/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chunk_dates(start: datetime, end: datetime, window_days: int = 30):
    cursor = start
    delta = timedelta(days=window_days)
    while cursor <= end:
        chunk_end = min(end, cursor + delta)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def download(start_date: str, end_date: str, out_json: str):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    all_events = []
    seen = set()

    for c_start, c_end in _chunk_dates(start, end, window_days=30):
        s = c_start.strftime("%Y-%m-%d")
        e = c_end.strftime("%Y-%m-%d")
        print(f"[donki] fetching {s} to {e}...", flush=True)
        payload = _fetch_range(s, e)
        if not isinstance(payload, list):
            continue
        for item in payload:
            key = (
                item.get("activityID")
                or f"{item.get('startTime')}::{item.get('sourceLocation')}"
            )
            if key in seen:
                continue
            seen.add(key)
            all_events.append(item)
        time.sleep(1.0)

    print(f"[donki] total events: {len(all_events)}", flush=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_events, f, indent=2)
    print(f"[donki] wrote {out_json}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download DONKI CME events (30-day chunked)."
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    download(args.start_date, args.end_date, args.out_json)
