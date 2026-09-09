import os
import re
from datetime import datetime, timedelta
from typing import List, Dict


def _parse_date(token: str) -> datetime:

    yymmdd = token[-6:]
    yy = int(yymmdd[:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    year = 1900 + yy if yy >= 70 else 2000 + yy
    return datetime(year, mm, dd)


def _parse_time(t: str) -> int:

    if t == "0000":
        return 0
    if len(t) != 4 or not t.isdigit():
        return -1
    return int(t[:2]) * 60 + int(t[2:4])


def _minutes_to_dt(date: datetime, minutes: int):
    if minutes < 0:
        return None
    if minutes >= 1440:
        minutes -= 1440
        date = date + timedelta(days=1)
    return date.replace(hour=minutes // 60, minute=minutes % 60)


def load_flare_reports(folder: str) -> List[Dict]:
    events = []
    for name in sorted(os.listdir(folder)):
        if not name.startswith("goes-xrs-report_") or not name.endswith(".txt"):
            continue
        path = os.path.join(folder, name)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                date_token, start_t, end_t, peak_t = (
                    parts[0],
                    parts[1],
                    parts[2],
                    parts[3],
                )
                flare_class = None

                for p in parts[4:]:
                    if re.fullmatch(r"[ABCMX]", p):
                        flare_class = p
                        break
                if flare_class is None:
                    continue
                date = _parse_date(date_token)
                start_min = _parse_time(start_t)
                peak_min = _parse_time(peak_t)
                end_min = _parse_time(end_t)
                if start_min < 0 or peak_min < 0:
                    continue
                start_dt = _minutes_to_dt(date, start_min)
                peak_dt = _minutes_to_dt(date, peak_min)
                end_dt = _minutes_to_dt(date, end_min)
                if start_dt is None or peak_dt is None:
                    continue
                events.append(
                    {
                        "start": start_dt,
                        "peak": peak_dt,
                        "end": end_dt,
                        "class": flare_class,
                    }
                )
    return events
