import argparse
import json
import math
import os
import re
from datetime import timedelta

import numpy as np
import pandas as pd


def _parse_time(ts: str):
    return pd.to_datetime(ts, errors="coerce", utc=True).tz_convert(None)


def _parse_source_location(loc: str):
    if not isinstance(loc, str) or not loc:
        return np.nan, np.nan

    match = re.match(r"^([NS])(\d{1,2})([EW])(\d{1,3})$", loc.strip())
    if match:
        lat = int(match.group(2)) * (1 if match.group(1) == "N" else -1)
        lon = int(match.group(4)) * (1 if match.group(3) == "W" else -1)
        return lat, lon
    match = re.match(r"^([NS])(\d{1,2})$", loc.strip())
    if match:
        lat = int(match.group(2)) * (1 if match.group(1) == "N" else -1)
        return lat, np.nan
    match = re.match(r"^([EW])(\d{1,3})$", loc.strip())
    if match:
        lon = int(match.group(2)) * (1 if match.group(1) == "W" else -1)
        return np.nan, lon
    return np.nan, np.nan


def _pick_analysis(item: dict):
    analyses = item.get("cmeAnalyses") or []
    if not analyses:
        return {}
    for entry in analyses:
        if entry.get("isMostAccurate"):
            return entry
    return analyses[0]


def _load_icme_catalog(path: str):
    if not path:
        return None
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    time_col = None
    for key in ("icme_start_time", "icme_start", "start_time", "start"):
        if key in cols:
            time_col = cols[key]
            break
    if not time_col:
        return None
    df["icme_start_time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df["icme_start_time"] = df["icme_start_time"].dt.tz_convert(None)
    df = df.dropna(subset=["icme_start_time"])

    sc_col = None
    for key in ("sc_insitu", "spacecraft", "sc", "insitu"):
        if key in cols:
            sc_col = cols[key]
            break
    if sc_col:
        sc = df[sc_col].astype(str).str.lower()
        df = df[sc.str.contains("wind|earth|l1", na=False)]

    df = df.sort_values("icme_start_time")
    return df


def _match_icme(
    cme_time: pd.Timestamp,
    icme_times: pd.Series,
    pred_hours: float | None = None,
    window_hours: float = 18,
    min_hours: float = 10,
    max_hours: float = 120,
):
    if icme_times is None or icme_times.empty:
        return None
    if pred_hours is not None and np.isfinite(pred_hours):
        lower_h = max(min_hours, pred_hours - window_hours)
        upper_h = min(max_hours, pred_hours + window_hours)
    else:
        lower_h = min_hours
        upper_h = max_hours
    lower = cme_time + timedelta(hours=lower_h)
    upper = cme_time + timedelta(hours=upper_h)
    candidates = icme_times[(icme_times >= lower) & (icme_times <= upper)]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def build_dataset(donki_json: str, icme_csv: str, out_csv: str, out_meta: str):
    with open(donki_json, "r", encoding="utf-8") as f:
        events = json.load(f)

    icme_df = _load_icme_catalog(icme_csv)
    icme_times = icme_df["icme_start_time"] if icme_df is not None else None
    if icme_times is not None and getattr(icme_times.dt, "tz", None) is not None:
        icme_times = icme_times.dt.tz_convert(None)

    rows = []
    for item in events:
        start_time = _parse_time(item.get("startTime"))
        if pd.isna(start_time):
            continue

        analysis = _pick_analysis(item)
        speed = pd.to_numeric(analysis.get("speed"), errors="coerce")
        half_angle = pd.to_numeric(analysis.get("halfAngle"), errors="coerce")
        if pd.isna(speed) or speed <= 0:
            continue
        if pd.isna(half_angle):
            continue
        lat = pd.to_numeric(analysis.get("latitude"), errors="coerce")
        lon = pd.to_numeric(analysis.get("longitude"), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            loc_lat, loc_lon = _parse_source_location(item.get("sourceLocation"))
            if pd.isna(lat):
                lat = loc_lat
            if pd.isna(lon):
                lon = loc_lon

        width = half_angle * 2 if pd.notna(half_angle) else np.nan
        is_halo = 1 if pd.notna(width) and width >= 360 else 0

        ar_num = pd.to_numeric(item.get("activeRegionNum"), errors="coerce")
        catalog = item.get("catalog") or ""
        cme_type = analysis.get("type") or ""

        transit_pred_hours = 149597870.7 / (speed * 3600.0)

        icme_time = None
        if icme_times is not None:
            icme_time = _match_icme(
                start_time, icme_times, pred_hours=transit_pred_hours
            )

        earth_impact = 1 if icme_time is not None else 0
        transit_hours = (
            (icme_time - start_time).total_seconds() / 3600
            if icme_time is not None
            else np.nan
        )

        rows.append(
            {
                "time": start_time,
                "speed": speed,
                "half_angle": half_angle,
                "width": width,
                "latitude": lat,
                "longitude": lon,
                "is_halo": is_halo,
                "active_region": ar_num,
                "catalog": catalog,
                "cme_type": cme_type,
                "earth_impact": earth_impact,
                "transit_hours": transit_hours,
                "transit_pred_hours": transit_pred_hours,
                "label_source": "icmecat" if icme_time is not None else "none",
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["speed", "half_angle"], how="all")
    df = df.sort_values("time")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    meta = {
        "rows": len(df),
        "earth_impact_pos": int(df["earth_impact"].sum()),
        "earth_impact_neg": int((df["earth_impact"] == 0).sum()),
        "features": [
            "speed",
            "half_angle",
            "width",
            "latitude",
            "longitude",
            "is_halo",
            "active_region",
            "catalog",
            "cme_type",
        ],
        "label": "earth_impact",
        "transit_label": "transit_hours",
    }
    if out_meta:
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"[cme] wrote dataset -> {out_csv}")
    if out_meta:
        print(f"[cme] wrote metadata -> {out_meta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build CME impact dataset from DONKI + ICMECAT"
    )
    parser.add_argument("--donki-json", required=True)
    parser.add_argument("--icme-csv", required=False, default=None)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-meta", required=False, default=None)
    args = parser.parse_args()

    build_dataset(args.donki_json, args.icme_csv, args.out_csv, args.out_meta)
