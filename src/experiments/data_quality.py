import json
import os
from datetime import datetime
from collections import deque
from io import StringIO

import numpy as np
import pandas as pd

from config import CONFIG

CRITICAL_COLS = ["bz_gsm", "flow_speed", "proton_density"]


def read_tail_csv(path: str, rows: int) -> pd.DataFrame:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        buffer = deque(maxlen=rows)
        for line in f:
            buffer.append(line)
    data = header + "".join(buffer)
    df = pd.read_csv(StringIO(data), parse_dates=["time"])
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
    return df


def _load_baseline(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_quality(csv_path: str, window_min: int | None = None) -> dict:
    window_min = window_min or CONFIG.quality_window_min
    rows = max(120, int(window_min))
    df = read_tail_csv(csv_path, rows)
    if df.empty:
        return {
            "ok": False,
            "reason": "no_data",
            "rows": 0,
            "stale_minutes": None,
            "max_gap_minutes": None,
            "missing_pct": None,
            "drift_score": None,
            "drift_features": [],
        }

    df = df.sort_values("time")
    now = datetime.utcnow()
    last_ts = df["time"].max().to_pydatetime()
    stale_minutes = (now - last_ts).total_seconds() / 60.0

    diffs = df["time"].diff().dropna().dt.total_seconds() / 60.0
    max_gap = float(diffs.max()) if not diffs.empty else 0.0

    missing_rates = {}
    for col in CRITICAL_COLS:
        if col in df.columns:
            missing_rates[col] = float(df[col].isna().mean())
    missing_pct = (
        float(np.mean(list(missing_rates.values()))) if missing_rates else None
    )

    drift_score, drift_features = compute_drift(df)

    ok = True
    reasons = []
    if stale_minutes is not None and stale_minutes > 10:
        ok = False
        reasons.append("stale_data")
    if max_gap is not None and max_gap > 10:
        ok = False
        reasons.append("gaps_detected")
    if missing_pct is not None and missing_pct > 0.2:
        ok = False
        reasons.append("missing_data")
    if drift_score is not None and drift_score > 3.0:
        reasons.append("drift")

    return {
        "ok": ok,
        "reason": ",".join(reasons) if reasons else "ok",
        "rows": int(len(df)),
        "stale_minutes": round(stale_minutes, 2),
        "max_gap_minutes": round(max_gap, 2),
        "missing_pct": round(missing_pct, 4) if missing_pct is not None else None,
        "missing_by_col": missing_rates,
        "drift_score": drift_score,
        "drift_features": drift_features,
        "last_timestamp": last_ts.isoformat(),
    }


def compute_drift(df: pd.DataFrame) -> tuple[float | None, list[str]]:
    baseline = _load_baseline(CONFIG.drift_baseline_path)
    stats = baseline.get("feature_stats") if baseline else None
    if not stats:
        return None, []

    scores = {}
    for col, meta in stats.items():
        if col not in df.columns:
            continue
        mean = meta.get("mean")
        std = meta.get("std")
        if std is None or std == 0:
            continue
        current_mean = float(pd.to_numeric(df[col], errors="coerce").dropna().mean())
        z = abs(current_mean - mean) / std
        scores[col] = z

    if not scores:
        return None, []
    max_feature = max(scores, key=scores.get)
    return float(scores[max_feature]), [max_feature]


def build_baseline(csv_path: str, output_path: str, feature_cols: list[str]):
    df = pd.read_csv(csv_path, parse_dates=["time"])
    stats = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        stats[col] = {"mean": float(series.mean()), "std": float(series.std())}
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "feature_stats": stats,
        "rows": int(len(df)),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
