import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_dst_daily(dst_csv: Path, final_only: bool) -> pd.DataFrame:
    df = pd.read_csv(dst_csv, parse_dates=["time"])
    df = df.dropna(subset=["time"])
    if "source" in df.columns and final_only:
        df["source"] = df["source"].astype(str).str.lower()
        df = df[df["source"].str.contains("final", na=False)]
    df["dst"] = pd.to_numeric(df["dst"], errors="coerce")
    df = df.dropna(subset=["dst"]).set_index("time").sort_index()
    daily = df.resample("1D").min().rename(columns={"dst": "dst_min"})
    daily.index = daily.index.tz_localize(None)
    return daily


def _climo_stats(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["doy"] = df.index.dayofyear
    stats = (
        df.groupby("doy")["dst_min"]
        .agg(
            climo_median="median",
            climo_p25=lambda x: np.nanpercentile(x, 25),
            climo_p75=lambda x: np.nanpercentile(x, 75),
            climo_mean="mean",
        )
        .reset_index()
    )
    return stats


def _storm_probability(daily: pd.DataFrame, threshold: float) -> pd.Series:
    df = daily.copy()
    df["doy"] = df.index.dayofyear
    probs = df.groupby("doy")["dst_min"].apply(lambda x: float(np.mean(x <= threshold)))
    return probs


def _risk_label(prob: float) -> str:
    if prob >= 0.4:
        return "High"
    if prob >= 0.2:
        return "Elevated"
    return "Low"


def forecast(
    dst_csv: str,
    out_csv: str,
    out_json: str | None,
    days: int,
    lookback_days: int,
    threshold: float,
    final_only: bool,
):
    dst_path = Path(dst_csv)
    if not dst_path.exists():
        raise FileNotFoundError(f"Dst CSV not found: {dst_csv}")

    daily = _load_dst_daily(dst_path, final_only=final_only)
    if daily.empty:
        raise ValueError("Dst daily series is empty.")

    last_day = daily.index.max()
    target_days = pd.date_range(
        last_day + pd.Timedelta(days=1), periods=days, freq="1D"
    )
    climo = _climo_stats(daily).set_index("doy")
    storm_prob = _storm_probability(daily, threshold=threshold)

    rows = []
    for day in target_days:
        rec_day = day - pd.Timedelta(days=lookback_days)
        rec_val = daily["dst_min"].get(rec_day, np.nan)
        doy = day.dayofyear
        climo_row = climo.loc[doy] if doy in climo.index else None
        climo_med = (
            float(climo_row["climo_median"]) if climo_row is not None else np.nan
        )
        climo_p25 = float(climo_row["climo_p25"]) if climo_row is not None else np.nan
        climo_p75 = float(climo_row["climo_p75"]) if climo_row is not None else np.nan

        if np.isfinite(rec_val):
            pred = float(rec_val)
            source = f"recurrence_{lookback_days}d"
            confidence = 0.7
        else:
            pred = float(climo_med) if np.isfinite(climo_med) else np.nan
            source = "climatology"
            confidence = 0.4

        prob = float(storm_prob.get(doy, np.nan))
        rows.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "dst_min_pred": pred,
                "dst_min_rec": None if not np.isfinite(rec_val) else float(rec_val),
                "dst_min_climo": (
                    None if not np.isfinite(climo_med) else float(climo_med)
                ),
                "climo_p25": None if not np.isfinite(climo_p25) else float(climo_p25),
                "climo_p75": None if not np.isfinite(climo_p75) else float(climo_p75),
                "storm_prob": None if not np.isfinite(prob) else prob,
                "risk": _risk_label(prob) if np.isfinite(prob) else "Unknown",
                "source": source,
                "confidence": confidence,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    summary = {
        "last_observed_day": last_day.strftime("%Y-%m-%d"),
        "forecast_days": days,
        "lookback_days": lookback_days,
        "storm_threshold": threshold,
        "final_only": final_only,
        "rows": len(out_df),
    }

    if out_json:
        out_json_path = Path(out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2)
        )

    return out_df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="30-day geomagnetic outlook (recurrence + climatology)"
    )
    parser.add_argument("--dst-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--lookback-days", type=int, default=27)
    parser.add_argument(
        "--threshold", type=float, default=-50.0, help="Dst storm threshold (nT)"
    )
    parser.add_argument(
        "--final-only", action="store_true", help="Use only Dst rows marked as final"
    )
    args = parser.parse_args()

    forecast(
        dst_csv=args.dst_csv,
        out_csv=args.out_csv,
        out_json=args.out_json,
        days=args.days,
        lookback_days=args.lookback_days,
        threshold=args.threshold,
        final_only=args.final_only,
    )
