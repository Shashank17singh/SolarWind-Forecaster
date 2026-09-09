import argparse
import os
import numpy as np
import pandas as pd
from flare_reports import load_flare_reports

FEATURE_COLS = [
    "bx_gse",
    "by_gse",
    "bz_gse",
    "by_gsm",
    "bz_gsm",
    "flow_speed",
    "vx_gse",
    "vy_gse",
    "vz_gse",
    "proton_density",
    "temperature",
    "flow_pressure",
    "electric_field",
    "plasma_beta",
    "alfven_mach",
    "ae",
    "al",
    "au",
    "sym_h",
    "asy_h",
]


def add_rolling_features(df: pd.DataFrame, window: int, prefix: str) -> pd.DataFrame:
    roll = df[FEATURE_COLS].rolling(window=window, min_periods=max(3, window // 5))
    out = {}
    for col in FEATURE_COLS:
        out[f"{col}_{prefix}_mean"] = roll[col].mean()
        out[f"{col}_{prefix}_std"] = roll[col].std()
        out[f"{col}_{prefix}_min"] = roll[col].min()
        out[f"{col}_{prefix}_max"] = roll[col].max()
        out[f"{col}_{prefix}_delta"] = df[col] - df[col].shift(window)
    return pd.DataFrame(out, index=df.index)


def label_flares(
    index: pd.DatetimeIndex, flare_events, horizon_min: int, class_filter=("M", "X")
) -> pd.Series:
    times = index.values.astype("datetime64[ns]")
    labels = np.zeros(len(times), dtype=bool)
    for ev in flare_events:
        if ev["class"] not in class_filter:
            continue
        peak = np.datetime64(ev["peak"])
        start = peak - np.timedelta64(horizon_min, "m")
        left = np.searchsorted(times, start, side="left")
        right = np.searchsorted(times, peak, side="left")
        if right > left:
            labels[left:right] = True
    return pd.Series(labels, index=index)


def build_dataset(
    omni_csv: str,
    flare_reports_dir: str,
    output_csv: str,
    horizon_min: int = 15,
    chunksize: int = 400000,
    skip_flare: bool = False,
) -> None:
    usecols = ["time"] + FEATURE_COLS
    dtype = {col: "float32" for col in FEATURE_COLS}

    flare_events = []
    if not skip_flare and flare_reports_dir and os.path.isdir(flare_reports_dir):
        flare_events = load_flare_reports(flare_reports_dir)

    iterator = pd.read_csv(
        omni_csv,
        parse_dates=["time"],
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    wrote_header = False
    prev_tail = None
    pending_index = None
    total_rows = 0
    chunk_id = 0

    for chunk in iterator:
        chunk_id += 1
        total_rows += len(chunk)
        print(f"[build] chunk {chunk_id} rows={len(chunk)} total={total_rows}")
        chunk = chunk.sort_values("time")
        combined = (
            chunk
            if prev_tail is None
            else pd.concat([prev_tail, chunk], ignore_index=True)
        )
        combined = combined.sort_values("time").set_index("time")

        base = combined[FEATURE_COLS].copy()
        feat_15 = add_rolling_features(combined, window=15, prefix="w15")
        feat_60 = add_rolling_features(combined, window=60, prefix="w60")
        features = pd.concat([base, feat_15, feat_60], axis=1)

        out = features.copy()
        out["symh_future"] = combined["sym_h"].shift(-horizon_min)
        out["storm_risk"] = (out["symh_future"] <= -50).astype(int)

        if flare_events:
            out["flare_mx_next_15m"] = label_flares(
                combined.index, flare_events, horizon_min=horizon_min
            ).astype(int)

        if pending_index is not None:
            pending_ready = out.loc[pending_index].dropna(subset=["symh_future"])
            if not pending_ready.empty:
                pending_ready.to_csv(
                    output_csv, mode="a", header=not wrote_header, index=True
                )
                wrote_header = True

        current_index = chunk["time"]
        out_chunk = out.loc[current_index]
        if len(out_chunk) > horizon_min:
            ready = out_chunk.iloc[:-horizon_min]
            pending_index = out_chunk.iloc[-horizon_min:].index
        else:
            ready = out_chunk.iloc[0:0]
            pending_index = out_chunk.index

        ready = ready.dropna(subset=["symh_future"])
        if not ready.empty:
            ready.to_csv(output_csv, mode="a", header=not wrote_header, index=True)
            wrote_header = True

        prev_tail = combined.tail(60).reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ML dataset from solar wind data"
    )
    parser.add_argument("--omni-csv", required=True)
    parser.add_argument("--flare-reports-dir", default=None)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--horizon-min", type=int, default=15)
    parser.add_argument("--chunksize", type=int, default=400000)
    parser.add_argument("--skip-flare", action="store_true")
    args = parser.parse_args()

    build_dataset(
        args.omni_csv,
        args.flare_reports_dir,
        args.output_csv,
        args.horizon_min,
        args.chunksize,
        args.skip_flare,
    )
