import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_dst_lstm_attention import FEATURE_COLS, DERIVED_FEATURES


def _hourly_omni(path: Path, feature_cols: list[str]) -> pd.DataFrame:
    usecols = ["time"] + feature_cols
    df = pd.read_csv(path, usecols=usecols, parse_dates=["time"])
    df = df.dropna(subset=["time"])
    df = df.set_index("time").sort_index().resample("1h").mean(numeric_only=True)
    return df


def _hourly_dst(path: Path, filter_final: bool) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["time", "dst", "source"], parse_dates=["time"])
    df = df.dropna(subset=["time"])
    if filter_final:
        df["source"] = df["source"].astype(str).str.lower()
        df = df[df["source"].str.contains("final", na=False)]
    df["dst"] = pd.to_numeric(df["dst"], errors="coerce")
    df = df.set_index("time").sort_index().resample("1h").mean(numeric_only=True)
    return df.rename(columns={"dst": "dst_kyoto"})


def _interpolate(
    df: pd.DataFrame, cols: list[str], method: str, limit: int
) -> pd.DataFrame:
    df = df.copy()
    if method == "time":
        df[cols] = df[cols].interpolate(method="time", limit=limit, limit_area="inside")
    else:
        df[cols] = df[cols].interpolate(
            method="spline", order=3, limit=limit, limit_area="inside"
        )
    return df


def clean(
    omni_csv: str,
    dst_csv: str,
    out_joined: str,
    out_omni: str | None,
    out_dst: str | None,
    interp_method: str,
    interp_limit: int,
    filter_final: bool,
):
    omni_path = Path(omni_csv)
    dst_path = Path(dst_csv)
    if not omni_path.exists():
        raise FileNotFoundError(f"OMNI CSV not found: {omni_csv}")
    if not dst_path.exists():
        raise FileNotFoundError(f"Dst CSV not found: {dst_csv}")

    base_cols = [c for c in FEATURE_COLS if c not in DERIVED_FEATURES]

    print("[clean] loading hourly OMNI...", flush=True)
    omni = _hourly_omni(omni_path, base_cols)
    print(
        f"[clean] OMNI rows={len(omni)} range={omni.index.min()} -> {omni.index.max()}",
        flush=True,
    )

    print("[clean] loading hourly Dst...", flush=True)
    dst = _hourly_dst(dst_path, filter_final)
    print(
        f"[clean] Dst rows={len(dst)} range={dst.index.min()} -> {dst.index.max()}",
        flush=True,
    )

    joined = omni.join(dst[["dst_kyoto"]], how="left")

    joined = _interpolate(joined, base_cols, interp_method, interp_limit)

    before = len(joined)
    joined = joined.dropna(subset=base_cols + ["dst_kyoto"])
    after = len(joined)
    print(f"[clean] dropped rows: {before - after} (kept {after})", flush=True)

    out_joined_path = Path(out_joined)
    out_joined_path.parent.mkdir(parents=True, exist_ok=True)
    joined.reset_index().to_csv(out_joined_path, index=False)
    print(f"[clean] wrote joined -> {out_joined_path}", flush=True)

    if out_omni:
        out_omni_path = Path(out_omni)
        out_omni_path.parent.mkdir(parents=True, exist_ok=True)
        joined.reset_index()[["time"] + base_cols].to_csv(out_omni_path, index=False)
        print(f"[clean] wrote omni -> {out_omni_path}", flush=True)
    if out_dst:
        out_dst_path = Path(out_dst)
        out_dst_path.parent.mkdir(parents=True, exist_ok=True)
        joined.reset_index()[["time", "dst_kyoto"]].rename(
            columns={"dst_kyoto": "dst"}
        ).to_csv(out_dst_path, index=False)
        print(f"[clean] wrote dst -> {out_dst_path}", flush=True)

    summary = {
        "rows_in": int(before),
        "rows_out": int(after),
        "dropped": int(before - after),
        "omni_range": [str(omni.index.min()), str(omni.index.max())],
        "dst_range": [str(dst.index.min()), str(dst.index.max())],
        "joined_range": [str(joined.index.min()), str(joined.index.max())],
        "interp_method": interp_method,
        "interp_limit": interp_limit,
        "filter_final": filter_final,
        "feature_cols": base_cols,
    }
    meta_path = out_joined_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(summary, indent=2))
    print(f"[clean] wrote meta -> {meta_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and align OMNI + Dst dataset")
    parser.add_argument("--omni-csv", required=True)
    parser.add_argument("--dst-csv", required=True)
    parser.add_argument("--out-joined", required=True)
    parser.add_argument("--out-omni", default=None)
    parser.add_argument("--out-dst", default=None)
    parser.add_argument("--interp-method", choices=["time", "spline"], default="time")
    parser.add_argument("--interp-limit", type=int, default=24)
    parser.add_argument(
        "--filter-final",
        action="store_true",
        help="Keep only Dst rows with source containing 'final'",
    )
    args = parser.parse_args()

    clean(
        omni_csv=args.omni_csv,
        dst_csv=args.dst_csv,
        out_joined=args.out_joined,
        out_omni=args.out_omni,
        out_dst=args.out_dst,
        interp_method=args.interp_method,
        interp_limit=args.interp_limit,
        filter_final=args.filter_final,
    )
