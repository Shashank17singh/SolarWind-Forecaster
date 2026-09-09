import argparse
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import tensorflow as tf


def _load_context(omni_csv: str, feature_cols: list[str], seq_len: int) -> pd.DataFrame:
    usecols = ["time"] + feature_cols
    df = pd.read_csv(omni_csv, usecols=usecols, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    df = df.resample("1h").mean()
    df = df.dropna(subset=feature_cols)
    if len(df) < seq_len:
        raise ValueError("Not enough history to build the initial sequence.")
    return df.iloc[-seq_len:].copy()


def _standardize(
    df: pd.DataFrame, mean: dict, std: dict, feature_cols: list[str]
) -> np.ndarray:
    out = df.copy()
    for col in feature_cols:
        out[col] = (out[col] - mean[col]) / (std[col] if std[col] else 1.0)
    return out[feature_cols].to_numpy(dtype=np.float32)


def _build_time_features(ts: pd.Timestamp, feature_cols: list[str]) -> dict:
    values = {}
    if "year" in feature_cols:
        values["year"] = ts.year
    if "doy" in feature_cols:
        values["doy"] = int(ts.dayofyear)
    if "hour" in feature_cols:
        values["hour"] = ts.hour
    return values


def forecast(
    omni_csv: str,
    model_path: str,
    meta_path: str,
    steps: int,
    out_csv: str,
):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]
    target_cols = meta["target_cols"]
    seq_len = int(meta["seq_len"])
    horizon_hours = int(meta["horizon_hours"])
    mean = meta["mean"]
    std = meta["std"]

    model = tf.keras.models.load_model(model_path)

    history = _load_context(omni_csv, feature_cols, seq_len)
    current_time = history.index[-1]

    outputs = []
    for _ in range(steps):
        X = _standardize(history, mean, std, feature_cols)
        X = np.expand_dims(X, axis=0)
        pred = model.predict(X, verbose=0).reshape(-1)

        current_time = current_time + timedelta(hours=horizon_hours)
        row = _build_time_features(current_time, feature_cols)
        for col, val in zip(target_cols, pred):
            row[col] = float(val)
        outputs.append({"time": current_time, **row})

        next_row = {col: row.get(col, np.nan) for col in feature_cols}
        history = pd.concat(
            [history.iloc[1:], pd.DataFrame(next_row, index=[current_time])],
            axis=0,
        )

    out_df = pd.DataFrame(outputs)
    out_df.to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forecast solar wind fields (multi-output)"
    )
    parser.add_argument("--omni-csv", default="data/processed/omni.csv")
    parser.add_argument("--model-path", default="models/solar_wind_lstm.keras")
    parser.add_argument("--meta-path", default="models/solar_wind_lstm_meta.json")
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of 6-hour steps to forecast"
    )
    parser.add_argument("--out-csv", default="models/solar_wind_forecast.csv")
    args = parser.parse_args()

    out = forecast(
        omni_csv=args.omni_csv,
        model_path=args.model_path,
        meta_path=args.meta_path,
        steps=args.steps,
        out_csv=args.out_csv,
    )
    print(f"[sw] wrote forecast -> {out}")
