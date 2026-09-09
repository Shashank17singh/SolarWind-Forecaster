import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf

from train_dst_lstm_attention import (
    FEATURE_COLS,
    _interpolate_missing,
    _load_dst_csv,
    _load_hourly_omni,
)


@dataclass
class Window:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


WINDOWS = [
    Window(
        "2023-02-14_to_2023-05-05",
        pd.Timestamp("2023-02-14"),
        pd.Timestamp("2023-05-05 23:00:00"),
    ),
    Window(
        "2024-04-24_to_2024-05-31",
        pd.Timestamp("2024-04-24"),
        pd.Timestamp("2024-05-31 23:00:00"),
    ),
]


def _standardize(df: pd.DataFrame, mean: dict, std: dict) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURE_COLS:
        out[col] = (out[col] - mean[col]) / (std[col] if std[col] else 1.0)
    return out


def _predict_series(
    df: pd.DataFrame,
    mean: dict,
    std: dict,
    seq_len: int,
    horizon_hours: int,
    model: tf.keras.Model,
):
    df = _standardize(df, mean, std)

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["dst_target"].to_numpy(dtype=np.float32)
    target_offset = seq_len - 1 + horizon_hours
    if len(y) <= target_offset:
        raise ValueError("Not enough rows to create sequences for evaluation.")

    targets = y[target_offset:]
    end_index = len(X) - target_offset - 1
    times = df.index[target_offset:]

    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=X,
        targets=targets,
        sequence_length=seq_len,
        sequence_stride=1,
        sampling_rate=1,
        start_index=0,
        end_index=end_index,
        batch_size=256,
        shuffle=False,
    )
    preds = model.predict(ds, verbose=0).reshape(-1)
    return times, targets[: len(preds)], preds


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(
    omni_csv: str,
    dst_csv: str,
    model_path: str,
    meta_path: str,
    out_dir: str,
    interp: bool,
    fill_missing: bool,
):
    t0 = pd.Timestamp.utcnow()
    print("[eval] loading model + metadata...", flush=True)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    seq_len = int(meta["seq_len"])
    horizon_hours = int(meta["horizon_hours"])
    mean = meta["mean"]
    std = meta["std"]

    print("[eval] loading solar wind + Dst...", flush=True)
    df = _load_hourly_omni(omni_csv)
    dst_df = _load_dst_csv(dst_csv)
    df = df.join(dst_df[["dst_kyoto"]], how="left")
    if interp:
        print("[eval] interpolating (limit=24h)...", flush=True)
        df = _interpolate_missing(df, limit=24)
    elif not fill_missing:
        print(
            "[eval] skipping interpolation; dropping rows with missing values",
            flush=True,
        )
        df = df.dropna()
    if fill_missing:
        for col in FEATURE_COLS:
            df[col] = df[col].fillna(mean.get(col, 0.0))
        print("[eval] filled missing feature values with training mean", flush=True)
    print(f"[eval] base dataframe rows={len(df)}", flush=True)

    df["dst_target"] = df["dst_kyoto"].shift(-horizon_hours)
    df = df.dropna(subset=["dst_target"])

    model = tf.keras.models.load_model(model_path)
    print("[eval] model loaded", flush=True)

    os.makedirs(out_dir, exist_ok=True)

    results = []
    for window in WINDOWS:
        sub = df[(df.index >= window.start) & (df.index <= window.end)].copy()
        if sub.empty:
            print(f"[eval] window {window.name}: no data")
            continue

        print(
            f"[eval] window {window.name}: building sequences + predicting...",
            flush=True,
        )
        times, truth, preds = _predict_series(
            sub,
            mean,
            std,
            seq_len,
            horizon_hours,
            model,
        )
        rmse = _rmse(truth, preds)
        corr = _corr(truth, preds)

        out_csv = os.path.join(out_dir, f"dst_eval_{window.name}.csv")
        pd.DataFrame(
            {"time": times[: len(preds)], "dst_true": truth, "dst_pred": preds}
        ).to_csv(out_csv, index=False)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(times[: len(preds)], truth, label="Dst (true)", color="#f59e0b")
            ax.plot(
                times[: len(preds)], preds, label="Model prediction", color="#2563eb"
            )
            ax.set_title(f"Dst Prediction: {window.name}")
            ax.set_ylabel("nT")
            ax.legend()
            fig.tight_layout()
            out_png = os.path.join(out_dir, f"dst_eval_{window.name}.png")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"[eval] plot failed for {window.name}: {exc}")

        results.append(
            {
                "window": window.name,
                "rmse": rmse,
                "corr": corr,
                "rows": len(preds),
                "csv": out_csv,
            }
        )
        print(
            f"[eval] {window.name}: rmse={rmse:.3f} corr={corr:.4f} rows={len(preds)}",
            flush=True,
        )

    if results:
        summary_path = os.path.join(out_dir, "dst_eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] wrote summary -> {summary_path}", flush=True)
    elapsed = (pd.Timestamp.utcnow() - t0).total_seconds()
    print(f"[eval] done in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Dst LSTM model on paper windows"
    )
    parser.add_argument("--omni-csv", default="data/processed/omni.csv")
    parser.add_argument("--dst-csv", default="data/indices/kyoto/dst_hourly.csv")
    parser.add_argument("--model-path", default="models/dst_lstm_attention.keras")
    parser.add_argument("--meta-path", default="models/dst_lstm_attention_meta.json")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument(
        "--no-interp",
        action="store_true",
        help="Skip interpolation and drop rows with missing values",
    )
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Fill missing feature values with training mean instead of dropping rows",
    )
    args = parser.parse_args()

    evaluate(
        omni_csv=args.omni_csv,
        dst_csv=args.dst_csv,
        model_path=args.model_path,
        meta_path=args.meta_path,
        out_dir=args.out_dir,
        interp=not args.no_interp,
        fill_missing=args.fill_missing,
    )
