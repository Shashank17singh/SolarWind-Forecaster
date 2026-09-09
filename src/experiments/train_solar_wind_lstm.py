import argparse
import json
import os
import time
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf

from config import CONFIG
from model_registry import register_model

TIME_COLS = [
    "doy",
    "hour",
]


SOLAR_WIND_COLS = [
    "b_mag",
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
    "magnetosonic_mach",
]


@dataclass
class SplitRanges:
    train_end: pd.Timestamp
    val_end: pd.Timestamp


def _parse_ts(value: str, label: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid {label}: {value}")
    return ts


def _load_hourly_omni(omni_csv: str, feature_cols: list[str]) -> pd.DataFrame:
    usecols = ["time"] + feature_cols
    df = pd.read_csv(omni_csv, usecols=usecols, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    df = df.resample("1h").mean()
    return df


def _interpolate_missing(
    df: pd.DataFrame, limit: int = 24, method: str = "spline"
) -> pd.DataFrame:
    numeric_cols = df.columns
    if method == "time":
        df[numeric_cols] = df[numeric_cols].interpolate(
            method="time",
            limit=limit,
            limit_area="inside",
        )
        return df
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="spline",
        order=3,
        limit=limit,
        limit_area="inside",
    )
    return df


def _split_by_time(df: pd.DataFrame, ranges: SplitRanges):
    train = df[df.index <= ranges.train_end]
    val = df[(df.index > ranges.train_end) & (df.index <= ranges.val_end)]
    test = df[df.index > ranges.val_end]
    return train, val, test


def _standardize(train_df: pd.DataFrame, df: pd.DataFrame, feature_cols: list[str]):
    mean = train_df[feature_cols].mean().fillna(0.0)
    std = train_df[feature_cols].std().replace(0, 1.0).fillna(1.0)
    out = df.copy()
    out[feature_cols] = (out[feature_cols] - mean) / std
    return out, mean, std


def _report_nans(df: pd.DataFrame, cols: list[str], label: str):
    nan_counts = df[cols].isna().sum().sort_values(ascending=False)
    top = nan_counts.head(15)
    total_rows = len(df)
    rows_any = int(df[cols].isna().any(axis=1).sum())
    print(
        f"[sw] NaN report ({label}) - rows={total_rows} rows_with_any_nan={rows_any}",
        flush=True,
    )
    for col, count in top.items():
        if count == 0:
            break
        print(f"[sw]   {col}: {int(count)}", flush=True)


def _make_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    seq_len: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_cols].to_numpy(dtype=np.float32)

    target_offset = seq_len - 1 + horizon
    if len(y) <= target_offset:
        raise ValueError(
            "Not enough rows to create sequences. Reduce seq_len or horizon."
        )

    targets = y[target_offset:]
    end_index = len(X) - target_offset - 1

    with tf.device("/CPU:0"):
        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=X,
            targets=targets,
            sequence_length=seq_len,
            sequence_stride=1,
            sampling_rate=1,
            start_index=0,
            end_index=end_index,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    return dataset


def _configure_gpu():
    if os.getenv("FORCE_CPU", "").lower() in {"1", "true", "yes"}:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        return
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def _build_model(seq_len: int, n_features: int, n_targets: int):
    inputs = tf.keras.Input(shape=(seq_len, n_features))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(
        inputs
    )
    x = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(
        x
    )
    x = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_targets)(x)
    return tf.keras.Model(inputs, outputs)


def _lr_schedule(epoch: int):
    block = (epoch // 5) % 2
    return 1e-3 if block == 0 else 1e-4


def _evaluate_metrics(
    model: tf.keras.Model,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    seq_len: int,
    horizon: int,
    batch_size: int,
):
    ds = _make_dataset(
        df,
        feature_cols,
        target_cols,
        seq_len,
        horizon,
        batch_size,
        shuffle=False,
    )
    preds = model.predict(ds, verbose=0)
    y = df[target_cols].to_numpy(dtype=np.float32)
    target_offset = seq_len - 1 + horizon
    y_true = y[target_offset : target_offset + len(preds)]

    rmse = np.sqrt(np.mean((y_true - preds) ** 2, axis=0))
    mae = np.mean(np.abs(y_true - preds), axis=0)
    metrics = {"overall_rmse": float(np.sqrt(np.mean((y_true - preds) ** 2)))}
    metrics["per_target_rmse"] = {
        col: float(val) for col, val in zip(target_cols, rmse)
    }
    metrics["per_target_mae"] = {col: float(val) for col, val in zip(target_cols, mae)}
    return metrics


def train(
    omni_csv: str,
    model_dir: str,
    train_end: str,
    val_end: str,
    seq_len: int,
    horizon_hours: int,
    epochs: int,
    batch_size: int,
    interp: bool,
    interp_method: str,
    include_year: bool,
    checkpoint_dir: str | None,
    model_name: str,
    resume: str | None,
    resume_meta: str | None,
    initial_epoch_override: int | None,
    nan_report: bool,
    nan_report_only: bool,
):
    _configure_gpu()
    t0 = time.time()
    feature_cols = TIME_COLS + SOLAR_WIND_COLS
    if include_year:
        feature_cols = ["year"] + feature_cols

    print("[sw] loading solar wind CSV...", flush=True)
    df = _load_hourly_omni(omni_csv, feature_cols)
    print(
        f"[sw] loaded + hourly resample rows={len(df)} in {time.time() - t0:.1f}s",
        flush=True,
    )

    t1 = time.time()
    if interp:
        print(
            f"[sw] interpolating missing values ({interp_method}, limit=24h)...",
            flush=True,
        )
        df = _interpolate_missing(df, limit=24, method=interp_method)
        print(f"[sw] interpolation done in {time.time() - t1:.1f}s", flush=True)
    else:
        print(
            "[sw] skipping interpolation; dropping rows with missing values", flush=True
        )
        df = df.dropna(subset=feature_cols)

    if nan_report:
        _report_nans(df, feature_cols + SOLAR_WIND_COLS, "post-interp")
    nan_before = df[feature_cols + SOLAR_WIND_COLS].isna().any(axis=1).sum()
    if nan_before:
        print(f"[sw] dropping {nan_before} rows with remaining NaNs", flush=True)
        df = df.dropna(subset=feature_cols + SOLAR_WIND_COLS)
    if nan_report:
        _report_nans(df, feature_cols + SOLAR_WIND_COLS, "post-drop")
    if nan_report_only:
        return

    ranges = SplitRanges(
        train_end=_parse_ts(train_end, "--train-end"),
        val_end=_parse_ts(val_end, "--val-end"),
    )
    train_df, val_df, test_df = _split_by_time(df, ranges)
    print(
        "[sw] rows: train={t} val={v} test={s}".format(
            t=len(train_df), v=len(val_df), s=len(test_df)
        ),
        flush=True,
    )

    train_df, mean, std = _standardize(train_df, train_df, feature_cols)
    val_df, _, _ = _standardize(train_df, val_df, feature_cols)
    test_df, _, _ = _standardize(train_df, test_df, feature_cols)

    train_ds = _make_dataset(
        train_df,
        feature_cols,
        SOLAR_WIND_COLS,
        seq_len,
        horizon_hours,
        batch_size,
        shuffle=True,
    )
    val_ds = None
    test_ds = None
    min_rows = seq_len + horizon_hours + 1
    if len(val_df) >= min_rows:
        val_ds = _make_dataset(
            val_df,
            feature_cols,
            SOLAR_WIND_COLS,
            seq_len,
            horizon_hours,
            batch_size,
            shuffle=False,
        )
    else:
        print("[sw] validation set too small; skipping val", flush=True)

    if len(test_df) >= min_rows:
        test_ds = _make_dataset(
            test_df,
            feature_cols,
            SOLAR_WIND_COLS,
            seq_len,
            horizon_hours,
            batch_size,
            shuffle=False,
        )
    else:
        print("[sw] test set too small; skipping test", flush=True)

    train_card = tf.data.experimental.cardinality(train_ds).numpy()
    val_card = (
        tf.data.experimental.cardinality(val_ds).numpy() if val_ds is not None else 0
    )
    test_card = (
        tf.data.experimental.cardinality(test_ds).numpy() if test_ds is not None else 0
    )
    print(
        f"[sw] batches: train={train_card} val={val_card} test={test_card}",
        flush=True,
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    if val_ds is not None:
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    if resume:
        print(f"[sw] loading model from {resume}", flush=True)
        model = tf.keras.models.load_model(resume)
        input_shape = model.input_shape
        if input_shape[1] != seq_len or input_shape[2] != len(feature_cols):
            raise ValueError(
                f"Resume model shape {input_shape} does not match seq_len={seq_len}, "
                f"features={len(feature_cols)}"
            )
    else:
        model = _build_model(seq_len, len(feature_cols), len(SOLAR_WIND_COLS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    callbacks = [tf.keras.callbacks.LearningRateScheduler(_lr_schedule, verbose=1)]
    if val_ds is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_rmse", patience=5, restore_best_weights=True
            )
        )
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        monitor = "val_rmse" if val_ds is not None else "rmse"
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f"{model_name}_best.keras"),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            )
        )
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f"{model_name}_latest.keras"),
                save_best_only=False,
                save_weights_only=False,
                verbose=0,
            )
        )

    initial_epoch = 0
    total_epochs = epochs
    if resume:
        meta_path = resume_meta or os.path.join(model_dir, f"{model_name}_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                prev_meta = json.load(f)
            prev_epochs = len(prev_meta.get("history", {}).get("loss", []))
            initial_epoch = prev_epochs
            total_epochs = prev_epochs + epochs
            print(
                f"[sw] resuming from epoch {prev_epochs}; training {epochs} more (total {total_epochs})",
                flush=True,
            )
        elif initial_epoch_override is not None:
            initial_epoch = int(initial_epoch_override)
            total_epochs = initial_epoch + epochs
            print(
                f"[sw] resuming from epoch {initial_epoch}; training {epochs} more (total {total_epochs})",
                flush=True,
            )
        else:
            print(
                "[sw] resume meta not found; restarting LR schedule at epoch 0",
                flush=True,
            )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=2,
    )

    test_metrics = None
    if test_ds is not None:
        test_metrics = _evaluate_metrics(
            model,
            test_df,
            feature_cols,
            SOLAR_WIND_COLS,
            seq_len,
            horizon_hours,
            batch_size,
        )
        print(f"[sw] test metrics -> {test_metrics}", flush=True)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.keras")
    model.save(model_path)
    print(f"[sw] saved model -> {model_path}", flush=True)

    meta = {
        "feature_cols": feature_cols,
        "target_cols": SOLAR_WIND_COLS,
        "seq_len": seq_len,
        "horizon_hours": horizon_hours,
        "train_end": train_end,
        "val_end": val_end,
        "interp_method": interp_method,
        "model_name": model_name,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "history": history.history,
        "test_metrics": test_metrics,
        "trained_epochs_total": total_epochs,
    }
    meta_path = os.path.join(model_dir, f"{model_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[sw] saved metadata -> {meta_path}", flush=True)

    register_model(
        name=model_name,
        version=datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        artifact_path=model_path,
        metrics=test_metrics or {},
        metadata={
            "train_end": train_end,
            "val_end": val_end,
            "seq_len": seq_len,
            "horizon_hours": horizon_hours,
            "feature_cols": feature_cols,
            "feature_spec_path": CONFIG.feature_spec_path,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train multi-output solar wind forecast model"
    )
    parser.add_argument("--omni-csv", default="data/processed/omni.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--val-end", default="2025-12-31")
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--horizon-hours", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--no-interp",
        action="store_true",
        help="Skip interpolation and drop rows with missing values",
    )
    parser.add_argument(
        "--interp-method",
        choices=["spline", "time"],
        default="spline",
        help="Interpolation method when --no-interp is not set",
    )
    parser.add_argument(
        "--include-year", action="store_true", help="Include year as an input feature"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to save best/latest checkpoints each epoch",
    )
    parser.add_argument("--model-name", default="solar_wind_lstm")
    parser.add_argument(
        "--nan-report", action="store_true", help="Print NaN report before training"
    )
    parser.add_argument(
        "--nan-report-only", action="store_true", help="Only report NaNs and exit"
    )
    parser.add_argument(
        "--resume", default=None, help="Path to a saved .keras model to resume training"
    )
    parser.add_argument(
        "--resume-meta", default=None, help="Path to meta.json to continue LR schedule"
    )
    parser.add_argument(
        "--initial-epoch",
        type=int,
        default=None,
        help="Manual initial epoch if meta is missing",
    )
    args = parser.parse_args()

    train(
        omni_csv=args.omni_csv,
        model_dir=args.model_dir,
        train_end=args.train_end,
        val_end=args.val_end,
        seq_len=args.seq_len,
        horizon_hours=args.horizon_hours,
        epochs=args.epochs,
        batch_size=args.batch_size,
        interp=not args.no_interp,
        interp_method=args.interp_method,
        include_year=args.include_year,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        resume=args.resume,
        resume_meta=args.resume_meta,
        initial_epoch_override=args.initial_epoch,
        nan_report=args.nan_report,
        nan_report_only=args.nan_report_only,
    )
