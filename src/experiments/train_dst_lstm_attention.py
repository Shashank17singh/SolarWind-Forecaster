import argparse
import json
import os
import re
from datetime import datetime
from urllib.request import urlopen, Request
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
import time

from config import CONFIG
from model_registry import register_model

DERIVED_FEATURES = {
    "v_np",
    "v2_np",
    "vbz_south",
    "bz_south",
    "bz_abs",
}


def _feature_spec_version() -> str | None:
    path = CONFIG.feature_spec_path
    if not path or not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return spec.get("version")
    except Exception:
        return None


SOLAR_CYCLE_COLS = [
    "solar_f107_smoothed",
    "solar_ssn_smoothed",
    "solar_cycle_phase",
]

FEATURE_COLS = [
    "year",
    "doy",
    "hour",
    "minute",
    "imf_sc_id",
    "sw_sc_id",
    "imf_npts",
    "sw_npts",
    "pct_interp",
    "timeshift_sec",
    "rms_timeshift_sec",
    "rms_phase_front_norm",
    "dbot_sec",
    "b_mag",
    "bx_gse",
    "by_gse",
    "bz_gse",
    "by_gsm",
    "bz_gsm",
    "rms_sd_b",
    "rms_sd_bvec",
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
    "sc_x_gse",
    "sc_y_gse",
    "sc_z_gse",
    "bsn_x",
    "bsn_y",
    "bsn_z",
    "ae",
    "al",
    "au",
    "sym_d",
    "asy_d",
    "asy_h",
    "pcn",
    "magnetosonic_mach",
    "v_np",
    "v2_np",
    "vbz_south",
    "bz_south",
    "bz_abs",
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


def _parse_feature_cols(value: str | None) -> list[str]:
    if not value:
        return FEATURE_COLS
    if os.path.exists(value):
        text = open(value, "r", encoding="utf-8").read()
    else:
        text = value
    tokens = [t.strip() for t in re.split(r"[,\s]+", text) if t.strip()]
    if not tokens:
        return FEATURE_COLS
    return tokens


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    if "flow_speed" in df.columns and "proton_density" in df.columns:
        if "v_np" in df.columns or "v_np" in DERIVED_FEATURES:
            df["v_np"] = df["flow_speed"] * df["proton_density"]
        if "v2_np" in df.columns or "v2_np" in DERIVED_FEATURES:
            df["v2_np"] = (df["flow_speed"] ** 2) * df["proton_density"]
    if "bz_gsm" in df.columns:
        if "bz_south" in df.columns or "bz_south" in DERIVED_FEATURES:
            df["bz_south"] = np.minimum(df["bz_gsm"], 0.0)
        if "bz_abs" in df.columns or "bz_abs" in DERIVED_FEATURES:
            df["bz_abs"] = np.abs(df["bz_gsm"])
    if "flow_speed" in df.columns and "bz_gsm" in df.columns:
        if "vbz_south" in df.columns or "vbz_south" in DERIVED_FEATURES:
            df["vbz_south"] = df["flow_speed"] * np.minimum(df["bz_gsm"], 0.0)
    return df


def _load_cycle_json(source: str) -> list[dict]:
    if not source:
        return []
    if source.startswith("http://") or source.startswith("https://"):
        req = Request(source, headers={"User-Agent": "space-weather-sentinel/1.0"})
        with urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    with open(source, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_cycle_rows(data: list[dict], kind: str) -> pd.DataFrame:
    rows = []
    for row in data or []:
        tag = row.get("time-tag") or row.get("time_tag") or row.get("time")
        if not tag:
            continue
        ts = pd.to_datetime(tag, errors="coerce")
        if pd.isna(ts):
            continue
        item = {"month": ts.strftime("%Y-%m")}
        if kind == "observed":
            item["solar_ssn_smoothed"] = (
                row.get("smoothed_ssn")
                or row.get("smoothed_swpc_ssn")
                or row.get("ssn")
            )
            item["solar_f107_smoothed"] = (
                row.get("smoothed_f10.7") or row.get("f10.7") or row.get("f107")
            )
        else:
            item["solar_ssn_smoothed"] = row.get("predicted_ssn") or row.get("ssn")
            item["solar_f107_smoothed"] = (
                row.get("predicted_f10.7") or row.get("f10.7") or row.get("f107")
            )
        rows.append(item)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["solar_ssn_smoothed"] = pd.to_numeric(df["solar_ssn_smoothed"], errors="coerce")
    df["solar_f107_smoothed"] = pd.to_numeric(
        df["solar_f107_smoothed"], errors="coerce"
    )
    df = df.dropna(subset=["month"])
    df = df.drop_duplicates(subset=["month"], keep="last")
    return df


def _attach_solar_cycle(
    df: pd.DataFrame, observed_src: str | None, predicted_src: str | None
) -> pd.DataFrame:
    obs_df = (
        _parse_cycle_rows(_load_cycle_json(observed_src), "observed")
        if observed_src
        else pd.DataFrame()
    )
    pred_df = (
        _parse_cycle_rows(_load_cycle_json(predicted_src), "predicted")
        if predicted_src
        else pd.DataFrame()
    )
    cycle_df = pd.concat([obs_df, pred_df], ignore_index=True)
    if cycle_df.empty:
        return df

    f107_vals = cycle_df["solar_f107_smoothed"].dropna().to_numpy()
    ssn_vals = cycle_df["solar_ssn_smoothed"].dropna().to_numpy()
    if len(f107_vals):
        min_val, max_val = float(f107_vals.min()), float(f107_vals.max())
        if max_val != min_val:
            cycle_df["solar_cycle_phase"] = (
                cycle_df["solar_f107_smoothed"] - min_val
            ) / (max_val - min_val)
        else:
            cycle_df["solar_cycle_phase"] = 0.5
    elif len(ssn_vals):
        min_val, max_val = float(ssn_vals.min()), float(ssn_vals.max())
        if max_val != min_val:
            cycle_df["solar_cycle_phase"] = (
                cycle_df["solar_ssn_smoothed"] - min_val
            ) / (max_val - min_val)
        else:
            cycle_df["solar_cycle_phase"] = 0.5
    else:
        cycle_df["solar_cycle_phase"] = 0.5

    df = df.copy()
    df["month_key"] = df.index.to_period("M").astype(str)
    df = df.join(cycle_df.set_index("month"), on="month_key")
    df = df.drop(columns=["month_key"])
    return df


def _load_hourly_omni(
    omni_csv: str, feature_cols: list[str] | None = None
) -> pd.DataFrame:
    cols = feature_cols or FEATURE_COLS
    base_cols = [
        c for c in cols if c not in DERIVED_FEATURES and c not in SOLAR_CYCLE_COLS
    ]
    usecols = ["time"] + base_cols + ["sym_h"]
    df = pd.read_csv(omni_csv, usecols=usecols, parse_dates=["time"])
    df = df.set_index("time").sort_index()

    df = df.resample("1h").mean()
    df = _add_derived_features(df)

    return df


def _load_dst_csv(dst_csv: str) -> pd.DataFrame:
    df = pd.read_csv(dst_csv, parse_dates=["time"])
    df = df.dropna(subset=["time"])
    df = df.set_index("time").sort_index()

    if "dst" not in df.columns:
        raise ValueError("Dst CSV missing required column: dst")
    df["dst"] = pd.to_numeric(df["dst"], errors="coerce")
    df = df[["dst"]]
    df = df.resample("1h").mean()
    df = df.rename(columns={"dst": "dst_kyoto"})
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


def _make_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
    horizon: int,
    batch_size: int,
    shuffle: bool,
):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

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


def _build_model(seq_len: int, n_features: int):
    inputs = tf.keras.Input(shape=(seq_len, n_features))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True))(
        inputs
    )
    x = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(170, return_sequences=True))(
        x
    )
    x = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)


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


def _lr_schedule(epoch: int):

    block = (epoch // 5) % 2
    return 1e-3 if block == 0 else 1e-4


def train(
    omni_csv: str,
    dst_csv: str | None,
    model_dir: str,
    train_end: str,
    val_end: str,
    seq_len: int,
    horizon_hours: int,
    epochs: int,
    batch_size: int,
    interp: bool,
    interp_method: str,
    feature_cols: list[str],
    model_name: str,
    resume: str | None,
    resume_meta: str | None,
    checkpoint_dir: str | None,
    early_stop_patience: int,
    solar_cycle_observed: str | None,
    solar_cycle_predicted: str | None,
):
    _configure_gpu()
    t0 = time.time()
    if any(col in SOLAR_CYCLE_COLS for col in feature_cols) and not (
        solar_cycle_observed or solar_cycle_predicted
    ):
        raise ValueError(
            "Solar cycle features requested but no solar cycle JSON provided."
        )
    print("[dst] loading solar wind CSV...", flush=True)
    df = _load_hourly_omni(omni_csv, feature_cols)
    if dst_csv:
        print(f"[dst] loading Dst index from {dst_csv} ...", flush=True)
        dst_df = _load_dst_csv(dst_csv)
        df = df.join(dst_df[["dst_kyoto"]], how="left")
    print(
        f"[dst] loaded + hourly resample rows={len(df)} in {time.time() - t0:.1f}s",
        flush=True,
    )

    if solar_cycle_observed or solar_cycle_predicted:
        print("[dst] attaching solar cycle progression features...", flush=True)
        df = _attach_solar_cycle(df, solar_cycle_observed, solar_cycle_predicted)

    t1 = time.time()
    if interp:
        print(
            f"[dst] interpolating missing values ({interp_method}, limit=24h)...",
            flush=True,
        )
        df = _interpolate_missing(df, limit=24, method=interp_method)
        print(f"[dst] interpolation done in {time.time() - t1:.1f}s", flush=True)
    else:
        print(
            "[dst] skipping interpolation; dropping rows with missing values",
            flush=True,
        )
        df = df.dropna(subset=feature_cols)

    target_col = "dst_kyoto" if dst_csv else "sym_h"
    print(f"[dst] building target horizon={horizon_hours}h", flush=True)
    df["dst_target"] = df[target_col].shift(-horizon_hours)
    df = df.dropna(subset=["dst_target"] + feature_cols)

    ranges = SplitRanges(
        train_end=_parse_ts(train_end, "--train-end"),
        val_end=_parse_ts(val_end, "--val-end"),
    )
    train_df, val_df, test_df = _split_by_time(df, ranges)
    print(
        "[dst] rows: train={t} val={v} test={s}".format(
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
        "dst_target",
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
            "dst_target",
            seq_len,
            horizon_hours,
            batch_size,
            shuffle=False,
        )
    else:
        print("[dst] validation set too small; skipping val", flush=True)

    if len(test_df) >= min_rows:
        test_ds = _make_dataset(
            test_df,
            feature_cols,
            "dst_target",
            seq_len,
            horizon_hours,
            batch_size,
            shuffle=False,
        )
    else:
        print("[dst] test set too small; skipping test", flush=True)

    train_card = tf.data.experimental.cardinality(train_ds).numpy()
    val_card = (
        tf.data.experimental.cardinality(val_ds).numpy() if val_ds is not None else 0
    )
    test_card = (
        tf.data.experimental.cardinality(test_ds).numpy() if test_ds is not None else 0
    )
    print(
        f"[dst] batches: train={train_card} val={val_card} test={test_card}",
        flush=True,
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    if val_ds is not None:
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    if resume:
        print(f"[dst] loading model from {resume}", flush=True)
        model = tf.keras.models.load_model(resume)

        input_shape = model.input_shape
        if input_shape[1] != seq_len or input_shape[2] != len(feature_cols):
            raise ValueError(
                f"Resume model shape {input_shape} does not match seq_len={seq_len}, "
                f"features={len(feature_cols)}"
            )
    else:
        model = _build_model(seq_len, len(feature_cols))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    callbacks = [tf.keras.callbacks.LearningRateScheduler(_lr_schedule, verbose=1)]
    if val_ds is not None and early_stop_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_rmse",
                patience=early_stop_patience,
                restore_best_weights=True,
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
                f"[dst] resuming from epoch {prev_epochs}; training {epochs} more (total {total_epochs})",
                flush=True,
            )
        else:
            print(
                "[dst] resume meta not found; restarting LR schedule at epoch 0",
                flush=True,
            )

    history = model.fit(
        train_ds,
        validation_data=val_ds if val_ds is not None else None,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=2,
    )

    test_metrics = None
    if test_ds is not None:
        test_metrics = model.evaluate(test_ds, return_dict=True)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.keras")
    model.save(model_path)

    spec_version = _feature_spec_version()
    meta = {
        "feature_cols": feature_cols,
        "seq_len": seq_len,
        "horizon_hours": horizon_hours,
        "train_end": train_end,
        "val_end": val_end,
        "dst_csv": dst_csv,
        "interp_method": interp_method,
        "model_name": model_name,
        "feature_spec_version": spec_version,
        "feature_spec_path": CONFIG.feature_spec_path,
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "history": history.history,
        "test_metrics": test_metrics,
        "trained_epochs_total": total_epochs,
    }
    with open(
        os.path.join(model_dir, f"{model_name}_meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, indent=2)

    registry_meta = {
        "train_end": train_end,
        "val_end": val_end,
        "seq_len": seq_len,
        "horizon_hours": horizon_hours,
        "feature_spec_version": spec_version,
        "feature_cols": feature_cols,
    }
    register_model(
        name=model_name,
        version=datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        artifact_path=model_path,
        metrics=test_metrics or {},
        metadata=registry_meta,
    )

    print(f"[dst] saved model -> {model_path}")
    print(f"[dst] test metrics -> {test_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LSTM+Attention Dst predictor (paper-inspired)"
    )
    parser.add_argument("--omni-csv", default="data/processed/omni.csv")
    parser.add_argument("--dst-csv", default=None)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--val-end", default="2023-12-31")
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--horizon-hours", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
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
        "--feature-cols",
        default=None,
        help="Comma-separated list or file path with feature columns",
    )
    parser.add_argument("--model-name", default="dst_lstm_attention")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Early stopping patience (0 disables early stopping)",
    )
    parser.add_argument(
        "--resume", default=None, help="Path to a saved .keras model to resume training"
    )
    parser.add_argument(
        "--resume-meta", default=None, help="Path to meta.json to continue LR schedule"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to save best/latest checkpoints each epoch",
    )
    parser.add_argument(
        "--solar-cycle-observed",
        default=None,
        help="Path or URL to observed solar cycle JSON",
    )
    parser.add_argument(
        "--solar-cycle-predicted",
        default=None,
        help="Path or URL to predicted solar cycle JSON",
    )
    args = parser.parse_args()

    feature_cols = _parse_feature_cols(args.feature_cols)
    if args.solar_cycle_observed or args.solar_cycle_predicted:
        for col in SOLAR_CYCLE_COLS:
            if col not in feature_cols:
                feature_cols.append(col)

    train(
        omni_csv=args.omni_csv,
        dst_csv=args.dst_csv,
        model_dir=args.model_dir,
        train_end=args.train_end,
        val_end=args.val_end,
        seq_len=args.seq_len,
        horizon_hours=args.horizon_hours,
        epochs=args.epochs,
        batch_size=args.batch_size,
        interp=not args.no_interp,
        interp_method=args.interp_method,
        feature_cols=feature_cols,
        model_name=args.model_name,
        resume=args.resume,
        resume_meta=args.resume_meta,
        checkpoint_dir=args.checkpoint_dir,
        early_stop_patience=args.early_stop_patience,
        solar_cycle_observed=args.solar_cycle_observed,
        solar_cycle_predicted=args.solar_cycle_predicted,
    )
