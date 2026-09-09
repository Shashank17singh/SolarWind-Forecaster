import argparse
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from model_registry import register_model
import pyarrow.parquet as pq
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression

LABEL_COLS = {"storm_risk", "symh_future", "flare_mx_next_15m"}


def _list_parquet_files(dir_path: str) -> list[Path]:
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet directory: {dir_path}")
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {dir_path}")
    return files


def _estimate_pos_weight(
    files: list[Path], label_col: str, max_rows: int, seed: int
) -> float:
    total = 0
    pos = 0
    for path in files:
        df = pd.read_parquet(path, columns=[label_col])
        if max_rows and total >= max_rows:
            break
        if max_rows:
            remaining = max_rows - total
            if remaining < len(df):
                df = df.sample(n=remaining, random_state=seed + total)
        y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype("int8")
        pos += int((y == 1).sum())
        total += len(y)
    if pos == 0:
        return 1.0
    return float((total - pos) / pos)


def _eval_classifier(
    model: lgb.Booster,
    files: list[Path],
    feature_cols: list[str],
    label_col: str,
    max_rows: int,
    seed: int,
):
    total = 0
    loss_sum = 0.0
    acc_sum = 0
    for path in files:
        if max_rows and total >= max_rows:
            break
        df = pd.read_parquet(path, columns=feature_cols + [label_col])
        if max_rows:
            remaining = max_rows - total
            if remaining < len(df):
                df = df.sample(n=remaining, random_state=seed + total)
        y = (
            pd.to_numeric(df[label_col], errors="coerce")
            .fillna(0)
            .astype("int8")
            .to_numpy()
        )
        X = df[feature_cols].to_numpy(dtype=np.float32)
        prob = model.predict(X)
        loss_sum += log_loss(y, prob, labels=[0, 1]) * len(y)
        acc_sum += ((prob >= 0.5).astype(np.int8) == y).sum()
        total += len(y)
    if total == 0:
        return {"log_loss": float("nan"), "accuracy": float("nan")}
    return {"log_loss": loss_sum / total, "accuracy": acc_sum / total}


def _eval_regressor(
    model: lgb.Booster,
    files: list[Path],
    feature_cols: list[str],
    label_col: str,
    max_rows: int,
    seed: int,
):
    total = 0
    abs_sum = 0.0
    sq_sum = 0.0
    for path in files:
        if max_rows and total >= max_rows:
            break
        df = pd.read_parquet(path, columns=feature_cols + [label_col])
        if max_rows:
            remaining = max_rows - total
            if remaining < len(df):
                df = df.sample(n=remaining, random_state=seed + total)
        y = (
            pd.to_numeric(df[label_col], errors="coerce")
            .fillna(0)
            .astype("float32")
            .to_numpy()
        )
        X = df[feature_cols].to_numpy(dtype=np.float32)
        pred = model.predict(X)
        diff = pred - y
        abs_sum += np.abs(diff).sum()
        sq_sum += np.square(diff).sum()
        total += len(y)
    if total == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    return {"mae": abs_sum / total, "rmse": np.sqrt(sq_sum / total)}


def _collect_probs(
    model: lgb.Booster,
    files: list[Path],
    feature_cols: list[str],
    label_col: str,
    max_rows: int,
    seed: int,
):
    probs = []
    labels = []
    total = 0
    for path in files:
        if max_rows and total >= max_rows:
            break
        df = pd.read_parquet(path, columns=feature_cols + [label_col])
        if max_rows:
            remaining = max_rows - total
            if remaining < len(df):
                df = df.sample(n=remaining, random_state=seed + total)
        y = (
            pd.to_numeric(df[label_col], errors="coerce")
            .fillna(0)
            .astype("int8")
            .to_numpy()
        )
        X = df[feature_cols].to_numpy(dtype=np.float32)
        prob = model.predict(X)
        probs.append(prob)
        labels.append(y)
        total += len(y)
    if not probs:
        return np.array([]), np.array([])
    return np.concatenate(probs), np.concatenate(labels)


def train_lgbm(
    parquet_dir: str,
    model_dir: str,
    rounds_per_shard: int,
    max_eval_rows: int,
    seed: int,
):
    train_files = _list_parquet_files(os.path.join(parquet_dir, "train"))
    val_files = _list_parquet_files(os.path.join(parquet_dir, "val"))
    test_files = _list_parquet_files(os.path.join(parquet_dir, "test"))

    schema_cols = pq.ParquetFile(train_files[0]).schema.names
    feature_cols = [c for c in schema_cols if c not in LABEL_COLS and c != "time"]
    flare_available = "flare_mx_next_15m" in schema_cols

    print(f"[lgbm] features={len(feature_cols)} flare={flare_available}", flush=True)

    storm_weight = _estimate_pos_weight(
        train_files, "storm_risk", max_rows=1_000_000, seed=seed
    )
    flare_weight = None
    if flare_available:
        flare_weight = _estimate_pos_weight(
            train_files, "flare_mx_next_15m", max_rows=1_000_000, seed=seed + 7
        )

    storm_params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "metric": "binary_logloss",
        "scale_pos_weight": storm_weight,
        "seed": seed,
        "verbosity": -1,
    }
    symh_params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "metric": "l1",
        "seed": seed,
        "verbosity": -1,
    }
    flare_params = None
    if flare_available:
        flare_params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "metric": "binary_logloss",
            "scale_pos_weight": flare_weight if flare_weight else 1.0,
            "seed": seed,
            "verbosity": -1,
        }

    storm_model = None
    symh_model = None
    flare_model = None

    for i, path in enumerate(train_files, start=1):
        df = pd.read_parquet(
            path,
            columns=feature_cols
            + ["storm_risk", "symh_future"]
            + (["flare_mx_next_15m"] if flare_available else []),
        )
        X = df[feature_cols].to_numpy(dtype=np.float32)
        y_storm = (
            pd.to_numeric(df["storm_risk"], errors="coerce")
            .fillna(0)
            .astype("int8")
            .to_numpy()
        )
        y_symh = (
            pd.to_numeric(df["symh_future"], errors="coerce")
            .fillna(0)
            .astype("float32")
            .to_numpy()
        )

        d_storm = lgb.Dataset(X, label=y_storm, free_raw_data=True)
        d_symh = lgb.Dataset(X, label=y_symh, free_raw_data=True)

        storm_model = lgb.train(
            storm_params,
            d_storm,
            num_boost_round=rounds_per_shard,
            init_model=storm_model,
            keep_training_booster=True,
        )
        symh_model = lgb.train(
            symh_params,
            d_symh,
            num_boost_round=rounds_per_shard,
            init_model=symh_model,
            keep_training_booster=True,
        )

        if flare_available:
            y_flare = (
                pd.to_numeric(df["flare_mx_next_15m"], errors="coerce")
                .fillna(0)
                .astype("int8")
                .to_numpy()
            )
            d_flare = lgb.Dataset(X, label=y_flare, free_raw_data=True)
            flare_model = lgb.train(
                flare_params,
                d_flare,
                num_boost_round=rounds_per_shard,
                init_model=flare_model,
                keep_training_booster=True,
            )

        if i % 5 == 0 or i == len(train_files):
            print(f"[lgbm] trained shard {i}/{len(train_files)}", flush=True)

    print("[lgbm] validation", flush=True)
    storm_val = _eval_classifier(
        storm_model, val_files, feature_cols, "storm_risk", max_eval_rows, seed
    )
    symh_val = _eval_regressor(
        symh_model, val_files, feature_cols, "symh_future", max_eval_rows, seed
    )
    print(
        f"Storm val log_loss={storm_val['log_loss']:.4f} acc={storm_val['accuracy']:.4f}",
        flush=True,
    )
    print(
        f"SYM/H val MAE={symh_val['mae']:.4f} RMSE={symh_val['rmse']:.4f}", flush=True
    )
    if flare_available and flare_model is not None:
        flare_val = _eval_classifier(
            flare_model,
            val_files,
            feature_cols,
            "flare_mx_next_15m",
            max_eval_rows,
            seed,
        )
        print(
            f"Flare val log_loss={flare_val['log_loss']:.4f} acc={flare_val['accuracy']:.4f}",
            flush=True,
        )

    print("[lgbm] test", flush=True)
    storm_test = _eval_classifier(
        storm_model, test_files, feature_cols, "storm_risk", max_eval_rows, seed
    )
    symh_test = _eval_regressor(
        symh_model, test_files, feature_cols, "symh_future", max_eval_rows, seed
    )
    print(
        f"Storm test log_loss={storm_test['log_loss']:.4f} acc={storm_test['accuracy']:.4f}",
        flush=True,
    )
    print(
        f"SYM/H test MAE={symh_test['mae']:.4f} RMSE={symh_test['rmse']:.4f}",
        flush=True,
    )
    if flare_available and flare_model is not None:
        flare_test = _eval_classifier(
            flare_model,
            test_files,
            feature_cols,
            "flare_mx_next_15m",
            max_eval_rows,
            seed,
        )
        print(
            f"Flare test log_loss={flare_test['log_loss']:.4f} acc={flare_test['accuracy']:.4f}",
            flush=True,
        )

    print("[lgbm] calibration", flush=True)
    storm_probs, storm_labels = _collect_probs(
        storm_model, val_files, feature_cols, "storm_risk", max_eval_rows, seed
    )
    storm_cal = None
    if len(storm_probs):
        storm_cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        storm_cal.fit(storm_probs, storm_labels)
    flare_cal = None
    if flare_available and flare_model is not None:
        flare_probs, flare_labels = _collect_probs(
            flare_model,
            val_files,
            feature_cols,
            "flare_mx_next_15m",
            max_eval_rows,
            seed + 3,
        )
        if len(flare_probs):
            flare_cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            flare_cal.fit(flare_probs, flare_labels)

    os.makedirs(model_dir, exist_ok=True)
    storm_path = os.path.join(model_dir, "storm_model.joblib")
    joblib.dump(
        {"model": storm_model, "features": feature_cols, "calibrator": storm_cal},
        storm_path,
    )
    symh_path = os.path.join(model_dir, "symh_model.joblib")
    joblib.dump({"model": symh_model, "features": feature_cols}, symh_path)
    if flare_available and flare_model is not None:
        flare_path = os.path.join(model_dir, "flare_model.joblib")
        joblib.dump(
            {"model": flare_model, "features": feature_cols, "calibrator": flare_cal},
            flare_path,
        )

    run_version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    register_model(
        name="storm_model_lgbm",
        version=run_version,
        artifact_path=storm_path,
        metrics={**storm_val, **storm_test},
        metadata={"feature_cols": feature_cols, "training": "lgbm_shards"},
    )
    register_model(
        name="symh_model_lgbm",
        version=run_version,
        artifact_path=symh_path,
        metrics={**symh_val, **symh_test},
        metadata={"feature_cols": feature_cols, "training": "lgbm_shards"},
    )
    if flare_available and flare_model is not None:
        register_model(
            name="flare_model_lgbm",
            version=run_version,
            artifact_path=flare_path,
            metrics={**flare_val, **flare_test},
            metadata={"feature_cols": feature_cols, "training": "lgbm_shards"},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LightGBM training over Parquet shards"
    )
    parser.add_argument("--parquet-dir", default="data/processed/parquet")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--rounds-per-shard", type=int, default=10)
    parser.add_argument("--max-eval-rows", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_lgbm(
        parquet_dir=args.parquet_dir,
        model_dir=args.model_dir,
        rounds_per_shard=args.rounds_per_shard,
        max_eval_rows=args.max_eval_rows,
        seed=args.seed,
    )
