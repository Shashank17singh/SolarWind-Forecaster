import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

LABEL_COLS = {"storm_risk", "symh_future", "flare_mx_next_15m"}


def _list_parquet_files(dir_path: str) -> list[Path]:
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet directory: {dir_path}")
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {dir_path}")
    return files


def _sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def _estimate_class_weights(
    train_files: list[Path],
    label_col: str,
    max_rows: int,
    seed: int,
) -> dict[int, float]:
    rows = 0
    counts = {0: 0, 1: 0}
    for path in train_files:
        df = pd.read_parquet(path, columns=[label_col])
        if max_rows:
            remaining = max_rows - rows
            df = _sample_df(df, min(remaining, len(df)), seed + rows)
        y = (
            pd.to_numeric(df[label_col], errors="coerce")
            .fillna(0)
            .astype("int8")
            .to_numpy()
        )
        counts[0] += int((y == 0).sum())
        counts[1] += int((y == 1).sum())
        rows += len(y)
        if max_rows and rows >= max_rows:
            break
    total = counts[0] + counts[1]
    if total == 0 or counts[0] == 0 or counts[1] == 0:
        return {0: 1.0, 1: 1.0}
    w0 = total / (2.0 * counts[0])
    w1 = total / (2.0 * counts[1])
    return {0: float(w0), 1: float(w1)}


def _extract_xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    np.nan_to_num(X, copy=False)
    y_storm = df["storm_risk"].astype("int8").to_numpy()
    y_symh = df["symh_future"].astype("float32").to_numpy()
    if "flare_mx_next_15m" in df.columns:
        y_flare = (
            pd.to_numeric(df["flare_mx_next_15m"], errors="coerce")
            .fillna(0)
            .astype("int8")
            .to_numpy()
        )
    else:
        y_flare = None
    return X, y_storm, y_symh, y_flare


def _fit_scaler(train_files: list[Path], feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    total = 0
    for path in train_files:
        df = pd.read_parquet(path)
        X = df[feature_cols].to_numpy(dtype=np.float32)
        np.nan_to_num(X, copy=False)
        scaler.partial_fit(X)
        total += len(df)
        print(f"[scale] {path.name} rows={len(df)} total={total}", flush=True)
    if total == 0:
        raise RuntimeError("No rows found while fitting scaler.")
    return scaler


def _eval_classifier(
    model: SGDClassifier,
    scaler: StandardScaler,
    files: list[Path],
    feature_cols: list[str],
    label_col: str,
    max_rows: int,
    seed: int,
):
    total = 0
    total_loss = 0.0
    total_acc = 0
    for i, path in enumerate(files, start=1):
        df = pd.read_parquet(path)
        if max_rows and total >= max_rows:
            break
        if max_rows:
            remaining = max_rows - total
            df = _sample_df(df, min(remaining, len(df)), seed + total)
        X = df[feature_cols].to_numpy(dtype=np.float32)
        np.nan_to_num(X, copy=False)
        y = (
            pd.to_numeric(df[label_col], errors="coerce")
            .fillna(0)
            .astype("int8")
            .to_numpy()
        )
        Xs = scaler.transform(X)
        prob = model.predict_proba(Xs)[:, 1]
        total_loss += log_loss(y, prob, labels=[0, 1]) * len(y)
        total_acc += ((prob >= 0.5).astype(np.int8) == y).sum()
        total += len(y)
        print(f"[eval] {label_col} file {i}/{len(files)} rows={len(y)}", flush=True)
    if total == 0:
        return {"log_loss": float("nan"), "accuracy": float("nan")}
    return {"log_loss": total_loss / total, "accuracy": total_acc / total}


def _eval_regressor(
    model: SGDRegressor,
    scaler: StandardScaler,
    files: list[Path],
    feature_cols: list[str],
    label_col: str,
    max_rows: int,
    seed: int,
):
    total = 0
    abs_sum = 0.0
    sq_sum = 0.0
    for i, path in enumerate(files, start=1):
        df = pd.read_parquet(path)
        if max_rows and total >= max_rows:
            break
        if max_rows:
            remaining = max_rows - total
            df = _sample_df(df, min(remaining, len(df)), seed + total)
        X = df[feature_cols].to_numpy(dtype=np.float32)
        np.nan_to_num(X, copy=False)
        y = (
            pd.to_numeric(df[label_col], errors="coerce")
            .fillna(0)
            .astype("float32")
            .to_numpy()
        )
        Xs = scaler.transform(X)
        pred = model.predict(Xs)
        diff = pred - y
        abs_sum += np.abs(diff).sum()
        sq_sum += np.square(diff).sum()
        total += len(y)
        print(f"[eval] {label_col} file {i}/{len(files)} rows={len(y)}", flush=True)
    if total == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    return {"mae": abs_sum / total, "rmse": np.sqrt(sq_sum / total)}


def train_streaming(
    parquet_dir: str,
    model_dir: str,
    epochs: int,
    max_eval_rows: int,
    seed: int,
):
    train_files = _list_parquet_files(os.path.join(parquet_dir, "train"))
    val_files = _list_parquet_files(os.path.join(parquet_dir, "val"))
    test_files = _list_parquet_files(os.path.join(parquet_dir, "test"))

    schema_cols = pq.ParquetFile(train_files[0]).schema.names
    feature_cols = [c for c in schema_cols if c not in LABEL_COLS and c != "time"]
    flare_available = "flare_mx_next_15m" in schema_cols

    print(f"[train] features={len(feature_cols)} flare={flare_available}", flush=True)

    scaler = _fit_scaler(train_files, feature_cols)

    storm = SGDClassifier(
        loss="log_loss",
        alpha=0.0001,
        learning_rate="optimal",
        random_state=seed,
    )
    symh = SGDRegressor(
        alpha=0.0001,
        learning_rate="invscaling",
        random_state=seed,
    )
    flare = None
    if flare_available:
        flare = SGDClassifier(
            loss="log_loss",
            alpha=0.0001,
            learning_rate="optimal",
            random_state=seed,
        )

    classes = np.array([0, 1], dtype=np.int8)
    storm_init = False
    flare_init = False

    print("[train] estimating class weights...", flush=True)
    storm_weights = _estimate_class_weights(
        train_files, "storm_risk", max_rows=1_000_000, seed=seed
    )
    flare_weights = None
    if flare is not None:
        flare_weights = _estimate_class_weights(
            train_files, "flare_mx_next_15m", max_rows=1_000_000, seed=seed + 7
        )
    print(f"[train] storm weights: {storm_weights}", flush=True)
    if flare_weights is not None:
        print(f"[train] flare weights: {flare_weights}", flush=True)

    for epoch in range(1, epochs + 1):
        print(f"[train] epoch {epoch}/{epochs}", flush=True)
        for i, path in enumerate(train_files, start=1):
            df = pd.read_parquet(path)
            X, y_storm, y_symh, y_flare = _extract_xy(df, feature_cols)
            Xs = scaler.transform(X)

            storm_sw = np.where(
                y_storm == 1, storm_weights[1], storm_weights[0]
            ).astype("float32")
            if not storm_init:
                storm.partial_fit(Xs, y_storm, classes=classes, sample_weight=storm_sw)
                storm_init = True
            else:
                storm.partial_fit(Xs, y_storm, sample_weight=storm_sw)

            symh.partial_fit(Xs, y_symh)

            if flare is not None and y_flare is not None and flare_weights is not None:
                flare_sw = np.where(
                    y_flare == 1, flare_weights[1], flare_weights[0]
                ).astype("float32")
                if not flare_init:
                    flare.partial_fit(
                        Xs, y_flare, classes=classes, sample_weight=flare_sw
                    )
                    flare_init = True
                else:
                    flare.partial_fit(Xs, y_flare, sample_weight=flare_sw)

            if i % 10 == 0 or i == len(train_files):
                print(
                    f"[train] epoch {epoch} file {i}/{len(train_files)} rows={len(df)}",
                    flush=True,
                )

    print("[eval] validation", flush=True)
    storm_val = _eval_classifier(
        storm, scaler, val_files, feature_cols, "storm_risk", max_eval_rows, seed
    )
    symh_val = _eval_regressor(
        symh, scaler, val_files, feature_cols, "symh_future", max_eval_rows, seed
    )
    print(
        f"Storm val log_loss={storm_val['log_loss']:.4f} acc={storm_val['accuracy']:.4f}",
        flush=True,
    )
    print(
        f"SYM/H val MAE={symh_val['mae']:.4f} RMSE={symh_val['rmse']:.4f}", flush=True
    )

    if flare is not None:
        flare_val = _eval_classifier(
            flare,
            scaler,
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

    print("[eval] test", flush=True)
    storm_test = _eval_classifier(
        storm, scaler, test_files, feature_cols, "storm_risk", max_eval_rows, seed
    )
    symh_test = _eval_regressor(
        symh, scaler, test_files, feature_cols, "symh_future", max_eval_rows, seed
    )
    print(
        f"Storm test log_loss={storm_test['log_loss']:.4f} acc={storm_test['accuracy']:.4f}",
        flush=True,
    )
    print(
        f"SYM/H test MAE={symh_test['mae']:.4f} RMSE={symh_test['rmse']:.4f}",
        flush=True,
    )

    if flare is not None:
        flare_test = _eval_classifier(
            flare,
            scaler,
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

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(
        {"model": storm, "features": feature_cols, "scaler": scaler},
        os.path.join(model_dir, "storm_model.joblib"),
    )
    joblib.dump(
        {"model": symh, "features": feature_cols, "scaler": scaler},
        os.path.join(model_dir, "symh_model.joblib"),
    )
    if flare is not None:
        joblib.dump(
            {"model": flare, "features": feature_cols, "scaler": scaler},
            os.path.join(model_dir, "flare_model.joblib"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Streaming training over Parquet shards"
    )
    parser.add_argument("--parquet-dir", default="data/processed/parquet")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-eval-rows", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_streaming(
        parquet_dir=args.parquet_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        max_eval_rows=args.max_eval_rows,
        seed=args.seed,
    )
