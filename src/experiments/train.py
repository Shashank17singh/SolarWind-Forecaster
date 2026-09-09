import argparse
import os
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from model_registry import register_model


def time_split(df: pd.DataFrame, train_end: str, val_end: str):
    train_end_ts = pd.to_datetime(train_end, errors="coerce")
    val_end_ts = pd.to_datetime(val_end, errors="coerce")
    if pd.isna(train_end_ts):
        raise ValueError(f"Invalid --train-end: {train_end}")
    if pd.isna(val_end_ts):
        raise ValueError(f"Invalid --val-end: {val_end}")

    train = df.loc[df.index <= train_end_ts]
    val = df.loc[(df.index > train_end_ts) & (df.index <= val_end_ts)]
    test = df.loc[df.index > val_end_ts]
    return train, val, test


def load_dataset(
    data_csv: str,
    chunksize: int = 200000,
    sample_step: int = 1,
    sample_fraction: float = 1.0,
    start_date: str | None = None,
    end_date: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_date, errors="coerce") if start_date else None
    end_ts = pd.to_datetime(end_date, errors="coerce") if end_date else None
    if start_date and pd.isna(start_ts):
        raise ValueError(f"Invalid --start-date: {start_date}")
    if end_date and pd.isna(end_ts):
        raise ValueError(f"Invalid --end-date: {end_date}")
    header = pd.read_csv(data_csv, nrows=1)
    cols = list(header.columns)
    numeric_cols = [c for c in cols if c != "time"]

    print(f"[train] loading dataset in chunks of {chunksize}...", flush=True)
    chunks = []
    row_offset = 0
    for i, chunk in enumerate(
        pd.read_csv(
            data_csv,
            usecols=cols,
            dtype=str,
            chunksize=chunksize,
        ),
        start=1,
    ):

        chunk = chunk[chunk["time"] != "time"].copy()

        chunk.loc[:, "time"] = pd.to_datetime(
            chunk["time"],
            format="mixed",
            errors="coerce",
            cache=True,
        )
        for col in numeric_cols:
            chunk.loc[:, col] = pd.to_numeric(chunk[col], errors="coerce")

        if start_ts is not None:
            chunk = chunk[chunk["time"] >= start_ts]
        if end_ts is not None:
            chunk = chunk[chunk["time"] <= end_ts]

        if sample_step > 1:
            idx = np.arange(len(chunk)) + row_offset
            chunk = chunk[(idx % sample_step) == 0]

        if 0 < sample_fraction < 1.0 and not chunk.empty:
            chunk = chunk.sample(frac=sample_fraction, random_state=seed)

        row_offset += len(chunk)
        if not chunk.empty:
            chunks.append(chunk)

        print(f"[train] read chunk {i} rows={len(chunk)}", flush=True)

    if not chunks:
        raise RuntimeError("No data loaded. Check date filters or sampling.")

    df = pd.concat(chunks, ignore_index=True).sort_values("time")
    return df


def train_models(
    data_csv: str,
    model_dir: str,
    train_end: str,
    val_end: str,
    chunksize: int,
    sample_step: int,
    sample_fraction: float,
    start_date: str | None,
    end_date: str | None,
):
    df = load_dataset(
        data_csv,
        chunksize=chunksize,
        sample_step=sample_step,
        sample_fraction=sample_fraction,
        start_date=start_date,
        end_date=end_date,
    )
    df = df.set_index("time")

    print(
        f"[train] rows={len(df)} cols={len(df.columns)} range={df.index.min()} -> {df.index.max()}",
        flush=True,
    )

    y_storm = df["storm_risk"].astype("int8")
    y_symh = df["symh_future"].astype("float32")
    flare_available = "flare_mx_next_15m" in df.columns
    y_flare = df["flare_mx_next_15m"] if flare_available else None

    feature_cols = [
        c
        for c in df.columns
        if c not in {"storm_risk", "symh_future", "flare_mx_next_15m"}
    ]
    X = df[feature_cols]

    train, val, test = time_split(df, train_end, val_end)
    print(
        f"[train] split sizes: train={len(train)} val={len(val)} test={len(test)}",
        flush=True,
    )

    X_train = train[feature_cols]
    X_val = val[feature_cols]
    X_test = test[feature_cols]

    os.makedirs(model_dir, exist_ok=True)

    run_version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    base_meta = {
        "train_end": train_end,
        "val_end": val_end,
        "feature_cols": feature_cols,
        "rows": int(len(df)),
    }

    print("[train] fitting storm classifier...", flush=True)
    t0 = time.time()
    clf = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=200, verbose=1
    )
    clf.fit(X_train, train["storm_risk"])
    print(f"[train] storm classifier fit in {time.time() - t0:.1f}s", flush=True)
    val_pred = clf.predict_proba(X_val)[:, 1]
    test_pred = clf.predict_proba(X_test)[:, 1]

    print("Storm risk metrics", flush=True)
    print("  val ROC-AUC:", roc_auc_score(val["storm_risk"], val_pred), flush=True)
    print(
        "  val PR-AUC:",
        average_precision_score(val["storm_risk"], val_pred),
        flush=True,
    )
    print("  test ROC-AUC:", roc_auc_score(test["storm_risk"], test_pred), flush=True)
    print(
        "  test PR-AUC:",
        average_precision_score(test["storm_risk"], test_pred),
        flush=True,
    )

    storm_path = os.path.join(model_dir, "storm_model.joblib")
    joblib.dump({"model": clf, "features": feature_cols}, storm_path)
    register_model(
        name="storm_model",
        version=run_version,
        artifact_path=storm_path,
        metrics={
            "val_roc_auc": float(roc_auc_score(val["storm_risk"], val_pred)),
            "val_pr_auc": float(average_precision_score(val["storm_risk"], val_pred)),
            "test_roc_auc": float(roc_auc_score(test["storm_risk"], test_pred)),
            "test_pr_auc": float(
                average_precision_score(test["storm_risk"], test_pred)
            ),
        },
        metadata=base_meta,
    )

    print("[train] fitting SYM/H regressor...", flush=True)
    t0 = time.time()
    reg = HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.05, max_iter=200, verbose=1
    )
    reg.fit(X_train, train["symh_future"])
    print(f"[train] SYM/H regressor fit in {time.time() - t0:.1f}s", flush=True)
    val_pred_r = reg.predict(X_val)
    test_pred_r = reg.predict(X_test)

    print("SYM/H metrics", flush=True)
    print("  val MAE:", mean_absolute_error(val["symh_future"], val_pred_r), flush=True)
    print(
        "  test MAE:", mean_absolute_error(test["symh_future"], test_pred_r), flush=True
    )

    symh_path = os.path.join(model_dir, "symh_model.joblib")
    joblib.dump({"model": reg, "features": feature_cols}, symh_path)
    register_model(
        name="symh_model",
        version=run_version,
        artifact_path=symh_path,
        metrics={
            "val_mae": float(mean_absolute_error(val["symh_future"], val_pred_r)),
            "test_mae": float(mean_absolute_error(test["symh_future"], test_pred_r)),
        },
        metadata=base_meta,
    )

    if flare_available:
        print("[train] fitting flare classifier...", flush=True)
        t0 = time.time()
        clf_f = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=200, verbose=1
        )
        clf_f.fit(X_train, train["flare_mx_next_15m"].astype("int8"))
        print(f"[train] flare classifier fit in {time.time() - t0:.1f}s", flush=True)
        val_pred_f = clf_f.predict_proba(X_val)[:, 1]
        test_pred_f = clf_f.predict_proba(X_test)[:, 1]

        print("Flare M/X metrics", flush=True)
        print(
            "  val ROC-AUC:",
            roc_auc_score(val["flare_mx_next_15m"], val_pred_f),
            flush=True,
        )
        print(
            "  val PR-AUC:",
            average_precision_score(val["flare_mx_next_15m"], val_pred_f),
            flush=True,
        )
        print(
            "  test ROC-AUC:",
            roc_auc_score(test["flare_mx_next_15m"], test_pred_f),
            flush=True,
        )
        print(
            "  test PR-AUC:",
            average_precision_score(test["flare_mx_next_15m"], test_pred_f),
            flush=True,
        )

        flare_path = os.path.join(model_dir, "flare_model.joblib")
        joblib.dump({"model": clf_f, "features": feature_cols}, flare_path)
        register_model(
            name="flare_model",
            version=run_version,
            artifact_path=flare_path,
            metrics={
                "val_roc_auc": float(
                    roc_auc_score(val["flare_mx_next_15m"], val_pred_f)
                ),
                "val_pr_auc": float(
                    average_precision_score(val["flare_mx_next_15m"], val_pred_f)
                ),
                "test_roc_auc": float(
                    roc_auc_score(test["flare_mx_next_15m"], test_pred_f)
                ),
                "test_pr_auc": float(
                    average_precision_score(test["flare_mx_next_15m"], test_pred_f)
                ),
            },
            metadata=base_meta,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train space weather models")
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--val-end", default="2021-12-31")
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    train_models(
        args.data_csv,
        args.model_dir,
        args.train_end,
        args.val_end,
        args.chunksize,
        args.sample_step,
        args.sample_fraction,
        args.start_date,
        args.end_date,
    )
