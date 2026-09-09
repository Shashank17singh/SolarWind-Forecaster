import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _parse_ts(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value}")
    return ts


def _split_by_time(df: pd.DataFrame, train_end: str, val_end: str):
    train_end = _parse_ts(train_end)
    val_end = _parse_ts(val_end)
    train = df[df.index <= train_end]
    val = df[(df.index > train_end) & (df.index <= val_end)]
    test = df[df.index > val_end]
    return train, val, test


def _estimate_pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        return 1.0
    return max(1.0, neg / pos)


def train(
    data_csv: str,
    model_dir: str,
    label_col: str,
    train_end: str,
    val_end: str,
):
    df = pd.read_csv(data_csv, parse_dates=["time"])
    df = df.set_index("time").sort_index()

    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    feature_cols = [c for c in df.columns if c != label_col]
    train_df, val_df, test_df = _split_by_time(df, train_end, val_end)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[label_col].to_numpy(dtype=np.int8)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df[label_col].to_numpy(dtype=np.int8)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df[label_col].to_numpy(dtype=np.int8)

    def _count(y):
        return int((y == 1).sum()), int((y == 0).sum())

    train_pos, train_neg = _count(y_train)
    val_pos, val_neg = _count(y_val)
    test_pos, test_neg = _count(y_test)
    print(f"[impact] train pos={train_pos} neg={train_neg}", flush=True)
    print(f"[impact] val pos={val_pos} neg={val_neg}", flush=True)
    print(f"[impact] test pos={test_pos} neg={test_neg}", flush=True)
    if val_pos == 0 or val_neg == 0:
        print(
            "[impact] WARNING: validation set has a single class; ROC/PR metrics will be None.",
            flush=True,
        )
    if test_pos == 0 or test_neg == 0:
        print(
            "[impact] WARNING: test set has a single class; ROC/PR metrics will be None.",
            flush=True,
        )

    pos_weight = _estimate_pos_weight(y_train)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "scale_pos_weight": pos_weight,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    dval = (
        lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)
        if len(val_df)
        else None
    )

    callbacks = [lgb.log_evaluation(period=50)]
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dval] if dval is not None else None,
        callbacks=callbacks,
    )

    metrics = {}
    if len(val_df):
        val_probs = model.predict(X_val)
        metrics["val_roc_auc"] = (
            roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else None
        )
        metrics["val_pr_auc"] = (
            average_precision_score(y_val, val_probs)
            if len(np.unique(y_val)) > 1
            else None
        )

    if len(test_df):
        test_probs = model.predict(X_test)
        metrics["test_roc_auc"] = (
            roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else None
        )
        metrics["test_pr_auc"] = (
            average_precision_score(y_test, test_probs)
            if len(np.unique(y_test)) > 1
            else None
        )

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sat_impact_model.txt")
    model.save_model(model_path)

    meta = {
        "label_col": label_col,
        "feature_cols": feature_cols,
        "train_end": train_end,
        "val_end": val_end,
        "pos_weight": pos_weight,
        "class_counts": {
            "train": {"pos": train_pos, "neg": train_neg},
            "val": {"pos": val_pos, "neg": val_neg},
            "test": {"pos": test_pos, "neg": test_neg},
        },
        "metrics": metrics,
    }
    meta_path = os.path.join(model_dir, "sat_impact_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[impact] saved model -> {model_path}")
    print(f"[impact] saved meta -> {meta_path}")
    print(f"[impact] metrics -> {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train satellite impact classifier from anomaly labels"
    )
    parser.add_argument("--data-csv", default="data/processed/sat_impact_dataset.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--label-col", default="sat_impact_next_6h")
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--val-end", default="2021-12-31")
    args = parser.parse_args()

    train(
        data_csv=args.data_csv,
        model_dir=args.model_dir,
        label_col=args.label_col,
        train_end=args.train_end,
        val_end=args.val_end,
    )
