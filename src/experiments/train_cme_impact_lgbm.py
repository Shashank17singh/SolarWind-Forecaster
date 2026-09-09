import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error


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


def _encode_categories(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].astype("category").cat.codes.replace({-1: np.nan})
    return out


def train(data_csv: str, model_dir: str, train_end: str, val_end: str):
    df = pd.read_csv(data_csv, parse_dates=["time"])
    df = df.set_index("time").sort_index()

    if "earth_impact" not in df.columns:
        raise ValueError("earth_impact label not found in dataset")

    cat_cols = [c for c in ("catalog", "cme_type") if c in df.columns]
    df = _encode_categories(df, cat_cols)

    raw_feature_cols = [
        c
        for c in df.columns
        if c not in {"earth_impact", "transit_hours", "label_source"}
    ]
    df[raw_feature_cols] = df[raw_feature_cols].apply(pd.to_numeric, errors="coerce")

    missing_frac = df[raw_feature_cols].isna().mean()
    feature_cols = [c for c in raw_feature_cols if missing_frac.get(c, 0) <= 0.5]

    train_df, val_df, test_df = _split_by_time(df, train_end, val_end)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["earth_impact"].to_numpy(dtype=np.int8)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["earth_impact"].to_numpy(dtype=np.int8)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["earth_impact"].to_numpy(dtype=np.int8)

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

    reg_metrics = {}
    reg_model = None
    pos_df = train_df[train_df["earth_impact"] == 1].dropna(subset=["transit_hours"])
    if len(pos_df):
        X_reg = pos_df[feature_cols].to_numpy(dtype=np.float32)
        y_reg = pos_df["transit_hours"].to_numpy(dtype=np.float32)
        reg_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
        }
        dreg = lgb.Dataset(X_reg, label=y_reg, free_raw_data=True)
        reg_model = lgb.train(reg_params, dreg, num_boost_round=300)

        if len(val_df):
            val_pos = val_df[val_df["earth_impact"] == 1].dropna(
                subset=["transit_hours"]
            )
            if len(val_pos):
                pred = reg_model.predict(
                    val_pos[feature_cols].to_numpy(dtype=np.float32)
                )
                reg_metrics["val_mae_hours"] = mean_absolute_error(
                    val_pos["transit_hours"], pred
                )
        if len(test_df):
            test_pos = test_df[test_df["earth_impact"] == 1].dropna(
                subset=["transit_hours"]
            )
            if len(test_pos):
                pred = reg_model.predict(
                    test_pos[feature_cols].to_numpy(dtype=np.float32)
                )
                reg_metrics["test_mae_hours"] = mean_absolute_error(
                    test_pos["transit_hours"], pred
                )

    os.makedirs(model_dir, exist_ok=True)
    clf_path = os.path.join(model_dir, "cme_impact_model.txt")
    model.save_model(clf_path)

    reg_path = None
    if reg_model is not None:
        reg_path = os.path.join(model_dir, "cme_transit_model.txt")
        reg_model.save_model(reg_path)

    meta = {
        "feature_cols": feature_cols,
        "train_end": train_end,
        "val_end": val_end,
        "pos_weight": pos_weight,
        "metrics": metrics,
        "regression_metrics": reg_metrics,
    }
    meta_path = os.path.join(model_dir, "cme_impact_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[cme] saved classifier -> {clf_path}")
    if reg_path:
        print(f"[cme] saved regressor -> {reg_path}")
    print(f"[cme] saved meta -> {meta_path}")
    print(f"[cme] metrics -> {metrics}")
    if reg_metrics:
        print(f"[cme] transit metrics -> {reg_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CME impact + transit time models"
    )
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--val-end", required=True)
    args = parser.parse_args()

    train(args.data_csv, args.model_dir, args.train_end, args.val_end)
