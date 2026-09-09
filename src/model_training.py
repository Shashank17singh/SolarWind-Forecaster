import argparse
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression
import pyarrow.parquet as pq

LABEL_COLS = {"storm_risk", "symh_future", "flare_mx_next_15m"}

def _list_parquet_files(dir_path: str) -> list[Path]:
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet directory: {dir_path}")
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {dir_path}")
    return files

def _load_sample_from_shards(files: list[Path], feature_cols: list[str], max_rows: int, seed: int):
    """Load a random sample of rows from parquet shards to fit in memory for scikit-learn."""
    dfs = []
    total_loaded = 0
    for path in files:
        if total_loaded >= max_rows:
            break
        df = pd.read_parquet(path)
        
        # Take a subset to avoid memory explosion if the shard is huge
        if len(df) > max_rows // len(files):
            df = df.sample(n=max_rows // len(files), random_state=seed)
            
        dfs.append(df)
        total_loaded += len(df)
    
    if not dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(dfs, ignore_index=True)
    if len(full_df) > max_rows:
        full_df = full_df.sample(n=max_rows, random_state=seed)
        
    return full_df

def _eval_classifier(model, X, y):
    if len(y) == 0:
        return {"log_loss": float("nan"), "accuracy": float("nan")}
    
    prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    loss = log_loss(y, prob, labels=[0, 1])
    acc = ((prob >= 0.5).astype(np.int8) == y).mean()
    return {"log_loss": loss, "accuracy": acc}

def _eval_regressor(model, X, y):
    if len(y) == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    return {"mae": mae, "rmse": rmse}


def train_gradient_boosting(
    parquet_dir: str,
    model_dir: str,
    max_train_rows: int,
    max_eval_rows: int,
    seed: int,
):
    train_files = _list_parquet_files(os.path.join(parquet_dir, "train"))
    val_files = _list_parquet_files(os.path.join(parquet_dir, "val"))
    test_files = _list_parquet_files(os.path.join(parquet_dir, "test"))

    schema_cols = pq.ParquetFile(train_files[0]).schema.names
    feature_cols = [c for c in schema_cols if c not in LABEL_COLS and c != "time"]
    flare_available = "flare_mx_next_15m" in schema_cols

    print(f"[sklearn] features={len(feature_cols)} flare={flare_available}", flush=True)

    print("Loading training data sample into memory...")
    train_df = _load_sample_from_shards(train_files, feature_cols, max_train_rows, seed)
    
    print("Loading validation data sample into memory...")
    val_df = _load_sample_from_shards(val_files, feature_cols, max_eval_rows, seed + 1)
    
    print("Loading test data sample into memory...")
    test_df = _load_sample_from_shards(test_files, feature_cols, max_eval_rows, seed + 2)

    X_train = train_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].fillna(0).to_numpy(dtype=np.float32)

    # SYM-H Regressor
    y_symh_train = pd.to_numeric(train_df["symh_future"], errors="coerce").fillna(0).astype("float32").to_numpy()
    y_symh_val = pd.to_numeric(val_df["symh_future"], errors="coerce").fillna(0).astype("float32").to_numpy()
    y_symh_test = pd.to_numeric(test_df["symh_future"], errors="coerce").fillna(0).astype("float32").to_numpy()

    print("[sklearn] Training SYM-H Regressor...", flush=True)
    symh_model = GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=seed
    )
    symh_model.fit(X_train, y_symh_train)

    # Storm Risk Classifier
    y_storm_train = pd.to_numeric(train_df["storm_risk"], errors="coerce").fillna(0).astype("int8").to_numpy()
    y_storm_val = pd.to_numeric(val_df["storm_risk"], errors="coerce").fillna(0).astype("int8").to_numpy()
    y_storm_test = pd.to_numeric(test_df["storm_risk"], errors="coerce").fillna(0).astype("int8").to_numpy()

    print("[sklearn] Training Storm Risk Classifier...", flush=True)
    storm_model = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=seed
    )
    storm_model.fit(X_train, y_storm_train)

    flare_model = None
    if flare_available:
        y_flare_train = pd.to_numeric(train_df["flare_mx_next_15m"], errors="coerce").fillna(0).astype("int8").to_numpy()
        y_flare_val = pd.to_numeric(val_df["flare_mx_next_15m"], errors="coerce").fillna(0).astype("int8").to_numpy()
        y_flare_test = pd.to_numeric(test_df["flare_mx_next_15m"], errors="coerce").fillna(0).astype("int8").to_numpy()

        print("[sklearn] Training Flare Risk Classifier...", flush=True)
        flare_model = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=seed
        )
        flare_model.fit(X_train, y_flare_train)

    print("[sklearn] Validation Evaluation", flush=True)
    storm_val = _eval_classifier(storm_model, X_val, y_storm_val)
    symh_val = _eval_regressor(symh_model, X_val, y_symh_val)
    
    print(f"Storm val log_loss={storm_val['log_loss']:.4f} acc={storm_val['accuracy']:.4f}", flush=True)
    print(f"SYM/H val MAE={symh_val['mae']:.4f} RMSE={symh_val['rmse']:.4f}", flush=True)

    print("[sklearn] Test Evaluation", flush=True)
    storm_test = _eval_classifier(storm_model, X_test, y_storm_test)
    symh_test = _eval_regressor(symh_model, X_test, y_symh_test)
    
    print(f"Storm test log_loss={storm_test['log_loss']:.4f} acc={storm_test['accuracy']:.4f}", flush=True)
    print(f"SYM/H test MAE={symh_test['mae']:.4f} RMSE={symh_test['rmse']:.4f}", flush=True)

    print("[sklearn] Calibration", flush=True)
    storm_probs = storm_model.predict_proba(X_val)[:, 1]
    storm_cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    storm_cal.fit(storm_probs, y_storm_val)
    
    flare_cal = None
    if flare_available:
        flare_probs = flare_model.predict_proba(X_val)[:, 1]
        flare_cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        flare_cal.fit(flare_probs, y_flare_val)

    os.makedirs(model_dir, exist_ok=True)
    storm_path = os.path.join(model_dir, "storm_model.joblib")
    joblib.dump({"model": storm_model, "features": feature_cols, "calibrator": storm_cal}, storm_path)
    
    symh_path = os.path.join(model_dir, "symh_model.joblib")
    joblib.dump({"model": symh_model, "features": feature_cols}, symh_path)
    
    if flare_available and flare_model is not None:
        flare_path = os.path.join(model_dir, "flare_model.joblib")
        joblib.dump({"model": flare_model, "features": feature_cols, "calibrator": flare_cal}, flare_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scikit-learn Gradient Boosting training")
    parser.add_argument("--parquet-dir", default="data/processed/parquet")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--max-train-rows", type=int, default=100000)
    parser.add_argument("--max-eval-rows", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_gradient_boosting(
        parquet_dir=args.parquet_dir,
        model_dir=args.model_dir,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        seed=args.seed,
    )
