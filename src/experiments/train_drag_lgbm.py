import argparse
import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _time_split(df: pd.DataFrame, train_end: str):
    df = df.sort_values("time")
    train_end_dt = pd.to_datetime(train_end)
    train = df[df["time"] <= train_end_dt]
    val = df[df["time"] > train_end_dt]
    return train, val


def main():
    parser = argparse.ArgumentParser(
        description="Train JB2008 drag proxy regression (3-hour horizon)"
    )
    parser.add_argument("--data-csv", default="data/processed/drag_dataset.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--min-child-samples", type=int, default=50)
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv, parse_dates=["time"])
    target = "dtc_target"
    drop_cols = {"time", target}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    train_df, val_df = _time_split(df, args.train_end)
    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_val = val_df[feature_cols]
    y_val = val_df[target]

    model = LGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)

    if len(val_df) > 0:
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        print(f"[drag] val MAE={mae:.3f} RMSE={rmse:.3f}")
    else:
        print("[drag] no validation rows (adjust --train-end)")

    bundle = {
        "model": model,
        "features": feature_cols,
        "target": target,
        "horizon_hours": 3,
    }
    os.makedirs(args.model_dir, exist_ok=True)
    out_path = os.path.join(args.model_dir, "drag_model.joblib")
    joblib.dump(bundle, out_path)
    print(f"[drag] saved {out_path}")


if __name__ == "__main__":
    main()
