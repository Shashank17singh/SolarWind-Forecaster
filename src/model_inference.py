import argparse
import joblib
import numpy as np
import pandas as pd
from feature_engineering import FEATURE_COLS, add_rolling_features


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").set_index("time")
    base = df[FEATURE_COLS].copy()
    feat_15 = add_rolling_features(df, window=15, prefix="w15")
    feat_60 = add_rolling_features(df, window=60, prefix="w60")
    features = pd.concat([base, feat_15, feat_60], axis=1)
    return features


def _prepare_input(bundle: dict, latest: pd.DataFrame):
    features = bundle["features"]
    X = latest[features]
    scaler = bundle.get("scaler")
    if scaler is None:
        return X
    X = X.fillna(0.0)
    X = X.to_numpy(dtype=np.float32)
    return scaler.transform(X)


def _predict_proba(bundle, X):
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        probs = calibrator.predict(probs)
    return probs


def _predict_value(model, X):
    return model.predict(X)


def predict(latest_csv: str, model_dir: str):
    df = pd.read_csv(latest_csv, parse_dates=["time"])
    features = make_features(df)
    latest = features.iloc[-1:]

    out = {}
    storm = joblib.load(f"{model_dir}/storm_model.joblib")
    storm_X = _prepare_input(storm, latest)
    out["storm_risk_prob"] = float(_predict_proba(storm, storm_X)[0])

    symh = joblib.load(f"{model_dir}/symh_model.joblib")
    symh_X = _prepare_input(symh, latest)
    out["symh_future"] = float(_predict_value(symh["model"], symh_X)[0])
    out["dst_future"] = out["symh_future"]

    try:
        flare = joblib.load(f"{model_dir}/flare_model.joblib")
        flare_X = _prepare_input(flare, latest)
        out["flare_mx_prob"] = float(_predict_proba(flare, flare_X)[0])
    except FileNotFoundError:
        pass

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 15-min space weather prediction")
    parser.add_argument(
        "--latest-csv",
        required=True,
        help="CSV of latest solar-wind-like rows with time column",
    )
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    preds = predict(args.latest_csv, args.model_dir)
    for k, v in preds.items():
        print(f"{k}: {v}")
