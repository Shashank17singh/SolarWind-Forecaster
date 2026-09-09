import argparse
import os

import joblib
import pandas as pd

from build_drag_dataset import build_dataset


def predict(indices_dir: str, swall_path: str, model_dir: str):
    model_path = os.path.join(model_dir, "drag_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing drag model at {model_path}")

    bundle = joblib.load(model_path)
    data = build_dataset(
        os.path.join(indices_dir, "SOLFSMY.TXT"),
        os.path.join(indices_dir, "DTCFILE.TXT"),
        swall_path,
        horizon_hours=bundle.get("horizon_hours", 3),
    )
    latest = data.sort_values("time").iloc[-1:]
    X = latest[bundle["features"]]
    pred = float(bundle["model"].predict(X)[0])
    return {
        "time": latest["time"].iloc[0].isoformat(),
        "dtc_pred_3h": pred,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict 3-hour drag proxy from JB2008 indices"
    )
    parser.add_argument("--indices-dir", default="data/indices/jb2008")
    parser.add_argument("--swall-path", default="data/indices/jb2008/SW-All.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    out = predict(args.indices_dir, args.swall_path, args.model_dir)
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
