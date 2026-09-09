import argparse
import os

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

from train import load_dataset, time_split


def evaluate(
    data_csv: str,
    model_dir: str,
    train_end: str,
    val_end: str,
    start_date: str | None,
    end_date: str | None,
):
    df = load_dataset(
        data_csv,
        chunksize=200000,
        sample_step=1,
        sample_fraction=1.0,
        start_date=start_date,
        end_date=end_date,
    )
    df = df.set_index("time")
    feature_cols = [
        c
        for c in df.columns
        if c not in {"storm_risk", "symh_future", "flare_mx_next_15m"}
    ]
    train, val, test = time_split(df, train_end, val_end)

    storm_path = os.path.join(model_dir, "storm_model.joblib")
    symh_path = os.path.join(model_dir, "symh_model.joblib")
    flare_path = os.path.join(model_dir, "flare_model.joblib")

    if os.path.exists(storm_path):
        storm = joblib.load(storm_path)["model"]
        val_pred = storm.predict_proba(val[feature_cols])[:, 1]
        test_pred = storm.predict_proba(test[feature_cols])[:, 1]
        print("[eval] storm val ROC-AUC", roc_auc_score(val["storm_risk"], val_pred))
        print(
            "[eval] storm val PR-AUC",
            average_precision_score(val["storm_risk"], val_pred),
        )
        print("[eval] storm test ROC-AUC", roc_auc_score(test["storm_risk"], test_pred))
        print(
            "[eval] storm test PR-AUC",
            average_precision_score(test["storm_risk"], test_pred),
        )

    if os.path.exists(symh_path):
        symh = joblib.load(symh_path)["model"]
        val_pred = symh.predict(val[feature_cols])
        test_pred = symh.predict(test[feature_cols])
        print("[eval] symh val MAE", mean_absolute_error(val["symh_future"], val_pred))
        print(
            "[eval] symh test MAE", mean_absolute_error(test["symh_future"], test_pred)
        )

    if os.path.exists(flare_path) and "flare_mx_next_15m" in df.columns:
        flare = joblib.load(flare_path)["model"]
        val_pred = flare.predict_proba(val[feature_cols])[:, 1]
        test_pred = flare.predict_proba(test[feature_cols])[:, 1]
        print(
            "[eval] flare val ROC-AUC",
            roc_auc_score(val["flare_mx_next_15m"], val_pred),
        )
        print(
            "[eval] flare test ROC-AUC",
            roc_auc_score(test["flare_mx_next_15m"], test_pred),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on dataset splits"
    )
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--val-end", default="2021-12-31")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    evaluate(
        data_csv=args.data_csv,
        model_dir=args.model_dir,
        train_end=args.train_end,
        val_end=args.val_end,
        start_date=args.start_date,
        end_date=args.end_date,
    )
