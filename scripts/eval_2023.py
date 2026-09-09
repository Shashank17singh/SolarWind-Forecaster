import numpy as np
import pandas as pd
import joblib
import time
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_absolute_error, mean_squared_error

DATA_CSV = "data/processed/dataset.csv"
START = "2023-01-01"
END = "2023-12-31 23:59:59"

storm_bundle = joblib.load("models/storm_model.joblib")
symh_bundle = joblib.load("models/symh_model.joblib")
try:
    flare_bundle = joblib.load("models/flare_model.joblib")
except FileNotFoundError:
    flare_bundle = None

features = storm_bundle.get("features")
if not features:
    raise SystemExit("Storm model bundle missing feature list.")

header = pd.read_csv(DATA_CSV, nrows=1)
cols = header.columns.tolist()
label_cols = ["storm_risk", "symh_future", "flare_mx_next_15m"]
needed = ["time"] + features + [c for c in label_cols if c in cols]
missing = [c for c in (features + ["storm_risk", "symh_future"]) if c not in cols]
if missing:
    raise SystemExit(f"Missing required columns in dataset.csv: {missing}")

start_ts = pd.Timestamp(START)
end_ts = pd.Timestamp(END)

storm_probs_list = []
storm_y_list = []
symh_pred_list = []
symh_y_list = []
flare_probs_list = []
flare_y_list = []

chunksize = 200000
chunk_id = 0
rows_total = 0
rows_used = 0
start_time = time.time()

for chunk in pd.read_csv(DATA_CSV, usecols=needed, chunksize=chunksize, dtype=str):
    chunk_id += 1
    rows_total += len(chunk)
    chunk = chunk[chunk["time"] != "time"].copy()
    if chunk.empty:
        continue
    chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
    chunk = chunk[chunk["time"].between(start_ts, end_ts, inclusive="both")]
    if chunk.empty:
        continue
    rows_used += len(chunk)

    for col in needed:
        if col == "time":
            continue
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    X = chunk[features].to_numpy(dtype=np.float32)

    storm_probs = storm_bundle["model"].predict(X)
    storm_cal = storm_bundle.get("calibrator")
    if storm_cal is not None:
        storm_probs = storm_cal.predict(storm_probs)
    storm_y = chunk["storm_risk"].to_numpy(dtype=np.int8)
    mask = np.isfinite(storm_probs) & np.isfinite(storm_y)
    storm_probs_list.append(storm_probs[mask])
    storm_y_list.append(storm_y[mask])

    symh_pred = symh_bundle["model"].predict(X)
    symh_y = chunk["symh_future"].to_numpy(dtype=np.float32)
    mask = np.isfinite(symh_pred) & np.isfinite(symh_y)
    symh_pred_list.append(symh_pred[mask])
    symh_y_list.append(symh_y[mask])

    if flare_bundle is not None and "flare_mx_next_15m" in chunk.columns:
        flare_probs = flare_bundle["model"].predict(X)
        flare_cal = flare_bundle.get("calibrator")
        if flare_cal is not None:
            flare_probs = flare_cal.predict(flare_probs)
        flare_y = chunk["flare_mx_next_15m"].to_numpy(dtype=np.int8)
        mask = np.isfinite(flare_probs) & np.isfinite(flare_y)
        flare_probs_list.append(flare_probs[mask])
        flare_y_list.append(flare_y[mask])

    if chunk_id % 5 == 0:
        elapsed = time.time() - start_time
        print(f"[eval] chunks={chunk_id} rows_total={rows_total} rows_2023={rows_used} elapsed={elapsed:.1f}s", flush=True)

if not storm_probs_list:
    raise SystemExit("No 2023 rows found in dataset.csv")

storm_probs = np.concatenate(storm_probs_list)
storm_y = np.concatenate(storm_y_list)
symh_pred = np.concatenate(symh_pred_list)
symh_y = np.concatenate(symh_y_list)

print("2023 rows:", len(storm_y))
print("Storm positives:", int(storm_y.sum()), "(%.4f%%)" % (100.0 * storm_y.mean()))
print("Storm accuracy:", accuracy_score(storm_y, (storm_probs >= 0.5).astype(np.int8)))
if len(np.unique(storm_y)) > 1:
    print("Storm ROC-AUC:", roc_auc_score(storm_y, storm_probs))
    print("Storm PR-AUC:", average_precision_score(storm_y, storm_probs))
else:
    print("Storm ROC-AUC: n/a (only one class present)")
    print("Storm PR-AUC: n/a (only one class present)")

print("SYM/H MAE:", mean_absolute_error(symh_y, symh_pred))
print("SYM/H RMSE:", np.sqrt(mean_squared_error(symh_y, symh_pred)))
if len(symh_y) > 1:
    print("SYM/H Corr:", np.corrcoef(symh_y, symh_pred)[0, 1])
else:
    print("SYM/H Corr: n/a")

if flare_probs_list:
    flare_probs = np.concatenate(flare_probs_list)
    flare_y = np.concatenate(flare_y_list)
    print("Flare positives:", int(flare_y.sum()), "(%.6f%%)" % (100.0 * flare_y.mean()))
    print("Flare accuracy:", accuracy_score(flare_y, (flare_probs >= 0.5).astype(np.int8)))
    if len(np.unique(flare_y)) > 1:
        print("Flare ROC-AUC:", roc_auc_score(flare_y, flare_probs))
        print("Flare PR-AUC:", average_precision_score(flare_y, flare_probs))
    else:
        print("Flare ROC-AUC: n/a (only one class present)")
        print("Flare PR-AUC: n/a (only one class present)")
else:
    print("Flare metrics: skipped (no flare labels in dataset)")
