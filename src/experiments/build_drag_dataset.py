import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def _parse_solfsmy(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            line_strip = line.strip()
            if not line_strip or line_strip.startswith("#") or len(line_strip) < 8:
                continue
            parts = line_strip.split()
            year = month = day = None
            vals = None

            if len(parts) >= 11 and parts[0].isdigit() and parts[1].isdigit():
                year = int(parts[0])
                doy = int(parts[1])
                try:
                    day_dt = datetime.strptime(f"{year}-{doy:03d}", "%Y-%j")
                except ValueError:
                    continue
                month = day_dt.month
                day = day_dt.day
                try:
                    vals = [float(x) for x in parts[3:11]]
                except ValueError:
                    continue
            else:

                date_raw = line_strip[:8]
                if not date_raw.isdigit():
                    continue
                year = int(date_raw[:4])
                month = int(date_raw[4:6])
                day = int(date_raw[6:8])
                rest = line_strip[8:].strip()
                rest_parts = rest.split()
                if len(rest_parts) < 8:
                    rest_parts = [
                        line_strip[8 + i * 6 : 8 + (i + 1) * 6].strip()
                        for i in range(8)
                    ]
                try:
                    vals = [float(x) for x in rest_parts[:8]]
                except ValueError:
                    continue

            rows.append(
                {
                    "date": datetime(year, month, day),
                    "f10": vals[0],
                    "f10_81": vals[1],
                    "s10": vals[2],
                    "s10_81": vals[3],
                    "m10": vals[4],
                    "m10_81": vals[5],
                    "y10": vals[6],
                    "y10_81": vals[7],
                }
            )
    if not rows:
        raise ValueError(f"No SOLFSMY rows parsed from {path}")
    return pd.DataFrame(rows)


def _parse_dtcfile(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("DTC"):
                continue
            parts = line.split()
            if len(parts) < 27:
                continue
            year = int(parts[1])
            doy = int(parts[2])
            try:
                day = datetime.strptime(f"{year}-{doy:03d}", "%Y-%j")
            except ValueError:
                continue
            values = parts[3:27]
            if len(values) < 24:
                continue
            for hour, value in enumerate(values):
                try:
                    dtc = float(value)
                except ValueError:
                    dtc = np.nan
                rows.append({"time": day + timedelta(hours=hour), "dtc": dtc})
    if not rows:
        raise ValueError(f"No DTCFILE rows parsed from {path}")
    return pd.DataFrame(rows)


def _parse_swall(path: str) -> pd.DataFrame:
    from io import StringIO

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = []
        for line in f:
            if not line:
                continue
            if line.startswith(
                ("DATATYPE", "VERSION", "UPDATED", "NUM_", "BEGIN", "END")
            ):
                continue
            if line.startswith("DATE") or line[0].isdigit():
                lines.append(line)
    if not lines:
        raise ValueError(f"SW-All.csv appears empty: {path}")
    df = pd.read_csv(StringIO("".join(lines)))
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("date_") or cols.get("date ")
    if date_col is None:
        raise ValueError("SW-All.csv missing DATE column")
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def _kp_to_float(value):
    if pd.isna(value):
        return np.nan
    try:
        v = float(value)
    except ValueError:
        return np.nan
    if v > 9:
        return v / 10.0
    return v


def _expand_swall(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    kp_cols = [cols.get(f"kp{i}") for i in range(1, 9)]
    ap_cols = [cols.get(f"ap{i}") for i in range(1, 9)]
    if any(c is None for c in kp_cols + ap_cols):
        raise ValueError("SW-All.csv missing KP/AP 3-hour columns")

    f10_obs = cols.get("f10.7_obs") or cols.get("f10obs") or cols.get("f107_obs")
    f10_adj = cols.get("f10.7_adj") or cols.get("f10adj") or cols.get("f107_adj")
    f10_obs_avg = (
        cols.get("f10.7_obs_avg") or cols.get("f10obs_avg") or cols.get("f107_obs_avg")
    )
    f10_adj_avg = (
        cols.get("f10.7_adj_avg") or cols.get("f10adj_avg") or cols.get("f107_adj_avg")
    )

    rows = []
    for _, row in df.iterrows():
        base_date = row["date"]
        for i in range(8):
            hour = i * 3
            kp = _kp_to_float(row[kp_cols[i]])
            ap = row[ap_cols[i]]
            rows.append(
                {
                    "time": base_date + timedelta(hours=hour),
                    "kp": kp,
                    "ap": float(ap) if not pd.isna(ap) else np.nan,
                    "f10_obs": float(row[f10_obs]) if f10_obs else np.nan,
                    "f10_adj": float(row[f10_adj]) if f10_adj else np.nan,
                    "f10_obs_avg": float(row[f10_obs_avg]) if f10_obs_avg else np.nan,
                    "f10_adj_avg": float(row[f10_adj_avg]) if f10_adj_avg else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_dataset(
    solfsmy_path: str, dtc_path: str, swall_path: str, horizon_hours: int
) -> pd.DataFrame:
    if horizon_hours % 3 != 0:
        raise ValueError("horizon-hours must be a multiple of 3")
    sol = _parse_solfsmy(solfsmy_path)
    dtc = _parse_dtcfile(dtc_path)
    swall = _parse_swall(swall_path)

    sw3 = _expand_swall(swall)
    sol["date_key"] = sol["date"].dt.date
    sw3["date_key"] = sw3["time"].dt.date
    data = sw3.merge(sol.drop(columns=["date"]), on="date_key", how="left")
    data = data.merge(dtc, on="time", how="left")
    data = data.drop(columns=["date_key"]).sort_values("time")

    data["dtc"] = pd.to_numeric(data["dtc"], errors="coerce")
    data = data.dropna(subset=["kp", "ap", "dtc"])

    step = horizon_hours // 3
    data["dtc_target"] = data["dtc"].shift(-step)

    for lag in (1, 2, 3):
        data[f"kp_lag{lag}"] = data["kp"].shift(lag)
        data[f"ap_lag{lag}"] = data["ap"].shift(lag)
        data[f"dtc_lag{lag}"] = data["dtc"].shift(lag)

    data = data.dropna(subset=["dtc_target", "kp_lag1", "ap_lag1", "dtc_lag1"])
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Build 3-hour drag dataset from JB2008 indices"
    )
    parser.add_argument("--indices-dir", default="data/indices/jb2008")
    parser.add_argument("--swall-path", default="data/indices/jb2008/SW-All.csv")
    parser.add_argument("--horizon-hours", type=int, default=3)
    parser.add_argument("--out-csv", default="data/processed/drag_dataset.csv")
    args = parser.parse_args()

    solfsmy_path = os.path.join(args.indices_dir, "SOLFSMY.TXT")
    dtc_path = os.path.join(args.indices_dir, "DTCFILE.TXT")

    data = build_dataset(solfsmy_path, dtc_path, args.swall_path, args.horizon_hours)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    data.to_csv(args.out_csv, index=False)
    print(f"[drag] wrote {len(data)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
