import argparse
import json
import math
import os
import time
from datetime import timedelta
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd

from build_dataset import FEATURE_COLS
from config import CONFIG
from data_quality import read_tail_csv

MAG_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
PLASMA_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"


def _fetch_json(url: str, timeout: int | None = None, retries: int = 3):
    timeout = timeout or CONFIG.request_timeout_sec
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "space-weather-sentinel/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                payload = resp.read().decode("utf-8")
            return json.loads(payload)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (2**attempt))


def _rows_to_df(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    if isinstance(rows[0], dict):
        df = pd.DataFrame(rows)

        if "active" in df.columns:
            active = df[df["active"] == True]
            df = active if not active.empty else df
        if "time_tag" not in df.columns:
            raise ValueError("Expected time_tag column in upstream data")
        df["time"] = pd.to_datetime(
            df["time_tag"], errors="coerce", utc=True
        ).dt.tz_convert(None)
        df = df.drop(columns=["time_tag"], errors="ignore")

        rename_map = {
            "proton_speed": "speed",
            "proton_density": "density",
            "proton_temperature": "temperature",
            "proton_vx_gse": "vx_gse",
            "proton_vy_gse": "vy_gse",
            "proton_vz_gse": "vz_gse",
            "bt": "bavg",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df

    if len(rows) < 2:
        return pd.DataFrame()
    header = rows[0]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    if "time_tag" not in df.columns:
        raise ValueError("Expected time_tag column in upstream data")
    df["time"] = pd.to_datetime(
        df["time_tag"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    df = df.drop(columns=["time_tag"])
    return df


def _align_time(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        return df
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce").dt.floor("min")
    out = out.dropna(subset=["time"])
    if out.empty:
        return out
    return out.groupby("time", as_index=False).last()


def _validate_df(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("No rows after ingestion.")
    if "time" not in df.columns:
        raise RuntimeError("Missing time column in upstream data.")
    missing_time = df["time"].isna().mean()
    if missing_time > 0.2:
        raise RuntimeError("Upstream time column is mostly missing.")


def _read_last_timestamp(csv_path: str) -> pd.Timestamp | None:
    if not os.path.exists(csv_path):
        return None
    block_size = 8192
    with open(csv_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        if pos == 0:
            return None
        buffer = b""
        while pos > 0:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos)
            buffer = f.read(read_size) + buffer
            if b"\n" in buffer:
                lines = buffer.splitlines()
                for line in reversed(lines):
                    if not line or line.startswith(b"time"):
                        continue
                    try:
                        last_time = line.split(b",", 1)[0].decode(
                            "utf-8", errors="ignore"
                        )
                        return pd.to_datetime(last_time, errors="coerce")
                    except Exception:
                        return None
        return None


def _ensure_header(csv_path: str):
    header = "time," + ",".join(FEATURE_COLS) + "\n"
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    if first.startswith("time,"):
        return
    tmp_path = csv_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as out, open(
        csv_path, "r", encoding="utf-8", errors="ignore"
    ) as src:
        out.write(header)
        for line in src:
            out.write(line)
    os.replace(tmp_path, csv_path)


def _compute_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"time": df["time"]})
    for col in FEATURE_COLS:
        out[col] = np.nan

    def _col(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series([np.nan] * len(df))
        return pd.to_numeric(df[name], errors="coerce")

    bx = _col("bx_gsm")
    by = _col("by_gsm")
    bz = _col("bz_gsm")
    bavg = _col("bavg")
    if bavg.isna().all():
        bavg = _col("bt")

    speed = _col("speed")
    density = _col("density")
    if density.isna().all():
        density = _col("proton_density")
    temperature = _col("temperature")

    out["bx_gsm"] = bx
    out["by_gsm"] = by
    out["bz_gsm"] = bz

    out["bx_gse"] = bx
    out["by_gse"] = by
    out["bz_gse"] = bz

    out["flow_speed"] = speed
    out["proton_density"] = density
    out["temperature"] = temperature

    out["vx_gse"] = -speed
    out["vy_gse"] = 0.0
    out["vz_gse"] = 0.0

    out["flow_pressure"] = 1.6726e-6 * density * speed * speed

    out["electric_field"] = -speed * bz * 1e-3

    bmag = bavg
    if bmag.isna().all():
        bmag = np.sqrt(bx * bx + by * by + bz * bz)

    mu0 = 4 * math.pi * 1e-7
    k_b = 1.380649e-23
    n_m3 = density * 1e6
    p_th = n_m3 * k_b * temperature
    b_t = bmag * 1e-9
    beta = (2 * mu0 * p_th) / (b_t * b_t)
    out["plasma_beta"] = beta.replace([np.inf, -np.inf], np.nan)

    m_p = 1.6726219e-27
    rho = n_m3 * m_p
    v_a = b_t / np.sqrt(mu0 * rho)
    out["alfven_mach"] = (speed * 1000 / v_a).replace([np.inf, -np.inf], np.nan)

    return out


def update_live_omni(
    output_csv: str, min_new_rows: int = 1, backfill_minutes: int = 60
):
    mag_rows = _fetch_json(MAG_URL)
    plasma_rows = _fetch_json(PLASMA_URL)

    mag = _rows_to_df(mag_rows)
    plasma = _rows_to_df(plasma_rows)
    if mag.empty or plasma.empty:
        raise RuntimeError("No data returned from upstream endpoints.")

    df = pd.merge(mag, plasma, on="time", how="outer").sort_values("time")
    df = _align_time(df)
    _validate_df(df)
    last_ts = _read_last_timestamp(output_csv)
    if last_ts is not None:
        if backfill_minutes > 0:
            start_ts = last_ts - timedelta(minutes=backfill_minutes)
            df = df[df["time"] > start_ts]
        else:
            df = df[df["time"] > last_ts]

    if df.empty or len(df) < min_new_rows:
        print("[update] no new rows to append")
        return

    out = _compute_fields(df)
    out = out.dropna(subset=["time"])
    out = out.dropna(subset=["bz_gsm", "flow_speed", "proton_density"], how="all")
    out = out.dropna(subset=["time"])
    out = out[["time"] + FEATURE_COLS]

    if backfill_minutes > 0 and os.path.exists(output_csv):
        tail = read_tail_csv(output_csv, rows=backfill_minutes * 2)
        if not tail.empty:
            merged = pd.concat([tail, out]).drop_duplicates(
                subset=["time"], keep="last"
            )
            merged = merged.sort_values("time")
            if last_ts is not None:
                out = merged[merged["time"] > last_ts]
            else:
                out = merged

    _ensure_header(output_csv)
    out.to_csv(output_csv, mode="a", header=not os.path.exists(output_csv), index=False)
    print(f"[update] appended {len(out)} rows to {output_csv}")


def fetch_live_omni_window(minutes: int = 180) -> pd.DataFrame:
    mag_rows = _fetch_json(MAG_URL)
    plasma_rows = _fetch_json(PLASMA_URL)

    mag = _rows_to_df(mag_rows)
    plasma = _rows_to_df(plasma_rows)
    if mag.empty or plasma.empty:
        raise RuntimeError("No data returned from upstream endpoints.")

    df = pd.merge(mag, plasma, on="time", how="outer").sort_values("time")
    df = _align_time(df)
    out = _compute_fields(df)
    out = out.dropna(subset=["time"])
    out = out.dropna(subset=["bz_gsm", "flow_speed", "proton_density"], how="all")
    out = out[["time"] + FEATURE_COLS]

    if minutes and minutes > 0:
        out = out.tail(max(minutes, 120))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Append near-real-time solar wind data to omni.csv"
    )
    parser.add_argument("--output-csv", default="data/processed/omni_live.csv")
    parser.add_argument("--min-new-rows", type=int, default=1)
    parser.add_argument("--backfill-minutes", type=int, default=60)
    args = parser.parse_args()

    update_live_omni(args.output_csv, args.min_new_rows, args.backfill_minutes)
