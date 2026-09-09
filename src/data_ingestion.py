import argparse
import os
import numpy as np
import pandas as pd

WIDTHS = [
    4,
    4,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    7,
    7,
    6,
    7,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    7,
    9,
    6,
    7,
    7,
    6,
    8,
    8,
    8,
    8,
    8,
    8,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    7,
    5,
]

COLUMNS = [
    "year",
    "doy",
    "hour",
    "minute",
    "imf_sc_id",
    "sw_sc_id",
    "imf_npts",
    "sw_npts",
    "pct_interp",
    "timeshift_sec",
    "rms_timeshift_sec",
    "rms_phase_front_norm",
    "dbot_sec",
    "b_mag",
    "bx_gse",
    "by_gse",
    "bz_gse",
    "by_gsm",
    "bz_gsm",
    "rms_sd_b",
    "rms_sd_bvec",
    "flow_speed",
    "vx_gse",
    "vy_gse",
    "vz_gse",
    "proton_density",
    "temperature",
    "flow_pressure",
    "electric_field",
    "plasma_beta",
    "alfven_mach",
    "sc_x_gse",
    "sc_y_gse",
    "sc_z_gse",
    "bsn_x",
    "bsn_y",
    "bsn_z",
    "ae",
    "al",
    "au",
    "sym_d",
    "sym_h",
    "asy_d",
    "asy_h",
    "pcn",
    "magnetosonic_mach",
]

MISSING_VALUES = {
    99999.9,
    9999.99,
    9999.9,
    999.99,
    999.9,
    999.99,
    999.0,
    9999.0,
    99999.0,
    9999999.0,
    9999999,
    99999,
    9999,
    999,
    -99999.9,
    -9999.99,
    -9999.9,
    -999.99,
    -999.9,
    -99999.0,
    -9999.0,
    -999.0,
}

FLOAT_COLS = {
    "rms_phase_front_norm",
    "b_mag",
    "bx_gse",
    "by_gse",
    "bz_gse",
    "by_gsm",
    "bz_gsm",
    "rms_sd_b",
    "rms_sd_bvec",
    "flow_speed",
    "vx_gse",
    "vy_gse",
    "vz_gse",
    "proton_density",
    "temperature",
    "flow_pressure",
    "electric_field",
    "plasma_beta",
    "alfven_mach",
    "sc_x_gse",
    "sc_y_gse",
    "sc_z_gse",
    "bsn_x",
    "bsn_y",
    "bsn_z",
    "pcn",
    "magnetosonic_mach",
}


def to_datetime(df: pd.DataFrame) -> pd.Series:
    year = df["year"].astype(int)
    doy = df["doy"].astype(int)
    hour = df["hour"].astype(int)
    minute = df["minute"].astype(int)
    base = pd.to_datetime(year.astype(str), format="%Y")
    dt = base + pd.to_timedelta(doy - 1, unit="D")
    dt = dt + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    return dt


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in {"year", "doy", "hour", "minute", "imf_sc_id", "sw_sc_id"}:
            continue
        df[col] = df[col].replace(list(MISSING_VALUES), np.nan)
    return df


def parse_omni_files(input_dir: str, output_csv: str, chunksize: int = 500000) -> None:
    files = sorted(
        f
        for f in os.listdir(input_dir)
        if f.startswith("omni_min") and f.endswith(".asc")
    )
    if not files:
        raise SystemExit("No solar wind files found in input directory")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    wrote_header = False
    total_rows = 0

    for fname in files:
        fpath = os.path.join(input_dir, fname)
        print(f"[parse] reading {fname}...", flush=True)
        for chunk in pd.read_fwf(
            fpath,
            widths=WIDTHS,
            names=COLUMNS,
            chunksize=chunksize,
            header=None,
        ):
            chunk = clean_missing(chunk)
            chunk.insert(0, "time", to_datetime(chunk))
            total_rows += len(chunk)
            print(
                f"[parse] {fname} chunk rows={len(chunk)} total={total_rows}",
                flush=True,
            )
            if not wrote_header:
                chunk.to_csv(output_csv, index=False, mode="w")
                wrote_header = True
            else:
                chunk.to_csv(output_csv, index=False, mode="a", header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse 1-min solar wind data to CSV")
    parser.add_argument(
        "--input-dir", required=True, help="Directory with omni_minYYYY.asc files"
    )
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument("--chunksize", type=int, default=500000)
    args = parser.parse_args()

    parse_omni_files(args.input_dir, args.output_csv, args.chunksize)
