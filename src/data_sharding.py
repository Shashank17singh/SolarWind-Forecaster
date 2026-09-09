import argparse
import os
from pathlib import Path

import pandas as pd


def _parse_ts(value: str | None, label: str) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid {label}: {value}")
    return ts


def _prepare_chunk(chunk: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    chunk = chunk[chunk["time"] != "time"].copy()
    chunk.loc[:, "time"] = pd.to_datetime(
        chunk["time"],
        format="mixed",
        errors="coerce",
        cache=True,
    )
    for col in numeric_cols:
        chunk.loc[:, col] = pd.to_numeric(chunk[col], errors="coerce")
    chunk = chunk.dropna(subset=["time"])
    return chunk


def _write_part(
    df: pd.DataFrame, out_dir: Path, part_idx: int, compression: str
) -> int:
    if df.empty:
        return part_idx
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"part-{part_idx:06d}.parquet"
    df.to_parquet(out_path, index=False, compression=compression)
    return part_idx + 1


def shard_dataset(
    data_csv: str,
    out_dir: str,
    train_end: str,
    val_end: str,
    chunksize: int,
    start_date: str | None,
    end_date: str | None,
    compression: str,
):
    train_end_ts = _parse_ts(train_end, "--train-end")
    val_end_ts = _parse_ts(val_end, "--val-end")
    start_ts = _parse_ts(start_date, "--start-date")
    end_ts = _parse_ts(end_date, "--end-date")

    header = pd.read_csv(data_csv, nrows=1)
    cols = list(header.columns)
    numeric_cols = [c for c in cols if c != "time"]

    out_root = Path(out_dir)
    train_dir = out_root / "train"
    val_dir = out_root / "val"
    test_dir = out_root / "test"

    train_idx = val_idx = test_idx = 0
    print(f"[shard] reading {data_csv} in chunks of {chunksize} rows...", flush=True)

    for i, chunk in enumerate(
        pd.read_csv(
            data_csv,
            usecols=cols,
            dtype=str,
            chunksize=chunksize,
        ),
        start=1,
    ):
        chunk = _prepare_chunk(chunk, numeric_cols)
        if start_ts is not None:
            chunk = chunk[chunk["time"] >= start_ts]
        if end_ts is not None:
            chunk = chunk[chunk["time"] <= end_ts]
        if chunk.empty:
            print(f"[shard] chunk {i}: empty after filters", flush=True)
            continue

        train_chunk = chunk[chunk["time"] <= train_end_ts]
        val_chunk = chunk[
            (chunk["time"] > train_end_ts) & (chunk["time"] <= val_end_ts)
        ]
        test_chunk = chunk[chunk["time"] > val_end_ts]

        train_idx = _write_part(train_chunk, train_dir, train_idx, compression)
        val_idx = _write_part(val_chunk, val_dir, val_idx, compression)
        test_idx = _write_part(test_chunk, test_dir, test_idx, compression)

        print(
            "[shard] chunk {i}: train={t} val={v} test={s}".format(
                i=i, t=len(train_chunk), v=len(val_chunk), s=len(test_chunk)
            ),
            flush=True,
        )

    print(
        f"[shard] done. parts: train={train_idx} val={val_idx} test={test_idx}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset CSV into Parquet shards by time"
    )
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--out-dir", default="data/processed/parquet")
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--val-end", default="2021-12-31")
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--compression", default="snappy")
    args = parser.parse_args()

    shard_dataset(
        data_csv=args.data_csv,
        out_dir=args.out_dir,
        train_end=args.train_end,
        val_end=args.val_end,
        chunksize=args.chunksize,
        start_date=args.start_date,
        end_date=args.end_date,
        compression=args.compression,
    )
