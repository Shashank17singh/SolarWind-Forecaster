import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf

from train_dst_lstm_attention import (
    FEATURE_COLS,
    _interpolate_missing,
    _load_dst_csv,
    _load_hourly_omni,
)


@dataclass
class Window:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


DEFAULT_WINDOWS = [
    Window(
        "2023-02-14_to_2023-05-05",
        pd.Timestamp("2023-02-14"),
        pd.Timestamp("2023-05-05 23:00:00"),
    ),
    Window(
        "2024-04-24_to_2024-05-31",
        pd.Timestamp("2024-04-24"),
        pd.Timestamp("2024-05-31 23:00:00"),
    ),
]


def _standardize(df: pd.DataFrame, mean: dict, std: dict) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURE_COLS:
        out[col] = (out[col] - mean[col]) / (std[col] if std[col] else 1.0)
    return out


def _predict_series(
    df: pd.DataFrame,
    mean: dict,
    std: dict,
    seq_len: int,
    horizon_hours: int,
    model: tf.keras.Model,
):
    df = _standardize(df, mean, std)

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["dst_target"].to_numpy(dtype=np.float32)
    target_offset = seq_len - 1 + horizon_hours
    if len(y) <= target_offset:
        raise ValueError("Not enough rows to create sequences for evaluation.")

    targets = y[target_offset:]
    end_index = len(X) - target_offset - 1
    times = df.index[target_offset:]

    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=X,
        targets=targets,
        sequence_length=seq_len,
        sequence_stride=1,
        sampling_rate=1,
        start_index=0,
        end_index=end_index,
        batch_size=256,
        shuffle=False,
    )
    preds = model.predict(ds, verbose=0).reshape(-1)
    return times[: len(preds)], targets[: len(preds)], preds


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _load_base_dataframe(
    omni_csv: str,
    dst_csv: str,
    interp: bool,
    fill_missing: bool,
    mean: dict | None,
) -> pd.DataFrame:
    df = _load_hourly_omni(omni_csv)
    dst_df = _load_dst_csv(dst_csv)
    df = df.join(dst_df[["dst_kyoto"]], how="left")
    if interp:
        df = _interpolate_missing(df, limit=24)
    elif not fill_missing:
        df = df.dropna()
    if fill_missing and mean is not None:
        for col in FEATURE_COLS:
            df[col] = df[col].fillna(mean.get(col, 0.0))
    return df


def _plot_correlation(
    df: pd.DataFrame,
    out_path: str,
    sample_rows: int | None,
    annotate: bool,
):
    import matplotlib.pyplot as plt

    data = df[FEATURE_COLS].copy()
    data = data.dropna()
    if sample_rows and len(data) > sample_rows:
        data = data.sample(n=sample_rows, random_state=42)

    corr = data.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.4 * n + 6, 0.4 * n + 6))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Correlation", rotation=90)

    if annotate:
        for i in range(n):
            for j in range(n):
                value = corr.iat[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black",
                )

    ax.set_title("Correlation Between Features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _load_baseline(baseline_csv: str, baseline_col: str) -> pd.DataFrame:
    base = pd.read_csv(baseline_csv, parse_dates=["time"])
    if baseline_col not in base.columns:
        raise ValueError(f"Baseline CSV missing column: {baseline_col}")
    base = base[["time", baseline_col]].copy()
    base[baseline_col] = pd.to_numeric(base[baseline_col], errors="coerce")
    base = base.dropna(subset=["time", baseline_col])
    base = base.set_index("time").sort_index()
    return base


def _plot_window(
    window: Window,
    plot_df: pd.DataFrame,
    out_path: str,
    two_panel: bool,
    title_suffix: str | None,
):
    import matplotlib.pyplot as plt

    rmse = _rmse(plot_df["dst_true"].to_numpy(), plot_df["dst_pred"].to_numpy())
    corr = _corr(plot_df["dst_true"].to_numpy(), plot_df["dst_pred"].to_numpy())

    if two_panel:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax = axes[0]
        ax.plot(plot_df.index, plot_df["dst_true"], label="Dst (true)", color="#f59e0b")
        ax.plot(
            plot_df.index,
            plot_df["dst_pred"],
            label="Model prediction",
            color="#2563eb",
        )
        if "dst_baseline" in plot_df.columns:
            ax.plot(
                plot_df.index,
                plot_df["dst_baseline"],
                label="Baseline",
                color="#10b981",
            )
        ax.set_ylabel("nT")
        ax.legend(loc="lower left")
        title = f"Dst Prediction: {window.name}"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        ax.set_title(f"{title} | RMSE={rmse:.3f} Corr={corr:.3f}")

        ax2 = axes[1]
        ax2.plot(
            plot_df.index,
            plot_df["dst_pred"],
            label="Model prediction",
            color="#2563eb",
        )
        ax2.set_ylabel("nT")
        ax2.set_xlabel("Time")
        ax2.legend(loc="lower left")
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(plot_df.index, plot_df["dst_true"], label="Dst (true)", color="#f59e0b")
        ax.plot(
            plot_df.index,
            plot_df["dst_pred"],
            label="Model prediction",
            color="#2563eb",
        )
        if "dst_baseline" in plot_df.columns:
            ax.plot(
                plot_df.index,
                plot_df["dst_baseline"],
                label="Baseline",
                color="#10b981",
            )
        ax.set_ylabel("nT")
        ax.set_xlabel("Time")
        title = f"Dst Prediction: {window.name}"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        ax.set_title(f"{title} | RMSE={rmse:.3f} Corr={corr:.3f}")
        ax.legend(loc="lower left")
        fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate correlation + evaluation plots for Dst model"
    )
    parser.add_argument("--omni-csv", default="data/processed/omni.csv")
    parser.add_argument("--dst-csv", default="data/indices/kyoto/dst_hourly.csv")
    parser.add_argument("--model-path", default="models/dst_lstm_attention.keras")
    parser.add_argument("--meta-path", default="models/dst_lstm_attention_meta.json")
    parser.add_argument("--out-dir", default="models/figures")
    parser.add_argument(
        "--no-interp",
        action="store_true",
        help="Skip interpolation and drop rows with missing values",
    )
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Fill missing feature values with training mean instead of dropping rows",
    )

    parser.add_argument(
        "--no-corr", action="store_true", help="Skip correlation heatmap"
    )
    parser.add_argument(
        "--corr-sample",
        type=int,
        default=200000,
        help="Row sample size for correlation",
    )
    parser.add_argument(
        "--corr-annot",
        action="store_true",
        help="Annotate correlation values in heatmap",
    )

    parser.add_argument(
        "--no-windows", action="store_true", help="Skip window evaluation plots"
    )
    parser.add_argument(
        "--start", help="Custom window start (YYYY-MM-DD or full timestamp)"
    )
    parser.add_argument(
        "--end", help="Custom window end (YYYY-MM-DD or full timestamp)"
    )
    parser.add_argument(
        "--name", help="Custom window name (default derived from start/end)"
    )
    parser.add_argument(
        "--two-panel", action="store_true", help="Use two-panel plot style (default)"
    )
    parser.add_argument(
        "--single-panel", action="store_true", help="Use single-panel plot style"
    )

    parser.add_argument(
        "--baseline-csv",
        help="Optional CSV with baseline predictions (columns: time, baseline)",
    )
    parser.add_argument(
        "--baseline-col",
        default="baseline",
        help="Baseline column name in baseline CSV",
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    interp = not args.no_interp
    do_corr = not args.no_corr
    do_windows = not args.no_windows

    custom_window = None
    if args.start or args.end:
        if not (args.start and args.end):
            raise SystemExit(
                "--start and --end must be provided together for custom window"
            )
        name = args.name or f"{args.start}_to_{args.end}"
        custom_window = Window(name, pd.Timestamp(args.start), pd.Timestamp(args.end))

    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    seq_len = int(meta["seq_len"])
    horizon_hours = int(meta["horizon_hours"])
    mean = meta["mean"]
    std = meta["std"]

    df = _load_base_dataframe(
        args.omni_csv, args.dst_csv, interp, args.fill_missing, mean
    )
    df["dst_target"] = df["dst_kyoto"].shift(-horizon_hours)
    df = df.dropna(subset=["dst_target"])

    if do_corr:
        corr_path = os.path.join(args.out_dir, "corr_features.png")
        _plot_correlation(df, corr_path, args.corr_sample, args.corr_annot)
        print(f"[plot] wrote correlation heatmap -> {corr_path}", flush=True)

    if do_windows:
        model = tf.keras.models.load_model(args.model_path)
        baseline_df = None
        if args.baseline_csv:
            baseline_df = _load_baseline(args.baseline_csv, args.baseline_col)

        windows = [custom_window] if custom_window else DEFAULT_WINDOWS
        two_panel = True
        if args.single_panel:
            two_panel = False
        elif args.two_panel:
            two_panel = True

        for window in windows:
            sub = df[(df.index >= window.start) & (df.index <= window.end)].copy()
            if sub.empty:
                print(f"[plot] window {window.name}: no data")
                continue

            times, truth, preds = _predict_series(
                sub,
                mean,
                std,
                seq_len,
                horizon_hours,
                model,
            )

            plot_df = pd.DataFrame(
                {
                    "dst_true": truth,
                    "dst_pred": preds,
                },
                index=pd.to_datetime(times),
            )

            if baseline_df is not None:
                plot_df = plot_df.join(
                    baseline_df.rename(columns={args.baseline_col: "dst_baseline"}),
                    how="left",
                )

            plot_df = plot_df.dropna(subset=["dst_true", "dst_pred"])
            out_csv = os.path.join(args.out_dir, f"dst_plot_{window.name}.csv")
            plot_df.to_csv(out_csv, index_label="time")

            suffix = "two_panel" if two_panel else "single_panel"
            out_png = os.path.join(args.out_dir, f"dst_plot_{window.name}_{suffix}.png")
            _plot_window(
                window, plot_df, out_png, two_panel=two_panel, title_suffix=None
            )
            print(f"[plot] wrote window plot -> {out_png}", flush=True)


if __name__ == "__main__":
    main()
