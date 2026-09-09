import argparse
import json
import os

from data_quality import build_baseline
from config import CONFIG


def _load_feature_cols(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature spec not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    block = spec.get("dst_realtime_features", {})
    return block.get("feature_cols", [])


def main():
    parser = argparse.ArgumentParser(
        description="Compute feature baseline statistics for drift detection"
    )
    parser.add_argument("--omni-csv", required=True)
    parser.add_argument("--output", default=CONFIG.drift_baseline_path)
    parser.add_argument("--feature-spec", default=CONFIG.feature_spec_path)
    args = parser.parse_args()

    cols = _load_feature_cols(args.feature_spec)
    if not cols:
        raise RuntimeError("No feature columns found in feature spec.")
    build_baseline(args.omni_csv, args.output, cols)
    print(f"[baseline] wrote {args.output}")


if __name__ == "__main__":
    main()
