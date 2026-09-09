import argparse
import os
from urllib.request import urlretrieve

JB2008_BASE = "https://sol.spacenvironment.net/JB2008/indices"
SOLFSMY_URL = f"{JB2008_BASE}/SOLFSMY.TXT"
DTCFILE_URL = f"{JB2008_BASE}/DTCFILE.TXT"
SWALL_URL = "https://celestrak.org/SpaceData/SW-All.csv"


def _download(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urlretrieve(url, path)
    print(f"[download] {url} -> {path}")


def main(out_dir: str, with_swall: bool):
    _download(SOLFSMY_URL, os.path.join(out_dir, "SOLFSMY.TXT"))
    _download(DTCFILE_URL, os.path.join(out_dir, "DTCFILE.TXT"))
    if with_swall:
        _download(SWALL_URL, os.path.join(out_dir, "SW-All.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download JB2008 indices files")
    parser.add_argument("--out-dir", default="data/indices/jb2008")
    parser.add_argument(
        "--with-swall", action="store_true", help="Also download SW-All.csv for Ap/Kp"
    )
    args = parser.parse_args()
    main(args.out_dir, args.with_swall)
