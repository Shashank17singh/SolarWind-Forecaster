import argparse
import json
from urllib.request import Request, urlopen

DEFAULT_ARTICLE_ID = "6356420"
API_URL = "https://api.figshare.com/v2/articles/{article_id}"


def _fetch_json(url: str, timeout: int = 30):
    req = Request(url, headers={"User-Agent": "space-weather-sentinel/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pick_file(files, fmt: str):
    fmt = fmt.lower()
    for item in files:
        name = item.get("name", "")
        if name.lower().endswith(f".{fmt}"):
            return item
    return files[0] if files else None


def download(article_id: str, out_path: str, fmt: str):
    meta = _fetch_json(API_URL.format(article_id=article_id))
    files = meta.get("files", [])
    if not files:
        raise RuntimeError("No files found for ICMECAT article.")
    file_info = _pick_file(files, fmt)
    if not file_info:
        raise RuntimeError("No suitable file found.")
    url = file_info.get("download_url")
    if not url:
        raise RuntimeError("Download URL missing.")

    req = Request(url, headers={"User-Agent": "space-weather-sentinel/1.0"})
    with urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"[icmecat] downloaded {file_info.get('name')} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download HELCATS ICMECAT dataset from figshare."
    )
    parser.add_argument("--article-id", default=DEFAULT_ARTICLE_ID)
    parser.add_argument("--out", required=True)
    parser.add_argument("--format", default="csv", help="csv or json")
    args = parser.parse_args()
    download(args.article_id, args.out, args.format)
