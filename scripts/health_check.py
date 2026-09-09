import argparse
import json
import sys
from urllib.request import urlopen, Request


def main():
    parser = argparse.ArgumentParser(description="Check API health endpoint")
    parser.add_argument("--url", default="http://localhost:8000/api/health")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    req = Request(args.url, headers={"User-Agent": "space-weather-health/1.0"})
    with urlopen(req, timeout=args.timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ok = data.get("ok", False)
    if not ok:
        print(json.dumps(data, indent=2))
        sys.exit(1)
    print("ok")


if __name__ == "__main__":
    main()
