"""Print accepted structured-query paths in deterministic order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_dir")
    parser.add_argument(
        "--one-per-band",
        action="store_true",
        help="Print the first accepted query at each target selectivity band",
    )
    parser.add_argument(
        "--target-band",
        type=float,
        default=None,
        help="Print only accepted queries at this target selectivity band",
    )
    args = parser.parse_args()
    query_dir = Path(args.query_dir)
    if not query_dir.is_dir():
        raise FileNotFoundError(query_dir)
    found = 0
    seen_bands = set()
    for path in sorted(query_dir.glob("q*.json")):
        with path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if query.get("accepted", True):
            band = float(query["target_band"])
            if args.target_band is not None and not abs(band - args.target_band) <= 1e-12:
                continue
            if args.one_per_band and band in seen_bands:
                continue
            print(path)
            found += 1
            seen_bands.add(band)
    if found == 0:
        raise ValueError(f"no accepted queries found in {query_dir}")


if __name__ == "__main__":
    main()
