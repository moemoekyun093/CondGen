"""Print accepted structured-query paths in deterministic order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabdiff.query_split import load_query_split


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
    parser.add_argument(
        "--test-supported-only",
        action="store_true",
        help="Print only queries marked as supported by the held-out test split",
    )
    parser.add_argument("--query-split-manifest", default=None)
    parser.add_argument("--query-split", choices=("train", "test"), default=None)
    args = parser.parse_args()
    if (args.query_split_manifest is None) != (args.query_split is None):
        raise ValueError(
            "--query-split-manifest and --query-split must be supplied together"
        )
    selected_ids = None
    if args.query_split_manifest is not None:
        selected_ids = set(
            load_query_split(args.query_split_manifest, args.query_split)
        )
    query_dir = Path(args.query_dir)
    if not query_dir.is_dir():
        raise FileNotFoundError(query_dir)
    found = 0
    seen_bands = set()
    for path in sorted(query_dir.glob("q*.json")):
        with path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if query.get("accepted", True):
            if selected_ids is not None and query["query_id"] not in selected_ids:
                continue
            if args.test_supported_only and not query.get("test_supported", False):
                continue
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
    if selected_ids is not None and found != len(selected_ids):
        raise ValueError(
            f"selected {len(selected_ids)} query ids but found {found} accepted files"
        )


if __name__ == "__main__":
    main()
