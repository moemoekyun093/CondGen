"""Create a small stratified train-query sample while retaining all test queries."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from tabdiff.query_split import load_query_split, query_id_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-per-band", type=int, default=5)
    parser.add_argument("--arity", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_per_band <= 0:
        raise ValueError("train-per-band must be positive")
    query_dir = Path(args.query_dir)
    queries = {}
    for path in sorted(query_dir.glob("q*.json")):
        with path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if query.get("accepted", True):
            queries[str(query["query_id"])] = query

    source_train = load_query_split(args.source_manifest, "train")
    source_test = load_query_split(args.source_manifest, "test")
    missing = (set(source_train) | set(source_test)) - set(queries)
    if missing:
        raise ValueError(f"query files missing for ids: {sorted(missing)[:5]}")

    eligible_train = [
        query_id
        for query_id in source_train
        if args.arity is None
        or int(queries[query_id].get("arity", len(queries[query_id]["predicates"])))
        == args.arity
    ]
    eligible_test = [
        query_id
        for query_id in source_test
        if args.arity is None
        or int(queries[query_id].get("arity", len(queries[query_id]["predicates"])))
        == args.arity
    ]
    if not eligible_train or not eligible_test:
        raise ValueError(f"arity filter {args.arity!r} leaves an empty partition")
    by_band = defaultdict(list)
    for query_id in eligible_train:
        by_band[float(queries[query_id]["target_band"])].append(query_id)
    rng = random.Random(args.seed)
    selected_train = []
    selected_counts = {}
    for band, query_ids in sorted(by_band.items()):
        query_ids = sorted(query_ids)
        selected = (
            query_ids
            if args.arity is not None
            else rng.sample(query_ids, min(args.train_per_band, len(query_ids)))
        )
        selected_train.extend(sorted(selected))
        selected_counts[str(band)] = len(selected)

    payload = {
        "version": 1,
        "kind": "query_generalization_diagnostic",
        "source_manifest": str(args.source_manifest),
        "seed": args.seed,
        "arity_filter": args.arity,
        "train_queries_per_target_band": args.train_per_band,
        "selected_train_counts_by_band": selected_counts,
        "partitions": {
            "train": sorted(selected_train),
            "test": sorted(eligible_test),
        },
    }
    payload["source_query_ids_sha256"] = query_id_digest(
        [*payload["partitions"]["train"], *payload["partitions"]["test"]]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(f"Selected {len(selected_train)} training queries across {len(by_band)} bands")
    print(f"Retained {len(eligible_test)} unseen test queries")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
