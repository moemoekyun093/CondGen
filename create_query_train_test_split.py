"""Create a deterministic stratified train/test split over accepted queries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tabdiff.query_split import canonical_query_fingerprint, query_id_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_rank(seed: int, query_id: str) -> str:
    return hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("test-fraction must lie strictly between zero and one")
    query_dir = Path(args.query_dir)
    queries = []
    for path in sorted(query_dir.glob("q*.json")):
        with path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if query.get("accepted", True):
            queries.append(query)
    if not queries:
        raise ValueError(f"no accepted queries found in {query_dir}")

    semantic_groups = defaultdict(list)
    for query in queries:
        semantic_groups[canonical_query_fingerprint(query)].append(query)
    representatives = []
    duplicate_aliases = []
    for fingerprint, group in sorted(semantic_groups.items()):
        # When identical predicates were generated under several nominal bands,
        # keep the label closest to their measured training selectivity.
        representative = min(
            group,
            key=lambda query: (
                abs(
                    float(query["selectivity"]["train"])
                    - float(query["target_band"])
                ),
                str(query["query_id"]),
            ),
        )
        representatives.append(representative)
        aliases = sorted(
            str(query["query_id"])
            for query in group
            if query is not representative
        )
        if aliases:
            duplicate_aliases.append(
                {
                    "fingerprint": fingerprint,
                    "representative": str(representative["query_id"]),
                    "excluded_aliases": aliases,
                }
            )

    strata = defaultdict(list)
    for query in representatives:
        key = (
            float(query["target_band"]),
            int(query.get("arity", len(query["predicates"]))),
            str(query.get("modality_mix", "unknown")),
        )
        strata[key].append(str(query["query_id"]))

    train_ids = []
    test_ids = []
    stratum_rows = []
    for key in sorted(strata):
        ranked = sorted(strata[key], key=lambda value: (stable_rank(args.seed, value), value))
        if len(ranked) == 1:
            test_count = 0
        else:
            test_count = round(len(ranked) * args.test_fraction)
            test_count = min(len(ranked) - 1, max(1, test_count))
        selected_test = sorted(ranked[:test_count])
        selected_train = sorted(ranked[test_count:])
        train_ids.extend(selected_train)
        test_ids.extend(selected_test)
        stratum_rows.append(
            {
                "target_band": key[0],
                "arity": key[1],
                "modality_mix": key[2],
                "total": len(ranked),
                "train": len(selected_train),
                "test": len(selected_test),
            }
        )

    train_ids.sort()
    test_ids.sort()
    all_ids = [*train_ids, *test_ids]
    manifest = {
        "version": 1,
        "dataset": queries[0].get("dataset"),
        "query_suite": queries[0].get("suite"),
        "query_directory": str(query_dir),
        "seed": args.seed,
        "requested_test_fraction": args.test_fraction,
        "stratification": ["target_band", "arity", "modality_mix"],
        "singleton_policy": "train_only",
        "source_query_ids_sha256": query_id_digest(all_ids),
        "counts": {
            "source_accepted": len(queries),
            "unique_predicate_sets": len(all_ids),
            "excluded_duplicate_aliases": len(queries) - len(all_ids),
            "total": len(all_ids),
            "train": len(train_ids),
            "test": len(test_ids),
        },
        "partitions": {"train": train_ids, "test": test_ids},
        "duplicate_aliases": duplicate_aliases,
        "strata": stratum_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(
        f"Saved query split to {output}: "
        f"train={len(train_ids)}, test={len(test_ids)}, total={len(all_ids)}, "
        f"excluded_duplicate_aliases={len(queries) - len(all_ids)}"
    )


if __name__ == "__main__":
    main()
