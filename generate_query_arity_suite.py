"""Derive nested partial-arity queries from one full-query selectivity band."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from tabdiff.doob_h_evaluation import raw_constraint_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-query-dir", required=True)
    parser.add_argument("--output-query-dir", required=True)
    parser.add_argument("--real-data", default="synthetic/shoppers/real.csv")
    parser.add_argument("--source-band", type=float, default=0.4)
    parser.add_argument("--arities", default="2,4,8,12,18")
    parser.add_argument("--seed", type=int, default=7301)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arities = sorted({int(value) for value in args.arities.split(",")})
    if not arities or arities[0] <= 0:
        raise ValueError("arities must be positive comma-separated integers")
    source_dir = Path(args.source_query_dir)
    output_dir = Path(args.output_query_dir)
    refs_dir = output_dir / "refs"
    split_root = output_dir.parent
    train_indices_path = split_root / "splits" / "train_idx.npy"
    real_path = Path(args.real_data)
    for path in (source_dir, train_indices_path, real_path):
        if not path.exists():
            raise FileNotFoundError(path)
    real = pd.read_csv(real_path)
    train_indices = np.load(train_indices_path).astype(np.int64)
    if train_indices.min() < 0 or train_indices.max() >= len(real):
        raise ValueError("core training indices do not address the real table")
    output_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    base_queries = []
    for path in sorted(source_dir.glob("qf_*.json")):
        with path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if query.get("accepted", True) and np.isclose(
            float(query["target_band"]), args.source_band, rtol=0.0, atol=1e-12
        ):
            base_queries.append(query)
    if not base_queries:
        raise ValueError(f"no accepted source queries at band {args.source_band}")

    manifest = []
    for base_query in base_queries:
        numerical = [
            predicate for predicate in base_query["predicates"]
            if predicate["modality"] == "numeric"
        ]
        categorical = [
            predicate for predicate in base_query["predicates"]
            if predicate["modality"] == "categorical"
        ]
        total = len(numerical) + len(categorical)
        if arities[-1] > total:
            raise ValueError(f"requested arity exceeds {total} for {base_query['query_id']}")
        numerical_order = list(range(len(numerical)))
        categorical_order = list(range(len(categorical)))
        random.Random(f"{args.seed}:{base_query['query_id']}:num").shuffle(numerical_order)
        random.Random(f"{args.seed}:{base_query['query_id']}:cat").shuffle(categorical_order)

        for arity in arities:
            numerical_count = round(arity * len(numerical) / total)
            numerical_count = min(len(numerical), max(0, numerical_count))
            categorical_count = arity - numerical_count
            if categorical_count > len(categorical):
                categorical_count = len(categorical)
                numerical_count = arity - categorical_count
            selected_num = set(numerical_order[:numerical_count])
            selected_cat = set(categorical_order[:categorical_count])
            active_names = {
                predicate["col"] for index, predicate in enumerate(numerical)
                if index in selected_num
            } | {
                predicate["col"] for index, predicate in enumerate(categorical)
                if index in selected_cat
            }
            predicates = [
                copy.deepcopy(predicate)
                for predicate in base_query["predicates"]
                if predicate["col"] in active_names
            ]
            query_id = f"qf_arity_{base_query['query_id']}_k{arity:02d}"
            derived = copy.deepcopy(base_query)
            derived.update(
                {
                    "query_id": query_id,
                    "suite": "arity_relaxation",
                    "source_query_id": base_query["query_id"],
                    "source_target_band": float(base_query["target_band"]),
                    "arity": arity,
                    "n_num_cols": numerical_count,
                    "n_cat_cols": categorical_count,
                    "predicates": predicates,
                    "relaxed_columns": [
                        predicate["col"] for predicate in base_query["predicates"]
                        if predicate["col"] not in active_names
                    ],
                    "accepted": True,
                }
            )
            report, mask = raw_constraint_report(real, derived)
            train_rows = train_indices[mask[train_indices]]
            derived["counts"] = {
                **derived.get("counts", {}),
                "arity_real": int(mask.sum()),
                "arity_train": int(len(train_rows)),
            }
            derived["selectivity"] = {
                **derived.get("selectivity", {}),
                "arity_real": float(mask.mean()),
                "arity_train": float(len(train_rows) / len(train_indices)),
            }
            with (output_dir / f"{query_id}.json").open("w", encoding="utf-8") as stream:
                json.dump(derived, stream, indent=2)
            np.savez_compressed(refs_dir / f"{query_id}.npz", train_rows=train_rows)
            manifest.append(
                {
                    "query_id": query_id,
                    "source_query_id": base_query["query_id"],
                    "arity": arity,
                    "n_num_cols": numerical_count,
                    "n_cat_cols": categorical_count,
                    "real_selectivity": report["joint_hit_rate"],
                    "train_support": len(train_rows),
                }
            )
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    pd.DataFrame(manifest).to_csv(output_dir / "manifest.csv", index=False)
    print(f"Generated {len(manifest)} nested arity queries in {output_dir}")


if __name__ == "__main__":
    main()
