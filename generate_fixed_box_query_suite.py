"""Convert the legacy fixed numerical box into nested structured queries."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from tabdiff.doob_h_evaluation import raw_constraint_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-query", required=True)
    parser.add_argument("--output-query-dir", required=True)
    parser.add_argument("--real-data", default="synthetic/shoppers/real.csv")
    parser.add_argument("--split-root", default="data90/shoppers")
    parser.add_argument("--column-order", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--num-orderings",
        type=int,
        default=12,
        help="Identity, reverse, then seeded unique random permutations",
    )
    parser.add_argument("--seed", type=int, default=7301)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixed_path = Path(args.fixed_query)
    output_dir = Path(args.output_query_dir)
    split_root = Path(args.split_root)
    real_path = Path(args.real_data)
    train_indices_path = split_root / "splits" / "train_idx.npy"
    for path in (fixed_path, real_path, train_indices_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with fixed_path.open("r", encoding="utf-8") as stream:
        fixed = json.load(stream)
    columns = fixed.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("legacy fixed query must contain a nonempty 'columns' list")
    by_index = {
        int(column.get("model_index", position)): column
        for position, column in enumerate(columns)
    }
    base_order = [int(value.strip()) for value in args.column_order.split(",")]
    if len(base_order) != len(set(base_order)) or any(
        index not in by_index for index in base_order
    ):
        raise ValueError("column-order must contain unique model indices from the query")
    if args.num_orderings <= 0:
        raise ValueError("num-orderings must be positive")

    orderings = [tuple(base_order)]
    if args.num_orderings > 1:
        reversed_order = tuple(reversed(base_order))
        if reversed_order not in orderings:
            orderings.append(reversed_order)
    rng = random.Random(args.seed)
    attempts = 0
    while len(orderings) < args.num_orderings:
        candidate = list(base_order)
        rng.shuffle(candidate)
        candidate_tuple = tuple(candidate)
        if candidate_tuple not in orderings:
            orderings.append(candidate_tuple)
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("could not construct the requested unique orderings")

    real = pd.read_csv(real_path)
    train_indices = np.load(train_indices_path).astype(np.int64)
    if train_indices.size == 0 or train_indices.min() < 0 or train_indices.max() >= len(real):
        raise ValueError("core training indices do not address the real table")
    refs_dir = output_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    # This directory is a generated artifact. Remove only files owned by this
    # generator so changing the permutation count cannot leave stale queries.
    for stale in output_dir.glob("qf_fixed_box_o*_k*.json"):
        stale.unlink()
    for stale in refs_dir.glob("qf_fixed_box_o*_k*.npz"):
        stale.unlink()

    manifest = []
    ordering_lines = []
    for ordering_index, order in enumerate(orderings):
        ordering_id = f"o{ordering_index:02d}"
        ordering_lines.append(
            f"{ordering_id}|{','.join(str(index) for index in order)}"
        )
        for level in range(1, len(order) + 1):
            selected = [by_index[index] for index in order[:level]]
            predicates = []
            for column in selected:
                lower = column.get("raw_lower")
                upper = column.get("raw_upper")
                if lower is None or upper is None:
                    raise ValueError(
                        "structured interval comparison requires two finite bounds"
                    )
                predicates.append(
                    {
                        "col": str(column["name"]),
                        "modality": "numeric",
                        "op": "between",
                        "values": [float(lower), float(upper)],
                    }
                )

            query_id = f"qf_fixed_box_{ordering_id}_k{level:02d}"
            query = {
                "query_id": query_id,
                "dataset": fixed.get("dataname", fixed.get("dataset", "shoppers")),
                "suite": "legacy_fixed_box_permuted_nesting",
                "ordering_id": ordering_id,
                "column_order": list(order),
                "arity": level,
                "n_num_cols": level,
                "n_cat_cols": 0,
                "predicates": predicates,
                "accepted": True,
            }
            report, mask = raw_constraint_report(real, query)
            train_rows = train_indices[mask[train_indices]]
            if train_rows.size == 0:
                raise ValueError(f"{query_id} has no core-training support")
            selectivity = float(report["joint_hit_rate"])
            query["target_band"] = selectivity
            query["counts"] = {
                "arity_real": int(mask.sum()),
                "arity_train": int(train_rows.size),
            }
            query["selectivity"] = {
                "arity_real": selectivity,
                "arity_train": float(train_rows.size / train_indices.size),
            }
            with (output_dir / f"{query_id}.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(query, stream, indent=2)
            np.savez_compressed(refs_dir / f"{query_id}.npz", train_rows=train_rows)
            manifest.append(
                {
                    "query_id": query_id,
                    "ordering_id": ordering_id,
                    "arity": level,
                    "real_selectivity": selectivity,
                    "train_support": int(train_rows.size),
                    "added_column": str(selected[-1]["name"]),
                }
            )

    pd.DataFrame(manifest).to_csv(output_dir / "manifest.csv", index=False)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    with (output_dir / "orderings.txt").open("w", encoding="utf-8") as stream:
        stream.write("\n".join(ordering_lines) + "\n")
    print(
        f"Generated {len(manifest)} fixed-box queries across "
        f"{len(orderings)} orderings in {output_dir}"
    )


if __name__ == "__main__":
    main()
