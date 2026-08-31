"""Export the exact model-space coordinates supplied to structured Doob guides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tabdiff.doob_h_runtime import load_doob_runtime, resolve_base_checkpoint
from tabdiff.doob_query_suite import load_structured_query_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", default="shoppers")
    parser.add_argument("--base-ckpt", default=None)
    parser.add_argument("--base-exp-name", default="ft_periodic_seed0")
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = resolve_base_checkpoint(
        args.dataname, args.base_ckpt, args.base_exp_name
    )
    runtime = load_doob_runtime(args.dataname, checkpoint, torch.device("cpu"))
    queries = load_structured_query_suite(args.query_dir, runtime)
    category_counts = [int(value) for value in runtime.dataset.categories]
    payload = {
        "version": 1,
        "dataname": args.dataname,
        "base_checkpoint": checkpoint,
        "coordinate_system": "exact_TabDiff_clean_numerical_transform",
        "category_counts": category_counts,
        "queries": {},
    }
    for query in queries:
        allowed_columns = []
        offset = 0
        for count in category_counts:
            allowed_columns.append(
                query.categorical_allowed[offset : offset + count].tolist()
            )
            offset += count
        payload["queries"][query.query_id] = {
            "numerical_lower": query.numerical_lower.tolist(),
            "numerical_upper": query.numerical_upper.tolist(),
            "numerical_active": query.numerical_active.tolist(),
            "categorical_allowed": allowed_columns,
            "categorical_active": query.categorical_active.tolist(),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream)
        stream.write("\n")
    print(f"Exported {len(queries)} model-space queries to {output}")


if __name__ == "__main__":
    main()
