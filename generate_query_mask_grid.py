"""Build a deterministic selectivity-by-predicate-mask evaluation grid."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from tabdiff.doob_h_evaluation import raw_constraint_report


def comma_floats(value: str) -> list[float]:
    return sorted({float(item.strip()) for item in value.split(",") if item.strip()})


def comma_ints(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def unique_masks(total: int, arity: int, count: int, seed: str) -> list[tuple[int, ...]]:
    """Draw reproducible, distinct uniform masks at one exact arity."""
    if not 0 <= arity <= total:
        raise ValueError(f"arity {arity} is outside 0..{total}")
    if count <= 0:
        raise ValueError("masks-per-arity must be positive")
    if arity in (0, total):
        return [tuple(range(total)) if arity == total else tuple()]
    rng = random.Random(seed)
    masks = set()
    target = min(count, math.comb(total, arity))
    while len(masks) < target:
        masks.add(tuple(sorted(rng.sample(range(total), arity))))
    return sorted(masks)


def band_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-query-dir", required=True)
    parser.add_argument("--output-query-dir", required=True)
    parser.add_argument("--real-data", default="synthetic/shoppers/real.csv")
    parser.add_argument("--bands", default="0.005,0.01,0.02,0.05,0.1,0.25,0.4")
    parser.add_argument("--arities", default="0,2,4,8,12,18")
    parser.add_argument("--queries-per-band", type=int, default=1)
    parser.add_argument("--masks-per-arity", type=int, default=3)
    parser.add_argument("--seed", type=int, default=8127)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bands = comma_floats(args.bands)
    arities = comma_ints(args.arities)
    if not bands or not arities:
        raise ValueError("bands and arities cannot be empty")
    if args.queries_per_band <= 0:
        raise ValueError("queries-per-band must be positive")

    source_dir = Path(args.source_query_dir)
    output_dir = Path(args.output_query_dir)
    refs_dir = output_dir / "refs"
    real_path = Path(args.real_data)
    train_indices_path = output_dir.parent / "splits" / "train_idx.npy"
    for path in (source_dir, real_path, train_indices_path):
        if not path.exists():
            raise FileNotFoundError(path)
    real = pd.read_csv(real_path)
    train_indices = np.load(train_indices_path).astype(np.int64)
    if len(train_indices) == 0 or train_indices.min() < 0 or train_indices.max() >= len(real):
        raise ValueError("core training indices do not address the real table")

    by_band: dict[float, list[dict]] = defaultdict(list)
    for path in sorted(source_dir.glob("q*.json")):
        with path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if query.get("accepted", True):
            by_band[float(query["target_band"])].append(query)

    selected_queries = []
    for band in bands:
        matching = next(
            (queries for value, queries in by_band.items() if np.isclose(value, band)),
            [],
        )
        if len(matching) < args.queries_per_band:
            raise ValueError(
                f"band {band:g} has {len(matching)} queries, fewer than requested "
                f"{args.queries_per_band}"
            )
        shuffled = list(matching)
        random.Random(f"{args.seed}:band:{band:g}").shuffle(shuffled)
        selected_queries.extend((band, query) for query in shuffled[: args.queries_per_band])

    generated: list[tuple[dict, np.ndarray, dict]] = []
    for band, parent in selected_queries:
        predicates = parent["predicates"]
        total = len(predicates)
        if arities[0] < 0 or arities[-1] > total:
            raise ValueError(
                f"requested arities do not fit the {total} predicates in {parent['query_id']}"
            )
        parent_rank = next(
            index
            for index, (candidate_band, candidate) in enumerate(selected_queries)
            if candidate_band == band and candidate["query_id"] == parent["query_id"]
        )
        within_band_rank = sum(
            1
            for previous_band, _ in selected_queries[:parent_rank]
            if np.isclose(previous_band, band)
        )
        for arity in arities:
            masks = unique_masks(
                total,
                arity,
                args.masks_per_arity,
                f"{args.seed}:{parent['query_id']}:{arity}",
            )
            for mask_rank, selected_indices in enumerate(masks):
                selected = set(selected_indices)
                active_predicates = [
                    copy.deepcopy(predicate)
                    for index, predicate in enumerate(predicates)
                    if index in selected
                ]
                query_id = (
                    f"qf_grid_b{band_tag(band)}_q{within_band_rank:02d}_"
                    f"k{arity:02d}_m{mask_rank:02d}"
                )
                derived = copy.deepcopy(parent)
                derived.update(
                    {
                        "query_id": query_id,
                        "suite": "selectivity_mask_grid",
                        "source_query_id": parent["query_id"],
                        "source_target_band": float(band),
                        "target_band": float(band),
                        "arity": arity,
                        "mask_id": mask_rank,
                        "predicate_mask": [
                            int(index in selected) for index in range(total)
                        ],
                        "active_columns": [
                            predicate["col"] for predicate in active_predicates
                        ],
                        "n_num_cols": sum(
                            predicate["modality"] == "numeric"
                            for predicate in active_predicates
                        ),
                        "n_cat_cols": sum(
                            predicate["modality"] == "categorical"
                            for predicate in active_predicates
                        ),
                        "predicates": active_predicates,
                        "accepted": True,
                    }
                )
                report, real_mask = raw_constraint_report(real, derived)
                train_rows = train_indices[real_mask[train_indices]]
                if len(train_rows) == 0:
                    raise ValueError(f"derived query {query_id} has no core support")
                derived["counts"] = {
                    **derived.get("counts", {}),
                    "masked_real": int(real_mask.sum()),
                    "masked_train": int(len(train_rows)),
                }
                derived["selectivity"] = {
                    **derived.get("selectivity", {}),
                    "masked_real": float(report["joint_hit_rate"]),
                    "masked_train": float(len(train_rows) / len(train_indices)),
                }
                manifest_row = {
                    "query_id": query_id,
                    "source_query_id": parent["query_id"],
                    "target_band": float(band),
                    "arity": arity,
                    "mask_id": mask_rank,
                    "n_num_cols": derived["n_num_cols"],
                    "n_cat_cols": derived["n_cat_cols"],
                    "effective_real_selectivity": report["joint_hit_rate"],
                    "train_support": len(train_rows),
                    "active_columns": ",".join(derived["active_columns"]),
                }
                generated.append((derived, train_rows, manifest_row))

    expected = {query["query_id"] for query, _, _ in generated}
    existing = {path.stem for path in output_dir.glob("q*.json")}
    stale = existing - expected
    if stale:
        raise ValueError(
            "output directory contains queries from a different grid; use a new "
            f"directory. Unexpected examples: {sorted(stale)[:3]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    for query, train_rows, _ in generated:
        query_id = query["query_id"]
        with (output_dir / f"{query_id}.json").open("w", encoding="utf-8") as stream:
            json.dump(query, stream, indent=2)
        np.savez_compressed(refs_dir / f"{query_id}.npz", train_rows=train_rows)
    manifest = pd.DataFrame([row for _, _, row in generated])
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest.to_dict(orient="records"), stream, indent=2)
    print(f"Generated {len(generated)} selectivity-mask queries in {output_dir}")
    print(manifest.groupby(["target_band", "arity"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
