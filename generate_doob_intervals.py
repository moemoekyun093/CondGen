"""Generate one deterministic fixed interval for every numerical column.

The common symmetric marginal tail probability is chosen by binary search so
that the intersection of all intervals contains at least the requested fraction
of transformed training rows.  The resulting JSON is the immutable query used
for both Doob-guide training and conditional-sample evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from tabdiff.doob_h_runtime import resolve_base_checkpoint
from tabdiff.models.doob_h_transform import NumericalBoxQuery
from utils_train import TabDiffDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", default="shoppers")
    parser.add_argument("--base-ckpt", default=None)
    parser.add_argument("--base-exp-name", default="learnable_schedule")
    parser.add_argument("--target-coverage", type=float, default=0.30)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_dataset(dataname: str, checkpoint_path: str):
    data_dir = f"data/{dataname}"
    info_path = os.path.join(data_dir, "info.json")
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.pkl")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"processed dataset info not found: {info_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"base checkpoint config not found: {config_path}")

    with open(info_path, "r", encoding="utf-8") as stream:
        info = json.load(stream)
    with open(config_path, "rb") as stream:
        config = pickle.load(stream)
    dataset = TabDiffDataset(
        dataname,
        data_dir,
        info,
        isTrain=True,
        dequant_dist=config["data"]["dequant_dist"],
        int_dequant_factor=config["data"]["int_dequant_factor"],
    )
    return dataset, info


def query_at_tail(x_num: torch.Tensor, tail_probability: float) -> NumericalBoxQuery:
    return NumericalBoxQuery.from_quantiles(
        x_num,
        lower_quantile=tail_probability,
        upper_quantile=1.0 - tail_probability,
    )


def choose_query(
    x_num: torch.Tensor,
    target_coverage: float,
    iterations: int = 50,
) -> tuple[NumericalBoxQuery, float, float]:
    """Find the narrowest common central quantile box retaining target coverage."""
    lower_tail = 0.0
    upper_tail = 0.499999
    best_query = query_at_tail(x_num, lower_tail)
    best_coverage = 1.0

    for _ in range(iterations):
        candidate_tail = (lower_tail + upper_tail) / 2.0
        candidate_query = query_at_tail(x_num, candidate_tail)
        coverage = candidate_query.contains(x_num).float().mean().item()
        if coverage >= target_coverage:
            lower_tail = candidate_tail
            best_query = candidate_query
            best_coverage = coverage
        else:
            upper_tail = candidate_tail

    return best_query, lower_tail, best_coverage


def numerical_column_indices(info: dict) -> list[int]:
    if info["task_type"] == "regression":
        return list(info["target_col_idx"]) + list(info["num_col_idx"])
    return list(info["num_col_idx"])


def column_name(info: dict, original_index: int) -> str:
    mapping = info.get("idx_name_mapping", {})
    if str(original_index) in mapping:
        return str(mapping[str(original_index)])
    if original_index in mapping:
        return str(mapping[original_index])
    names = info.get("column_names")
    if names is not None:
        return str(names[original_index])
    return f"column_{original_index}"


def main() -> None:
    args = parse_args()
    if not 0.0 < args.target_coverage < 1.0:
        raise ValueError("target-coverage must be strictly between 0 and 1")

    checkpoint_path = resolve_base_checkpoint(
        args.dataname,
        args.base_ckpt,
        args.base_exp_name,
    )
    dataset, info = load_dataset(args.dataname, checkpoint_path)
    d_numerical = dataset.d_numerical
    x_num = dataset.X[:, :d_numerical].float().contiguous()
    query, tail_probability, achieved_coverage = choose_query(
        x_num,
        args.target_coverage,
    )

    normalized_bounds = np.stack(
        [query.lower.numpy(), query.upper.numpy()],
        axis=0,
    )
    raw_bounds = dataset.num_inverse(normalized_bounds).astype(np.float64)
    raw_bounds = dataset.int_inverse(raw_bounds).astype(np.float64)
    raw_lower = np.minimum(raw_bounds[0], raw_bounds[1])
    raw_upper = np.maximum(raw_bounds[0], raw_bounds[1])

    original_indices = numerical_column_indices(info)
    if len(original_indices) != d_numerical:
        raise ValueError(
            "processed numerical order does not match metadata: "
            f"{d_numerical} transformed columns versus {len(original_indices)} indices"
        )
    columns = []
    for model_index, original_index in enumerate(original_indices):
        columns.append(
            {
                "model_index": model_index,
                "original_index": original_index,
                "name": column_name(info, original_index),
                "normalized_lower": float(query.lower[model_index]),
                "normalized_upper": float(query.upper[model_index]),
                "raw_lower": float(raw_lower[model_index]),
                "raw_upper": float(raw_upper[model_index]),
            }
        )

    data_fingerprint = hashlib.sha256(x_num.numpy().tobytes()).hexdigest()
    constraint_payload = {
        "dataname": args.dataname,
        "method": "symmetric_marginal_quantiles_with_target_joint_coverage",
        "target_joint_coverage": args.target_coverage,
        "achieved_training_joint_coverage": achieved_coverage,
        "symmetric_tail_probability": tail_probability,
        "training_rows": len(x_num),
        "data_sha256": data_fingerprint,
        "space": "TabDiff quantile-normalized numerical coordinates",
        "all_numerical_columns": True,
        **query.to_dict(),
        "columns": columns,
    }
    constraint_id_source = json.dumps(
        {"dataname": args.dataname, **query.to_dict()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    constraint_payload["constraint_id"] = hashlib.sha256(
        constraint_id_source
    ).hexdigest()[:16]

    output = Path(
        args.output
        or f"constraints/{args.dataname}/fixed_numerical_intervals.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(constraint_payload, stream, indent=2)

    print(f"Constraint ID: {constraint_payload['constraint_id']}")
    print(f"Dataset: {args.dataname}")
    print(f"Numerical columns constrained: {d_numerical}")
    print(f"Target joint coverage: {args.target_coverage:.2%}")
    print(f"Achieved training joint coverage: {achieved_coverage:.2%}")
    print(f"Symmetric marginal tail probability: {tail_probability:.6f}")
    print(f"Saved deterministic intervals to {output}")


if __name__ == "__main__":
    main()
