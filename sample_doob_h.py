"""Generate samples with a trained mask-conditioned Doob h-transform."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from tabdiff.doob_h_runtime import load_doob_runtime, resolve_base_checkpoint
from tabdiff.doob_h_evaluation import raw_constraint_hits
from tabdiff.models.doob_h_transform import (
    CategoricalHTransformGuide,
    NumericalBoxQuery,
    NumericalDoobHGuide,
    NumericalHScoreGuide,
)
from tabdiff.trainer import recover_data, split_num_cat_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide-ckpt", required=True)
    parser.add_argument("--base-ckpt", default=None)
    parser.add_argument("--base-exp-name", default="learnable_schedule")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-correction", type=float, default=5.0)
    parser.add_argument("--max-log-h-ratio", type=float, default=10.0)
    parser.add_argument("--h-candidate-batch-size", type=int, default=65536)
    parser.add_argument(
        "--active-columns",
        default=None,
        help="Comma-separated active numerical model indices; random when omitted",
    )
    parser.add_argument(
        "--all-columns-active",
        action="store_true",
        help="Activate every numerical interval",
    )
    parser.add_argument(
        "--no-columns-active",
        action="store_true",
        help="Deactivate every numerical interval (unconditional anchor)",
    )
    parser.add_argument("--column-active-probability", type=float, default=0.5)
    parser.add_argument(
        "--categorical-start-mode",
        choices=("full", "section4_posterior"),
        default="section4_posterior",
        help=(
            "Use Section 4's posterior reverse start time when fixed categorical "
            "equalities are supplied; otherwise the sampler falls back to full time"
        ),
    )
    parser.add_argument(
        "--fixed-categorical",
        action="append",
        default=[],
        metavar="COLUMN=CLASS",
        help=(
            "Fix a model-space categorical column to a model-space class index; "
            "repeat for multiple equalities (example: --fixed-categorical 0=2)"
        ),
    )
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_fixed_categorical(specs: list[str]) -> dict[int, int]:
    fixed = {}
    for spec in specs:
        try:
            column_text, class_text = spec.split("=", maxsplit=1)
            column = int(column_text)
            class_index = int(class_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid fixed categorical equality {spec!r}; expected COLUMN=CLASS"
            ) from error
        if column in fixed and fixed[column] != class_index:
            raise ValueError(f"categorical column {column} was fixed to two values")
        fixed[column] = class_index
    return fixed


def choose_query_active_mask(
    specification: str | None,
    d_numerical: int,
    probability: float,
    device: torch.device,
    all_columns_active: bool = False,
    no_columns_active: bool = False,
) -> torch.Tensor:
    if not 0.0 < probability < 1.0:
        raise ValueError("column-active-probability must be between 0 and 1")
    mask = torch.zeros(d_numerical, dtype=torch.float32, device=device)
    if sum((all_columns_active, no_columns_active, specification is not None)) > 1:
        raise ValueError(
            "choose only one of all-columns-active, no-columns-active, or active-columns"
        )
    if all_columns_active:
        mask.fill_(1.0)
    elif no_columns_active:
        pass
    elif specification is not None:
        try:
            columns = [
                int(value.strip())
                for value in specification.split(",")
                if value.strip()
            ]
        except ValueError as error:
            raise ValueError("active-columns must be comma-separated integers") from error
        if not columns:
            raise ValueError("active-columns must select at least one column")
        if min(columns) < 0 or max(columns) >= d_numerical:
            raise ValueError(
                f"active numerical columns must lie in [0, {d_numerical - 1}]"
            )
        mask[columns] = 1.0
    else:
        mask = (torch.rand(d_numerical, device=device) < probability).float()
    return mask


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num-samples and batch-size must be positive")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    guide_state = torch_load(args.guide_ckpt, device)
    metadata = guide_state["metadata"]
    dataname = metadata["dataname"]
    metadata_base = metadata.get("base_checkpoint")
    requested_base = args.base_ckpt
    if requested_base is None and metadata_base and os.path.isfile(metadata_base):
        requested_base = metadata_base
    base_checkpoint = resolve_base_checkpoint(
        dataname,
        requested_base,
        args.base_exp_name,
    )
    runtime = load_doob_runtime(dataname, base_checkpoint, device)

    if "numerical_guide" in guide_state:
        numerical_guide = NumericalHScoreGuide(
            **metadata["numerical_guide"]
        ).to(device)
        categorical_guide = CategoricalHTransformGuide(
            **metadata["categorical_guide"]
        ).to(device)
        numerical_guide.load_state_dict(guide_state["numerical_guide"])
        categorical_guide.load_state_dict(guide_state["categorical_guide"])
    else:
        # Compatibility with checkpoints created before the two-guide split.
        legacy_guide = NumericalDoobHGuide(**metadata["guide"]).to(device)
        legacy_guide.load_state_dict(guide_state["guide"])
        numerical_guide = legacy_guide
        categorical_guide = legacy_guide
    numerical_guide.eval()
    categorical_guide.eval()
    d_numerical = runtime.dataset.d_numerical
    query_active_mask = choose_query_active_mask(
        args.active_columns,
        d_numerical,
        args.column_active_probability,
        device,
        all_columns_active=args.all_columns_active,
        no_columns_active=args.no_columns_active,
    )
    training_mode = metadata.get("training", {}).get("mode")
    if training_mode == "all_constrained" and not bool(query_active_mask.bool().all()):
        raise ValueError(
            "this checkpoint was trained only with the all-ones constraint mask; "
            "partial-mask sampling is intentionally disabled until partial training"
        )
    active_columns = torch.nonzero(
        query_active_mask,
        as_tuple=False,
    ).flatten().tolist()
    runtime.diffusion.set_doob_h_guides(
        numerical_guide,
        categorical_guide,
        strength=1.0,
        max_correction=args.max_correction,
        max_log_ratio=args.max_log_h_ratio,
        candidate_batch_size=args.h_candidate_batch_size,
        query_active_mask=query_active_mask,
    )
    fixed_categorical = parse_fixed_categorical(args.fixed_categorical)
    if args.categorical_start_mode == "section4_posterior" and not fixed_categorical:
        print(
            "No fixed categorical equalities were supplied; Section 4 q_C(t) is "
            "undefined, so sampling starts from t=1 as usual"
        )
    if args.num_timesteps is not None:
        if args.num_timesteps < 2:
            raise ValueError("num-timesteps must be at least 2")
        runtime.diffusion.num_timesteps = args.num_timesteps

    samples = runtime.diffusion.sample_all(
        args.num_samples,
        min(args.batch_size, args.num_samples),
        fixed_categorical=fixed_categorical,
        categorical_start_mode=args.categorical_start_mode,
    )
    query = NumericalBoxQuery.from_dict(metadata["query"])
    normalized_joint_hit_rate = (
        query.contains(
            samples[:, :d_numerical],
            query_active_mask.cpu(),
        ).float().mean().item()
    )
    lower = query.lower[None, :]
    upper = query.upper[None, :]
    per_value_satisfied_all = (
        (samples[:, :d_numerical] >= lower)
        & (samples[:, :d_numerical] <= upper)
    )
    per_column_hit_rate = per_value_satisfied_all.float().mean(dim=0)

    syn_num, syn_cat, syn_target = split_num_cat_target(
        samples,
        runtime.info,
        runtime.dataset.num_inverse,
        runtime.dataset.int_inverse,
        runtime.dataset.cat_inverse,
    )
    frame = recover_data(syn_num, syn_cat, syn_target, runtime.info)
    index_to_name = {
        int(key): value for key, value in runtime.info["idx_name_mapping"].items()
    }
    frame.rename(columns=index_to_name, inplace=True)

    column_specs = metadata["query"].get("columns", [])
    if len(column_specs) != d_numerical:
        raise ValueError(
            "saved query does not contain raw bounds for every numerical column: "
            f"expected {d_numerical}, found {len(column_specs)}"
        )
    active_column_specs = [column_specs[index] for index in active_columns]
    raw_per_value_satisfied, raw_rows_satisfied = raw_constraint_hits(
        frame,
        active_column_specs,
    )
    raw_per_column_hit_rate = raw_per_value_satisfied.mean(axis=0)
    raw_joint_hit_rate = raw_rows_satisfied.mean()

    output = Path(
        args.output
        or f"conditional_samples/{dataname}/doob_h_partial_fixed_box.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    active_query = {
        **metadata["query"],
        "constraint_id": (
            f"{metadata['query'].get('constraint_id', 'fixed_box')}"
            f"_active_{'_'.join(str(index) for index in active_columns)}"
        ),
        "columns": active_column_specs,
        "query_active_mask": query_active_mask.int().cpu().tolist(),
        "active_numerical_columns": active_columns,
    }
    active_query_output = output.with_suffix(".query.json")
    with open(active_query_output, "w", encoding="utf-8") as stream:
        json.dump(active_query, stream, indent=2)

    training_partial_hit_rate = query.contains(
        runtime.dataset.X[:, :d_numerical].float(),
        query_active_mask.cpu(),
    ).float().mean().item()
    constraint_report = {
        "constraint_id": metadata["query"].get("constraint_id"),
        "num_samples": len(frame),
        "evaluation_space": "raw generated table",
        "joint_hit_rate": float(raw_joint_hit_rate),
        "raw_joint_hit_rate": float(raw_joint_hit_rate),
        "normalized_joint_hit_rate": normalized_joint_hit_rate,
        "training_partial_hit_rate": training_partial_hit_rate,
        "all_rows_satisfy": bool(raw_rows_satisfied.all()),
        "query_active_mask": query_active_mask.int().cpu().tolist(),
        "active_numerical_columns": active_columns,
        "categorical_doob_transform": bool(
            metadata.get("training", {}).get("categorical_doob_transform", False)
        ),
        "categorical_start_mode": args.categorical_start_mode,
        "fixed_categorical_model_indices": {
            str(column): value for column, value in fixed_categorical.items()
        },
        "section4_posterior_start_used": bool(
            args.categorical_start_mode == "section4_posterior"
            and fixed_categorical
        ),
        "per_column": [],
    }
    for active_position, index in enumerate(active_columns):
        normalized_column_hit_rate = per_column_hit_rate[index].item()
        saved_column = column_specs[index]
        constraint_report["per_column"].append(
            {
                "model_index": index,
                "name": saved_column.get("name", f"numerical_{index}"),
                "active": True,
                "hit_rate": float(raw_per_column_hit_rate[active_position]),
                "raw_hit_rate": float(raw_per_column_hit_rate[active_position]),
                "normalized_hit_rate": normalized_column_hit_rate,
                "normalized_lower": float(query.lower[index]),
                "normalized_upper": float(query.upper[index]),
                "raw_lower": saved_column.get("raw_lower"),
                "raw_upper": saved_column.get("raw_upper"),
            }
        )
    report_output = output.with_suffix(".constraints.json")
    with open(report_output, "w", encoding="utf-8") as stream:
        json.dump(constraint_report, stream, indent=2)

    print(f"Generated {len(frame)} conditional rows")
    print(f"Active numerical model columns: {active_columns}")
    print(f"Raw-space numerical query hit rate: {raw_joint_hit_rate:.2%}")
    print(f"Normalized-space diagnostic hit rate: {normalized_joint_hit_rate:.2%}")
    print(f"Training-data partial-query hit rate: {training_partial_hit_rate:.2%}")
    print(f"Saved {output}")
    print(f"Saved active query {active_query_output}")
    print(f"Saved constraint report {report_output}")


if __name__ == "__main__":
    main()
