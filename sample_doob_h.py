"""Generate samples with a trained fixed-query numerical Doob h-transform."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from tabdiff.doob_h_runtime import load_doob_runtime, resolve_base_checkpoint
from tabdiff.models.doob_h_transform import NumericalBoxQuery, NumericalDoobHGuide
from tabdiff.trainer import recover_data, split_num_cat_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide-ckpt", required=True)
    parser.add_argument("--base-ckpt", default=None)
    parser.add_argument("--base-exp-name", default="learnable_schedule")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-correction", type=float, default=5.0)
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


def raw_constraint_hits(frame, column_specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate saved numerical intervals in the final, user-facing raw space."""
    hits = []
    for spec in column_specs:
        name = spec.get("name")
        lower = spec.get("raw_lower")
        upper = spec.get("raw_upper")
        if name not in frame.columns or lower is None or upper is None:
            raise ValueError(f"raw constraint metadata is incomplete for column {name!r}")
        values = frame[name].to_numpy(dtype=np.float64)
        scale = max(1.0, abs(float(lower)), abs(float(upper)))
        tolerance = 1e-7 * scale
        hits.append(
            (values >= float(lower) - tolerance)
            & (values <= float(upper) + tolerance)
        )
    per_value = np.stack(hits, axis=1)
    return per_value, per_value.all(axis=1)


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

    guide = NumericalDoobHGuide(**metadata["guide"]).to(device)
    guide.load_state_dict(guide_state["guide"])
    guide.eval()
    runtime.diffusion.set_numerical_h_guide(
        guide,
        strength=1.0,
        max_correction=args.max_correction,
    )
    if args.num_timesteps is not None:
        if args.num_timesteps < 2:
            raise ValueError("num-timesteps must be at least 2")
        runtime.diffusion.num_timesteps = args.num_timesteps

    samples = runtime.diffusion.sample_all(
        args.num_samples,
        min(args.batch_size, args.num_samples),
    )
    query = NumericalBoxQuery.from_dict(metadata["query"])
    d_numerical = runtime.dataset.d_numerical
    normalized_joint_hit_rate = (
        query.contains(samples[:, :d_numerical]).float().mean().item()
    )
    lower = query.lower[None, :]
    upper = query.upper[None, :]
    per_value_satisfied = (
        (samples[:, :d_numerical] >= lower)
        & (samples[:, :d_numerical] <= upper)
    )
    per_column_hit_rate = per_value_satisfied.float().mean(dim=0)

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
    raw_per_value_satisfied, raw_rows_satisfied = raw_constraint_hits(
        frame,
        column_specs,
    )
    raw_per_column_hit_rate = raw_per_value_satisfied.mean(axis=0)
    raw_joint_hit_rate = raw_rows_satisfied.mean()

    output = Path(
        args.output
        or f"conditional_samples/{dataname}/doob_h_fixed_box.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    constraint_report = {
        "constraint_id": metadata["query"].get("constraint_id"),
        "num_samples": len(frame),
        "evaluation_space": "raw generated table",
        "joint_hit_rate": float(raw_joint_hit_rate),
        "raw_joint_hit_rate": float(raw_joint_hit_rate),
        "normalized_joint_hit_rate": normalized_joint_hit_rate,
        "all_rows_satisfy": bool(raw_rows_satisfied.all()),
        "per_column": [],
    }
    for index, normalized_column_hit_rate in enumerate(per_column_hit_rate.tolist()):
        saved_column = column_specs[index] if index < len(column_specs) else {}
        constraint_report["per_column"].append(
            {
                "model_index": index,
                "name": saved_column.get("name", f"numerical_{index}"),
                "hit_rate": float(raw_per_column_hit_rate[index]),
                "raw_hit_rate": float(raw_per_column_hit_rate[index]),
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
    print(f"Raw-space numerical query hit rate: {raw_joint_hit_rate:.2%}")
    print(f"Normalized-space diagnostic hit rate: {normalized_joint_hit_rate:.2%}")
    print(f"Training-data query hit rate: {metadata['query']['training_hit_rate']:.2%}")
    print(f"Saved {output}")
    print(f"Saved constraint report {report_output}")


if __name__ == "__main__":
    main()
