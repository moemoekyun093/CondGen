"""Sample one full-arity interval/set query with structured Doob guides."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from tabdiff.doob_h_runtime import (
    frozen_ft_tokenizer,
    load_doob_runtime,
    resolve_base_checkpoint,
)
from tabdiff.doob_h_evaluation import raw_constraint_report
from tabdiff.doob_query_masking import (
    mask_query_kwargs,
    masked_query_specification,
    parse_predicate_mask,
)
from tabdiff.doob_query_suite import load_structured_query_suite
from tabdiff.doob_query_suite import _model_column_names
from tabdiff.models.doob_h_transform import (
    StructuredCategoricalHTransformGuide,
    StructuredNumericalHScoreGuide,
)
from tabdiff.trainer import recover_data, split_num_cat_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide-ckpt", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--base-ckpt", default=None)
    parser.add_argument("--base-exp-name", default="ft_periodic_seed0")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-correction", type=float, default=5.0)
    parser.add_argument("--max-log-h-ratio", type=float, default=10.0)
    parser.add_argument("--h-candidate-batch-size", type=int, default=65536)
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--active-columns",
        default=None,
        help="Comma-separated constrained column names, or 'all'/'none'",
    )
    mask_group.add_argument(
        "--predicate-mask",
        default=None,
        help="Binary mask in query predicate order, e.g. 101001",
    )
    parser.add_argument(
        "--diagnose-guidance",
        action="store_true",
        help="Save raw pre-clipping numerical correction statistics",
    )
    return parser.parse_args()


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    if min(args.num_samples, args.batch_size, args.h_candidate_batch_size) <= 0:
        raise ValueError("sample and batch sizes must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    state = torch_load(args.guide_ckpt, device)
    if state.get("architecture") != "structured_query_v1":
        raise ValueError("checkpoint is not a structured-query guide")
    metadata = state["metadata"]
    requested_base = args.base_ckpt
    saved_base = metadata.get("base_checkpoint")
    if requested_base is None and saved_base and os.path.isfile(saved_base):
        requested_base = saved_base
    base_checkpoint = resolve_base_checkpoint(
        metadata["dataname"], requested_base, args.base_exp_name
    )
    runtime = load_doob_runtime(metadata["dataname"], base_checkpoint, device)
    tokenizer = frozen_ft_tokenizer(runtime)

    query_path = Path(args.query_file)
    with query_path.open("r", encoding="utf-8") as stream:
        query_specification = json.load(stream)
    query_id = query_specification["query_id"]
    query = load_structured_query_suite(
        query_path.parent,
        runtime,
        query_ids=[query_id],
    )[0]
    predicate_mask = parse_predicate_mask(
        query.specification,
        active_columns=args.active_columns,
        predicate_mask=args.predicate_mask,
        device=device,
    )
    numerical_names, categorical_names = _model_column_names(runtime.info)
    active_query_kwargs = mask_query_kwargs(
        query.model_kwargs(device, torch.float32),
        query.specification,
        predicate_mask,
        numerical_names,
        categorical_names,
    )
    active_specification = masked_query_specification(
        query.specification,
        predicate_mask,
    )

    numerical_config = dict(metadata["numerical_guide"])
    categorical_config = dict(metadata["categorical_guide"])
    numerical_config.pop("output_kind", None)
    categorical_config.pop("output_kind", None)
    numerical = StructuredNumericalHScoreGuide(
        base_tokenizer=tokenizer, **numerical_config
    ).to(device)
    categorical = StructuredCategoricalHTransformGuide(
        base_tokenizer=tokenizer, **categorical_config
    ).to(device)
    numerical.load_state_dict(state["numerical_guide"])
    categorical.load_state_dict(state["categorical_guide"])
    numerical.eval()
    categorical.eval()
    runtime.diffusion.set_doob_h_guides(
        numerical,
        categorical,
        strength=1.0,
        max_correction=args.max_correction,
        max_log_ratio=args.max_log_h_ratio,
        candidate_batch_size=args.h_candidate_batch_size,
        query_conditioning=active_query_kwargs,
    )
    if args.diagnose_guidance:
        runtime.diffusion.enable_numerical_h_guide_diagnostics()
    if args.num_timesteps is not None:
        if args.num_timesteps < 2:
            raise ValueError("num-timesteps must be at least 2")
        runtime.diffusion.num_timesteps = args.num_timesteps

    print(f"Query: {query_id}")
    print(
        "Active predicates: "
        f"{int(predicate_mask.sum().item())}/{predicate_mask.numel()} "
        f"({active_specification['active_columns']})"
    )
    print("Numerical constraints: clean intervals with monotone endpoint encodings")
    print("Categorical constraints: sums of ReLU-frozen base category lookups")
    print("Categorical start: ordinary t=1 masked prior; no equality shortcut")
    samples = runtime.diffusion.sample_all(
        args.num_samples,
        min(args.batch_size, args.num_samples),
        fixed_categorical={},
        categorical_start_mode="full",
    )
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
    constraint_report, joint = raw_constraint_report(frame, active_specification)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    with output.with_suffix(".query.json").open("w", encoding="utf-8") as stream:
        json.dump(active_specification, stream, indent=2)
    report = {
        **constraint_report,
        "query_id": query_id,
        "raw_joint_hit_rate": float(joint.mean()),
        "raw_joint_violation_rate": float(1.0 - joint.mean()),
    }
    with output.with_suffix(".constraints.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    if args.diagnose_guidance:
        guidance_report = runtime.diffusion.numerical_h_guide_diagnostics()
        numerical_names, _ = _model_column_names(runtime.info)
        for column, name in zip(guidance_report["per_column"], numerical_names):
            column["name"] = name
        guidance_report.update(
            {
                "query_id": query_id,
                "max_correction": args.max_correction,
                "guidance_strength": 1.0,
                "correction_kind": "direct denoiser correction before coordinate-wise clipping",
            }
        )
        with output.with_suffix(".guidance.json").open("w", encoding="utf-8") as stream:
            json.dump(guidance_report, stream, indent=2)
        print(
            "Numerical correction clipping rate: "
            f"{guidance_report['overall_clip_rate']:.2%}"
        )
    print(f"Raw full-query hit rate: {joint.mean():.2%}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
