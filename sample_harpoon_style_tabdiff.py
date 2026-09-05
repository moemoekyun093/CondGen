"""Sample a structured query using HARPOON-style guidance on frozen TabDiff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tabdiff.doob_h_evaluation import raw_constraint_report, raw_modality_constraint_report
from tabdiff.doob_h_runtime import load_doob_runtime, resolve_base_checkpoint
from tabdiff.doob_query_suite import load_structured_query_suite
from tabdiff.trainer import recover_data, split_num_cat_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", default="shoppers")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--base-ckpt", default=None)
    parser.add_argument("--base-exp-name", default="ft_periodic_seed0")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sample_query(
    runtime,
    query,
    *,
    num_samples: int,
    batch_size: int,
    eta: float,
    seed: int,
):
    """Generate one query without rebuilding the frozen TabDiff runtime."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = runtime.diffusion.device
    runtime.diffusion.set_harpoon_style_guidance(
        query.model_kwargs(device, torch.float32),
        strength=eta,
    )
    samples = runtime.diffusion.sample_all(
        num_samples,
        min(batch_size, num_samples),
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
    return frame


def save_query_result(
    frame,
    query,
    output: str | Path,
    *,
    eta: float,
    num_timesteps: int,
    base_checkpoint: str,
) -> float:
    """Write samples and diagnostics, returning the raw joint hit rate."""
    constraint_report, joint = raw_constraint_report(frame, query.specification)
    modality_report = raw_modality_constraint_report(frame, query.specification)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    with output.with_suffix(".query.json").open("w", encoding="utf-8") as stream:
        json.dump(query.specification, stream, indent=2)
    report = {
        **constraint_report,
        **modality_report,
        "query_id": query.query_id,
        "raw_joint_hit_rate": float(joint.mean()),
        "raw_joint_violation_rate": float(1.0 - joint.mean()),
        "method": "harpoon_style_guidance_on_tabdiff",
        "manifold_guidance": "grad_x_t L_inf(Q_t(x_t), query) through frozen denoiser",
        "guidance_eta": eta,
        "num_timesteps": num_timesteps,
        "base_checkpoint": base_checkpoint,
    }
    with output.with_suffix(".constraints.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return float(joint.mean())


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample and batch sizes must be positive")
    if args.num_timesteps < 2:
        raise ValueError("num-timesteps must be at least 2")
    if args.eta <= 0:
        raise ValueError("eta must be positive")

    device = torch.device(args.device)
    base_checkpoint = resolve_base_checkpoint(
        args.dataname, args.base_ckpt, args.base_exp_name
    )
    runtime = load_doob_runtime(args.dataname, base_checkpoint, device)
    query_path = Path(args.query_file)
    with query_path.open("r", encoding="utf-8") as stream:
        specification = json.load(stream)
    query = load_structured_query_suite(
        query_path.parent,
        runtime,
        query_ids=[specification["query_id"]],
    )[0]

    runtime.diffusion.num_timesteps = args.num_timesteps
    print(f"Query: {query.query_id}")
    print("Method: HARPOON-style manifold guidance on frozen TabDiff")
    print("Constraint loss: squared ReLU intervals plus categorical set-mass loss")
    print("Manifold term: full constraint gradient through the frozen dirty estimate Q_t")
    print(f"Guidance eta: {args.eta}")
    print(f"Reverse steps: {args.num_timesteps}")
    frame = sample_query(
        runtime,
        query,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        eta=args.eta,
        seed=args.seed,
    )
    hit_rate = save_query_result(
        frame,
        query,
        args.output,
        eta=args.eta,
        num_timesteps=args.num_timesteps,
        base_checkpoint=base_checkpoint,
    )
    print(f"Raw full-query hit rate: {hit_rate:.2%}")
    output = Path(args.output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
