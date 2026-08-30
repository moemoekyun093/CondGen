"""Train structured Doob guides over a suite of interval/set queries."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from tabdiff.doob_h_runtime import (
    frozen_ft_tokenizer,
    load_doob_runtime,
    resolve_base_checkpoint,
)
from tabdiff.doob_query_curriculum import (
    BUCKET_ORDER,
    QueryCurriculumSampler,
    parse_bucket_probabilities,
)
from tabdiff.doob_query_masking import (
    eligible_indices_for_predicate_mask,
    mask_query_kwargs,
    predicate_hit_matrix,
    sample_predicate_mask,
)
from tabdiff.doob_query_suite import _model_column_names, load_structured_query_suite
from tabdiff.query_split import load_query_split
from tabdiff.models.doob_h_transform import (
    StructuredCategoricalHTransformGuide,
    StructuredNumericalHScoreGuide,
    categorical_candidate_log_h,
    guided_categorical_log_probs,
    sample_conditional_batch,
)
from train_doob_h import corrupt_mixed_state
from utils_train import update_ema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", default="shoppers")
    parser.add_argument("--base-ckpt", default=None)
    parser.add_argument("--base-exp-name", default="ft_periodic_seed0")
    parser.add_argument("--query-dir", default="data90/shoppers/queries_full")
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument(
        "--query-split-manifest",
        default=None,
        help="Optional manifest defining disjoint train/test query-id partitions",
    )
    parser.add_argument(
        "--query-split",
        choices=("train", "test"),
        default=None,
        help="Partition to load from --query-split-manifest",
    )
    parser.add_argument(
        "--target-band",
        type=float,
        default=None,
        help="Train only accepted queries with this target selectivity band",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--gradient-loss-weight", type=float, default=1.0)
    parser.add_argument("--categorical-loss-weight", type=float, default=1.0)
    parser.add_argument("--h-candidate-batch-size", type=int, default=16384)
    parser.add_argument("--d-token", type=int, default=48)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--factor", type=float, default=2.0)
    parser.add_argument("--bound-embedding-dim", type=int, default=8)
    parser.add_argument("--active-embedding-dim", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-warmup", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--query-sampling-mode",
        choices=("curriculum", "uniform"),
        default="curriculum",
        help="Hybrid selectivity curriculum or legacy uniform-over-query sampling",
    )
    parser.add_argument("--curriculum-warmup-steps", type=int, default=2000)
    parser.add_argument("--curriculum-transition-steps", type=int, default=4000)
    parser.add_argument(
        "--curriculum-warmup-probabilities",
        default="0.70,0.25,0.05",
        help="Broad,medium,tight probabilities during warm-up",
    )
    parser.add_argument(
        "--curriculum-final-probabilities",
        default="0.25,0.35,0.40",
        help="Broad,medium,tight probabilities in the final mixture",
    )
    parser.add_argument("--curriculum-tight-max-band", type=float, default=0.01)
    parser.add_argument("--curriculum-broad-min-band", type=float, default=0.10)
    parser.add_argument(
        "--curriculum-selectivity-source",
        choices=("target_band", "realized_train"),
        default="target_band",
        help="Stratify by requested target or measured training selectivity",
    )
    parser.add_argument(
        "--curriculum-reference-metadata",
        default=None,
        help="Reuse the selectivity schedule stored in an earlier metadata.json",
    )
    parser.add_argument(
        "--predicate-mask-mode",
        choices=("full", "mixed"),
        default="full",
        help="Legacy full queries or shared per-step predicate subsets",
    )
    parser.add_argument("--random-predicate-active-probability", type=float, default=0.5)
    parser.add_argument("--all-active-query-probability", type=float, default=0.1)
    parser.add_argument("--all-inactive-query-probability", type=float, default=0.1)
    return parser.parse_args()


def apply_reference_curriculum(args):
    if args.curriculum_reference_metadata is None:
        return
    with open(args.curriculum_reference_metadata, "r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    training = metadata.get("training", {})
    keys = (
        "query_sampling_mode",
        "curriculum_warmup_steps",
        "curriculum_transition_steps",
        "curriculum_warmup_probabilities",
        "curriculum_final_probabilities",
        "curriculum_tight_max_band",
        "curriculum_broad_min_band",
    )
    missing = [key for key in keys if key not in training]
    if missing:
        raise ValueError(
            "reference metadata is missing curriculum fields: " + ", ".join(missing)
        )
    for key in keys:
        setattr(args, key, training[key])


def expand_query(query, batch_size: int, device, dtype):
    return {
        name: value[None, :].expand(batch_size, -1)
        for name, value in query.model_kwargs(device, dtype).items()
    }


def structured_joint_batch(
    runtime,
    numerical_guide,
    categorical_guide,
    x0,
    query_kwargs,
    candidate_batch_size,
):
    batch_size = len(x0)
    d_numerical = runtime.dataset.d_numerical
    t = torch.rand(batch_size, device=x0.device, dtype=x0.dtype)
    (
        x_num_t,
        x_cat_t,
        x_cat_t_soft,
        sigma_num,
        sigma_cat,
        dsigma_cat,
    ) = corrupt_mixed_state(runtime, x0, t)
    if runtime.diffusion._denoise_fn.precond:
        sigma_data = runtime.diffusion._denoise_fn.denoise_fn_D.sigma_data
        state_numerical_scale = (sigma_data**2 + sigma_num**2).rsqrt()
    else:
        state_numerical_scale = torch.ones_like(sigma_num)
    guide_kwargs = {
        **query_kwargs,
        "state_numerical_scale": state_numerical_scale,
    }
    with torch.no_grad():
        base_denoised, base_raw_logits = runtime.diffusion._denoise_fn(
            x_num_t, x_cat_t_soft, t, sigma=sigma_num
        )
        numerical_target = x0[:, :d_numerical] - base_denoised
        base_log_probs = runtime.diffusion._subs_parameterization(
            base_raw_logits, x_cat_t
        )

    numerical_prediction = numerical_guide(
        x_num_t, x_cat_t_soft, t, **guide_kwargs
    )
    gradient_loss = F.mse_loss(numerical_prediction, numerical_target)
    candidate_log_h = categorical_candidate_log_h(
        categorical_guide,
        x_num_t,
        x_cat_t,
        t,
        runtime.diffusion.num_classes,
        runtime.diffusion.mask_index,
        runtime.diffusion.to_one_hot,
        candidate_batch_size=candidate_batch_size,
        **guide_kwargs,
    )
    guided_log_probs = guided_categorical_log_probs(base_log_probs, candidate_log_h)
    categorical_loss = runtime.diffusion._absorbed_closs(
        guided_log_probs,
        x0[:, d_numerical:].long(),
        sigma_cat,
        dsigma_cat,
    ).mean()
    return gradient_loss, categorical_loss


def save_checkpoint(path, numerical, categorical, metadata, step, loss, ema):
    torch.save(
        {
            "architecture": "structured_query_v1",
            "numerical_guide": numerical.state_dict(),
            "categorical_guide": categorical.state_dict(),
            "metadata": metadata,
            "step": step,
            "training_loss": loss,
            "ema": ema,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    apply_reference_curriculum(args)
    if (args.query_split_manifest is None) != (args.query_split is None):
        raise ValueError(
            "--query-split-manifest and --query-split must be supplied together"
        )
    if args.query_split_manifest is not None and args.query_id:
        raise ValueError("query-id cannot be combined with a query split manifest")
    if min(args.steps, args.batch_size, args.h_candidate_batch_size) <= 0:
        raise ValueError("steps and batch sizes must be positive")
    if args.predicate_mask_mode == "mixed":
        mask_probabilities = (
            args.random_predicate_active_probability,
            args.all_active_query_probability,
            args.all_inactive_query_probability,
        )
        if any(value < 0 or value > 1 for value in mask_probabilities):
            raise ValueError("predicate-mask probabilities must lie in [0, 1]")
        if sum(mask_probabilities[1:]) > 1:
            raise ValueError("all-active and all-inactive probabilities exceed one")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    base_checkpoint = resolve_base_checkpoint(
        args.dataname, args.base_ckpt, args.base_exp_name
    )
    runtime = load_doob_runtime(args.dataname, base_checkpoint, device)
    tokenizer = frozen_ft_tokenizer(runtime)
    selected_query_ids = args.query_id or None
    if args.query_split_manifest is not None:
        selected_query_ids = load_query_split(
            args.query_split_manifest,
            args.query_split,
        )
    queries = load_structured_query_suite(
        args.query_dir,
        runtime,
        query_ids=selected_query_ids,
        target_band=args.target_band,
    )
    if selected_query_ids is not None:
        loaded_ids = {query.query_id for query in queries}
        missing_ids = sorted(set(selected_query_ids) - loaded_ids)
        if missing_ids:
            raise ValueError(
                "query selection contains ids absent from the accepted suite: "
                f"{missing_ids[:5]}"
            )
    numerical_names, categorical_names = _model_column_names(runtime.info)
    core_indices = None
    query_hit_matrices = None
    if args.predicate_mask_mode == "mixed":
        core_indices_path = Path(args.query_dir).parent / "splits" / "train_idx.npy"
        if not core_indices_path.is_file():
            raise FileNotFoundError(
                f"query-suite core split is missing: {core_indices_path}"
            )
        core_indices = np.load(core_indices_path).astype(np.int64)
        real_path = Path("synthetic") / args.dataname / "real.csv"
        if not real_path.is_file():
            raise FileNotFoundError(f"raw training table is missing: {real_path}")
        real_frame = pd.read_csv(real_path)
        if len(real_frame) != len(runtime.dataset):
            raise ValueError(
                "raw real table and transformed training tensor are misaligned"
            )
        query_hit_matrices = {
            query.query_id: predicate_hit_matrix(real_frame, query.specification)
            for query in queries
        }
    curriculum = None
    if args.query_sampling_mode == "curriculum":
        warmup_probabilities = parse_bucket_probabilities(
            args.curriculum_warmup_probabilities
        )
        final_probabilities = parse_bucket_probabilities(
            args.curriculum_final_probabilities
        )
        curriculum = QueryCurriculumSampler(
            queries,
            total_steps=args.steps,
            warmup_steps=args.curriculum_warmup_steps,
            transition_steps=args.curriculum_transition_steps,
            warmup_probabilities=warmup_probabilities,
            final_probabilities=final_probabilities,
            tight_max_band=args.curriculum_tight_max_band,
            broad_min_band=args.curriculum_broad_min_band,
            selectivity_source=args.curriculum_selectivity_source,
        )
    x_all = runtime.dataset.X.float().to(device)
    architecture = {
        "d_numerical": runtime.dataset.d_numerical,
        "categories": (runtime.dataset.categories + 1).tolist(),
        "base_d_token": int(tokenizer.d_token),
        "d_token": args.d_token,
        "num_layers": args.num_layers,
        "n_head": args.n_head,
        "factor": args.factor,
        "bound_embedding_dim": args.bound_embedding_dim,
        "active_embedding_dim": args.active_embedding_dim,
    }
    numerical = StructuredNumericalHScoreGuide(
        base_tokenizer=tokenizer, **architecture
    ).to(device)
    categorical = StructuredCategoricalHTransformGuide(
        base_tokenizer=tokenizer, **architecture
    ).to(device)
    ema_numerical = StructuredNumericalHScoreGuide(
        base_tokenizer=tokenizer, **architecture
    ).to(device)
    ema_categorical = StructuredCategoricalHTransformGuide(
        base_tokenizer=tokenizer, **architecture
    ).to(device)
    ema_numerical.load_state_dict(numerical.state_dict())
    ema_categorical.load_state_dict(categorical.state_dict())
    for model in (ema_numerical, ema_categorical):
        for parameter in model.parameters():
            parameter.detach_()

    optimizer = torch.optim.AdamW(
        list(numerical.parameters()) + list(categorical.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataname": args.dataname,
        "base_checkpoint": runtime.checkpoint_path,
        "architecture": "structured_query_v1",
        "numerical_guide": {**architecture, "output_kind": "numerical"},
        "categorical_guide": {**architecture, "output_kind": "categorical"},
        "query_suite": {
            "directory": str(Path(args.query_dir)),
            "query_ids": [query.query_id for query in queries],
            "split_manifest": args.query_split_manifest,
            "split": args.query_split,
            "target_band": args.target_band,
            "sampling": (
                "curriculum_bucket_then_uniform_band_then_uniform_query_then_"
                "uniform_support_with_replacement"
                if curriculum is not None
                else "uniform_over_queries_then_uniform_support_with_replacement"
            ),
            "curriculum_buckets": (
                curriculum.summary() if curriculum is not None else None
            ),
            "predicate_masking": {
                "mode": args.predicate_mask_mode,
                "all_active_probability": args.all_active_query_probability,
                "all_inactive_probability": args.all_inactive_query_probability,
                "random_subset_probability": (
                    1.0
                    - args.all_active_query_probability
                    - args.all_inactive_query_probability
                ),
                "random_predicate_active_probability": (
                    args.random_predicate_active_probability
                ),
                "shared_by_optimizer_step_batch": True,
            },
        },
        "objective": {
            "numerical": "unchanged direct sigma^2 grad log-h correction MSE",
            "categorical": "unchanged conditional Generator Matching absorbed loss",
            "nested_query_auxiliary_loss": False,
        },
        "checkpoint_selection": (
            "minimum EMA-smoothed training loss during final curriculum mixture"
            if curriculum is not None
            else "minimum EMA-smoothed training loss after checkpoint warmup"
        ),
        "training": vars(args),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    base_count = sum(p.numel() for p in runtime.diffusion._denoise_fn.parameters())
    guide_count = sum(p.numel() for p in numerical.parameters()) + sum(
        p.numel() for p in categorical.parameters()
    )
    print(f"Loaded {len(queries)} accepted structured queries")
    supports = np.asarray([len(query.eligible_indices) for query in queries])
    arities = np.asarray([len(query.specification["predicates"]) for query in queries])
    print(
        "Query support min/median/max: "
        f"{supports.min()}/{int(np.median(supports))}/{supports.max()}"
    )
    print(
        f"Query arity min/max: {arities.min()}/{arities.max()} | "
        f"curriculum selectivity: {args.curriculum_selectivity_source}"
    )
    if curriculum is None:
        print("Query sampling: legacy uniform over queries")
    else:
        print(f"Query curriculum buckets: {curriculum.summary()}")
        print(
            "Warm-up probabilities (broad,medium,tight): "
            f"{curriculum.warmup_probabilities} for "
            f"{curriculum.warmup_steps} steps"
        )
        print(
            "Linear transition to final probabilities: "
            f"{curriculum.final_probabilities} over "
            f"{curriculum.transition_steps} steps"
        )
        print(
            f"Final stratified mixture begins at step {curriculum.final_phase_start}"
        )
    print("Endpoint sampling: uniform satisfying rows, with replacement")
    if args.predicate_mask_mode == "mixed":
        print(
            "Predicate masking: "
            f"all-active={args.all_active_query_probability:.0%}, "
            f"all-inactive={args.all_inactive_query_probability:.0%}, "
            "random-subset="
            f"{1-args.all_active_query_probability-args.all_inactive_query_probability:.0%} "
            f"with per-predicate p={args.random_predicate_active_probability:g}"
        )
    else:
        print("Predicate masking: full query on every optimizer step")
    print("Frozen base tokenizer and ReLU category lookups shared by both guides")
    print(f"Combined trainable guide parameters: {guide_count:,} ({guide_count/base_count:.2%} of base)")
    print("No containment regularizer or BCE objective")

    best = float("inf")
    smoothed_loss = None
    query_counts = {query.query_id: 0 for query in queries}
    band_counts = {}
    bucket_counts = {bucket: 0 for bucket in BUCKET_ORDER}
    phase_counts = {"warmup": 0, "transition": 0, "final_mixture": 0, "uniform": 0}
    mask_counts = {"all_active": 0, "all_inactive": 0, "random_subset": 0}
    arity_counts = {}
    best_eligible_step = args.checkpoint_warmup
    if curriculum is not None:
        # Losses from different curriculum phases are not directly comparable.
        # Select the best checkpoint only under the final target mixture.
        best_eligible_step = max(best_eligible_step, curriculum.final_phase_start)
    for step in range(1, args.steps + 1):
        if curriculum is None:
            query = random.choice(queries)
            sampled_band = query.specification["target_band"]
            bucket = "uniform"
            phase = "uniform"
        else:
            query, bucket, sampled_band, phase = curriculum.sample(step, random)
        query_counts[query.query_id] += 1
        band_key = str(sampled_band)
        band_counts[band_key] = band_counts.get(band_key, 0) + 1
        if bucket in bucket_counts:
            bucket_counts[bucket] += 1
        phase_counts[phase] += 1
        if args.predicate_mask_mode == "mixed":
            predicate_mask, mask_kind = sample_predicate_mask(
                len(query.specification["predicates"]),
                device=device,
                random_active_probability=args.random_predicate_active_probability,
                all_active_probability=args.all_active_query_probability,
                all_inactive_probability=args.all_inactive_query_probability,
            )
            eligible = eligible_indices_for_predicate_mask(
                query_hit_matrices[query.query_id],
                core_indices,
                predicate_mask,
            ).to(device)
        else:
            predicate_mask = torch.ones(
                len(query.specification["predicates"]),
                dtype=torch.bool,
                device=device,
            )
            mask_kind = "all_active"
            # Preserve the original full-query training path exactly.
            eligible = query.eligible_indices.to(device)
        mask_counts[mask_kind] += 1
        active_arity = int(predicate_mask.sum().item())
        arity_counts[str(active_arity)] = arity_counts.get(str(active_arity), 0) + 1
        x0 = sample_conditional_batch(x_all, eligible, args.batch_size)
        single_query_kwargs = mask_query_kwargs(
            query.model_kwargs(device, x_all.dtype),
            query.specification,
            predicate_mask,
            numerical_names,
            categorical_names,
        )
        query_kwargs = {
            name: value[None, :].expand(args.batch_size, -1)
            for name, value in single_query_kwargs.items()
        }
        optimizer.zero_grad(set_to_none=True)
        gradient_loss, categorical_loss = structured_joint_batch(
            runtime,
            numerical,
            categorical,
            x0,
            query_kwargs,
            args.h_candidate_batch_size,
        )
        loss = (
            args.gradient_loss_weight * gradient_loss
            + args.categorical_loss_weight * categorical_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        update_ema(ema_numerical.parameters(), numerical.parameters(), args.ema_decay)
        update_ema(ema_categorical.parameters(), categorical.parameters(), args.ema_decay)

        loss_value = loss.item()
        smoothed_loss = (
            loss_value
            if smoothed_loss is None
            else 0.99 * smoothed_loss + 0.01 * loss_value
        )
        if step >= best_eligible_step and smoothed_loss < best:
            best = smoothed_loss
            save_checkpoint(
                output_dir / "best_guide.pt",
                ema_numerical,
                ema_categorical,
                metadata,
                step,
                smoothed_loss,
                True,
            )
        if step % args.checkpoint_every == 0:
            save_checkpoint(
                output_dir / f"guide_{step}.pt",
                ema_numerical,
                ema_categorical,
                metadata,
                step,
                loss_value,
                True,
            )
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step:05d} query={query.query_id} support={len(eligible)} "
                f"mask={mask_kind} arity={active_arity} "
                f"phase={phase} bucket={bucket} band={sampled_band} "
                f"total={loss_value:.6f} gradient_mse={gradient_loss.item():.6f} "
                f"categorical_loss={categorical_loss.item():.6f} "
                f"smoothed_total={smoothed_loss:.6f}",
                flush=True,
            )

    save_checkpoint(
        output_dir / "last_guide.pt",
        ema_numerical,
        ema_categorical,
        metadata,
        args.steps,
        loss_value,
        True,
    )
    if not (output_dir / "best_guide.pt").is_file():
        save_checkpoint(
            output_dir / "best_guide.pt",
            ema_numerical,
            ema_categorical,
            metadata,
            args.steps,
            smoothed_loss,
            True,
        )
        best = smoothed_loss
    with (output_dir / "query_counts.json").open("w", encoding="utf-8") as stream:
        json.dump(query_counts, stream, indent=2)
    with (output_dir / "sampling_counts.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "queries": query_counts,
                "bands": band_counts,
                "buckets": bucket_counts,
                "phases": phase_counts,
                "predicate_masks": mask_counts,
                "active_arities": arity_counts,
            },
            stream,
            indent=2,
        )
    print(f"Finished {args.steps} optimizer steps; best training loss={best:.6f}")


if __name__ == "__main__":
    main()
