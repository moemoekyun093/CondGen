"""Train structured Doob guides over a suite of interval/set queries."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tabdiff.doob_h_runtime import (
    frozen_ft_tokenizer,
    load_doob_runtime,
    resolve_base_checkpoint,
)
from tabdiff.doob_query_suite import load_structured_query_suite
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
        "--target-band",
        type=float,
        default=None,
        help="Train only accepted queries with this target selectivity band",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=6000)
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
    return parser.parse_args()


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
    if min(args.steps, args.batch_size, args.h_candidate_batch_size) <= 0:
        raise ValueError("steps and batch sizes must be positive")
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
    queries = load_structured_query_suite(
        args.query_dir,
        runtime,
        query_ids=args.query_id or None,
        target_band=args.target_band,
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
            "target_band": args.target_band,
            "sampling": "uniform_over_queries_then_uniform_support_with_replacement",
        },
        "objective": {
            "numerical": "unchanged direct sigma^2 grad log-h correction MSE",
            "categorical": "unchanged conditional Generator Matching absorbed loss",
            "nested_query_auxiliary_loss": False,
        },
        "training": vars(args),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    base_count = sum(p.numel() for p in runtime.diffusion._denoise_fn.parameters())
    guide_count = sum(p.numel() for p in numerical.parameters()) + sum(
        p.numel() for p in categorical.parameters()
    )
    print(f"Loaded {len(queries)} accepted full-arity queries")
    for query in queries:
        print(
            f"  {query.query_id}: target_band="
            f"{query.specification.get('target_band')} support={len(query.eligible_indices)}"
        )
    print("Query sampling: uniform over queries; endpoint sampling: uniform with replacement")
    print("Frozen base tokenizer and ReLU category lookups shared by both guides")
    print(f"Combined trainable guide parameters: {guide_count:,} ({guide_count/base_count:.2%} of base)")
    print("No containment regularizer or BCE objective")

    best = float("inf")
    smoothed_loss = None
    query_counts = {query.query_id: 0 for query in queries}
    for step in range(1, args.steps + 1):
        query = random.choice(queries)
        query_counts[query.query_id] += 1
        eligible = query.eligible_indices.to(device)
        x0 = sample_conditional_batch(x_all, eligible, args.batch_size)
        query_kwargs = expand_query(query, args.batch_size, device, x_all.dtype)
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
        if step >= args.checkpoint_warmup and smoothed_loss < best:
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
    print(f"Finished {args.steps} optimizer steps; best training loss={best:.6f}")


if __name__ == "__main__":
    main()
