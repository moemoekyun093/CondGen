"""Train separate numerical h-score and categorical log-h guides.

The first verification stage fixes the active vector to all ones. Conditional
endpoint rows supervise a direct ``sigma^2 grad_x log h`` predictor with the
Section 5 denoiser-difference target. A separate scalar network supplies
categorical child/current h-ratios to TabDiff's original absorbed loss. Every
endpoint is drawn uniformly from the rows satisfying the complete box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tabdiff.doob_h_runtime import load_doob_runtime, resolve_base_checkpoint
from tabdiff.models.doob_h_transform import (
    CategoricalHTransformGuide,
    NumericalBoxQuery,
    NumericalHScoreGuide,
    categorical_log_h_ratios,
)
from utils_train import update_ema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", choices=("shoppers",), default="shoppers")
    parser.add_argument("--base-ckpt", default=None)
    parser.add_argument("--base-exp-name", default="learnable_schedule")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--diagnostic-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--gradient-loss-weight", type=float, default=1.0)
    parser.add_argument("--categorical-loss-weight", type=float, default=1.0)
    parser.add_argument("--h-candidate-batch-size", type=int, default=16384)
    parser.add_argument("--diagnostic-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-warmup", type=int, default=4000)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--reduce-lr-patience", type=int, default=20)
    parser.add_argument("--lr-factor", type=float, default=0.9)
    parser.add_argument("--d-token", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--factor", type=float, default=2.0)
    parser.add_argument("--n-frequencies", type=int, default=16)
    return parser.parse_args()


def corrupt_mixed_state(runtime, x0: torch.Tensor, t: torch.Tensor):
    d_numerical = runtime.dataset.d_numerical
    x0_num = x0[:, :d_numerical]
    x0_cat = x0[:, d_numerical:].long()
    sigma_num = runtime.diffusion.num_schedule.total_noise(t[:, None])
    x_num_t = x0_num + sigma_num * torch.randn_like(x0_num)

    if x0_cat.shape[1] == 0:
        empty_cat = x0_cat
        empty_soft = x_num_t.new_zeros((len(x0), 0))
        empty_sigma = x_num_t.new_zeros((len(x0), 0))
        return (
            x_num_t,
            empty_cat,
            empty_soft,
            sigma_num,
            empty_sigma,
            empty_sigma,
        )

    sigma_cat = runtime.diffusion.cat_schedule.total_noise(t[:, None])
    dsigma_cat = runtime.diffusion.cat_schedule.rate_noise(t[:, None])
    move_chance = -torch.expm1(-sigma_cat)
    strategy = (
        "soft"
        if runtime.diffusion.cat_scheduler == "log_linear_per_column"
        else "hard"
    )
    x_cat_t, x_cat_t_soft = runtime.diffusion.q_xt(
        x0_cat,
        move_chance,
        strategy=strategy,
    )
    if strategy == "hard":
        x_cat_t_soft = runtime.diffusion.to_one_hot(x_cat_t).to(x_num_t.dtype)
    return x_num_t, x_cat_t, x_cat_t_soft, sigma_num, sigma_cat, dsigma_cat


def categorical_doob_loss(
    runtime,
    categorical_guide,
    x_num_t: torch.Tensor,
    x_cat_t: torch.Tensor,
    t: torch.Tensor,
    x0_cat: torch.Tensor,
    base_raw_logits: torch.Tensor,
    sigma_cat: torch.Tensor,
    dsigma_cat: torch.Tensor,
    active_mask: torch.Tensor,
    candidate_batch_size: int,
):
    """Reuse TabDiff's absorbed loss after multiplying by the shared h ratio."""
    if x0_cat.shape[1] == 0:
        return x_num_t.sum() * 0.0

    with torch.no_grad():
        base_log_probs = runtime.diffusion._subs_parameterization(
            base_raw_logits,
            x_cat_t,
        )

    log_ratios = categorical_log_h_ratios(
        categorical_guide,
        x_num_t,
        x_cat_t,
        t,
        runtime.diffusion.num_classes,
        runtime.diffusion.mask_index,
        runtime.diffusion.to_one_hot,
        query_active_mask=active_mask,
        candidate_batch_size=candidate_batch_size,
    )

    # Multiplication by h(child)/h(current) in probability space is addition
    # of the log-ratio here. Renormalization gives the guided token law used by
    # TabDiff's original categorical diffusion loss.
    guided_log_probs = base_log_probs + log_ratios
    guided_log_probs = guided_log_probs - torch.logsumexp(
        guided_log_probs,
        dim=-1,
        keepdim=True,
    )
    return runtime.diffusion._absorbed_closs(
        guided_log_probs,
        x0_cat,
        sigma_cat,
        dsigma_cat,
    ).mean()


def joint_batch(
    runtime,
    numerical_guide,
    categorical_guide,
    x0: torch.Tensor,
    candidate_batch_size: int,
):
    """Train continuous and categorical guidance on conditional endpoints."""
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
    active_mask = x0.new_ones((batch_size, d_numerical))
    with torch.no_grad():
        base_denoised, base_raw_logits = runtime.diffusion._denoise_fn(
            x_num_t,
            x_cat_t_soft,
            t,
            sigma=sigma_num,
        )
        target = x0[:, :d_numerical] - base_denoised
    prediction = numerical_guide(
        x_num_t,
        x_cat_t_soft,
        t,
        query_active_mask=active_mask,
    )
    gradient_loss = F.mse_loss(prediction, target)
    categorical_loss = categorical_doob_loss(
        runtime,
        categorical_guide,
        x_num_t,
        x_cat_t,
        t,
        x0[:, d_numerical:].long(),
        base_raw_logits,
        sigma_cat,
        dsigma_cat,
        active_mask,
        candidate_batch_size,
    )
    return gradient_loss, categorical_loss, prediction, target


@torch.no_grad()
def joint_diagnostic(
    runtime,
    numerical_guide,
    categorical_guide,
    x_positive,
    batch_size,
    seed,
    candidate_batch_size,
):
    numerical_guide.eval()
    categorical_guide.eval()
    cuda_devices = (
        [
            x_positive.device.index
            if x_positive.device.index is not None
            else torch.cuda.current_device()
        ]
        if x_positive.is_cuda
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        indices = torch.arange(
            min(batch_size, len(x_positive)),
            device=x_positive.device,
        )
        gradient_loss, categorical_loss, prediction, target = joint_batch(
            runtime,
            numerical_guide,
            categorical_guide,
            x_positive[indices],
            candidate_batch_size=candidate_batch_size,
        )
    numerical_guide.train()
    categorical_guide.train()
    return {
        "gradient_mse": gradient_loss.item(),
        "categorical_loss": categorical_loss.item(),
        "mean_correction_norm": prediction.norm(dim=1).mean().item(),
        "mean_target_norm": target.norm(dim=1).mean().item(),
        "finite": bool(torch.isfinite(prediction).all()),
    }


def save_checkpoint(
    path,
    numerical_guide,
    categorical_guide,
    metadata,
    epoch,
    loss,
    ema,
):
    torch.save(
        {
            "numerical_guide": numerical_guide.state_dict(),
            "categorical_guide": categorical_guide.state_dict(),
            "metadata": metadata,
            "step": epoch,
            "epoch": epoch,
            "training_mse": loss,
            "ema": ema,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if min(
        args.epochs,
        args.diagnostic_batch_size,
        args.h_candidate_batch_size,
        args.diagnostic_every,
        args.log_every,
    ) <= 0:
        raise ValueError("epochs, diagnostic batch size, and reporting intervals must be positive")
    if args.gradient_loss_weight <= 0 or args.categorical_loss_weight <= 0:
        raise ValueError("gradient and categorical loss weights must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema-decay must be in [0,1)")
    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError("lr-factor must be in (0,1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    base_checkpoint = resolve_base_checkpoint(
        args.dataname,
        args.base_ckpt,
        args.base_exp_name,
    )
    runtime = load_doob_runtime(args.dataname, base_checkpoint, device)
    x_all = runtime.dataset.X.float().to(device)
    d_numerical = runtime.dataset.d_numerical
    x_num_cpu = runtime.dataset.X[:, :d_numerical].float().contiguous()

    with open(args.query_file, "r", encoding="utf-8") as stream:
        query_spec = json.load(stream)
    if query_spec.get("dataname") != args.dataname:
        raise ValueError("query file belongs to a different dataset")
    expected_fingerprint = query_spec.get("data_sha256")
    actual_fingerprint = hashlib.sha256(x_num_cpu.numpy().tobytes()).hexdigest()
    if expected_fingerprint and expected_fingerprint != actual_fingerprint:
        raise ValueError("query file was generated from different transformed data")
    query = NumericalBoxQuery.from_dict(query_spec)
    all_active = torch.ones(d_numerical)
    event_all = query.contains(x_num_cpu, all_active)
    event_rate = event_all.float().mean().item()
    if not 0.0 < event_rate < 1.0:
        raise ValueError("all-constrained event needs both positive and negative rows")
    x_positive = x_all[event_all.to(device)]

    architecture_kwargs = {
        "d_numerical": d_numerical,
        "categories": (runtime.dataset.categories + 1).tolist(),
        "d_token": args.d_token,
        "num_layers": args.num_layers,
        "n_head": args.n_head,
        "factor": args.factor,
        "n_frequencies": args.n_frequencies,
        "freq_sigma": 0.05,
        # The all-constrained query is fixed and intentionally implicit in this
        # first verification checkpoint. Partial-mask conditioning comes later.
        "query_mask_conditioning": False,
    }
    numerical_guide = NumericalHScoreGuide(**architecture_kwargs).to(device)
    categorical_guide = CategoricalHTransformGuide(**architecture_kwargs).to(device)
    ema_numerical_guide = deepcopy(numerical_guide)
    ema_categorical_guide = deepcopy(categorical_guide)
    for parameter in ema_numerical_guide.parameters():
        parameter.detach_()
    for parameter in ema_categorical_guide.parameters():
        parameter.detach_()

    optimizer = torch.optim.AdamW(
        list(numerical_guide.parameters()) + list(categorical_guide.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.reduce_lr_patience,
    )

    output_dir = Path(
        args.output_dir
        or "tabdiff/ckpt/shoppers/ft_periodic_seed0/doob_h_all_constrained_two_guides"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataname": args.dataname,
        "base_checkpoint": runtime.checkpoint_path,
        "numerical_guide": architecture_kwargs,
        "categorical_guide": architecture_kwargs,
        "query": {
            **query_spec,
            **query.to_dict(),
            "query_active_mask": [1] * d_numerical,
            "active_numerical_columns": list(range(d_numerical)),
        },
        "objective": {
            "separate_parameterizations": True,
            "section_5_conditional_rows_only": True,
            "numerical_sampling": (
                "separate FT-periodic G_psi predicts sigma(t)^2 grad_x_num log h"
            ),
            "categorical_training": (
                "TabDiff absorbed categorical loss after multiplying frozen base "
                "token probabilities by h(child)/h(current)"
            ),
            "categorical_sampling": "R_h=R*h(child)/h(current)",
        },
        "training": {
            "mode": "all_constrained",
            "epochs": args.epochs,
            "rows_per_epoch": len(x_positive),
            "optimizer_updates": args.epochs,
            "full_conditional_pass_per_epoch": True,
            "sampling_with_replacement": False,
            "fixed_conditional_diagnostic_every": args.diagnostic_every,
            "event_rate": event_rate,
            "positive_rows": int(event_all.sum()),
            "negative_rows": int((~event_all).sum()),
            "ema_decay": args.ema_decay,
            "gradient_loss_weight": args.gradient_loss_weight,
            "categorical_loss_weight": args.categorical_loss_weight,
            "h_candidate_batch_size": args.h_candidate_batch_size,
        },
        "seed": args.seed,
    }
    with open(output_dir / "query.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    print("Training separate numerical h-score and categorical log-h guides")
    print(
        f"All-constrained event: {int(event_all.sum())}/{len(event_all)} "
        f"({event_rate:.2%})"
    )
    print(
        "Numerical guide parameters: "
        f"{sum(p.numel() for p in numerical_guide.parameters()):,}"
    )
    print(
        "Categorical guide parameters: "
        f"{sum(p.numel() for p in categorical_guide.parameters()):,}"
    )
    print(
        f"Epochs: {args.epochs}; every epoch uses all {len(x_positive)} "
        "all-constrained rows exactly once"
    )
    print("Categorical objective: TabDiff absorbed loss with scalar-h ratios")
    print("No unrestricted rows and no BCE/classification objective")
    print(f"Fixed EMA joint diagnostic every {args.diagnostic_every} epochs")
    print(f"Writing checkpoints to {output_dir}")

    best_ema = float("inf")
    last_loss = float("nan")
    last_ema_loss = float("nan")
    numerical_guide.train()
    categorical_guide.train()
    for epoch in range(1, args.epochs + 1):
        positive_indices = torch.randperm(len(x_positive), device=device)
        optimizer.zero_grad(set_to_none=True)
        gradient_loss, categorical_loss, _, _ = joint_batch(
            runtime,
            numerical_guide,
            categorical_guide,
            x_positive[positive_indices],
            candidate_batch_size=args.h_candidate_batch_size,
        )
        loss = (
            args.gradient_loss_weight * gradient_loss
            + args.categorical_loss_weight * categorical_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite joint loss at epoch {epoch}: "
                f"numerical={gradient_loss.item()}, "
                f"categorical={categorical_loss.item()}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(numerical_guide.parameters())
            + list(categorical_guide.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        update_ema(
            ema_numerical_guide.parameters(),
            numerical_guide.parameters(),
            rate=args.ema_decay,
        )
        update_ema(
            ema_categorical_guide.parameters(),
            categorical_guide.parameters(),
            rate=args.ema_decay,
        )
        last_loss = loss.item()

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"epoch={epoch:05d}/{args.epochs:05d} total={last_loss:.6f} "
                f"gradient_mse={gradient_loss.item():.6f} "
                f"categorical_loss={categorical_loss.item():.6f}"
            )

        if epoch % args.diagnostic_every == 0 or epoch == args.epochs:
            raw_metrics = joint_diagnostic(
                runtime,
                numerical_guide,
                categorical_guide,
                x_positive,
                min(args.diagnostic_batch_size, len(x_positive)),
                args.seed + 100_000,
                args.h_candidate_batch_size,
            )
            ema_metrics = joint_diagnostic(
                runtime,
                ema_numerical_guide,
                ema_categorical_guide,
                x_positive,
                min(args.diagnostic_batch_size, len(x_positive)),
                args.seed + 100_000,
                args.h_candidate_batch_size,
            )
            last_ema_loss = (
                args.gradient_loss_weight * ema_metrics["gradient_mse"]
                + args.categorical_loss_weight * ema_metrics["categorical_loss"]
            )
            scheduler.step(last_ema_loss)
            print(
                f"diagnostic epoch={epoch:05d} ema_total={last_ema_loss:.6f} "
                f"raw_gradient_mse={raw_metrics['gradient_mse']:.6f} "
                f"ema_gradient_mse={ema_metrics['gradient_mse']:.6f} "
                f"raw_categorical_loss={raw_metrics['categorical_loss']:.6f} "
                f"ema_categorical_loss={ema_metrics['categorical_loss']:.6f} "
                f"ema_gradient_finite={ema_metrics['finite']} "
                f"lr={optimizer.param_groups[0]['lr']:.3e}"
            )
            if epoch > args.checkpoint_warmup and last_ema_loss < best_ema:
                best_ema = last_ema_loss
                save_checkpoint(
                    output_dir / "best_guide.pt",
                    ema_numerical_guide,
                    ema_categorical_guide,
                    metadata,
                    epoch,
                    last_ema_loss,
                    ema=True,
                )

        if epoch % args.checkpoint_every == 0:
            save_checkpoint(
                output_dir / f"guide_epoch_{epoch}.pt",
                numerical_guide,
                categorical_guide,
                metadata,
                epoch,
                last_loss,
                ema=False,
            )
            save_checkpoint(
                output_dir / f"ema_guide_epoch_{epoch}.pt",
                ema_numerical_guide,
                ema_categorical_guide,
                metadata,
                epoch,
                last_ema_loss,
                ema=True,
            )

    save_checkpoint(
        output_dir / "last_guide.pt",
        numerical_guide,
        categorical_guide,
        metadata,
        args.epochs,
        last_loss,
        ema=False,
    )
    save_checkpoint(
        output_dir / "last_ema_guide.pt",
        ema_numerical_guide,
        ema_categorical_guide,
        metadata,
        args.epochs,
        last_ema_loss,
        ema=True,
    )
    if not (output_dir / "best_guide.pt").exists():
        best_ema = last_ema_loss
        save_checkpoint(
            output_dir / "best_guide.pt",
            ema_numerical_guide,
            ema_categorical_guide,
            metadata,
            args.epochs,
            last_ema_loss,
            ema=True,
        )
    print(f"Best EMA conditional joint loss: {best_ema:.6f}")
    print(f"Sampling checkpoint: {output_dir / 'best_guide.pt'}")


if __name__ == "__main__":
    main()
