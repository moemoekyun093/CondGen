"""Train one fixed-query numerical Doob h-transform guide off-policy.

The base TabDiff model and its noise schedules are frozen.  Training endpoints
are the rows satisfying one broad box over all normalized numerical columns.
The guide is not parameterized by query bounds: its only inputs are x_t and t.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from tabdiff.doob_h_runtime import load_doob_runtime, resolve_base_checkpoint
from tabdiff.models.doob_h_transform import NumericalBoxQuery, NumericalDoobHGuide
from utils_train import update_ema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", default="shoppers")
    parser.add_argument("--base-ckpt", default=None)
    parser.add_argument("--base-exp-name", default="learnable_schedule")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--query-file",
        default=None,
        help="Saved fixed numerical interval JSON from generate_doob_intervals.py",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--reduce-lr-patience", type=int, default=50)
    parser.add_argument("--lr-factor", type=float, default=0.9)
    parser.add_argument("--checkpoint-warmup", type=int, default=4000)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--lower-quantile", type=float, default=0.005)
    parser.add_argument("--upper-quantile", type=float, default=0.995)
    parser.add_argument("--d-token", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--factor", type=float, default=2.0)
    parser.add_argument("--n-frequencies", type=int, default=16)
    return parser.parse_args()


def correction_batch(runtime, guide, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return prediction and Section-5 denoiser-space target for one endpoint batch."""
    batch_size = x0.shape[0]
    device = x0.device
    t = torch.rand(batch_size, device=device, dtype=x0.dtype)
    d_numerical = runtime.dataset.d_numerical
    x0_num = x0[:, :d_numerical]
    x0_cat = x0[:, d_numerical:].long()
    sigma = runtime.diffusion.num_schedule.total_noise(t[:, None])
    x_t = x0_num + sigma * torch.randn_like(x0_num)

    if x0_cat.shape[1] > 0:
        sigma_cat = runtime.diffusion.cat_schedule.total_noise(t[:, None])
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
            x_cat_t_soft = runtime.diffusion.to_one_hot(x_cat_t).to(x_t.dtype)
    else:
        x_cat_t_soft = x_t.new_zeros((batch_size, 0))
    with torch.no_grad():
        base_denoised, _ = runtime.diffusion._denoise_fn(
            x_t,
            x_cat_t_soft,
            t,
            sigma=sigma,
        )

    # From Eq. 5.1 for Gaussian forward noise:
    #   grad log h = (x0 - D_base) / sigma^2.
    # We learn sigma^2 grad log h, which has the same pointwise minimizer and
    # remains well scaled as sigma approaches sigma_min.
    target_correction = x0_num - base_denoised
    predicted_correction = guide(x_t, x_cat_t_soft, t)
    return predicted_correction, target_correction


@torch.no_grad()
def full_training_loss(runtime, guide, loader: DataLoader, device: torch.device) -> float:
    """Evaluate on all constrained training rows, following TabDiff's trainer."""
    guide.eval()
    total_loss = 0.0
    total_rows = 0
    for (x0,) in loader:
        x0 = x0.to(device)
        prediction, target = correction_batch(runtime, guide, x0)
        total_loss += F.mse_loss(prediction, target).item() * len(x0)
        total_rows += len(x0)
    guide.train()
    return total_loss / total_rows


def save_guide_checkpoint(
    path: Path,
    guide: NumericalDoobHGuide,
    metadata: dict,
    epoch: int,
    training_mse: float,
    ema: bool,
) -> None:
    torch.save(
        {
            "guide": guide.state_dict(),
            "metadata": metadata,
            "epoch": epoch,
            "training_mse": training_mse,
            "ema": ema,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema-decay must be in [0, 1)")
    if args.reduce_lr_patience < 0:
        raise ValueError("reduce-lr-patience must be non-negative")
    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError("lr-factor must be between 0 and 1")
    if args.checkpoint_warmup < 0 or args.checkpoint_every <= 0:
        raise ValueError("checkpoint-warmup must be non-negative and checkpoint-every positive")

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
    d_numerical = runtime.dataset.d_numerical
    x_all = runtime.dataset.X.float()
    x_num = x_all[:, :d_numerical]

    query_spec = None
    if args.query_file is not None:
        with open(args.query_file, "r", encoding="utf-8") as stream:
            query_spec = json.load(stream)
        if query_spec.get("dataname") != args.dataname:
            raise ValueError(
                f"query file is for {query_spec.get('dataname')}, not {args.dataname}"
            )
        expected_fingerprint = query_spec.get("data_sha256")
        actual_fingerprint = hashlib.sha256(
            x_num.contiguous().numpy().tobytes()
        ).hexdigest()
        if expected_fingerprint and expected_fingerprint != actual_fingerprint:
            raise ValueError(
                "query file was generated from different transformed training data"
            )
        query = NumericalBoxQuery.from_dict(query_spec)
    else:
        query = NumericalBoxQuery.from_quantiles(
            x_num,
            lower_quantile=args.lower_quantile,
            upper_quantile=args.upper_quantile,
        )
    positive_mask = query.contains(x_num)
    positive = x_all[positive_mask]
    hit_rate = positive_mask.float().mean().item()
    if len(positive) < max(32, args.batch_size // 4):
        raise ValueError(
            f"the fixed query retained only {len(positive)} rows ({hit_rate:.2%}); "
            "use broader quantiles"
        )

    generator = torch.Generator().manual_seed(args.seed)
    training = positive
    train_loader = DataLoader(
        TensorDataset(training),
        batch_size=min(args.batch_size, len(training)),
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    loss_loader = DataLoader(
        TensorDataset(training),
        batch_size=min(args.batch_size, len(training)),
        shuffle=False,
        drop_last=False,
    )

    guide_kwargs = {
        "d_numerical": d_numerical,
        "categories": (runtime.dataset.categories + 1).tolist(),
        "d_token": args.d_token,
        "num_layers": args.num_layers,
        "n_head": args.n_head,
        "factor": args.factor,
        "n_frequencies": args.n_frequencies,
        "freq_sigma": 0.05,
    }
    guide = NumericalDoobHGuide(**guide_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        guide.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.reduce_lr_patience,
    )
    ema_guide = deepcopy(guide)
    for parameter in ema_guide.parameters():
        parameter.detach_()

    output_dir = Path(
        args.output_dir or f"tabdiff/ckpt/{args.dataname}/doob_h_fixed_box"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataname": args.dataname,
        "base_checkpoint": runtime.checkpoint_path,
        "guide": guide_kwargs,
        "query": {
            **(query_spec or {}),
            **query.to_dict(),
            "space": "TabDiff quantile-normalized numerical coordinates",
            "all_numerical_columns": True,
            "source_file": args.query_file,
            "training_hit_rate": hit_rate,
            "positive_rows": len(positive),
        },
        "objective": "MSE on sigma^2 * grad_x log h = x0 - D_base(x_t,t)",
        "seed": args.seed,
        "training": {
            "epochs": args.epochs,
            "batch_size": min(args.batch_size, len(training)),
            "batches_per_epoch": len(train_loader),
            "optimizer_updates": args.epochs * len(train_loader),
            "all_constrained_rows": True,
            "validation_split": False,
            "ema_decay": args.ema_decay,
            "checkpoint_warmup": args.checkpoint_warmup,
            "checkpoint_every": args.checkpoint_every,
        },
    }
    with open(output_dir / "query.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    parameter_count = sum(parameter.numel() for parameter in guide.parameters())
    print(f"Fixed query retains {len(positive)}/{len(x_num)} rows ({hit_rate:.2%})")
    print(f"Guide parameters: {parameter_count:,}")
    print(
        f"Training for {args.epochs} epochs x {len(train_loader)} batches/epoch "
        f"= {args.epochs * len(train_loader)} optimizer updates"
    )
    print(f"Writing checkpoints to {output_dir}")

    best_training = float("inf")
    best_ema_training = float("inf")
    last_epoch = 0
    last_training_mse = float("nan")
    last_ema_training_mse = float("nan")
    guide.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_rows = 0
        for (x0,) in train_loader:
            x0 = x0.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, target = correction_batch(runtime, guide, x0)
            loss = F.mse_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(guide.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(x0)
            total_rows += len(x0)

        training_mse = total_loss / total_rows
        if not math.isfinite(training_mse):
            print(f"Stopping at epoch {epoch}: non-finite training MSE {training_mse}")
            break

        scheduler.step(training_mse)
        update_ema(ema_guide.parameters(), guide.parameters(), rate=args.ema_decay)
        ema_training_mse = full_training_loss(runtime, ema_guide, loss_loader, device)
        if not math.isfinite(ema_training_mse):
            print(f"Stopping at epoch {epoch}: non-finite EMA training MSE {ema_training_mse}")
            break
        last_epoch = epoch
        last_training_mse = training_mse
        last_ema_training_mse = ema_training_mse
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:05d}/{args.epochs:05d} train_mse={training_mse:.6f} "
            f"ema_train_mse={ema_training_mse:.6f} lr={lr:.3e}"
        )

        if epoch > args.checkpoint_warmup and training_mse < best_training:
            best_training = training_mse
            save_guide_checkpoint(
                output_dir / "best_raw_guide.pt",
                guide,
                metadata,
                epoch,
                training_mse,
                ema=False,
            )

        if epoch > args.checkpoint_warmup and ema_training_mse < best_ema_training:
            best_ema_training = ema_training_mse
            save_guide_checkpoint(
                output_dir / "best_guide.pt",
                ema_guide,
                metadata,
                epoch,
                ema_training_mse,
                ema=True,
            )

        if epoch % args.checkpoint_every == 0:
            save_guide_checkpoint(
                output_dir / f"guide_epoch_{epoch}.pt",
                guide,
                metadata,
                epoch,
                training_mse,
                ema=False,
            )
            save_guide_checkpoint(
                output_dir / f"ema_guide_epoch_{epoch}.pt",
                ema_guide,
                metadata,
                epoch,
                ema_training_mse,
                ema=True,
            )

    if last_epoch == 0:
        raise RuntimeError("training stopped before completing one finite epoch")

    save_guide_checkpoint(
        output_dir / "last_guide.pt",
        guide,
        metadata,
        last_epoch,
        last_training_mse,
        ema=False,
    )
    save_guide_checkpoint(
        output_dir / "last_ema_guide.pt",
        ema_guide,
        metadata,
        last_epoch,
        last_ema_training_mse,
        ema=True,
    )
    if not (output_dir / "best_guide.pt").exists() or last_epoch <= args.checkpoint_warmup:
        save_guide_checkpoint(
            output_dir / "best_guide.pt",
            ema_guide,
            metadata,
            last_epoch,
            last_ema_training_mse,
            ema=True,
        )
        print("Run ended before EMA best-checkpoint selection; using final EMA guide")
    print(f"Best raw training MSE after warmup: {best_training:.6f}")
    print(f"Best EMA training MSE after warmup: {best_ema_training:.6f}")
    print(f"Sampling checkpoint: {output_dir / 'best_guide.pt'}")


if __name__ == "__main__":
    main()
