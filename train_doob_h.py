"""Train one fixed-query numerical Doob h-transform guide off-policy.

The base TabDiff model and its noise schedules are frozen.  Training endpoints
are the rows satisfying one broad box over all normalized numerical columns.
The guide is not parameterized by query bounds: its only inputs are x_t and t.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from tabdiff.doob_h_runtime import load_doob_runtime, resolve_base_checkpoint
from tabdiff.models.doob_h_transform import NumericalBoxQuery, NumericalDoobHGuide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", default="news")
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
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lower-quantile", type=float, default=0.005)
    parser.add_argument("--upper-quantile", type=float, default=0.995)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=100)
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
def validation_loss(runtime, guide, validation: torch.Tensor, batch_size: int) -> float:
    guide.eval()
    losses = []
    for start in range(0, len(validation), batch_size):
        x0 = validation[start : start + batch_size]
        prediction, target = correction_batch(runtime, guide, x0)
        losses.append(F.mse_loss(prediction, target).item())
    guide.train()
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5")

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
    permutation = torch.randperm(len(positive), generator=generator)
    positive = positive[permutation]
    validation_size = max(1, int(len(positive) * args.validation_fraction))
    validation = positive[:validation_size].to(device)
    training = positive[validation_size:]
    loader = DataLoader(
        TensorDataset(training),
        batch_size=min(args.batch_size, len(training)),
        shuffle=True,
        drop_last=False,
        generator=generator,
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
    }
    with open(output_dir / "query.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    parameter_count = sum(parameter.numel() for parameter in guide.parameters())
    print(f"Fixed query retains {len(positive)}/{len(x_num)} rows ({hit_rate:.2%})")
    print(f"Guide parameters: {parameter_count:,}")
    print(f"Writing checkpoints to {output_dir}")

    best_validation = float("inf")
    iterator = iter(loader)
    guide.train()
    for step in range(1, args.steps + 1):
        try:
            (x0,) = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            (x0,) = next(iterator)
        x0 = x0.to(device)

        optimizer.zero_grad(set_to_none=True)
        prediction, target = correction_batch(runtime, guide, x0)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(guide.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            val_loss = validation_loss(runtime, guide, validation, args.batch_size)
            print(f"step={step:05d} train_mse={loss.item():.6f} val_mse={val_loss:.6f}")
            if val_loss < best_validation:
                best_validation = val_loss
                torch.save(
                    {
                        "guide": guide.state_dict(),
                        "metadata": metadata,
                        "step": step,
                        "validation_mse": val_loss,
                    },
                    output_dir / "best_guide.pt",
                )

    print(f"Best validation MSE: {best_validation:.6f}")
    print(f"Saved {output_dir / 'best_guide.pt'}")


if __name__ == "__main__":
    main()
