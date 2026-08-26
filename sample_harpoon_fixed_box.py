"""Sample the official HARPOON OHE diffusion under our saved interval query."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from harpoon_runtime import load_harpoon


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--harpoon-root", default="baselines/harpoon")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--hid-dim", type=int, default=1024)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--beta-0", type=float, default=0.0001)
    parser.add_argument("--beta-t", type=float, default=0.02)
    parser.add_argument("--guidance-scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0 or args.timesteps < 2:
        raise ValueError("num-samples, batch-size, and timesteps must be positive")
    if args.guidance_scale < 0:
        raise ValueError("guidance-scale must be non-negative")

    harpoon_root = (PROJECT_ROOT / args.harpoon_root).resolve()
    if not harpoon_root.is_dir():
        raise FileNotFoundError(harpoon_root)
    query_path = (PROJECT_ROOT / args.query_file).resolve()
    with open(query_path, "r", encoding="utf-8") as stream:
        query = json.load(stream)
    if query.get("dataname") not in (None, args.dataname):
        raise ValueError("query file belongs to a different dataset")
    column_specs = query.get("columns", [])
    if not column_specs:
        raise ValueError("query contains no active numerical columns")

    previous_cwd = Path.cwd()
    os.chdir(harpoon_root)
    try:
        Preprocessor, MLPDiffusion, calc_diffusion_hyperparams = load_harpoon(
            harpoon_root
        )

        device = torch.device(args.device)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        prepper = Preprocessor(args.dataname)
        train_encoded = prepper.encodeDf("OHE", prepper.df_train)
        numerical_count = prepper.numerical_indices_np_end
        numerical_mean = np.mean(train_encoded[:, :numerical_count], axis=0)
        numerical_std = np.std(train_encoded[:, :numerical_count], axis=0)
        numerical_std[numerical_std == 0] = 1.0
        mean = np.concatenate(
            [numerical_mean, np.zeros(train_encoded.shape[1] - numerical_count)]
        )
        std = np.concatenate(
            [numerical_std, np.ones(train_encoded.shape[1] - numerical_count)]
        )

        numeric_original_indices = list(prepper.info["num_col_idx"])
        numeric_names = [prepper.df_train.columns[index] for index in numeric_original_indices]
        name_to_numeric_index = {
            str(name): index for index, name in enumerate(numeric_names)
        }
        constraint_indices = []
        standardized_lower = []
        standardized_upper = []
        for spec in column_specs:
            name = str(spec["name"])
            if name not in name_to_numeric_index:
                raise ValueError(f"query column {name!r} is not numerical in HARPOON")
            index = name_to_numeric_index[name]
            constraint_indices.append(index)
            standardized_lower.append(
                (float(spec["raw_lower"]) - numerical_mean[index])
                / numerical_std[index]
            )
            standardized_upper.append(
                (float(spec["raw_upper"]) - numerical_mean[index])
                / numerical_std[index]
            )
        constraint_indices = torch.tensor(
            constraint_indices, device=device, dtype=torch.long
        )
        lower = torch.tensor(standardized_lower, device=device, dtype=torch.float32)
        upper = torch.tensor(standardized_upper, device=device, dtype=torch.float32)

        checkpoint = Path(
            args.checkpoint
            or f"saved_models/{args.dataname}/diffputer_selfmade.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = MLPDiffusion(train_encoded.shape[1], args.hid_dim).to(device)
        model.load_state_dict(torch_load(checkpoint, device))
        model.eval()
        diffusion = calc_diffusion_hyperparams(
            args.timesteps,
            args.beta_0,
            args.beta_t,
        )

        generated_parts = []
        remaining = args.num_samples
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)
            x_t = torch.randn(
                (batch_size, train_encoded.shape[1]),
                device=device,
                dtype=torch.float32,
            )
            for step in range(args.timesteps - 1, -1, -1):
                timesteps = torch.full(
                    (batch_size,), step, device=device, dtype=torch.long
                )
                alpha_t = diffusion["Alpha"][step].to(device)
                alpha_bar_t = diffusion["Alpha_bar"][step].to(device)
                alpha_bar_previous = (
                    diffusion["Alpha_bar"][step - 1].to(device)
                    if step >= 1
                    else torch.tensor(1.0, device=device)
                )
                if args.guidance_scale > 0:
                    with torch.enable_grad():
                        x_t = x_t.detach().requires_grad_(True)
                        predicted_noise = model(x_t, timesteps)
                        x0_hat = (
                            x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise
                        ) / torch.sqrt(alpha_bar_t)
                        constrained = x0_hat[:, constraint_indices]
                        condition_loss = (
                            torch.relu(lower - constrained).square()
                            + torch.relu(constrained - upper).square()
                        ).sum(dim=1)
                        gradient = torch.autograd.grad(
                            condition_loss.sum(),
                            x_t,
                        )[0]
                else:
                    with torch.no_grad():
                        predicted_noise = model(x_t, timesteps)
                        gradient = torch.zeros_like(x_t)
                with torch.no_grad():
                    x_t = (x_t / torch.sqrt(alpha_t)) - (
                        (1 - alpha_t)
                        / (torch.sqrt(alpha_t) * torch.sqrt(1 - alpha_bar_t))
                    ) * predicted_noise
                    if step > 0:
                        posterior_variance = (1 - alpha_t) * (
                            (1 - alpha_bar_previous) / (1 - alpha_bar_t)
                        )
                        x_t += posterior_variance * torch.randn_like(x_t)
                    x_t -= args.guidance_scale * gradient
            generated_parts.append(x_t.cpu().numpy())
            remaining -= batch_size

        encoded = np.concatenate(generated_parts, axis=0)[: args.num_samples]
        encoded = encoded * std + mean
        decoded = prepper.decodeNp("OHE", encoded)
        numerical_names = [
            prepper.df_train.columns[index] for index in prepper.info["num_col_idx"]
        ]
        categorical_names = [
            prepper.df_train.columns[index] for index in prepper.info["cat_col_idx"]
        ]
        generated = pd.DataFrame(
            decoded,
            columns=numerical_names + categorical_names,
        )
        generated = generated[list(prepper.df_train.columns)]
    finally:
        os.chdir(previous_cwd)

    output = Path(
        args.output
        or f"conditional_samples/{args.dataname}/harpoon_partial.csv"
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(output, index=False)

    satisfied = np.ones(len(generated), dtype=bool)
    for spec in column_specs:
        values = pd.to_numeric(generated[spec["name"]], errors="coerce")
        satisfied &= values.between(
            float(spec["raw_lower"]),
            float(spec["raw_upper"]),
            inclusive="both",
        ).fillna(False).to_numpy()
    report = {
        "method": "HARPOON official OHE DDPM with squared-hinge inference guidance",
        "source_commit": "40dc8cee26e215e86523045fdafb7c1ad89c3fd7",
        "dataname": args.dataname,
        "query_file": str(query_path),
        "active_columns": [spec["name"] for spec in column_specs],
        "num_samples": len(generated),
        "raw_joint_hit_rate": float(satisfied.mean()),
        "guidance_scale": args.guidance_scale,
        "timesteps": args.timesteps,
    }
    with open(output.with_suffix(".constraints.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Generated {len(generated)} HARPOON rows")
    print(f"Raw-space partial-query hit rate: {satisfied.mean():.2%}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
