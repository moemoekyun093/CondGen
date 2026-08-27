"""Run HARPOON Algorithm 1 with test-time inequality guidance on Shoppers."""

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
    parser.add_argument("--dataname", choices=("shoppers",), default="shoppers")
    parser.add_argument(
        "--lower-bound",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Test-time lower-bound inequality; repeat for multiple columns",
    )
    parser.add_argument(
        "--upper-bound",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Test-time upper-bound inequality; repeat for multiple columns",
    )
    parser.add_argument(
        "--query-file",
        default=None,
        help="Fixed-interval JSON; mutually exclusive with explicit bound arguments",
    )
    parser.add_argument(
        "--active-columns",
        default=None,
        help="Comma-separated numerical model indices selected from --query-file",
    )
    parser.add_argument("--harpoon-root", default="baselines/harpoon")
    parser.add_argument(
        "--runtime-root",
        default=None,
        help="Directory containing HARPOON datasets and saved_models; defaults to --harpoon-root",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Defaults to the number of Shoppers test rows satisfying the constraint",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hid-dim", type=int, default=1024)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--beta-0", type=float, default=0.0001)
    parser.add_argument("--beta-t", type=float, default=0.02)
    parser.add_argument("--guidance-scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_bound_specs(
    lower_specs: list[str],
    upper_specs: list[str],
) -> list[dict]:
    """Parse paper-style one-sided or two-sided test-time inequalities."""
    constraints: dict[str, dict] = {}
    if not lower_specs and not upper_specs:
        # Exact Shoppers range task from Appendix E.2 / Table 9.
        lower_specs = ["Administrative=4"]
    for side, specs in (("raw_lower", lower_specs), ("raw_upper", upper_specs)):
        for specification in specs:
            try:
                name, value_text = specification.rsplit("=", maxsplit=1)
                name = name.strip()
                value = float(value_text)
            except (AttributeError, ValueError) as error:
                raise ValueError(
                    f"invalid inequality {specification!r}; expected COLUMN=VALUE"
                ) from error
            if not name or not np.isfinite(value):
                raise ValueError(f"invalid inequality {specification!r}")
            constraint = constraints.setdefault(
                name,
                {"name": name, "raw_lower": None, "raw_upper": None},
            )
            if constraint[side] is not None and constraint[side] != value:
                raise ValueError(f"{name!r} has two different {side} values")
            constraint[side] = value
    for constraint in constraints.values():
        lower = constraint["raw_lower"]
        upper = constraint["raw_upper"]
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"lower bound exceeds upper bound for {constraint['name']!r}")
    return list(constraints.values())


def load_query_specs(query_file: str, active_columns: str | None) -> list[dict]:
    """Load a nested numerical query without changing the upstream HARPOON code."""
    with open(query_file, "r", encoding="utf-8") as stream:
        query = json.load(stream)
    columns = query.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("query file must contain a non-empty columns list")

    by_index = {}
    for position, column in enumerate(columns):
        model_index = int(column.get("model_index", position))
        if model_index in by_index:
            raise ValueError(f"duplicate model_index {model_index} in query file")
        by_index[model_index] = column

    if active_columns is None or active_columns.strip().lower() == "all":
        selected_indices = sorted(by_index)
    else:
        selected_indices = [
            int(value.strip())
            for value in active_columns.split(",")
            if value.strip()
        ]
    if not selected_indices:
        raise ValueError("at least one active query column is required")
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("active query columns must be unique")

    specs = []
    for model_index in selected_indices:
        if model_index not in by_index:
            raise ValueError(f"model_index {model_index} is absent from query file")
        column = by_index[model_index]
        specs.append(
            {
                "model_index": model_index,
                "name": str(column["name"]),
                "raw_lower": column.get("raw_lower"),
                "raw_upper": column.get("raw_upper"),
            }
        )
    return specs


def summed_squared_relu_loss(
    constrained: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Official HARPOON AND rule extended across numerical inequalities."""
    lower_loss = torch.relu(lower - constrained).square()
    upper_loss = torch.relu(constrained - upper).square()
    return (lower_loss + upper_loss).sum(dim=1)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("num-samples must be positive when provided")
    if args.batch_size <= 0 or args.timesteps < 2:
        raise ValueError("batch-size must be positive and timesteps at least 2")
    if args.guidance_scale < 0:
        raise ValueError("guidance-scale must be non-negative")

    harpoon_root = (PROJECT_ROOT / args.harpoon_root).resolve()
    if not harpoon_root.is_dir():
        raise FileNotFoundError(harpoon_root)
    runtime_root = (
        Path(args.runtime_root).resolve()
        if args.runtime_root
        else harpoon_root
    )
    if not runtime_root.is_dir():
        raise FileNotFoundError(runtime_root)
    if args.query_file and (args.lower_bound or args.upper_bound):
        raise ValueError("--query-file cannot be combined with explicit bounds")
    if args.active_columns is not None and not args.query_file:
        raise ValueError("--active-columns requires --query-file")
    column_specs = (
        load_query_specs(args.query_file, args.active_columns)
        if args.query_file
        else parse_bound_specs(args.lower_bound, args.upper_bound)
    )

    previous_cwd = Path.cwd()
    os.chdir(runtime_root)
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
        reference_satisfied = np.ones(len(prepper.df_test), dtype=bool)
        for spec in column_specs:
            if spec["name"] not in prepper.df_test.columns:
                raise ValueError(f"constraint column {spec['name']!r} is missing")
            values = pd.to_numeric(prepper.df_test[spec["name"]], errors="coerce")
            column_satisfied = values.notna()
            if spec["raw_lower"] is not None:
                column_satisfied &= values >= float(spec["raw_lower"])
            if spec["raw_upper"] is not None:
                column_satisfied &= values <= float(spec["raw_upper"])
            reference_satisfied &= column_satisfied.to_numpy()
        reference_rows = int(reference_satisfied.sum())
        num_samples = args.num_samples or reference_rows
        if num_samples <= 0:
            raise ValueError("no Shoppers test rows satisfy the test-time constraint")
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
                None
                if spec["raw_lower"] is None
                else (float(spec["raw_lower"]) - numerical_mean[index])
                / numerical_std[index]
            )
            standardized_upper.append(
                None
                if spec["raw_upper"] is None
                else (float(spec["raw_upper"]) - numerical_mean[index])
                / numerical_std[index]
            )
        constraint_indices = torch.tensor(
            constraint_indices, device=device, dtype=torch.long
        )
        lower = torch.tensor(
            [float("-inf") if value is None else value for value in standardized_lower],
            device=device,
            dtype=torch.float32,
        )
        upper = torch.tensor(
            [float("inf") if value is None else value for value in standardized_upper],
            device=device,
            dtype=torch.float32,
        )

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
        remaining = num_samples
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
                if args.guidance_scale > 0:
                    with torch.enable_grad():
                        x_t = x_t.detach().requires_grad_(True)
                        predicted_noise = model(x_t, timesteps)
                        x0_hat = (
                            x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise
                        ) / torch.sqrt(alpha_bar_t)
                        constrained = x0_hat[:, constraint_indices]
                        # The released general-constraint sampler squares its
                        # ReLU range loss and adds independent constraint losses
                        # for an AND query. Extend exactly that additive rule to
                        # every active numerical interval.
                        inference_loss = summed_squared_relu_loss(
                            constrained,
                            lower,
                            upper,
                        )
                        gradient = torch.autograd.grad(
                            inference_loss.sum(),
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
                        # Match the stochastic term in the released HARPOON
                        # general-constraint sampler.
                        alpha_bar_t_1 = diffusion["Alpha_bar"][step - 1].to(device)
                        posterior_variance = (1 - alpha_t) * (
                            (1 - alpha_bar_t_1) / (1 - alpha_bar_t)
                        )
                        x_t += posterior_variance * torch.randn_like(x_t)
                    # Algorithm 1, line 11: tangential test-time correction.
                    x_t -= args.guidance_scale * gradient
            generated_parts.append(x_t.cpu().numpy())
            remaining -= batch_size

        encoded = np.concatenate(generated_parts, axis=0)[:num_samples]
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
        or "conditional_samples/shoppers/harpoon_paper_range.csv"
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(output, index=False)

    satisfied = np.ones(len(generated), dtype=bool)
    for spec in column_specs:
        values = pd.to_numeric(generated[spec["name"]], errors="coerce")
        column_satisfied = values.notna()
        if spec["raw_lower"] is not None:
            column_satisfied &= values >= float(spec["raw_lower"])
        if spec["raw_upper"] is not None:
            column_satisfied &= values <= float(spec["raw_upper"])
        satisfied &= column_satisfied.to_numpy()
    query = {
        "constraint_id": "harpoon_test_time_inequality",
        "dataname": "shoppers",
        "source": "HARPOON Appendix D Eq. 9 and Appendix E.2",
        "columns": column_specs,
    }
    query_output = output.with_suffix(".query.json")
    with open(query_output, "w", encoding="utf-8") as stream:
        json.dump(query, stream, indent=2)
    report = {
        "method": "HARPOON Algorithm 1 test-time manifold guidance",
        "source_commit": "40dc8cee26e215e86523045fdafb7c1ad89c3fd7",
        "paper": "arXiv:2602.07875v3",
        "dataname": "shoppers",
        "constraints": column_specs,
        "inference_loss": "sum_j ReLU(lower_j-x0_hat_j)^2 + ReLU(x0_hat_j-upper_j)^2",
        "constraint_combination": "AND by addition, matching the released HARPOON general-constraint sampler",
        "gradient": "grad_x_t L_inf(Q_t(x_t), c)",
        "update_order": "unconditional DDPM step, then -eta*gradient",
        "num_samples": len(generated),
        "paper_protocol_reference_rows": reference_rows,
        "sample_count_overridden": args.num_samples is not None,
        "raw_joint_hit_rate": float(satisfied.mean()),
        "active_numerical_columns": [
            int(spec.get("model_index", index))
            for index, spec in enumerate(column_specs)
        ],
        "eta": args.guidance_scale,
        "timesteps": args.timesteps,
    }
    with open(output.with_suffix(".constraints.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Generated {len(generated)} HARPOON rows")
    print(f"Constraint-matching Shoppers test rows: {reference_rows}")
    print(f"Raw-space inequality hit rate: {satisfied.mean():.2%}")
    print(f"Saved {output}")
    print(f"Saved test-time constraint {query_output}")


if __name__ == "__main__":
    main()
