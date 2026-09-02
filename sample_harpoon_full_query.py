"""Sample HARPOON with an external full-table AND query conditioner.

The frozen HARPOON checkout is only imported.  This adapter extends its
released test-time guidance rule to several numerical intervals and
categorical allowed sets without modifying upstream code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from harpoon_runtime import load_harpoon
from tabdiff.doob_h_evaluation import raw_constraint_report


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument(
        "--allow-partial-query",
        action="store_true",
        help="Permit omitted columns; omitted predicates contribute no HARPOON loss",
    )
    parser.add_argument("--harpoon-root", default="baselines/harpoon")
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", required=True)
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


def canonical_category(value: object) -> str:
    """Canonical key matching JSON strings to pandas/sklearn scalar values."""
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def categorical_allowed_set_loss(
    block: torch.Tensor,
    allowed_indices: torch.Tensor,
) -> torch.Tensor:
    """Minimum official OHE L1 equality loss over an allowed category set."""
    if block.ndim != 2 or allowed_indices.ndim != 1 or allowed_indices.numel() == 0:
        raise ValueError("categorical block must be 2-D and allowed set non-empty")
    targets = torch.nn.functional.one_hot(
        allowed_indices,
        num_classes=block.shape[1],
    ).to(dtype=block.dtype)
    distances = torch.abs(block[:, None, :] - targets[None, :, :]).sum(dim=2)
    return distances.min(dim=1).values


def full_query_guidance_loss(
    x0_hat: torch.Tensor,
    numerical_indices: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    categorical_constraints: list[tuple[int, int, torch.Tensor]],
) -> torch.Tensor:
    """HARPOON's additive AND loss for intervals and allowed category sets."""
    loss = torch.zeros(x0_hat.shape[0], device=x0_hat.device, dtype=x0_hat.dtype)
    if numerical_indices.numel():
        values = x0_hat[:, numerical_indices]
        loss = loss + (
            torch.relu(lower - values).square()
            + torch.relu(values - upper).square()
        ).sum(dim=1)
    for start, end, allowed_indices in categorical_constraints:
        loss = loss + categorical_allowed_set_loss(
            x0_hat[:, start:end],
            allowed_indices,
        )
    return loss


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_query(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        query = json.load(stream)
    predicates = query.get("predicates")
    if not isinstance(predicates, list) or not predicates:
        raise ValueError("full-query JSON must contain a non-empty predicates list")
    return query


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0 or args.timesteps < 2:
        raise ValueError("positive sample/batch sizes and at least two timesteps are required")
    if args.guidance_scale < 0:
        raise ValueError("guidance-scale must be non-negative")

    query_path = Path(args.query_file).resolve()
    harpoon_root = (PROJECT_ROOT / args.harpoon_root).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    for path in (query_path, harpoon_root, runtime_root):
        if not path.exists():
            raise FileNotFoundError(path)
    query = load_query(query_path)

    previous_cwd = Path.cwd()
    os.chdir(runtime_root)
    try:
        Preprocessor, MLPDiffusion, calc_diffusion_hyperparams = load_harpoon(harpoon_root)
        device = torch.device(args.device)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        prepper = Preprocessor(args.dataname)
        train_encoded = prepper.encodeDf("OHE", prepper.df_train)
        numerical_count = int(prepper.numerical_indices_np_end)
        numerical_mean = np.mean(train_encoded[:, :numerical_count], axis=0)
        numerical_std = np.std(train_encoded[:, :numerical_count], axis=0)
        numerical_std[numerical_std == 0] = 1.0
        mean = np.concatenate(
            [numerical_mean, np.zeros(train_encoded.shape[1] - numerical_count)]
        )
        std = np.concatenate(
            [numerical_std, np.ones(train_encoded.shape[1] - numerical_count)]
        )

        numeric_names = [
            str(prepper.df_train.columns[index])
            for index in prepper.info["num_col_idx"]
        ]
        categorical_names = [
            str(prepper.df_train.columns[index])
            for index in prepper.info["cat_col_idx"]
        ]
        modeled_names = numeric_names + categorical_names
        table_names = [str(name) for name in prepper.df_train.columns]
        if sorted(modeled_names) != sorted(table_names) or len(modeled_names) != len(table_names):
            raise ValueError(
                "HARPOON runtime Info does not model every table column (including the "
                "classification target). Rerun prepare_harpoon_data.py into the runtime root."
            )

        predicates_by_name: dict[str, dict] = {}
        for predicate in query["predicates"]:
            name = str(predicate.get("col"))
            if name in predicates_by_name:
                raise ValueError(f"duplicate predicate for {name!r}")
            predicates_by_name[name] = predicate
        missing = sorted(set(table_names) - set(predicates_by_name))
        extra = sorted(set(predicates_by_name) - set(table_names))
        if extra or (missing and not args.allow_partial_query):
            raise ValueError(
                f"query columns are incompatible with the table; missing={missing}, extra={extra}. "
                "Use --allow-partial-query to relax omitted columns."
            )

        numerical_indices_list: list[int] = []
        standardized_lower: list[float] = []
        standardized_upper: list[float] = []
        for index, name in enumerate(numeric_names):
            if name not in predicates_by_name:
                continue
            predicate = predicates_by_name[name]
            if predicate.get("modality") != "numeric" or predicate.get("op") != "between":
                raise ValueError(f"expected numeric between predicate for {name!r}")
            values = predicate.get("values", [])
            if len(values) != 2 or float(values[0]) > float(values[1]):
                raise ValueError(f"invalid interval for {name!r}")
            numerical_indices_list.append(index)
            standardized_lower.append(
                (float(values[0]) - numerical_mean[index]) / numerical_std[index]
            )
            standardized_upper.append(
                (float(values[1]) - numerical_mean[index]) / numerical_std[index]
            )

        numerical_indices = torch.tensor(
            numerical_indices_list, device=device, dtype=torch.long
        )
        lower = torch.tensor(standardized_lower, device=device, dtype=torch.float32)
        upper = torch.tensor(standardized_upper, device=device, dtype=torch.float32)

        categorical_constraints: list[tuple[int, int, torch.Tensor]] = []
        ohe_offset = numerical_count
        for column_index, (name, categories) in enumerate(
            zip(categorical_names, prepper.OneHotEncoder.categories_)
        ):
            width = len(categories)
            if name not in predicates_by_name:
                ohe_offset += width
                continue
            predicate = predicates_by_name[name]
            if predicate.get("modality") != "categorical" or predicate.get("op") != "in":
                raise ValueError(f"expected categorical in predicate for {name!r}")
            category_lookup = {
                canonical_category(value): index for index, value in enumerate(categories)
            }
            requested = [canonical_category(value) for value in predicate.get("values", [])]
            unknown = sorted(set(requested) - set(category_lookup))
            if not requested or unknown:
                raise ValueError(
                    f"invalid allowed set for {name!r}; unknown={unknown}, "
                    f"known={sorted(category_lookup)}"
                )
            allowed_indices = torch.tensor(
                sorted({category_lookup[value] for value in requested}),
                device=device,
                dtype=torch.long,
            )
            categorical_constraints.append(
                (ohe_offset, ohe_offset + width, allowed_indices)
            )
            ohe_offset += width
        if ohe_offset != train_encoded.shape[1]:
            raise ValueError("computed OHE slices do not cover the encoded HARPOON state")

        checkpoint = Path(
            args.checkpoint
            or f"saved_models/{args.dataname}/diffputer_selfmade.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = MLPDiffusion(train_encoded.shape[1], args.hid_dim).to(device)
        model.load_state_dict(torch_load(checkpoint, device))
        model.eval()
        diffusion = calc_diffusion_hyperparams(args.timesteps, args.beta_0, args.beta_t)

        generated_parts = []
        remaining = args.num_samples
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)
            x_t = torch.randn(
                (batch_size, train_encoded.shape[1]), device=device, dtype=torch.float32
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
                        inference_loss = full_query_guidance_loss(
                            x0_hat,
                            numerical_indices,
                            lower,
                            upper,
                            categorical_constraints,
                        )
                        gradient = torch.autograd.grad(inference_loss.sum(), x_t)[0]
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
                        alpha_bar_previous = diffusion["Alpha_bar"][step - 1].to(device)
                        posterior_variance = (1 - alpha_t) * (
                            (1 - alpha_bar_previous) / (1 - alpha_bar_t)
                        )
                        x_t += posterior_variance * torch.randn_like(x_t)
                    x_t -= args.guidance_scale * gradient
            generated_parts.append(x_t.detach().cpu().numpy())
            remaining -= batch_size

        encoded = np.concatenate(generated_parts, axis=0)[: args.num_samples]
        decoded = prepper.decodeNp("OHE", encoded * std + mean)
        generated = pd.DataFrame(decoded, columns=numeric_names + categorical_names)
        generated = generated[table_names]
        test_reference = prepper.df_test.copy()
    finally:
        os.chdir(previous_cwd)

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(output, index=False)
    generated_report, _ = raw_constraint_report(generated, query)
    reference_report, _ = raw_constraint_report(test_reference, query)
    with open(output.with_suffix(".query.json"), "w", encoding="utf-8") as stream:
        json.dump(query, stream, indent=2)
    report = {
        "method": "HARPOON Algorithm 1 external full-query adapter",
        "source_commit": "40dc8cee26e215e86523045fdafb7c1ad89c3fd7",
        "query_id": query.get("query_id"),
        "dataname": args.dataname,
        "num_samples": len(generated),
        "eta": args.guidance_scale,
        "timesteps": args.timesteps,
        "numerical_loss": "sum_j ReLU(lower_j-x0_hat_j)^2 + ReLU(x0_hat_j-upper_j)^2",
        "categorical_set_loss": "sum_j min_{k in S_j} ||x0_hat_j-e_{j,k}||_1",
        "constraint_combination": "full AND by addition",
        "raw_query_hit_rate": generated_report["joint_hit_rate"],
        "arity": len(query["predicates"]),
        "test_reference_rows_satisfying": reference_report["rows_satisfying"],
        "test_reference_rows": reference_report["num_rows"],
    }
    with open(output.with_suffix(".constraints.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Generated {len(generated)} HARPOON rows")
    print(f"Raw full-query hit rate: {generated_report['joint_hit_rate']:.2%}")
    print(
        "Test conditional support: "
        f"{reference_report['rows_satisfying']}/{reference_report['num_rows']}"
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
