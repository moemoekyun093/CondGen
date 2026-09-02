"""Native imputation-style conditioning for HARPOON's released baselines."""

from __future__ import annotations

import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tabdiff.baseline_data import BaselineTable, query_categorical_observations


def _harpoon_imports(root: Path):
    root = root.resolve()
    sys.path.insert(0, str(root))
    from model import MLPDiffusion
    from utils import calc_diffusion_hyperparams

    return MLPDiffusion, calc_diffusion_hyperparams


def sample_diffputer_repaint(
    *,
    table: BaselineTable,
    query: dict,
    checkpoint: Path,
    harpoon_root: Path,
    num_rows: int,
    batch_size: int,
    seed: int,
    device: str,
    hid_dim: int = 1024,
    timesteps: int = 200,
    beta_0: float = 1e-4,
    beta_t: float = 2e-2,
) -> tuple[pd.DataFrame, dict]:
    """Run the RePaint-style OHE DDPM called DiffPuter in HARPOON Table 3."""
    MLPDiffusion, calc_diffusion_hyperparams = _harpoon_imports(harpoon_root)
    rng = np.random.default_rng(seed)
    observed, metadata = query_categorical_observations(table, query, num_rows, rng)
    mean, std = table.standardization()
    train_encoded = table.encode(table.train)
    in_dim = train_encoded.shape[1]
    torch_device = torch.device(device)
    model = MLPDiffusion(in_dim, hid_dim).to(torch_device)
    payload = torch.load(checkpoint, map_location=torch_device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        config = payload.get("config", {})
        hid_dim = int(config.get("hid_dim", hid_dim))
        timesteps = int(config.get("timesteps", timesteps))
        beta_0 = float(config.get("beta_0", beta_0))
        beta_t = float(config.get("beta_t", beta_t))
        model = MLPDiffusion(in_dim, hid_dim).to(torch_device)
    else:
        state_dict = payload
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = calc_diffusion_hyperparams(timesteps, beta_0, beta_t)

    categorical_slices = table.categorical_slices()
    outputs = []
    generator = torch.Generator(device=torch_device).manual_seed(seed)
    for start in range(0, num_rows, batch_size):
        stop = min(start + batch_size, num_rows)
        size = stop - start
        known = np.zeros((size, in_dim), dtype=np.float64)
        missing = np.ones((size, in_dim), dtype=np.float32)
        for column, encoded_slice in categorical_slices.items():
            values = observed.iloc[start:stop][column]
            known_rows = values.notna().to_numpy()
            if not known_rows.any():
                continue
            categories = table.encoder.categories_[table.categorical_columns.index(column)]
            lookup = {str(value): index for index, value in enumerate(categories)}
            local = np.array([lookup.get(str(value), -1) for value in values], dtype=int)
            rows = np.flatnonzero(known_rows)
            known[rows, encoded_slice.start + local[rows]] = 1.0
            missing[rows, encoded_slice] = 0.0
        known = (known - mean) / std
        known_t = torch.tensor(known, dtype=torch.float32, device=torch_device)
        missing_t = torch.tensor(missing, dtype=torch.float32, device=torch_device)
        x_t = torch.randn((size, in_dim), generator=generator, device=torch_device)
        with torch.no_grad():
            for step in range(timesteps - 1, -1, -1):
                t = torch.full((size,), step, device=torch_device, dtype=torch.long)
                alpha = diffusion["Alpha"][step].to(torch_device)
                alpha_bar = diffusion["Alpha_bar"][step].to(torch_device)
                previous_bar = (
                    diffusion["Alpha_bar"][step - 1].to(torch_device)
                    if step > 0
                    else torch.tensor(1.0, device=torch_device)
                )
                predicted_noise = model(x_t, t)
                x_t = x_t / torch.sqrt(alpha) - (
                    (1 - alpha) / (torch.sqrt(alpha) * torch.sqrt(1 - alpha_bar))
                ) * predicted_noise
                if step > 0:
                    variance = (1 - alpha) * (1 - previous_bar) / (1 - alpha_bar)
                    x_t = x_t + variance * torch.randn(
                        x_t.shape, generator=generator, device=torch_device
                    )
                known_noisy = torch.sqrt(previous_bar) * known_t + torch.sqrt(
                    1 - previous_bar
                ) * torch.randn(known_t.shape, generator=generator, device=torch_device)
                x_t = (1 - missing_t) * known_noisy + missing_t * x_t
        outputs.append(x_t.cpu().numpy() * std + mean)
    frame = table.decode(np.concatenate(outputs, axis=0))
    metadata.update(
        {
            "method": "diffputer_harpoon_repaint",
            "upstream_protocol": "HARPOON sampling_repaint_generalconstraints.py",
            "timesteps": timesteps,
            "seed": seed,
        }
    )
    return frame, metadata


def sample_great(
    *,
    table: BaselineTable,
    query: dict,
    model_dir: Path,
    great_root: Path,
    num_rows: int,
    batch_size: int,
    seed: int,
    max_retries: int = 5,
) -> tuple[pd.DataFrame, dict]:
    sys.path.insert(0, str(great_root.resolve()))
    from be_great import GReaT

    # GReaT 0.0.9 uses the removed NumPy alias internally. Keep this
    # compatibility shim outside the pinned upstream checkout.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    observed, metadata = query_categorical_observations(table, query, num_rows, rng)
    model = GReaT.load_from_dir(str(model_dir))
    chunks = []
    fallback_rows = 0
    for start in range(0, num_rows, batch_size):
        masked = observed.iloc[start : start + batch_size].copy()
        try:
            imputed = model.impute(
                masked,
                k=min(200, len(masked)),
                max_retries=max_retries,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except ValueError as error:
            if "No objects to concatenate" not in str(error):
                raise
            imputed = pd.DataFrame(columns=table.model_columns)
        # Version 0.0.9 drops rows that exhaust retries. Preserve exact output
        # cardinality and expose the count rather than failing the entire suite.
        imputed = imputed.reindex(masked.index)
        missing = imputed.isna().any(axis=1)
        if missing.any():
            replacements = table.train.loc[:, table.model_columns].sample(
                n=int(missing.sum()), replace=True, random_state=seed + start
            )
            replacements.index = imputed.index[missing]
            for column in table.categorical_columns:
                constrained = masked.loc[missing, column].notna()
                constrained_index = constrained[constrained].index
                replacements.loc[constrained_index, column] = masked.loc[
                    constrained_index, column
                ]
            imputed.loc[missing, :] = replacements
            fallback_rows += int(missing.sum())
        chunks.append(imputed)
    generated = pd.concat(chunks).sort_index().reset_index(drop=True)
    generated = generated.loc[:, table.model_columns]
    repairs = {"numeric_nonfinite": 0, "categorical_unknown": 0}
    for column in table.numerical_columns:
        values = pd.to_numeric(generated[column], errors="coerce")
        invalid = ~np.isfinite(values.to_numpy(dtype=float))
        repairs["numeric_nonfinite"] += int(invalid.sum())
        if invalid.any():
            values.loc[invalid] = rng.choice(table.train[column].dropna(), invalid.sum())
        generated[column] = values
    for column, categories in zip(table.categorical_columns, table.encoder.categories_):
        valid = {str(value) for value in categories}
        values = generated[column].astype(str)
        invalid = ~values.isin(valid)
        repairs["categorical_unknown"] += int(invalid.sum())
        if invalid.any():
            values.loc[invalid] = rng.choice(table.train[column].astype(str), invalid.sum())
        generated[column] = values
    repairs["rows_exhausting_generation_retries"] = fallback_rows
    metadata.update({"method": "great_native_imputation", "seed": seed, "repairs": repairs})
    return generated.loc[:, table.train.columns], metadata
