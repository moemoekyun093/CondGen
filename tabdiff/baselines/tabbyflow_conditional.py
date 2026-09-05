"""Training-free hard-query conditioning for a frozen TabbyFlow model.

TabbyFlow predicts a mean-field endpoint distribution: an isotropic Gaussian
for the numerical block and one categorical distribution per categorical
column.  For an axis-aligned query, conditioning that endpoint distribution is
analytic.  Numerical means become truncated-Gaussian means and categorical
probabilities are restricted to their allowed sets and renormalized.

This module intentionally imports the frozen reference implementation from
``external/tabbyflow_ft_periodic_reference``.  It does not modify or duplicate
the upstream model and it has no trainable parameters of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SIGMA_MIN = 0.01
REFERENCE_MODEL = (
    Path(__file__).resolve().parents[2]
    / "external"
    / "tabbyflow_ft_periodic_reference"
    / "tab-equiv-cond"
    / "tabequiv"
    / "models"
    / "ft_periodic.py"
)


@dataclass(frozen=True)
class EncodedQuery:
    query_id: str
    specification: dict[str, Any]
    numerical_lower: torch.Tensor
    numerical_upper: torch.Tensor
    numerical_active: torch.Tensor
    categorical_allowed: tuple[torch.Tensor, ...]
    categorical_active: torch.Tensor


@dataclass(frozen=True)
class TabbyFlowSchema:
    frame: pd.DataFrame
    info: dict[str, Any]
    numerical_indices: tuple[int, ...]
    categorical_indices: tuple[int, ...]
    categorical_levels: tuple[tuple[str, ...], ...]

    @property
    def columns(self) -> list[str]:
        return list(self.frame.columns)


def _load_module(path: Path, name: str) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import reference module from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_schema(data_dir: str | Path, info_file: str | Path) -> TabbyFlowSchema:
    """Reconstruct the exact raw column/category ordering used during training."""
    data_dir = Path(data_dir)
    with Path(info_file).open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    frames = []
    for name in ("train.csv", "val.csv", "test.csv"):
        path = data_dir / name
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"no train/val/test CSVs found in {data_dir}")
    frame = pd.concat(frames, ignore_index=True)
    numerical = list(map(int, info["num_col_idx"]))
    categorical = list(map(int, info["cat_col_idx"]))
    target = list(map(int, info.get("target_col_idx", [])))
    if info.get("task_type") == "regression":
        numerical = sorted(set(numerical) | set(target))
    else:
        categorical = sorted(set(categorical) | set(target))
    levels = tuple(
        tuple(sorted(set(frame.iloc[:, index].astype(str))))
        for index in categorical
    )
    return TabbyFlowSchema(
        frame=frame,
        info=info,
        numerical_indices=tuple(numerical),
        categorical_indices=tuple(categorical),
        categorical_levels=levels,
    )


def load_frozen_model(
    run_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
    reference_model: str | Path = REFERENCE_MODEL,
) -> tuple[nn.Module, dict[str, Any], Path]:
    """Build FT-periodic from the run config and load the EMA checkpoint strictly."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("family_key") != "unimod_efvfm":
        raise ValueError("the run is not an official unimod_efvfm/TabbyFlow run")
    if config.get("denoiser") != "ft_periodic":
        raise ValueError("this adapter currently requires the FT-periodic TabbyFlow arm")
    checkpoint_path = (
        Path(checkpoint)
        if checkpoint is not None
        else run_dir / config.get("selected_checkpoint", "checkpoints/best_ema.pt")
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    reference = _load_module(Path(reference_model), "_tabbyflow_ft_periodic_reference")
    model = reference.UniModMLPFTPeriodic(
        d_numerical=int(config["d_num"]),
        categories=[int(value) for value in config["categories"]],
        **dict(config["ft_kw"]),
    )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("ema") or payload.get("raw")
    if state is None:
        state = payload

    prefixes = ("_vf_fn.", "module._vf_fn.", "ema_model._vf_fn.")
    core_state = None
    for prefix in prefixes:
        selected = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if selected:
            core_state = selected
            break
    if core_state is None:
        core_state = state
    missing, unexpected = model.load_state_dict(core_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch: {len(missing)} missing and "
            f"{len(unexpected)} unexpected keys; first missing={list(missing)[:3]}, "
            f"first unexpected={list(unexpected)[:3]}"
        )
    model = model.to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config, checkpoint_path


def resolve_transform(run_dir: str | Path, transform_file: str | Path | None):
    """Load the training-fitted QuantileTransformer without trusting stale abs paths."""
    import joblib

    path = Path(transform_file) if transform_file else Path(run_dir) / "num_transform.joblib"
    if not path.is_file():
        raise FileNotFoundError(
            f"numerical transform not found at {path}; pass --transform-file explicitly"
        )
    return joblib.load(path), path


def encode_query(
    query: dict[str, Any],
    schema: TabbyFlowSchema,
    numerical_transform,
    *,
    d_numerical: int,
    categories: list[int] | tuple[int, ...],
) -> EncodedQuery:
    """Encode raw interval/set predicates into the frozen model's column order."""
    if len(schema.numerical_indices) != d_numerical:
        raise ValueError("config and schema disagree on the number of numerical columns")
    categories = [int(value) for value in categories]
    if tuple(map(len, schema.categorical_levels)) != tuple(categories):
        raise ValueError(
            "config category cardinalities do not match the levels reconstructed "
            "from the complete dataset"
        )
    num_names = [schema.columns[index] for index in schema.numerical_indices]
    cat_names = [schema.columns[index] for index in schema.categorical_indices]
    num_lookup = {name: index for index, name in enumerate(num_names)}
    cat_lookup = {name: index for index, name in enumerate(cat_names)}

    raw_lower = np.zeros(d_numerical, dtype=np.float64)
    raw_upper = np.zeros(d_numerical, dtype=np.float64)
    num_active = np.zeros(d_numerical, dtype=bool)
    cat_active = np.zeros(len(categories), dtype=bool)
    allowed = [np.ones(count, dtype=bool) for count in categories]

    for predicate in query["predicates"]:
        name = str(predicate["col"])
        modality = predicate["modality"]
        if modality == "numeric" and predicate.get("op") == "between":
            if name not in num_lookup:
                raise ValueError(f"unknown numerical query column {name!r}")
            index = num_lookup[name]
            lower, upper = map(float, predicate["values"])
            if lower > upper:
                raise ValueError(f"reversed interval for {name}: {lower} > {upper}")
            raw_lower[index], raw_upper[index] = lower, upper
            num_active[index] = True
        elif modality == "categorical" and predicate.get("op") == "in":
            if name not in cat_lookup:
                raise ValueError(f"unknown categorical query column {name!r}")
            index = cat_lookup[name]
            level_lookup = {
                str(value).strip(): position
                for position, value in enumerate(schema.categorical_levels[index])
            }
            selected = np.zeros(categories[index], dtype=bool)
            for value in predicate["values"]:
                key = str(value).strip()
                if key not in level_lookup:
                    raise ValueError(f"unknown category {value!r} for {name}")
                selected[level_lookup[key]] = True
            if not selected.any():
                raise ValueError(f"empty allowed set for {name}")
            allowed[index] = selected
            cat_active[index] = True
        else:
            raise ValueError(f"unsupported predicate for {name!r}: {predicate}")

    transformed = numerical_transform.transform(np.stack((raw_lower, raw_upper)))
    lower = transformed[0].astype(np.float32)
    upper = transformed[1].astype(np.float32)
    return EncodedQuery(
        query_id=str(query["query_id"]),
        specification=query,
        numerical_lower=torch.from_numpy(lower),
        numerical_upper=torch.from_numpy(upper),
        numerical_active=torch.from_numpy(num_active),
        categorical_allowed=tuple(torch.from_numpy(value) for value in allowed),
        categorical_active=torch.from_numpy(cat_active),
    )


def truncated_normal_mean(
    mean: torch.Tensor,
    std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Mean of N(mean, std^2) restricted to [lower, upper], computed in fp64."""
    original_dtype = mean.dtype
    mu = mean.double()
    sigma = std.double().clamp_min(torch.finfo(torch.float64).tiny)
    lo = lower.double()
    hi = upper.double()
    alpha = (lo - mu) / sigma
    beta = (hi - mu) / sigma
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    cdf_alpha = 0.5 * (1.0 + torch.erf(alpha * inv_sqrt_two))
    cdf_beta = 0.5 * (1.0 + torch.erf(beta * inv_sqrt_two))
    normalizer = cdf_beta - cdf_alpha
    inv_sqrt_two_pi = 1.0 / math.sqrt(2.0 * math.pi)
    pdf_alpha = torch.exp(-0.5 * alpha.square()) * inv_sqrt_two_pi
    pdf_beta = torch.exp(-0.5 * beta.square()) * inv_sqrt_two_pi
    candidate = mu + sigma * (pdf_alpha - pdf_beta) / normalizer.clamp_min(1e-14)

    # If both CDF values numerically coincide in an extreme tail, the conditional
    # mass lies next to the interval endpoint closest to the original mean.
    tail_fallback = torch.where(alpha > 0, lo, torch.where(beta < 0, hi, mu))
    candidate = torch.where(
        (normalizer > 1e-14) & torch.isfinite(candidate), candidate, tail_fallback
    )
    candidate = torch.maximum(torch.minimum(candidate, hi), lo)
    return candidate.to(original_dtype)


class ConditionalTabbyFlowVelocity(nn.Module):
    """Analytic Doob/endpoint-conditioning adapter around a frozen TabbyFlow net."""

    def __init__(self, model: nn.Module, query: EncodedQuery, sigma_min: float = SIGMA_MIN):
        super().__init__()
        self.model = model
        self.d_numerical = int(model.d_numerical)
        self.categories = [int(value) for value in model.categories]
        self.sigma_min = float(sigma_min)
        self.register_buffer("numerical_lower", query.numerical_lower)
        self.register_buffer("numerical_upper", query.numerical_upper)
        self.register_buffer("numerical_active", query.numerical_active.bool())
        self.register_buffer("categorical_active", query.categorical_active.bool())
        for index, value in enumerate(query.categorical_allowed):
            self.register_buffer(f"categorical_allowed_{index}", value.bool())

    def conditioned_predictions(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch_t = t.reshape(-1).expand(x.shape[0]) if t.numel() == 1 else t.reshape(-1)
        x_num = x[:, : self.d_numerical]
        x_cat = x[:, self.d_numerical :]
        mean, logits = self.model(x_num, x_cat, batch_t)

        if self.d_numerical:
            # This is the covariance used by ExpVFM._mvgloss in the imported model.
            variance = 1.0 - (1.0 - self.sigma_min) * batch_t.square()
            std = variance.clamp_min(1e-12).sqrt().unsqueeze(1)
            conditioned = truncated_normal_mean(
                mean,
                std,
                self.numerical_lower.unsqueeze(0),
                self.numerical_upper.unsqueeze(0),
            )
            mean = torch.where(self.numerical_active.unsqueeze(0), conditioned, mean)

        probabilities = []
        offset = 0
        for index, count in enumerate(self.categories):
            column_logits = logits[:, offset : offset + count]
            if self.categorical_active[index]:
                allowed = getattr(self, f"categorical_allowed_{index}")
                column_logits = column_logits.masked_fill(~allowed.unsqueeze(0), -torch.inf)
            probabilities.append(F.softmax(column_logits, dim=-1))
            offset += count
        return mean, probabilities

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_t = t.reshape(-1).expand(x.shape[0]) if t.numel() == 1 else t.reshape(-1)
        x_num = x[:, : self.d_numerical]
        x_cat = x[:, self.d_numerical :]
        mean, probabilities = self.conditioned_predictions(t, x)
        coefficient = 1.0 - self.sigma_min
        denominator = 1.0 - coefficient * batch_t.unsqueeze(1)
        v_num = (mean - coefficient * x_num) / denominator
        v_cat_parts = []
        offset = 0
        for probabilities_k, count in zip(probabilities, self.categories):
            x_k = x_cat[:, offset : offset + count]
            v_cat_parts.append((probabilities_k - coefficient * x_k) / denominator)
            offset += count
        v_cat = torch.cat(v_cat_parts, dim=1) if v_cat_parts else x_cat
        return torch.cat((v_num, v_cat), dim=1)


@torch.no_grad()
def sample_conditioned_state(
    velocity: ConditionalTabbyFlowVelocity,
    num_samples: int,
    *,
    batch_size: int,
    seed: int,
    solver: str = "heun",
    steps: int = 50,
    terminal_time: float = 0.999,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> torch.Tensor:
    """Sample raw ``[numericals | one-hot blocks]`` states from the conditional ODE."""
    if num_samples <= 0 or batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive")
    if solver not in {"dopri5", "euler", "heun"}:
        raise ValueError(f"unsupported solver {solver!r}")
    if solver != "dopri5" and steps <= 0:
        raise ValueError("steps must be positive for a fixed-step solver")
    device = next(velocity.model.parameters()).device
    dtype = next(velocity.model.parameters()).dtype
    dimension = velocity.d_numerical + sum(velocity.categories)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    outputs = []
    for start in range(0, num_samples, batch_size):
        size = min(batch_size, num_samples - start)
        state = torch.randn(size, dimension, generator=generator, device=device, dtype=dtype)
        if solver == "dopri5":
            from torchdiffeq import odeint

            times = torch.tensor([0.0, terminal_time], device=device, dtype=dtype)
            state = odeint(
                velocity, state, times, method="dopri5", rtol=rtol, atol=atol
            )[-1]
        else:
            dt = terminal_time / steps
            for index in range(steps):
                t = torch.tensor(index * dt, device=device, dtype=dtype)
                first = velocity(t, state)
                if solver == "euler":
                    state = state + dt * first
                else:
                    predicted = state + dt * first
                    second = velocity(t + dt, predicted)
                    state = state + 0.5 * dt * (first + second)
        outputs.append(state.cpu())
    return torch.cat(outputs, dim=0)


def decode_state(
    state: torch.Tensor,
    schema: TabbyFlowSchema,
    numerical_transform,
    categories: list[int] | tuple[int, ...],
) -> pd.DataFrame:
    """Decode a TabbyFlow state back to the original table schema and units."""
    values = state.detach().cpu().float().numpy()
    d_numerical = len(schema.numerical_indices)
    numerical = numerical_transform.inverse_transform(values[:, :d_numerical])
    output = pd.DataFrame(index=range(len(values)), columns=schema.columns, dtype=object)
    for local, raw_index in enumerate(schema.numerical_indices):
        source = pd.to_numeric(schema.frame.iloc[:, raw_index], errors="coerce").dropna()
        column = numerical[:, local]
        if len(source) and (source % 1 == 0).all():
            column = np.rint(column).clip(float(source.min()), float(source.max())).astype(np.int64)
        else:
            if len(source) and (source == 0).any():
                nonzero = np.abs(source[source != 0].to_numpy(float))
                scale = float(nonzero.min()) if len(nonzero) else 1.0
                column = np.where(np.abs(column) < 0.5 * scale, 0.0, column)
        output.iloc[:, raw_index] = column

    offset = d_numerical
    categories = [int(value) for value in categories]
    for local, (raw_index, count) in enumerate(zip(schema.categorical_indices, categories)):
        codes = values[:, offset : offset + count].argmax(axis=1)
        labels = np.asarray(schema.categorical_levels[local], dtype=object)[codes]
        source = schema.frame.iloc[:, raw_index]
        if pd.api.types.is_bool_dtype(source):
            decoded = np.asarray([str(value) == "True" for value in labels], dtype=bool)
        elif pd.api.types.is_integer_dtype(source):
            decoded = labels.astype(np.int64)
        elif pd.api.types.is_float_dtype(source):
            decoded = labels.astype(float)
        else:
            decoded = labels
        output.iloc[:, raw_index] = decoded
        offset += count
    return output.infer_objects()
