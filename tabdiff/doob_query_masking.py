"""Predicate masking for structured-query Doob training and sampling."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from tabdiff.doob_query_suite import _category_key


def sample_predicate_mask(
    num_predicates: int,
    *,
    device: torch.device,
    random_active_probability: float,
    all_active_probability: float,
    all_inactive_probability: float,
) -> tuple[torch.Tensor, str]:
    """Draw one predicate mask shared by an entire optimizer-step batch."""
    probabilities = (
        random_active_probability,
        all_active_probability,
        all_inactive_probability,
    )
    if num_predicates <= 0:
        raise ValueError("a structured query must contain at least one predicate")
    if any(value < 0 or value > 1 for value in probabilities):
        raise ValueError("predicate-mask probabilities must lie in [0, 1]")
    if all_active_probability + all_inactive_probability > 1:
        raise ValueError("all-active and all-inactive probabilities cannot exceed one")
    draw = torch.rand((), device=device).item()
    if draw < all_active_probability:
        return torch.ones(num_predicates, dtype=torch.bool, device=device), "all_active"
    if draw < all_active_probability + all_inactive_probability:
        return torch.zeros(num_predicates, dtype=torch.bool, device=device), "all_inactive"
    return (
        torch.rand(num_predicates, device=device) < random_active_probability
    ), "random_subset"


def predicate_hit_matrix(frame: pd.DataFrame, specification: dict) -> np.ndarray:
    """Return raw-space satisfaction for every row and query predicate."""
    hits = []
    for predicate in specification.get("predicates", []):
        name = predicate["col"]
        if name not in frame.columns:
            raise ValueError(f"query column {name!r} is absent from the real table")
        if predicate["modality"] == "numeric" and predicate.get("op") == "between":
            lower, upper = map(float, predicate["values"])
            values = frame[name].to_numpy(dtype=np.float64)
            tolerance = 1e-7 * max(1.0, abs(lower), abs(upper))
            hit = (
                np.isfinite(values)
                & (values >= lower - tolerance)
                & (values <= upper + tolerance)
            )
        elif predicate["modality"] == "categorical" and predicate.get("op") == "in":
            allowed = {_category_key(value) for value in predicate["values"]}
            hit = frame[name].map(_category_key).isin(allowed).to_numpy()
        else:
            raise ValueError(f"unsupported predicate for {name!r}")
        hits.append(hit)
    if not hits:
        raise ValueError("structured query contains no predicates")
    return np.stack(hits, axis=1)


def eligible_indices_for_predicate_mask(
    hit_matrix: np.ndarray,
    core_indices: np.ndarray,
    predicate_mask: torch.Tensor,
) -> torch.Tensor:
    """Return core-training row ids satisfying every active predicate."""
    active = predicate_mask.detach().cpu().numpy().astype(bool)
    if hit_matrix.ndim != 2 or hit_matrix.shape[1] != len(active):
        raise ValueError("predicate hit matrix and mask have incompatible shapes")
    if active.any():
        selected = hit_matrix[core_indices][:, active].all(axis=1)
        eligible = core_indices[selected]
    else:
        eligible = core_indices
    if len(eligible) == 0:
        raise ValueError("sampled predicate mask has no core-training support")
    return torch.from_numpy(np.asarray(eligible, dtype=np.int64))


def mask_query_kwargs(
    query_kwargs: dict[str, torch.Tensor],
    specification: dict,
    predicate_mask: torch.Tensor,
    numerical_names: list[str],
    categorical_names: list[str],
) -> dict[str, torch.Tensor]:
    """Apply a specification-order predicate mask to model-order active flags."""
    predicates = specification.get("predicates", [])
    if predicate_mask.numel() != len(predicates):
        raise ValueError("predicate mask length does not match the query")
    output = {name: value.clone() for name, value in query_kwargs.items()}
    numerical_active = torch.zeros_like(output["query_numerical_active"])
    categorical_active = torch.zeros_like(output["query_categorical_active"])
    numerical_index = {name: index for index, name in enumerate(numerical_names)}
    categorical_index = {name: index for index, name in enumerate(categorical_names)}
    for is_active, predicate in zip(predicate_mask, predicates):
        if not bool(is_active.item()):
            continue
        name = predicate["col"]
        if predicate["modality"] == "numeric":
            numerical_active[numerical_index[name]] = 1
        elif predicate["modality"] == "categorical":
            categorical_active[categorical_index[name]] = 1
        else:
            raise ValueError(f"unsupported modality for {name!r}")
    output["query_numerical_active"] = numerical_active
    output["query_categorical_active"] = categorical_active
    return output


def masked_query_specification(specification: dict, predicate_mask: torch.Tensor) -> dict:
    """Create raw diagnostic metadata for only the selected predicates."""
    predicates = specification.get("predicates", [])
    active = predicate_mask.detach().cpu().numpy().astype(bool)
    if len(active) != len(predicates):
        raise ValueError("predicate mask length does not match the query")
    output = deepcopy(specification)
    output["parent_query_id"] = specification.get("query_id")
    output["parent_target_band"] = specification.get("target_band")
    output["predicate_mask"] = active.astype(int).tolist()
    output["predicates"] = [
        deepcopy(predicate)
        for predicate, keep in zip(predicates, active)
        if keep
    ]
    output["arity"] = int(active.sum())
    output.pop("columns", None)
    output["active_columns"] = [predicate["col"] for predicate in output["predicates"]]
    return output


def parse_predicate_mask(
    specification: dict,
    *,
    active_columns: str | None,
    predicate_mask: str | None,
    device: torch.device,
) -> torch.Tensor:
    """Resolve a user-facing column list or binary mask for sampling."""
    predicates = specification.get("predicates", [])
    if active_columns is not None and predicate_mask is not None:
        raise ValueError("provide either active-columns or predicate-mask, not both")
    if predicate_mask is not None:
        compact = predicate_mask.replace(",", "").replace(" ", "")
        if len(compact) != len(predicates) or any(value not in "01" for value in compact):
            raise ValueError(
                f"predicate-mask must contain {len(predicates)} binary entries"
            )
        return torch.tensor([value == "1" for value in compact], device=device)
    if active_columns is None or active_columns.strip().lower() == "all":
        return torch.ones(len(predicates), dtype=torch.bool, device=device)
    if active_columns.strip().lower() == "none":
        return torch.zeros(len(predicates), dtype=torch.bool, device=device)
    requested = {value.strip() for value in active_columns.split(",") if value.strip()}
    available = {predicate["col"] for predicate in predicates}
    unknown = requested - available
    if unknown:
        raise ValueError(f"columns are not predicates in this query: {sorted(unknown)}")
    return torch.tensor(
        [predicate["col"] in requested for predicate in predicates],
        dtype=torch.bool,
        device=device,
    )
