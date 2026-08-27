"""Model-space loading for full-arity numerical-interval/category-set queries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class StructuredQuery:
    query_id: str
    specification: Dict[str, Any]
    numerical_lower: Tensor
    numerical_upper: Tensor
    numerical_active: Tensor
    categorical_allowed: Tensor
    categorical_active: Tensor
    eligible_indices: Tensor

    def model_kwargs(self, device: torch.device, dtype: torch.dtype) -> Dict[str, Tensor]:
        return {
            "query_lower": self.numerical_lower.to(device=device, dtype=dtype),
            "query_upper": self.numerical_upper.to(device=device, dtype=dtype),
            "query_numerical_active": self.numerical_active.to(device=device, dtype=dtype),
            "query_categorical_allowed": self.categorical_allowed.to(
                device=device, dtype=dtype
            ),
            "query_categorical_active": self.categorical_active.to(
                device=device, dtype=dtype
            ),
        }


def _model_column_names(info: Dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return feature names in the internal transformed tensor order.

    TabDiff prepends the target to the modality that contains it. In a
    classification dataset this makes categorical model order
    ``target, categorical features`` even though the recovered raw table puts
    the target last.
    """
    mapping = {int(key): value for key, value in info["idx_name_mapping"].items()}
    numerical_indices = list(info["num_col_idx"])
    categorical_indices = list(info["cat_col_idx"])
    target_indices = list(info["target_col_idx"])
    if info["task_type"] == "regression":
        numerical_indices = target_indices + numerical_indices
    else:
        categorical_indices = target_indices + categorical_indices
    return (
        [mapping[int(index)] for index in numerical_indices],
        [mapping[int(index)] for index in categorical_indices],
    )


def _categorical_value_maps(dataset) -> list[dict[str, int]]:
    counts = [int(value) for value in dataset.categories]
    n_columns = len(counts)
    max_count = max(counts, default=0)
    encoded = np.zeros((max_count, n_columns), dtype=np.int64)
    mappings = []
    for column, count in enumerate(counts):
        encoded[:count, column] = np.arange(count)
    decoded = dataset.cat_inverse(encoded.copy())
    for column, count in enumerate(counts):
        mapping = {}
        for class_index in range(count):
            value = decoded[class_index, column]
            mapping[_category_key(value)] = class_index
        mappings.append(mapping)
    return mappings


def _category_key(value: Any) -> str:
    """Canonicalize JSON/scikit-learn representations of category labels."""
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def _transform_numerical_bounds(dataset, lower: np.ndarray, upper: np.ndarray):
    if dataset.num_transform is None:
        return lower.astype(np.float32), upper.astype(np.float32)
    # Bounds describe clean x_0 values. Do not apply the stochastic integer
    # dequantizer here; use the deterministic fitted normalization directly.
    transformed = dataset.num_transform.transform(np.stack((lower, upper)))
    return transformed[0].astype(np.float32), transformed[1].astype(np.float32)


def load_structured_query_suite(
    query_dir: str | Path,
    runtime,
    *,
    query_ids: Iterable[str] | None = None,
    target_band: float | None = None,
) -> list[StructuredQuery]:
    """Load accepted query JSONs and their exact core-training references."""
    query_dir = Path(query_dir)
    if not query_dir.is_dir():
        raise FileNotFoundError(f"query directory does not exist: {query_dir}")
    requested = None if query_ids is None else set(query_ids)
    dataset = runtime.dataset
    d_numerical = dataset.d_numerical
    category_counts = [int(value) for value in dataset.categories]
    numerical_names, categorical_names = _model_column_names(runtime.info)
    if len(numerical_names) != d_numerical:
        raise ValueError("metadata numerical order does not match transformed data")
    if len(categorical_names) != len(category_counts):
        raise ValueError("metadata categorical order does not match transformed data")
    numerical_index = {name: index for index, name in enumerate(numerical_names)}
    categorical_index = {name: index for index, name in enumerate(categorical_names)}
    category_maps = _categorical_value_maps(dataset)

    split_root = query_dir.parent
    core_indices_path = split_root / "splits" / "train_idx.npy"
    if not core_indices_path.is_file():
        raise FileNotFoundError(f"query-suite core split is missing: {core_indices_path}")
    core_indices = np.load(core_indices_path)

    queries = []
    for path in sorted(query_dir.glob("qf_*.json")):
        with path.open("r", encoding="utf-8") as stream:
            specification = json.load(stream)
        query_id = specification["query_id"]
        if requested is not None and query_id not in requested:
            continue
        if target_band is not None and not np.isclose(
            float(specification.get("target_band", float("nan"))),
            float(target_band),
            rtol=0.0,
            atol=1e-12,
        ):
            continue
        if specification.get("dataset") != runtime.dataset.dataname:
            raise ValueError(f"{path} belongs to a different dataset")
        if not specification.get("accepted", True):
            continue

        raw_lower = np.zeros(d_numerical, dtype=np.float64)
        raw_upper = np.zeros(d_numerical, dtype=np.float64)
        numerical_active = np.zeros(d_numerical, dtype=np.float32)
        categorical_active = np.zeros(len(category_counts), dtype=np.float32)
        allowed_parts = [np.zeros(count, dtype=np.float32) for count in category_counts]

        for predicate in specification["predicates"]:
            name = predicate["col"]
            if predicate["modality"] == "numeric":
                if predicate.get("op") != "between":
                    raise ValueError(f"unsupported numerical predicate in {query_id}")
                index = numerical_index[name]
                raw_lower[index], raw_upper[index] = map(float, predicate["values"])
                numerical_active[index] = 1.0
            elif predicate["modality"] == "categorical":
                if predicate.get("op") != "in":
                    raise ValueError(f"unsupported categorical predicate in {query_id}")
                index = categorical_index[name]
                categorical_active[index] = 1.0
                for raw_value in predicate["values"]:
                    key = _category_key(raw_value)
                    if key not in category_maps[index]:
                        raise ValueError(
                            f"unknown value {raw_value!r} for {name} in {query_id}"
                        )
                    allowed_parts[index][category_maps[index][key]] = 1.0
            else:
                raise ValueError(f"unsupported predicate modality in {query_id}")

        lower, upper = _transform_numerical_bounds(dataset, raw_lower, raw_upper)
        reference_path = query_dir / "refs" / f"{query_id}.npz"
        if not reference_path.is_file():
            raise FileNotFoundError(f"training reference is missing: {reference_path}")
        with np.load(reference_path) as reference:
            train_rows = reference["train_rows"].astype(np.int64)
        # Reference arrays store row ids in the original provided TabDiff
        # training partition (not positions inside ``train_idx``).
        eligible_indices = train_rows
        if eligible_indices.size == 0:
            raise ValueError(f"query {query_id} has no core-training support")
        if not np.isin(eligible_indices, core_indices).all():
            raise ValueError(f"query {query_id} references rows outside the core split")
        if eligible_indices.min() < 0 or eligible_indices.max() >= len(dataset):
            raise ValueError(
                f"query-suite indices for {query_id} do not address the TabDiff training set"
            )

        queries.append(
            StructuredQuery(
                query_id=query_id,
                specification=specification,
                numerical_lower=torch.from_numpy(lower),
                numerical_upper=torch.from_numpy(upper),
                numerical_active=torch.from_numpy(numerical_active),
                categorical_allowed=torch.from_numpy(np.concatenate(allowed_parts)),
                categorical_active=torch.from_numpy(categorical_active),
                eligible_indices=torch.from_numpy(eligible_indices),
            )
        )
    if requested:
        found = {query.query_id for query in queries}
        missing = requested - found
        if missing:
            raise ValueError(f"requested queries were not loaded: {sorted(missing)}")
    if not queries:
        band_message = "" if target_band is None else f" at target band {target_band}"
        raise ValueError(
            f"no accepted structured queries found in {query_dir}{band_message}"
        )
    return queries
