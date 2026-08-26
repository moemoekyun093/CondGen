"""Raw-space diagnostics shared by Doob sample generation and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def raw_constraint_hits(
    frame: pd.DataFrame,
    column_specs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-value and per-row satisfaction in user-facing raw units."""
    hits = []
    for spec in column_specs:
        name = spec.get("name")
        lower = spec.get("raw_lower")
        upper = spec.get("raw_upper")
        if name not in frame.columns or (lower is None and upper is None):
            raise ValueError(f"raw constraint metadata is incomplete for column {name!r}")
        values = frame[name].to_numpy(dtype=np.float64)
        finite_bounds = [float(value) for value in (lower, upper) if value is not None]
        scale = max(1.0, *(abs(value) for value in finite_bounds))
        tolerance = 1e-7 * scale
        column_hits = np.isfinite(values)
        if lower is not None:
            column_hits &= values >= float(lower) - tolerance
        if upper is not None:
            column_hits &= values <= float(upper) + tolerance
        hits.append(column_hits)
    if not hits:
        raise ValueError("query does not contain any numerical column constraints")
    per_value = np.stack(hits, axis=1)
    return per_value, per_value.all(axis=1)


def raw_constraint_report(
    frame: pd.DataFrame,
    query: dict,
) -> tuple[dict, np.ndarray]:
    """Build a serializable raw-space report and return the joint row mask."""
    column_specs = query.get("columns", [])
    per_value, rows_satisfied = raw_constraint_hits(frame, column_specs)
    per_column_rates = per_value.mean(axis=0)
    report = {
        "constraint_id": query.get("constraint_id"),
        "evaluation_space": "raw generated table",
        "num_rows": len(frame),
        "joint_hit_rate": float(rows_satisfied.mean()),
        "all_rows_satisfy": bool(rows_satisfied.all()),
        "rows_satisfying": int(rows_satisfied.sum()),
        "per_column": [],
    }
    for index, spec in enumerate(column_specs):
        report["per_column"].append(
            {
                "model_index": spec.get("model_index", index),
                "name": spec.get("name", f"numerical_{index}"),
                "hit_rate": float(per_column_rates[index]),
                "raw_lower": spec.get("raw_lower"),
                "raw_upper": spec.get("raw_upper"),
            }
        )
    return report, rows_satisfied


def compare_correlation_matrices(
    left: pd.DataFrame,
    right: pd.DataFrame,
    top_k: int = 10,
) -> dict:
    """Compare upper triangles while safely excluding undefined correlations."""
    if list(left.columns) != list(right.columns):
        raise ValueError("correlation matrices have different columns")
    row_idx, col_idx = np.triu_indices(len(left.columns), k=1)
    left_values = left.to_numpy()[row_idx, col_idx]
    right_values = right.to_numpy()[row_idx, col_idx]
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    differences = right_values - left_values
    valid_differences = differences[valid]
    valid_left = left_values[valid]
    valid_right = right_values[valid]
    if valid.sum() >= 2 and np.std(valid_left) > 0 and np.std(valid_right) > 0:
        structure_correlation = float(np.corrcoef(valid_left, valid_right)[0, 1])
    else:
        structure_correlation = None

    valid_positions = np.flatnonzero(valid)
    ranked = valid_positions[np.argsort(np.abs(differences[valid_positions]))[::-1]]
    top_changes = []
    for position in ranked[:top_k]:
        i = row_idx[position]
        j = col_idx[position]
        top_changes.append(
            {
                "column_1": str(left.columns[i]),
                "column_2": str(left.columns[j]),
                "left_correlation": float(left_values[position]),
                "right_correlation": float(right_values[position]),
                "difference": float(differences[position]),
                "absolute_difference": float(abs(differences[position])),
            }
        )
    return {
        "matrix_upper_triangle_correlation": structure_correlation,
        "mean_absolute_correlation_change": (
            float(np.mean(np.abs(valid_differences))) if len(valid_differences) else None
        ),
        "max_absolute_correlation_change": (
            float(np.max(np.abs(valid_differences))) if len(valid_differences) else None
        ),
        "compared_pairs": int(valid.sum()),
        "undefined_pairs_excluded": int(len(valid) - valid.sum()),
        "top_absolute_changes": top_changes,
    }
