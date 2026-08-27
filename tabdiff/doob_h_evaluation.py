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
        # The all-inactive mask is the unconditional anchor: its empty
        # intersection is satisfied by every row.
        return (
            np.empty((len(frame), 0), dtype=bool),
            np.ones(len(frame), dtype=bool),
        )
    per_value = np.stack(hits, axis=1)
    return per_value, per_value.all(axis=1)


def raw_constraint_report(
    frame: pd.DataFrame,
    query: dict,
) -> tuple[dict, np.ndarray]:
    """Build a serializable raw-space report and return the joint row mask."""
    column_specs = query.get("columns", [])
    predicates = query.get("predicates", [])
    if predicates:
        hits = []
        column_specs = []
        for predicate in predicates:
            name = predicate["col"]
            if name not in frame.columns:
                raise ValueError(f"query column {name!r} is absent from the table")
            if predicate["modality"] == "numeric" and predicate.get("op") == "between":
                lower, upper = map(float, predicate["values"])
                values = frame[name].to_numpy(dtype=np.float64)
                scale = max(1.0, abs(lower), abs(upper))
                tolerance = 1e-7 * scale
                hit = (
                    np.isfinite(values)
                    & (values >= lower - tolerance)
                    & (values <= upper + tolerance)
                )
                column_specs.append(
                    {"name": name, "raw_lower": lower, "raw_upper": upper}
                )
            elif predicate["modality"] == "categorical" and predicate.get("op") == "in":
                allowed = {str(value) for value in predicate["values"]}
                hit = frame[name].astype(str).isin(allowed).to_numpy()
                column_specs.append({"name": name, "allowed_values": sorted(allowed)})
            else:
                raise ValueError(f"unsupported query predicate for {name!r}")
            hits.append(hit)
        per_value = np.stack(hits, axis=1) if hits else np.empty((len(frame), 0), bool)
        rows_satisfied = per_value.all(axis=1) if hits else np.ones(len(frame), bool)
    else:
        per_value, rows_satisfied = raw_constraint_hits(frame, column_specs)
    per_column_rates = per_value.mean(axis=0)
    report = {
        "constraint_id": query.get("constraint_id", query.get("query_id")),
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
                "allowed_values": spec.get("allowed_values"),
            }
        )
    return report, rows_satisfied


def raw_modality_constraint_report(frame: pd.DataFrame, query: dict) -> dict:
    """Separate joint and per-predicate miss rates by query modality."""
    predicates = query.get("predicates", [])
    if not predicates:
        raise ValueError("modality diagnostics require predicate-style query metadata")
    output = {}
    for modality in ("numeric", "categorical"):
        selected = [
            predicate
            for predicate in predicates
            if predicate.get("modality") == modality
        ]
        if not selected:
            output[modality] = {
                "num_constraints": 0,
                "joint_hit_rate": 1.0,
                "joint_miss_rate": 0.0,
                "mean_per_constraint_hit_rate": 1.0,
                "mean_per_constraint_miss_rate": 0.0,
            }
            continue
        report, _ = raw_constraint_report(frame, {"predicates": selected})
        per_constraint_hits = [
            float(column["hit_rate"]) for column in report["per_column"]
        ]
        joint_hit_rate = float(report["joint_hit_rate"])
        mean_hit_rate = float(np.mean(per_constraint_hits))
        output[modality] = {
            "num_constraints": len(selected),
            "joint_hit_rate": joint_hit_rate,
            "joint_miss_rate": 1.0 - joint_hit_rate,
            "mean_per_constraint_hit_rate": mean_hit_rate,
            "mean_per_constraint_miss_rate": 1.0 - mean_hit_rate,
        }
    return output


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
