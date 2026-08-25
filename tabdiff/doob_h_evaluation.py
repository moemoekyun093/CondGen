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
        if name not in frame.columns or lower is None or upper is None:
            raise ValueError(f"raw constraint metadata is incomplete for column {name!r}")
        values = frame[name].to_numpy(dtype=np.float64)
        scale = max(1.0, abs(float(lower)), abs(float(upper)))
        tolerance = 1e-7 * scale
        hits.append(
            (values >= float(lower) - tolerance)
            & (values <= float(upper) + tolerance)
        )
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
