"""Measure and plot how far fixed-query numerical violations miss their intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def distribution_summary(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=40)
    args = parser.parse_args()
    if args.bins < 2:
        raise ValueError("bins must be at least 2")

    samples = pd.read_csv(args.samples)
    with Path(args.query_file).open("r", encoding="utf-8") as stream:
        query = json.load(stream)
    numerical = [
        predicate
        for predicate in query["predicates"]
        if predicate["modality"] == "numeric"
    ]
    if not numerical:
        raise ValueError("the selected query has no numerical interval")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_column = []
    summaries = []
    for predicate in numerical:
        column = str(predicate["col"])
        lower, upper = map(float, predicate["values"])
        width = upper - lower
        if width <= 0:
            raise ValueError(f"{column} has a non-positive interval width")
        values = pd.to_numeric(samples[column]).to_numpy(dtype=float)
        below_distance = np.maximum(lower - values, 0.0)
        above_distance = np.maximum(values - upper, 0.0)
        distance = below_distance + above_distance
        relative_distance = distance / width
        side = np.full(len(values), "inside", dtype=object)
        side[below_distance > 0] = "below"
        side[above_distance > 0] = "above"
        violated = distance > 0
        per_column.append(
            pd.DataFrame(
                {
                    "row_index": np.arange(len(values)),
                    "column": column,
                    "raw_value": values,
                    "raw_lower": lower,
                    "raw_upper": upper,
                    "raw_interval_width": width,
                    "violation_side": side,
                    "is_violation": violated,
                    "raw_distance_to_interval": distance,
                    "distance_in_interval_widths": relative_distance,
                }
            )
        )
        violating_distance = distance[violated]
        violating_relative = relative_distance[violated]
        summaries.append(
            {
                "column": column,
                "raw_lower": lower,
                "raw_upper": upper,
                "raw_interval_width": width,
                "generated_rows": len(values),
                "violating_rows": int(violated.sum()),
                "violation_rate": float(violated.mean()),
                "below_rows": int((side == "below").sum()),
                "above_rows": int((side == "above").sum()),
                "raw_distance_among_violations": distribution_summary(
                    violating_distance
                ),
                "interval_widths_among_violations": distribution_summary(
                    violating_relative
                ),
            }
        )

    details = pd.concat(per_column, ignore_index=True)
    details.to_csv(output_dir / "numerical_violation_distances.csv", index=False)
    with (output_dir / "numerical_violation_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "query_id": query["query_id"],
                "distance_definition": "max(lower-x, 0, x-upper)",
                "columns": summaries,
            },
            stream,
            indent=2,
        )
        stream.write("\n")

    n_columns = len(numerical)
    fig, axes = plt.subplots(
        n_columns,
        3,
        figsize=(15, 4.2 * n_columns),
        squeeze=False,
        constrained_layout=True,
    )
    for row, (predicate, summary) in enumerate(zip(numerical, summaries)):
        column = str(predicate["col"])
        selected = details[details["column"] == column]
        violating = selected[selected["is_violation"]]
        lower, upper = summary["raw_lower"], summary["raw_upper"]

        axes[row, 0].hist(selected["raw_value"], bins=args.bins, color="tab:blue", alpha=0.8)
        axes[row, 0].axvspan(lower, upper, color="tab:green", alpha=0.25, label="valid interval")
        axes[row, 0].axvline(lower, color="tab:green", linestyle="--")
        axes[row, 0].axvline(upper, color="tab:green", linestyle="--")
        axes[row, 0].set_title(f"{column}: generated values")
        axes[row, 0].set_xlabel("Raw value")
        axes[row, 0].set_ylabel("Rows")
        axes[row, 0].legend()

        axes[row, 1].hist(
            violating["raw_distance_to_interval"],
            bins=args.bins,
            color="tab:orange",
            alpha=0.85,
        )
        axes[row, 1].set_title(
            f"Raw miss distance ({summary['violating_rows']} violations)"
        )
        axes[row, 1].set_xlabel("Distance to nearest valid endpoint")
        axes[row, 1].set_ylabel("Violating rows")

        axes[row, 2].hist(
            violating["distance_in_interval_widths"],
            bins=args.bins,
            color="tab:red",
            alpha=0.8,
        )
        axes[row, 2].set_title("Miss distance relative to interval width")
        axes[row, 2].set_xlabel("Number of interval widths outside")
        axes[row, 2].set_ylabel("Violating rows")

    fig.savefig(output_dir / "violation_distance_histogram.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"query_id": query["query_id"], "columns": summaries}, indent=2))
    print(f"Saved violation-magnitude diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
