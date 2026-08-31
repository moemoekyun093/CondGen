"""Measure and plot fixed-query numerical violation magnitude in two spaces."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def distribution_summary(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {key: None for key in ("mean", "median", "p90", "p95", "max")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def numerical_column_names(info: dict) -> list[str]:
    mapping = {int(key): value for key, value in info["idx_name_mapping"].items()}
    indices = list(info["num_col_idx"])
    if info["task_type"] == "regression":
        indices = list(info["target_col_idx"]) + indices
    return [mapping[int(index)] for index in indices]


def load_tabdiff_numerical_transform(args: argparse.Namespace, info: dict):
    # Imported lazily so syntax checks do not need the cluster's torch runtime.
    from utils_train import TabDiffDataset

    with Path(args.base_config).open("rb") as stream:
        config = pickle.load(stream)
    dataset = TabDiffDataset(
        args.dataname,
        args.data_dir,
        info,
        isTrain=True,
        dequant_dist=config["data"]["dequant_dist"],
        int_dequant_factor=config["data"]["int_dequant_factor"],
    )
    if dataset.num_transform is None:
        raise ValueError("TabDiff has no fitted numerical transformation")
    return dataset.num_transform


def transform_frame(frame, numerical_names, numerical_transform) -> np.ndarray:
    missing = [name for name in numerical_names if name not in frame.columns]
    if missing:
        raise ValueError(f"sample CSV is missing numerical columns: {missing}")
    raw = frame[numerical_names].apply(pd.to_numeric).to_numpy(dtype=float)
    return numerical_transform.transform(raw)


def transformed_bounds(predicates, numerical_names, numerical_transform):
    # QuantileTransformer operates feature-wise. Zero placeholders reproduce
    # the query loader and cannot affect an active column's transformed bound.
    lower = np.zeros(len(numerical_names), dtype=float)
    upper = np.zeros(len(numerical_names), dtype=float)
    by_name = {name: index for index, name in enumerate(numerical_names)}
    for predicate in predicates:
        index = by_name[str(predicate["col"])]
        lower[index], upper[index] = map(float, predicate["values"])
    transformed = numerical_transform.transform(np.stack((lower, upper)))
    return {
        str(predicate["col"]): (
            float(transformed[0, by_name[str(predicate["col"])]]),
            float(transformed[1, by_name[str(predicate["col"])]]),
        )
        for predicate in predicates
    }


def column_diagnostics(*, values, column, method, lower, upper, space):
    width = upper - lower
    if width <= 0:
        raise ValueError(f"{column} has a non-positive {space} interval width")
    below_distance = np.maximum(lower - values, 0.0)
    above_distance = np.maximum(values - upper, 0.0)
    distance = below_distance + above_distance
    relative_distance = distance / width
    side = np.full(len(values), "inside", dtype=object)
    side[below_distance > 0] = "below"
    side[above_distance > 0] = "above"
    violated = distance > 0
    details = pd.DataFrame(
        {
            "row_index": np.arange(len(values)),
            "method": method,
            "column": column,
            "space": space,
            "value": values,
            "lower": lower,
            "upper": upper,
            "interval_width": width,
            "violation_side": side,
            "is_violation": violated,
            "distance_to_interval": distance,
            "distance_in_interval_widths": relative_distance,
        }
    )
    summary = {
        "method": method,
        "column": column,
        "space": space,
        "lower": lower,
        "upper": upper,
        "interval_width": width,
        "generated_rows": len(values),
        "violating_rows": int(violated.sum()),
        "violation_rate": float(violated.mean()),
        "below_rows": int((side == "below").sum()),
        "above_rows": int((side == "above").sum()),
        "distance_among_violations": distribution_summary(distance[violated]),
        "interval_widths_among_violations": distribution_summary(
            relative_distance[violated]
        ),
    }
    return details, summary


def plot_space(details, predicates, summaries, *, space, bins, output):
    methods = list(dict.fromkeys(details["method"]))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    selected_space = details[details["space"] == space]
    space_summaries = [item for item in summaries if item["space"] == space]
    fig, axes = plt.subplots(
        len(predicates), 3, figsize=(15, 4.2 * len(predicates)),
        squeeze=False, constrained_layout=True,
    )
    for row, predicate in enumerate(predicates):
        column = str(predicate["col"])
        selected_column = selected_space[selected_space["column"] == column]
        summary = next(item for item in space_summaries if item["column"] == column)
        lower, upper = summary["lower"], summary["upper"]
        violating_column = selected_column[selected_column["is_violation"]]
        value_edges = np.histogram_bin_edges(selected_column["value"], bins=bins)
        if len(violating_column):
            distance_edges = np.histogram_bin_edges(
                violating_column["distance_to_interval"], bins=bins
            )
            relative_edges = np.histogram_bin_edges(
                violating_column["distance_in_interval_widths"], bins=bins
            )
        else:
            distance_edges = relative_edges = np.linspace(0.0, 1.0, bins + 1)
        for method_index, method in enumerate(methods):
            selected = selected_column[selected_column["method"] == method]
            violating = selected[selected["is_violation"]]
            color = colors[method_index % len(colors)]
            axes[row, 0].hist(
                selected["value"], bins=value_edges, histtype="step", linewidth=1.8,
                color=color, label=method,
            )
            axes[row, 1].hist(
                violating["distance_to_interval"], bins=distance_edges, histtype="step",
                linewidth=1.8, color=color,
                label=f"{method} (n={len(violating)})",
            )
            axes[row, 2].hist(
                violating["distance_in_interval_widths"], bins=relative_edges,
                histtype="step", linewidth=1.8, color=color, label=method,
            )
        axes[row, 0].axvspan(
            lower, upper, color="tab:green", alpha=0.18, label="valid interval"
        )
        axes[row, 0].axvline(lower, color="tab:green", linestyle="--")
        axes[row, 0].axvline(upper, color="tab:green", linestyle="--")
        axes[row, 0].set_title(f"{column}: generated values")
        axes[row, 0].set_xlabel(f"{space} value")
        axes[row, 0].set_ylabel("Rows")
        axes[row, 0].legend()
        axes[row, 1].set_title(f"{space.capitalize()} miss distance")
        axes[row, 1].set_xlabel("Distance to nearest valid endpoint")
        axes[row, 1].set_ylabel("Violating rows")
        axes[row, 1].legend()
        axes[row, 2].set_title("Miss distance relative to interval width")
        axes[row, 2].set_xlabel("Number of interval widths outside")
        axes[row, 2].set_ylabel("Violating rows")
        axes[row, 2].legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--method-label", default="Doob")
    parser.add_argument("--comparison-samples", default=None)
    parser.add_argument("--comparison-label", default="HARPOON")
    parser.add_argument("--print-comparison-rows", type=int, default=20)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--info-file", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--bins", type=int, default=40)
    args = parser.parse_args()
    if args.bins < 2 or args.print_comparison_rows < 0:
        raise ValueError("bins must be >=2 and printed rows must be non-negative")

    with Path(args.query_file).open("r", encoding="utf-8") as stream:
        query = json.load(stream)
    with Path(args.info_file).open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    predicates = [
        predicate for predicate in query["predicates"]
        if predicate["modality"] == "numeric"
    ]
    if not predicates:
        raise ValueError("the selected query has no numerical interval")
    numerical_names = numerical_column_names(info)
    numerical_transform = load_tabdiff_numerical_transform(args, info)
    model_bounds = transformed_bounds(predicates, numerical_names, numerical_transform)

    frames = [(args.method_label, pd.read_csv(args.samples))]
    if args.comparison_samples:
        comparison_path = Path(args.comparison_samples)
        if not comparison_path.is_file():
            raise FileNotFoundError(f"comparison samples not found: {comparison_path}")
        comparison = pd.read_csv(comparison_path)
        frames.append((args.comparison_label, comparison))
        columns = [str(predicate["col"]) for predicate in predicates]
        print(
            f"First {min(args.print_comparison_rows, len(comparison))} "
            f"{args.comparison_label} rows for active numerical columns:"
        )
        print(comparison[columns].head(args.print_comparison_rows).to_string(index=True))

    details_parts = []
    summaries = []
    for method, frame in frames:
        transformed = transform_frame(frame, numerical_names, numerical_transform)
        transformed_by_name = {
            name: transformed[:, index] for index, name in enumerate(numerical_names)
        }
        for predicate in predicates:
            column = str(predicate["col"])
            raw_values = pd.to_numeric(frame[column]).to_numpy(dtype=float)
            raw_lower, raw_upper = map(float, predicate["values"])
            for values, lower, upper, space in (
                (raw_values, raw_lower, raw_upper, "raw"),
                (
                    transformed_by_name[column], model_bounds[column][0],
                    model_bounds[column][1], "transformed",
                ),
            ):
                part, summary = column_diagnostics(
                    values=values, column=column, method=method,
                    lower=lower, upper=upper, space=space,
                )
                details_parts.append(part)
                summaries.append(summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details = pd.concat(details_parts, ignore_index=True)
    details.to_csv(output_dir / "numerical_violation_distances.csv", index=False)
    with (output_dir / "numerical_violation_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "query_id": query["query_id"],
                "distance_definition": "max(lower-x, 0, x-upper)",
                "spaces": {
                    "raw": "original CSV data space",
                    "transformed": "TabDiff fitted quantile-normalized model space",
                },
                "summaries": summaries,
            }, stream, indent=2,
        )
        stream.write("\n")

    plot_space(
        details, predicates, summaries, space="raw", bins=args.bins,
        output=output_dir / "violation_distance_histogram_raw.png",
    )
    plot_space(
        details, predicates, summaries, space="transformed", bins=args.bins,
        output=output_dir / "violation_distance_histogram_transformed.png",
    )
    print(json.dumps({"query_id": query["query_id"], "summaries": summaries}, indent=2))
    print(f"Saved raw and transformed violation diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
