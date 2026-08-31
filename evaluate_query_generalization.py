"""Evaluate train/test query satisfaction and nearest-train-query distance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tabdiff.doob_h_evaluation import (
    raw_constraint_report,
    raw_modality_constraint_report,
)
from tabdiff.query_split import load_query_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--diagnostic-manifest", required=True)
    parser.add_argument("--train-samples", required=True)
    parser.add_argument("--test-samples", required=True)
    parser.add_argument("--query-coordinates", required=True)
    parser.add_argument("--num-plot-bins", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_queries(query_dir: Path) -> dict[str, dict]:
    queries = {}
    for path in sorted(query_dir.glob("q*.json")):
        with path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if query.get("accepted", True):
            queries[str(query["query_id"])] = query
    return queries


def query_distance(
    left: dict, right: dict, numerical_scales: np.ndarray
) -> tuple[float, float, float, float]:
    """Return overall, active mismatch, numerical, and categorical distances."""
    left_num_active = np.asarray(left["numerical_active"], dtype=bool)
    right_num_active = np.asarray(right["numerical_active"], dtype=bool)
    left_cat_active = np.asarray(left["categorical_active"], dtype=bool)
    right_cat_active = np.asarray(right["categorical_active"], dtype=bool)
    num_union = left_num_active | right_num_active
    cat_union = left_cat_active | right_cat_active
    union_size = int(num_union.sum() + cat_union.sum())
    if union_size == 0:
        return 0.0, 0.0, float("nan"), float("nan")
    per_column = []
    numerical = []
    categorical = []
    mismatches = 0
    left_lower = np.asarray(left["numerical_lower"], dtype=float)
    right_lower = np.asarray(right["numerical_lower"], dtype=float)
    left_upper = np.asarray(left["numerical_upper"], dtype=float)
    right_upper = np.asarray(right["numerical_upper"], dtype=float)
    for column in np.flatnonzero(num_union):
        if left_num_active[column] != right_num_active[column]:
            distance = 1.0
            mismatches += 1
        else:
            scale = numerical_scales[column]
            distance = 0.5 * (
                abs(left_lower[column] - right_lower[column]) / scale
                + abs(left_upper[column] - right_upper[column]) / scale
            )
            distance = min(float(distance), 1.0)
            numerical.append(distance)
        per_column.append(distance)
    for column in np.flatnonzero(cat_union):
        if left_cat_active[column] != right_cat_active[column]:
            distance = 1.0
            mismatches += 1
        else:
            left_set = np.asarray(left["categorical_allowed"][column], dtype=bool)
            right_set = np.asarray(right["categorical_allowed"][column], dtype=bool)
            intersection = np.logical_and(left_set, right_set).sum()
            union = np.logical_or(left_set, right_set).sum()
            distance = 1.0 - intersection / union
            categorical.append(distance)
        per_column.append(distance)
    return (
        float(np.mean(per_column)),
        mismatches / union_size,
        float(np.mean(numerical)) if numerical else float("nan"),
        float(np.mean(categorical)) if categorical else float("nan"),
    )


def nearest_training_query(
    test_query_id: str,
    training_query_ids: list[str],
    coordinates: dict[str, dict],
    numerical_scales: np.ndarray,
) -> dict:
    candidates = []
    for train_query_id in training_query_ids:
        distances = query_distance(
            coordinates[test_query_id], coordinates[train_query_id], numerical_scales
        )
        candidates.append((distances[0], train_query_id, distances))
    _, query_id, distances = min(candidates, key=lambda value: (value[0], value[1]))
    return {
        "nearest_train_query_id": query_id,
        "nearest_train_distance": distances[0],
        "nearest_active_mismatch_rate": distances[1],
        "nearest_numerical_distance": distances[2],
        "nearest_categorical_distance": distances[3],
    }


def evaluate_samples(query: dict, sample_path: Path) -> dict:
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)
    samples = pd.read_csv(sample_path)
    report, _ = raw_constraint_report(samples, query)
    modality = raw_modality_constraint_report(samples, query)
    return {
        "generated_rows": len(samples),
        "raw_joint_hit_rate": float(report["joint_hit_rate"]),
        "violation_rate": 1.0 - float(report["joint_hit_rate"]),
        "numeric_joint_miss_rate": modality["numeric"]["joint_miss_rate"],
        "categorical_joint_miss_rate": modality["categorical"]["joint_miss_rate"],
        "numeric_mean_column_miss_rate": modality["numeric"][
            "mean_per_constraint_miss_rate"
        ],
        "categorical_mean_column_miss_rate": modality["categorical"][
            "mean_per_constraint_miss_rate"
        ],
    }


def make_plots(frame: pd.DataFrame, output_dir: Path, num_plot_bins: int) -> None:
    import matplotlib.pyplot as plt

    labels = (
        ("violation_rate", "Joint violation"),
        ("numeric_joint_miss_rate", "Numerical joint miss"),
        ("categorical_joint_miss_rate", "Categorical joint miss"),
    )

    def plot_group(group_column, x_label, filename, *, log_x=False):
        selected_frame = frame.dropna(subset=[group_column])
        grouped = selected_frame.groupby(["split", group_column], sort=True)[
            [metric for metric, _ in labels]
        ].agg(["mean", "std"])
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        for axis, (metric, title) in zip(axes, labels):
            for split, color in (("train", "tab:blue"), ("test", "tab:orange")):
                if split not in grouped.index.get_level_values(0):
                    continue
                selected = grouped.loc[split]
                x = selected.index.to_numpy(dtype=float)
                mean = selected[(metric, "mean")].to_numpy(dtype=float)
                std = selected[(metric, "std")].fillna(0).to_numpy(dtype=float)
                axis.plot(x, mean, marker="o", label=split, color=color)
                axis.fill_between(
                    x,
                    np.maximum(0, mean - std),
                    np.minimum(1, mean + std),
                    alpha=0.15,
                    color=color,
                )
            if log_x:
                axis.set_xscale("log")
            axis.set_ylim(0, 1)
            axis.set_title(title)
            axis.set_xlabel(x_label)
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Miss rate")
        axes[0].legend()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    plot_group(
        "target_band",
        "Target selectivity band",
        "train_vs_test_by_selectivity.png",
        log_x=True,
    )
    plot_group("arity", "Query arity", "train_vs_test_by_arity.png")
    plot_group(
        "interval_width_bin_midpoint",
        "Mean active interval width in transformed space",
        "train_vs_test_by_interval_width.png",
    )

    test = frame[frame["split"] == "test"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for axis, (metric, title) in zip(axes, labels):
        axis.scatter(test["nearest_train_distance"], test[metric], alpha=0.65, s=24)
        finite = test[["nearest_train_distance", metric]].dropna()
        correlation = finite.corr(method="spearman").iloc[0, 1] if len(finite) >= 2 else float("nan")
        if finite["nearest_train_distance"].nunique() >= 2:
            distance_bins = pd.qcut(
                finite["nearest_train_distance"],
                q=min(num_plot_bins, finite["nearest_train_distance"].nunique()),
                duplicates="drop",
            )
            binned = finite.groupby(distance_bins, observed=True).agg(
                nearest_train_distance=("nearest_train_distance", "mean"),
                metric_mean=(metric, "mean"),
            )
            axis.plot(
                binned["nearest_train_distance"],
                binned["metric_mean"],
                color="black",
                marker="o",
                linewidth=2,
                label=f"{num_plot_bins}-bin mean",
            )
        axis.set_title(f"{title}; Spearman={correlation:.3f}")
        axis.set_xlabel("Nearest training-query distance")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Miss rate")
    fig.savefig(output_dir / "test_miss_vs_nearest_train_distance.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.num_plot_bins < 2:
        raise ValueError("num-plot-bins must be at least 2")
    query_dir = Path(args.query_dir)
    queries = load_queries(query_dir)
    all_train_ids = load_query_split(args.source_manifest, "train")
    with Path(args.diagnostic_manifest).open("r", encoding="utf-8") as stream:
        diagnostic_manifest = json.load(stream)
    arity_filter = diagnostic_manifest.get("arity_filter")
    if arity_filter is not None:
        all_train_ids = [
            query_id
            for query_id in all_train_ids
            if int(
                queries[query_id].get(
                    "arity", len(queries[query_id]["predicates"])
                )
            )
            == int(arity_filter)
        ]
    sampled_train_ids = load_query_split(args.diagnostic_manifest, "train")
    test_ids = load_query_split(args.diagnostic_manifest, "test")
    with Path(args.query_coordinates).open("r", encoding="utf-8") as stream:
        coordinate_payload = json.load(stream)
    coordinates = coordinate_payload["queries"]
    missing_coordinates = (set(all_train_ids) | set(test_ids)) - set(coordinates)
    if missing_coordinates:
        raise ValueError(
            f"model coordinates missing for {sorted(missing_coordinates)[:5]}"
        )
    coordinate_values = list(coordinates.values())
    d_numerical = len(coordinate_values[0]["numerical_active"])
    numerical_scales = np.ones(d_numerical, dtype=float)
    for column in range(d_numerical):
        endpoints = []
        for value in coordinate_values:
            if value["numerical_active"][column]:
                endpoints.extend(
                    (value["numerical_lower"][column], value["numerical_upper"][column])
                )
        if endpoints:
            numerical_scales[column] = max(max(endpoints) - min(endpoints), 1e-6)
    rows = []
    for split, query_ids, sample_dir in (
        ("train", sampled_train_ids, Path(args.train_samples)),
        ("test", test_ids, Path(args.test_samples)),
    ):
        for query_id in query_ids:
            query = queries[query_id]
            coordinate = coordinates[query_id]
            active = np.asarray(coordinate["numerical_active"], dtype=bool)
            widths = (
                np.asarray(coordinate["numerical_upper"], dtype=float)
                - np.asarray(coordinate["numerical_lower"], dtype=float)
            )[active]
            row = {
                "split": split,
                "query_id": query_id,
                "target_band": float(query["target_band"]),
                "arity": int(query.get("arity", len(query["predicates"]))),
                "modality_mix": str(query.get("modality_mix", "unknown")),
                "num_active_numerical_intervals": int(active.sum()),
                "mean_normalized_interval_width": (
                    float(widths.mean()) if len(widths) else float("nan")
                ),
                "min_normalized_interval_width": (
                    float(widths.min()) if len(widths) else float("nan")
                ),
                "max_normalized_interval_width": (
                    float(widths.max()) if len(widths) else float("nan")
                ),
                **evaluate_samples(query, sample_dir / f"{query_id}.csv"),
            }
            if split == "test":
                row.update(
                    nearest_training_query(
                        query_id, all_train_ids, coordinates, numerical_scales
                    )
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    interval_widths = frame["mean_normalized_interval_width"].dropna()
    quantile_edges = np.unique(
        np.quantile(
            interval_widths,
            np.linspace(0.0, 1.0, args.num_plot_bins + 1),
        )
    )
    frame["interval_width_bin"] = pd.NA
    frame["interval_width_bin_midpoint"] = np.nan
    if len(quantile_edges) >= 2:
        bins = pd.cut(
            frame["mean_normalized_interval_width"],
            bins=quantile_edges,
            include_lowest=True,
            duplicates="drop",
        )
        frame["interval_width_bin"] = bins.astype("string")
        frame["interval_width_bin_midpoint"] = [
            interval.mid if pd.notna(interval) else np.nan for interval in bins
        ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_query.csv", index=False)
    test_neighbours = frame[frame["split"] == "test"].copy()
    test_neighbours.to_csv(output_dir / "test_nearest_training_queries.csv", index=False)
    metrics = [
        "violation_rate",
        "numeric_joint_miss_rate",
        "categorical_joint_miss_rate",
        "numeric_mean_column_miss_rate",
        "categorical_mean_column_miss_rate",
    ]
    grouped = frame.groupby(["split", "target_band"], sort=True)[metrics].agg(
        ["mean", "std", "count"]
    )
    grouped.columns = ["_".join(column) for column in grouped.columns]
    grouped.reset_index().to_csv(output_dir / "train_vs_test_by_selectivity.csv", index=False)
    by_arity = frame.groupby(["split", "arity"], sort=True)[metrics].agg(
        ["mean", "std", "count"]
    )
    by_arity.columns = ["_".join(column) for column in by_arity.columns]
    by_arity.reset_index().to_csv(
        output_dir / "train_vs_test_by_arity.csv", index=False
    )
    by_width = frame.dropna(subset=["interval_width_bin_midpoint"]).groupby(
        ["split", "interval_width_bin", "interval_width_bin_midpoint"],
        sort=True,
    )[metrics].agg(["mean", "std", "count"])
    by_width.columns = ["_".join(column) for column in by_width.columns]
    by_width.reset_index().to_csv(
        output_dir / "train_vs_test_by_interval_width.csv", index=False
    )
    correlations = []
    for metric in metrics:
        selected = test_neighbours[["nearest_train_distance", metric]].dropna()
        correlations.append(
            {
                "metric": metric,
                "spearman_with_nearest_train_distance": selected.corr(method="spearman").iloc[0, 1],
                "num_queries": len(selected),
            }
        )
    pd.DataFrame(correlations).to_csv(output_dir / "distance_correlations.csv", index=False)
    make_plots(frame, output_dir, args.num_plot_bins)
    print(f"Evaluated {len(sampled_train_ids)} sampled training queries")
    print(f"Evaluated {len(test_ids)} unseen test queries")
    print(f"Nearest neighbours searched over all {len(all_train_ids)} training queries")
    print(f"Saved diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
