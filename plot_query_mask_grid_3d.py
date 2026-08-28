"""Plot selectivity-by-mask-arity metric surfaces from per-query evaluation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def surface_data(grouped: pd.DataFrame, metric: str):
    pivot = grouped.pivot(index="arity", columns="target_band", values=metric)
    pivot = pivot.sort_index().sort_index(axis=1)
    if pivot.isna().any().any():
        raise ValueError(f"incomplete selectivity/arity grid for {metric}")
    selectivities = pivot.columns.to_numpy(dtype=float)
    arities = pivot.index.to_numpy(dtype=float)
    x, y = np.meshgrid(np.log10(selectivities), arities)
    return x, y, pivot.to_numpy(dtype=float), selectivities


def plot_metric_group(frame: pd.DataFrame, output: Path, definitions, title: str) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(6.4 * len(definitions), 5.5))
    for index, (metric, metric_title) in enumerate(definitions, start=1):
        axis = figure.add_subplot(1, len(definitions), index, projection="3d")
        x, y, z, selectivities = surface_data(frame, metric)
        axis.plot_surface(x, y, z, cmap="viridis", alpha=0.82, edgecolor="none")
        axis.scatter(x.ravel(), y.ravel(), z.ravel(), color="black", s=14)
        axis.set_xticks(np.log10(selectivities))
        axis.set_xticklabels([f"{value:g}" for value in selectivities], rotation=25)
        axis.set_xlabel("Parent-query selectivity")
        axis.set_ylabel("Active predicates")
        axis.set_zlabel("Score")
        axis.set_zlim(0.0, 1.0)
        axis.set_title(metric_title)
        axis.view_init(elev=25, azim=-135)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    per_query = pd.read_csv(args.per_query)
    required = {
        "method", "target_band", "arity", "violation_rate", "shape", "trend",
        "numeric_joint_miss_rate", "categorical_joint_miss_rate",
    }
    if not required.issubset(per_query.columns):
        raise ValueError(f"per-query CSV is missing: {sorted(required - set(per_query))}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        "violation_rate", "shape", "trend", "numeric_joint_miss_rate",
        "categorical_joint_miss_rate", "conditional_real_rate",
    ]
    grouped = (
        per_query.groupby(["method", "target_band", "arity"], sort=True)[metrics]
        .agg(["mean", "std", "count"])
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped.to_csv(output_dir / "by_selectivity_and_arity.csv", index=False)

    for method in grouped["method"].unique():
        selected = grouped[grouped["method"] == method]
        label = safe_label(str(method))
        plot_metric_group(
            selected,
            output_dir / f"{label}_constraint_quality_surfaces_3d.png",
            (
                ("violation_rate_mean", "Joint violation (lower is better)"),
                ("shape_mean", "Column Shape (higher is better)"),
                ("trend_mean", "Column-pair Trend (higher is better)"),
            ),
            f"{method}: query selectivity × active-mask arity",
        )
        plot_metric_group(
            selected,
            output_dir / f"{label}_modality_miss_surfaces_3d.png",
            (
                ("numeric_joint_miss_rate_mean", "Numerical joint miss"),
                ("categorical_joint_miss_rate_mean", "Categorical joint miss"),
            ),
            f"{method}: modality-specific constraint misses",
        )
    print(f"Saved 3D query-mask surfaces to {output_dir}")


if __name__ == "__main__":
    main()
