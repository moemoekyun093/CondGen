"""Render readable heatmaps and profile curves for a query-mask evaluation grid."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


QUALITY_METRICS = (
    ("violation_rate", "Joint violation", False),
    ("shape", "Column Shape", True),
    ("trend", "Column-pair Trend", True),
)
MODALITY_METRICS = (
    ("numeric_joint_miss_rate", "Numerical joint miss", False),
    ("categorical_joint_miss_rate", "Categorical joint miss", False),
)
FILTERED_QUALITY_METRICS = (
    ("filtered_shape", "Feasible-only Column Shape", True),
    ("filtered_trend", "Feasible-only Column-pair Trend", True),
)


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def metric_matrix(frame: pd.DataFrame, metric: str, statistic: str = "mean"):
    column = f"{metric}_{statistic}"
    pivot = frame.pivot(index="arity", columns="target_band", values=column)
    return pivot.sort_index().sort_index(axis=1)


def annotate_heatmap(axis, values: np.ndarray, *, percent: bool) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                label = "n/a"
            else:
                label = f"{100 * value:.1f}%" if percent else f"{value:.3f}"
            axis.text(column, row, label, ha="center", va="center", fontsize=8)


def format_heatmap_axis(axis, pivot, title: str) -> None:
    axis.set_xticks(np.arange(len(pivot.columns)))
    axis.set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=35)
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_yticklabels([str(int(value)) for value in pivot.index])
    axis.set_xlabel("Parent-query selectivity")
    axis.set_ylabel("Active predicates")
    axis.set_title(title)


def comparison_heatmaps(
    grouped: pd.DataFrame,
    output: Path,
    definitions,
    primary: str,
    baseline: str,
    figure_title: str,
    allow_missing: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(definitions), 3,
        figsize=(15.5, 4.2 * len(definitions)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row, (metric, title, higher_is_better) in enumerate(definitions):
        primary_pivot = metric_matrix(grouped[grouped["method"] == primary], metric)
        baseline_pivot = metric_matrix(grouped[grouped["method"] == baseline], metric)
        if not primary_pivot.index.equals(baseline_pivot.index) or not primary_pivot.columns.equals(
            baseline_pivot.columns
        ):
            raise ValueError(f"methods have different grid cells for {metric}")
        if not allow_missing and (
            primary_pivot.isna().any().any() or baseline_pivot.isna().any().any()
        ):
            raise ValueError(f"missing grid values for {metric}")
        absolute_cmap = plt.get_cmap(
            "RdYlGn" if higher_is_better else "RdYlGn_r"
        ).copy()
        absolute_cmap.set_bad("lightgray")
        absolute_images = []
        for column, (method, pivot) in enumerate(
            ((primary, primary_pivot), (baseline, baseline_pivot))
        ):
            image = axes[row, column].imshow(
                np.ma.masked_invalid(pivot.to_numpy()),
                vmin=0.0, vmax=1.0, cmap=absolute_cmap, aspect="auto"
            )
            absolute_images.append(image)
            annotate_heatmap(
                axes[row, column], pivot.to_numpy(), percent=not higher_is_better
            )
            format_heatmap_axis(axes[row, column], pivot, f"{method}: {title}")
        figure.colorbar(
            absolute_images[0], ax=[axes[row, 0], axes[row, 1]],
            shrink=0.72, pad=0.015, label="Score",
        )

        # Positive always means that the primary method is better.
        advantage = (
            primary_pivot - baseline_pivot
            if higher_is_better
            else baseline_pivot - primary_pivot
        )
        finite_advantage = np.abs(advantage.to_numpy())[
            np.isfinite(advantage.to_numpy())
        ]
        limit = max(
            0.01,
            float(finite_advantage.max()) if len(finite_advantage) else 0.01,
        )
        delta_cmap = plt.get_cmap("RdYlGn").copy()
        delta_cmap.set_bad("lightgray")
        delta_image = axes[row, 2].imshow(
            np.ma.masked_invalid(advantage.to_numpy()), vmin=-limit, vmax=limit,
            cmap=delta_cmap, aspect="auto",
        )
        annotate_heatmap(axes[row, 2], advantage.to_numpy(), percent=True)
        format_heatmap_axis(
            axes[row, 2], advantage,
            f"{primary} advantage over {baseline}",
        )
        figure.colorbar(
            delta_image, ax=axes[row, 2], shrink=0.72, pad=0.015,
            label="Advantage (+ is better)",
        )
    figure.suptitle(figure_title, fontsize=15)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def metric_profiles(
    grouped: pd.DataFrame,
    output_dir: Path,
    definition,
    primary: str,
    baseline: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    metric, title, higher_is_better = definition
    bands = sorted(grouped["target_band"].unique())
    columns = 4
    rows = int(np.ceil(len(bands) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(15.5, 4.0 * rows), sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    colors = {primary: "tab:blue", baseline: "tab:orange"}
    visible_values = []
    for axis, band in zip(axes, bands):
        for method in (primary, baseline):
            selected = grouped[
                (grouped["method"] == method)
                & np.isclose(grouped["target_band"], band)
            ].sort_values("arity")
            x = selected["arity"].to_numpy(dtype=float)
            mean = selected[f"{metric}_mean"].to_numpy(dtype=float)
            std = selected[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
            axis.plot(x, mean, marker="o", linewidth=2.2, color=colors[method], label=method)
            axis.fill_between(
                x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                color=colors[method], alpha=0.14,
            )
            visible_values.extend((mean - std).tolist())
            visible_values.extend((mean + std).tolist())
        axis.set_title(f"Parent selectivity {band:g}")
        axis.set_xlabel("Active predicates")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
    for axis in axes[len(bands):]:
        axis.set_visible(False)
    lower = max(0.0, min(visible_values) - 0.04)
    upper = min(1.0, max(visible_values) + 0.04)
    if upper - lower < 0.15:
        midpoint = (upper + lower) / 2
        lower, upper = max(0, midpoint - 0.075), min(1, midpoint + 0.075)
    for axis in axes[: len(bands)]:
        axis.set_ylim(lower, upper)
        if not higher_is_better:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    direction = "higher" if higher_is_better else "lower"
    figure.suptitle(f"{title} by active-mask size ({direction} is better)", fontsize=15)
    figure.savefig(
        output_dir / f"{safe_label(metric)}_profiles_by_selectivity.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(figure)


def effective_selectivity_heatmap(grouped: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    # This reference quantity is identical across methods, apart from floating error.
    reference = (
        grouped.groupby(["target_band", "arity"], as_index=False)[
            "conditional_real_rate_mean"
        ].mean()
    )
    pivot = reference.pivot(
        index="arity", columns="target_band", values="conditional_real_rate_mean"
    ).sort_index().sort_index(axis=1)
    figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    image = axis.imshow(pivot.to_numpy(), vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    annotate_heatmap(axis, pivot.to_numpy(), percent=True)
    format_heatmap_axis(axis, pivot, "Actual fraction of real rows satisfying each masked query")
    figure.colorbar(image, ax=axis, label="Effective query selectivity")
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-query", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-method", required=True)
    parser.add_argument("--baseline-method", required=True)
    args = parser.parse_args()
    per_query = pd.read_csv(args.per_query)
    metrics = [
        "violation_rate", "shape", "trend", "numeric_joint_miss_rate",
        "categorical_joint_miss_rate", "conditional_real_rate",
        "filtered_shape", "filtered_trend",
    ]
    required = {"method", "target_band", "arity", *metrics}
    if not required.issubset(per_query.columns):
        raise ValueError(f"per-query CSV is missing: {sorted(required - set(per_query))}")
    available = set(per_query["method"])
    for method in (args.primary_method, args.baseline_method):
        if method not in available:
            raise ValueError(f"method {method!r} is absent from the evaluation")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = (
        per_query.groupby(["method", "target_band", "arity"], sort=True)[metrics]
        .agg(["mean", "std", "count"])
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped.to_csv(output_dir / "by_selectivity_and_arity.csv", index=False)

    comparison_heatmaps(
        grouped,
        output_dir / "method_comparison_constraint_quality_heatmaps.png",
        QUALITY_METRICS,
        args.primary_method,
        args.baseline_method,
        "Doob versus HARPOON across query selectivity and active-mask size",
    )
    comparison_heatmaps(
        grouped,
        output_dir / "method_comparison_modality_miss_heatmaps.png",
        MODALITY_METRICS,
        args.primary_method,
        args.baseline_method,
        "Which modality causes constraint misses?",
    )
    comparison_heatmaps(
        grouped,
        output_dir / "method_comparison_feasible_only_quality_heatmaps.png",
        FILTERED_QUALITY_METRICS,
        args.primary_method,
        args.baseline_method,
        "Shape and Trend after removing constraint-violating generations",
        allow_missing=True,
    )
    for definition in QUALITY_METRICS:
        metric_profiles(
            grouped, output_dir, definition,
            args.primary_method, args.baseline_method,
        )
    effective_selectivity_heatmap(
        grouped, output_dir / "effective_masked_query_selectivity_heatmap.png"
    )
    print(f"Saved readable query-mask plots to {output_dir}")


if __name__ == "__main__":
    main()
