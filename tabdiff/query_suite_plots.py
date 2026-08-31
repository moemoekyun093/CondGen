"""Plotting utilities for aggregated conditional query-suite results."""

from pathlib import Path

import numpy as np
import pandas as pd


def make_query_suite_plots(grouped: pd.DataFrame, output_dir: Path, group_by: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    definitions = (
        ("violation_rate", "Joint violations (lower is better)", "Violation rate"),
        ("shape", "Column Shape (higher is better)", "Shape score"),
        ("trend", "Column-pair Trend (higher is better)", "Trend score"),
    )
    for label in grouped["method"].unique():
        selected = grouped[grouped["method"] == label].sort_values(group_by)
        x = selected[group_by].to_numpy(dtype=float)
        for axis, (metric, _, _) in zip(axes, definitions):
            mean = selected[f"{metric}_mean"].to_numpy(dtype=float)
            std = selected[f"{metric}_std"].to_numpy(dtype=float)
            axis.plot(x, mean, marker="o", linewidth=2, label=label)
            axis.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.14)
    for axis, (_, title, ylabel) in zip(axes, definitions):
        if group_by == "target_band":
            axis.set_xscale("log")
        axis.set_ylim(0, 1)
        axis.set_xlabel(
            "Target selectivity band"
            if group_by == "target_band"
            else "Mean active interval width in transformed space"
            if group_by == "mean_interval_width_bin_midpoint"
            else "Active predicates"
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend()
    figure.tight_layout()
    filename_group = (
        "mean_interval_width"
        if group_by == "mean_interval_width_bin_midpoint"
        else group_by
    )
    figure.savefig(output_dir / f"query_suite_by_{filename_group}.png", dpi=180)
    plt.close(figure)

    definitions = (
        ("c2st", "Logistic C2ST similarity"),
        ("c2st_xgb", "XGBoost C2ST similarity"),
        ("alpha_precision", "Alpha Precision"),
        ("beta_recall", "Beta Recall"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharex=True)
    for label in grouped["method"].unique():
        selected = grouped[grouped["method"] == label].sort_values(group_by)
        x = selected[group_by].to_numpy(dtype=float)
        for axis, (metric, _) in zip(axes.flat, definitions):
            mean = selected[f"{metric}_mean"].to_numpy(dtype=float)
            std = selected[f"{metric}_std"].to_numpy(dtype=float)
            axis.plot(x, mean, marker="o", linewidth=2, label=label)
            axis.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.14)
    for axis, (_, title) in zip(axes.flat, definitions):
        if group_by == "target_band":
            axis.set_xscale("log")
        axis.set_ylim(0, 1)
        axis.set_xlabel(
            "Target selectivity band"
            if group_by == "target_band"
            else "Mean active interval width in transformed space"
            if group_by == "mean_interval_width_bin_midpoint"
            else "Active predicates"
        )
        axis.set_ylabel("Score (higher is better)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes.flat[0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"quality_metrics_by_{filename_group}.png", dpi=180)
    plt.close(figure)
