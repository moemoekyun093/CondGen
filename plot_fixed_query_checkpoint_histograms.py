"""Plot TabDiff-space histograms throughout fixed-query guide training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_fixed_query_violation_magnitude import (
    load_tabdiff_numerical_transform,
    numerical_column_names,
    transform_frame,
    transformed_bounds,
)


def parse_series(values: list[str]) -> list[tuple[str, Path]]:
    output = []
    for value in values:
        if "=" not in value:
            raise ValueError("series must be LABEL=DIRECTORY")
        label, directory = value.split("=", 1)
        output.append((label, Path(directory)))
    return output


def plot_histograms(records, steps, lower, upper, bins, output, zoom=False):
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), constrained_layout=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    width = upper - lower
    for axis, step in zip(axes.flat, steps):
        at_step = [(label, values) for label, saved_step, values in records if saved_step == step]
        pooled = np.concatenate([values for _, values in at_step])
        if zoom:
            plot_lower = lower - 5.0 * width
            plot_upper = upper + 5.0 * width
            in_view = pooled[(pooled >= plot_lower) & (pooled <= plot_upper)]
            edges_source = in_view if len(in_view) else np.asarray([plot_lower, plot_upper])
            edges = np.histogram_bin_edges(edges_source, bins=bins, range=(plot_lower, plot_upper))
        else:
            edges = np.histogram_bin_edges(pooled, bins=bins)
        for index, (label, values) in enumerate(at_step):
            axis.hist(
                values, bins=edges, histtype="step", linewidth=1.8,
                color=colors[index % len(colors)], label=label,
            )
        axis.axvspan(lower, upper, color="tab:green", alpha=0.20, label="valid interval")
        axis.axvline(lower, color="tab:green", linestyle="--", linewidth=1.0)
        axis.axvline(upper, color="tab:green", linestyle="--", linewidth=1.0)
        axis.set_title(f"Training step {step}")
        axis.set_xlabel("TabDiff quantile-normalized value")
        axis.set_ylabel("Generated rows")
        if zoom:
            axis.set_xlim(plot_lower, plot_upper)
        axis.legend(fontsize=8)
    title = "Fixed-query generated distributions in TabDiff model space"
    if zoom:
        title += " (interval zoom)"
    fig.suptitle(title, fontsize=16)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def safe_filename(value: str) -> str:
    """Return a stable filename component for a user-facing method label."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def plot_final_method(
    *, label, values, lower, upper, bins, raw_violation_rate,
    transformed_violation_rate, step, output,
):
    """Plot one method by itself at the requested final checkpoint."""
    width = upper - lower
    zoom_lower = lower - 5.0 * width
    zoom_upper = upper + 5.0 * width
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, zoom in zip(axes, (False, True)):
        if zoom:
            in_view = values[(values >= zoom_lower) & (values <= zoom_upper)]
            source = in_view if len(in_view) else np.asarray([zoom_lower, zoom_upper])
            edges = np.histogram_bin_edges(
                source, bins=bins, range=(zoom_lower, zoom_upper)
            )
        else:
            edges = np.histogram_bin_edges(values, bins=bins)
        axis.hist(values, bins=edges, color="tab:blue", alpha=0.78)
        axis.axvspan(lower, upper, color="tab:green", alpha=0.22)
        axis.axvline(lower, color="tab:green", linestyle="--", linewidth=1.2)
        axis.axvline(upper, color="tab:green", linestyle="--", linewidth=1.2)
        axis.set_title("Interval zoom" if zoom else "Full generated distribution")
        axis.set_xlabel("TabDiff quantile-normalized value")
        axis.set_ylabel("Generated rows")
        if zoom:
            axis.set_xlim(zoom_lower, zoom_upper)
    fig.suptitle(
        f"{label} — training step {step}\n"
        f"Raw-space violation rate: {raw_violation_rate:.2%}  |  "
        f"Transformed-space miss: {transformed_violation_rate:.2%}",
        fontsize=14,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", action="append", required=True)
    parser.add_argument("--steps", default="200,400,600,800,1000,1200,1400,1600,1800,2000")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--info-file", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()
    steps = [int(value) for value in args.steps.split(",")]
    if not steps or args.bins < 2:
        raise ValueError("at least one checkpoint and two histogram bins are required")

    with Path(args.query_file).open("r", encoding="utf-8") as stream:
        query = json.load(stream)
    predicates = [item for item in query["predicates"] if item["modality"] == "numeric"]
    if len(predicates) != 1:
        raise ValueError("fixed-query trajectory currently requires one numerical predicate")
    with Path(args.info_file).open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    transform_args = SimpleNamespace(
        dataname=args.dataname,
        data_dir=args.data_dir,
        base_config=args.base_config,
    )
    numerical_transform = load_tabdiff_numerical_transform(transform_args, info)
    numerical_names = numerical_column_names(info)
    bounds = transformed_bounds(predicates, numerical_names, numerical_transform)
    column = str(predicates[0]["col"])
    raw_lower, raw_upper = (float(value) for value in predicates[0]["values"])
    lower, upper = bounds[column]
    column_index = numerical_names.index(column)

    records = []
    metric_rows = []
    for label, directory in parse_series(args.series):
        for step in steps:
            sample_path = directory / f"step_{step:04d}.csv"
            if not sample_path.is_file():
                raise FileNotFoundError(sample_path)
            frame = pd.read_csv(sample_path)
            raw_values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            raw_violated = (
                ~np.isfinite(raw_values)
                | (raw_values < raw_lower)
                | (raw_values > raw_upper)
            )
            transformed = transform_frame(frame, numerical_names, numerical_transform)
            values = transformed[:, column_index]
            distance = np.maximum(lower - values, 0.0) + np.maximum(values - upper, 0.0)
            violated = distance > 0
            raw_violation_rate = float(raw_violated.mean())
            records.append((label, step, values))
            metric_rows.append(
                {
                    "method": label,
                    "training_step": step,
                    "generated_rows": len(values),
                    "transformed_lower": lower,
                    "transformed_upper": upper,
                    "transformed_interval_width": upper - lower,
                    "raw_violation_rate": raw_violation_rate,
                    "transformed_violation_rate": float(violated.mean()),
                    "mean_violation_distance": (
                        float(distance[violated].mean()) if violated.any() else 0.0
                    ),
                    "p95_violation_distance": (
                        float(np.quantile(distance[violated], 0.95)) if violated.any() else 0.0
                    ),
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(
        output_dir / "transformed_metrics_by_checkpoint.csv", index=False
    )
    if len(steps) == 1:
        step = steps[0]
        final_metrics = metrics[metrics["training_step"] == step].copy()
        final_metrics.to_csv(
            output_dir / f"step_{step:04d}_violation_rates.csv", index=False
        )
        for label, saved_step, values in records:
            if saved_step != step:
                continue
            row = final_metrics[final_metrics["method"] == label].iloc[0]
            plot_final_method(
                label=label,
                values=values,
                lower=lower,
                upper=upper,
                bins=args.bins,
                raw_violation_rate=float(row["raw_violation_rate"]),
                transformed_violation_rate=float(row["transformed_violation_rate"]),
                step=step,
                output=output_dir
                / f"transformed_histogram_step_{step:04d}_{safe_filename(label)}.png",
            )
    else:
        plot_histograms(
            records, steps, lower, upper, args.bins,
            output_dir / "transformed_histograms_by_checkpoint.png",
        )
        plot_histograms(
            records, steps, lower, upper, args.bins,
            output_dir / "transformed_histograms_by_checkpoint_zoom.png", zoom=True,
        )
    print(metrics.to_string(index=False))
    print(f"Saved transformed-space checkpoint plots to {output_dir}")


if __name__ == "__main__":
    main()
