"""Evaluate center/log-width guide variability and the nearby unconditional mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from evaluate_fixed_query_violation_magnitude import (
    load_tabdiff_numerical_transform,
    numerical_column_names,
    transform_frame,
    transformed_bounds,
)


def raw_violation(frame, column, lower, upper):
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
    return ~np.isfinite(values) | (values < lower) | (values > upper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--seeds", default="74000,74001,74002,74003,74004")
    parser.add_argument("--unconditional-samples", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--info-file", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",")]
    with Path(args.query_file).open("r", encoding="utf-8") as stream:
        query = json.load(stream)
    predicates = [p for p in query["predicates"] if p["modality"] == "numeric"]
    if len(predicates) != 1:
        raise ValueError("multiseed diagnostic requires one numerical predicate")
    predicate = predicates[0]
    column = str(predicate["col"])
    raw_lower, raw_upper = map(float, predicate["values"])
    with Path(args.info_file).open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    transform = load_tabdiff_numerical_transform(
        SimpleNamespace(
            dataname=args.dataname,
            data_dir=args.data_dir,
            base_config=args.base_config,
        ),
        info,
    )
    numerical_names = numerical_column_names(info)
    column_index = numerical_names.index(column)
    lower, upper = transformed_bounds(predicates, numerical_names, transform)[column]
    width = upper - lower
    view_lower, view_upper = lower - 5 * width, upper + 5 * width

    rows = []
    guided_values = []
    per_seed = []
    for seed in seeds:
        path = Path(args.sample_dir) / f"seed_{seed}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        violated = raw_violation(frame, column, raw_lower, raw_upper)
        values = transform_frame(frame, numerical_names, transform)[:, column_index]
        guided_values.append(values)
        rows.append(
            {
                "seed": seed,
                "generated_rows": len(frame),
                "raw_hit_rate": float(1 - violated.mean()),
                "raw_violation_rate": float(violated.mean()),
            }
        )
        per_seed.append((seed, values, float(violated.mean())))

    metrics = pd.DataFrame(rows)
    unconditional = pd.read_csv(args.unconditional_samples)
    unconditional_violated = raw_violation(
        unconditional, column, raw_lower, raw_upper
    )
    unconditional_values = transform_frame(
        unconditional, numerical_names, transform
    )[:, column_index]
    pooled = np.concatenate(guided_values)

    fine_edges = np.linspace(view_lower, view_upper, max(200, args.bins * 4) + 1)
    unconditional_counts, _ = np.histogram(unconditional_values, bins=fine_edges)
    mode_index = int(np.argmax(unconditional_counts))
    mode_lower, mode_upper = fine_edges[mode_index], fine_edges[mode_index + 1]
    mode_center = 0.5 * (mode_lower + mode_upper)
    mode_position = (
        "below" if mode_center < lower else "above" if mode_center > upper else "inside"
    )
    spike = pd.DataFrame(
        [
            {
                "column": column,
                "query_transformed_lower": lower,
                "query_transformed_upper": upper,
                "unconditional_mode_bin_lower": mode_lower,
                "unconditional_mode_bin_upper": mode_upper,
                "unconditional_mode_bin_center": mode_center,
                "mode_position_relative_to_query": mode_position,
                "mode_bin_rows": int(unconditional_counts[mode_index]),
                "unconditional_raw_violation_rate": float(unconditional_violated.mean()),
            }
        ]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "center_logwidth_multiseed_metrics.csv", index=False)
    aggregate = pd.DataFrame(
        [
            {
                "num_seeds": len(metrics),
                "rows_per_seed": int(metrics["generated_rows"].iloc[0]),
                "violation_rate_mean": metrics["raw_violation_rate"].mean(),
                "violation_rate_std": metrics["raw_violation_rate"].std(ddof=1),
                "violation_rate_min": metrics["raw_violation_rate"].min(),
                "violation_rate_max": metrics["raw_violation_rate"].max(),
                "pooled_violation_rate": 1.0 - metrics["raw_hit_rate"].mean(),
            }
        ]
    )
    aggregate.to_csv(output_dir / "center_logwidth_multiseed_summary.csv", index=False)
    spike.to_csv(output_dir / "unconditional_spike_diagnostic.csv", index=False)

    fig, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    axis.scatter(metrics["seed"].astype(str), metrics["raw_violation_rate"], s=55)
    mean = metrics["raw_violation_rate"].mean()
    std = metrics["raw_violation_rate"].std(ddof=1)
    axis.axhline(mean, color="tab:blue", label=f"mean={mean:.2%}, std={std:.2%}")
    axis.axhspan(mean - std, mean + std, color="tab:blue", alpha=0.15)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Sampling seed")
    axis.set_ylabel("Raw constraint violation rate")
    axis.legend()
    fig.savefig(output_dir / "violation_rate_across_seeds.png", dpi=190)
    plt.close(fig)

    edges = np.linspace(view_lower, view_upper, max(100, args.bins * 2) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True, sharey=True)
    plot_groups = (
        (axes[0], unconditional_values, "Unconditional FT-periodic", "0.3",
         float(unconditional_violated.mean())),
        (axes[1], pooled, f"Center/log-width MLP pooled ({len(seeds)} seeds)",
         "tab:blue", float(metrics["raw_violation_rate"].mean())),
    )
    for axis, values, label, color, violation in plot_groups:
        in_view = int(((values >= view_lower) & (values <= view_upper)).sum())
        axis.hist(values, bins=edges, color=color, alpha=0.80, density=True)
        axis.axvspan(lower, upper, color="tab:green", alpha=0.22)
        axis.axvline(lower, color="tab:green", linestyle="--")
        axis.axvline(upper, color="tab:green", linestyle="--")
        axis.axvspan(mode_lower, mode_upper, color="tab:red", alpha=0.22)
        axis.set_xlim(view_lower, view_upper)
        axis.set_title(
            f"{label}\nviolation={violation:.2%}; zoom contains {in_view}/{len(values)}"
        )
        axis.set_xlabel("TabDiff quantile-normalized value")
        axis.set_ylabel("Probability density in shared bins")
    fig.suptitle(
        f"Unconditional spike (red) is {mode_position} the query interval (green)"
    )
    fig.savefig(output_dir / "unconditional_vs_center_logwidth_pooled_zoom.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True, sharex=True, sharey=True)
    for axis, (seed, values, violation) in zip(axes.flat, per_seed):
        axis.hist(values, bins=edges, color="tab:blue", alpha=0.80)
        axis.axvspan(lower, upper, color="tab:green", alpha=0.22)
        axis.set_title(f"seed {seed}: violation={violation:.2%}")
        axis.set_xlim(view_lower, view_upper)
    for axis in axes.flat[len(per_seed):]:
        axis.set_visible(False)
    fig.suptitle("Center/log-width MLP sampling variability (50 steps, λ=1)")
    fig.savefig(output_dir / "center_logwidth_histograms_across_seeds.png", dpi=190)
    plt.close(fig)

    print(metrics.to_string(index=False))
    print(aggregate.to_string(index=False))
    print(spike.to_string(index=False))
    print(f"Saved multiseed diagnostic to {output_dir}")


if __name__ == "__main__":
    main()
