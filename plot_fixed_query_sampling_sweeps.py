"""Plot reverse-step and guidance-strength sweeps for one fixed-query guide."""

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


def parse_csv_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def sample_path(root: Path, steps: int, strength: float) -> Path:
    strength_text = f"{strength:g}"
    return root / f"steps_{steps:03d}_lambda_{strength_text}.csv"


def read_values(path, column, numerical_names, transform, column_index, raw_bounds):
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    raw = pd.to_numeric(frame[column], errors="coerce").to_numpy()
    raw_lower, raw_upper = raw_bounds
    violated = (
        ~np.isfinite(raw) | (raw < raw_lower) | (raw > raw_upper)
    )
    transformed = transform_frame(frame, numerical_names, transform)[:, column_index]
    return transformed, float(violated.mean()), len(frame)


def histogram_grid(records, lower, upper, bins, title, output):
    width = upper - lower
    view_lower, view_upper = lower - 5 * width, upper + 5 * width
    edges = np.linspace(view_lower, view_upper, max(100, bins * 2) + 1)
    columns = min(3, len(records))
    rows = int(np.ceil(len(records) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.2 * columns, 4.2 * rows),
        squeeze=False, constrained_layout=True, sharey=True,
    )
    for axis, record in zip(axes.flat, records):
        in_view = int(
            ((record["values"] >= view_lower) & (record["values"] <= view_upper)).sum()
        )
        axis.hist(record["values"], bins=edges, color="tab:blue", alpha=0.78)
        axis.axvspan(lower, upper, color="tab:green", alpha=0.24)
        axis.axvline(lower, color="tab:green", linestyle="--", linewidth=1.1)
        axis.axvline(upper, color="tab:green", linestyle="--", linewidth=1.1)
        axis.set_xlim(view_lower, view_upper)
        axis.set_title(
            f"{record['panel']}\n"
            f"raw violation = {record['violation_rate']:.2%} | "
            f"zoom contains {in_view}/{record['generated_rows']}"
        )
        axis.set_xlabel("TabDiff quantile-normalized value")
        axis.set_ylabel("Generated rows")
    for axis in axes.flat[len(records):]:
        axis.set_visible(False)
    fig.suptitle(title, fontsize=15)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def line_plot(rows, x, xlabel, output):
    frame = pd.DataFrame(rows).sort_values(x)
    fig, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    axis.plot(frame[x], frame["raw_violation_rate"], marker="o", linewidth=2)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Raw constraint violation rate")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylim(0, min(1.0, max(0.05, frame["raw_violation_rate"].max() * 1.15)))
    axis.grid(alpha=0.25)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--info-file", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reverse-steps", default="50,75,100,150,200")
    parser.add_argument("--guidance-strengths", default="1,2,5")
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()

    reverse_steps = parse_csv_list(args.reverse_steps, int)
    strengths = parse_csv_list(args.guidance_strengths, float)
    with Path(args.query_file).open("r", encoding="utf-8") as stream:
        query = json.load(stream)
    predicates = [p for p in query["predicates"] if p["modality"] == "numeric"]
    if len(predicates) != 1:
        raise ValueError("sampling sweep plot requires exactly one numerical predicate")
    predicate = predicates[0]
    column = str(predicate["col"])
    raw_bounds = tuple(float(value) for value in predicate["values"])
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
    root = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    step_records = []
    for steps in reverse_steps:
        path = sample_path(root, steps, 1.0)
        values, violation, count = read_values(
            path, column, numerical_names, transform, column_index, raw_bounds
        )
        step_records.append(
            {"panel": f"{steps} reverse steps, λ=1", "values": values,
             "violation_rate": violation, "generated_rows": count}
        )
        metric_rows.append(
            {"sweep": "reverse_steps", "reverse_steps": steps,
             "guidance_strength": 1.0, "raw_violation_rate": violation,
             "generated_rows": count}
        )

    strength_records = []
    for strength in strengths:
        path = sample_path(root, 50, strength)
        values, violation, count = read_values(
            path, column, numerical_names, transform, column_index, raw_bounds
        )
        strength_records.append(
            {"panel": f"50 reverse steps, λ={strength:g}", "values": values,
             "violation_rate": violation, "generated_rows": count}
        )
        metric_rows.append(
            {"sweep": "guidance_strength", "reverse_steps": 50,
             "guidance_strength": strength, "raw_violation_rate": violation,
             "generated_rows": count}
        )

    histogram_grid(
        step_records, lower, upper, args.bins,
        "Ordinary MLP center/log-width: reverse-step sweep (λ=1)",
        output_dir / "center_logwidth_mlp_histograms_by_reverse_steps.png",
    )
    histogram_grid(
        strength_records, lower, upper, args.bins,
        "Ordinary MLP center/log-width: guidance-strength sweep (50 steps)",
        output_dir / "center_logwidth_mlp_histograms_by_guidance_strength.png",
    )
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(output_dir / "sampling_sweep_violation_rates.csv", index=False)
    line_plot(
        metric_frame[metric_frame["sweep"] == "reverse_steps"],
        "reverse_steps", "Reverse-diffusion steps",
        output_dir / "violation_rate_by_reverse_steps.png",
    )
    line_plot(
        metric_frame[metric_frame["sweep"] == "guidance_strength"],
        "guidance_strength", "Guidance strength λ (50 reverse steps)",
        output_dir / "violation_rate_by_guidance_strength.png",
    )
    print(metric_frame.to_string(index=False))
    print(f"Saved center/log-width MLP sampling sweeps to {output_dir}")


if __name__ == "__main__":
    main()
