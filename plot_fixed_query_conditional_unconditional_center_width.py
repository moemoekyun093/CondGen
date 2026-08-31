"""Compare conditional real data, unconditional samples, and center/log-width Doob samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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


def raw_hits(frame: pd.DataFrame, column: str, lower: float, upper: float):
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
    return np.isfinite(values) & (values >= lower) & (values <= upper)


def matched_sample(frame: pd.DataFrame, rows: int, rng) -> pd.DataFrame:
    if len(frame) == 0:
        raise ValueError("cannot sample from an empty frame")
    indices = rng.choice(len(frame), size=rows, replace=len(frame) < rows)
    return frame.iloc[indices].reset_index(drop=True)


def plot_panels(records, lower, upper, bins, output, zoom):
    width = upper - lower
    if zoom:
        plot_lower, plot_upper = lower - 5 * width, upper + 5 * width
        edges = np.linspace(plot_lower, plot_upper, max(100, bins * 2) + 1)
    else:
        pooled = np.concatenate([record["values"] for record in records])
        edges = np.histogram_bin_edges(pooled, bins=bins)
        plot_lower = plot_upper = None
    fig, axes = plt.subplots(
        1, 3, figsize=(17, 5.2), constrained_layout=True, sharey=True
    )
    colors = ("tab:green", "0.35", "tab:blue")
    for axis, record, color in zip(axes, records, colors):
        in_view = (
            len(record["values"])
            if not zoom
            else int(
                ((record["values"] >= plot_lower) & (record["values"] <= plot_upper)).sum()
            )
        )
        axis.hist(record["values"], bins=edges, color=color, alpha=0.80)
        axis.axvspan(lower, upper, color="tab:green", alpha=0.20)
        axis.axvline(lower, color="tab:green", linestyle="--", linewidth=1.1)
        axis.axvline(upper, color="tab:green", linestyle="--", linewidth=1.1)
        if zoom:
            axis.set_xlim(plot_lower, plot_upper)
        axis.set_title(
            f"{record['label']}\nraw violation = {record['violation_rate']:.2%} | "
            f"shown {in_view}/{len(record['values'])}"
        )
        axis.set_xlabel("TabDiff quantile-normalized value")
        axis.set_ylabel("Rows")
    suffix = "extreme interval zoom" if zoom else "full transformed distribution"
    fig.suptitle(f"Conditional real vs unconditional vs center/log-width MLP — {suffix}")
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--real-data", required=True)
    parser.add_argument("--unconditional-samples", required=True)
    parser.add_argument("--center-width-samples", required=True)
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--info-file", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=83000)
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()
    if args.rows <= 0 or args.bins < 2:
        raise ValueError("rows and bins must be positive")

    with Path(args.query_file).open("r", encoding="utf-8") as stream:
        query = json.load(stream)
    predicates = [item for item in query["predicates"] if item["modality"] == "numeric"]
    if len(predicates) != 1:
        raise ValueError("comparison requires exactly one numerical predicate")
    predicate = predicates[0]
    column = str(predicate["col"])
    raw_lower, raw_upper = (float(value) for value in predicate["values"])
    with Path(args.info_file).open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    numerical_names = numerical_column_names(info)
    column_index = numerical_names.index(column)
    transform = load_tabdiff_numerical_transform(
        SimpleNamespace(
            dataname=args.dataname,
            data_dir=args.data_dir,
            base_config=args.base_config,
        ),
        info,
    )
    transformed_lower, transformed_upper = transformed_bounds(
        predicates, numerical_names, transform
    )[column]

    real = pd.read_csv(args.real_data)
    unconditional = pd.read_csv(args.unconditional_samples)
    center_width = pd.read_csv(args.center_width_samples)
    real_conditional = real.loc[raw_hits(real, column, raw_lower, raw_upper)].copy()
    rng = np.random.default_rng(args.seed)
    conditional_sample = matched_sample(real_conditional, args.rows, rng)
    groups = (
        ("Conditional real (sampled with replacement)", conditional_sample),
        ("Unconditional FT-periodic", matched_sample(unconditional, args.rows, rng)),
        ("Center/log-width MLP (50 steps, λ=1)", matched_sample(center_width, args.rows, rng)),
    )

    records = []
    summary = []
    for label, frame in groups:
        hits = raw_hits(frame, column, raw_lower, raw_upper)
        values = transform_frame(frame, numerical_names, transform)[:, column_index]
        records.append(
            {"label": label, "values": values, "violation_rate": float(1 - hits.mean())}
        )
        summary.append(
            {
                "method": label,
                "rows_plotted": len(frame),
                "raw_hit_rate": float(hits.mean()),
                "raw_violation_rate": float(1 - hits.mean()),
                "conditional_real_pool_rows": len(real_conditional),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditional_sample.to_csv(output_dir / "sampled_conditional_real.csv", index=False)
    pd.DataFrame(summary).to_csv(output_dir / "side_by_side_summary.csv", index=False)
    plot_panels(
        records, transformed_lower, transformed_upper, args.bins,
        output_dir / "conditional_unconditional_center_width_full.png", False,
    )
    plot_panels(
        records, transformed_lower, transformed_upper, args.bins,
        output_dir / "conditional_unconditional_center_width_extreme_zoom.png", True,
    )
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Saved side-by-side comparison to {output_dir}")


if __name__ == "__main__":
    main()
