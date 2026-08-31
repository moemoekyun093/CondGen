"""Plot unconditional density around every tight arity-1 numerical query."""

from __future__ import annotations

import argparse
import json
import math
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


def load_queries(directory, max_target_band, max_realized_selectivity):
    selected = []
    for path in sorted(Path(directory).glob("*.json")):
        with path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if not isinstance(query, dict) or "predicates" not in query:
            continue
        predicates = query["predicates"]
        arity = int(query.get("arity", len(predicates)))
        if arity != 1 or len(predicates) != 1:
            continue
        predicate = predicates[0]
        if predicate.get("modality") != "numeric" or predicate.get("op") != "between":
            continue
        target_band = query.get("target_band")
        selectivity = query.get("selectivity")
        realized_train = selectivity.get("train") if isinstance(selectivity, dict) else None
        if target_band is None or float(target_band) > max_target_band:
            continue
        if realized_train is None or float(realized_train) > max_realized_selectivity:
            continue
        if not query.get("accepted", True):
            continue
        selected.append((path, query, predicate, float(target_band), float(realized_train)))
    if not selected:
        raise ValueError("no accepted tight arity-1 numerical queries matched the filters")
    return selected


def position_and_distance(center, lower, upper):
    width = upper - lower
    if center < lower:
        return "below", (lower - center) / width
    if center > upper:
        return "above", (center - upper) / width
    return "inside", 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--unconditional-samples", required=True)
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--info-file", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-target-band", type=float, default=0.02)
    parser.add_argument("--max-realized-selectivity", type=float, default=0.025)
    parser.add_argument("--window-widths", type=float, default=5.0)
    parser.add_argument("--bins", type=int, default=200)
    args = parser.parse_args()
    if args.bins < 20 or args.window_widths <= 0:
        raise ValueError("bins must be >=20 and window-widths must be positive")

    queries = load_queries(
        args.query_dir, args.max_target_band, args.max_realized_selectivity
    )
    with Path(args.info_file).open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    numerical_names = numerical_column_names(info)
    transform = load_tabdiff_numerical_transform(
        SimpleNamespace(
            dataname=args.dataname,
            data_dir=args.data_dir,
            base_config=args.base_config,
        ),
        info,
    )
    unconditional = pd.read_csv(args.unconditional_samples)
    transformed = transform_frame(unconditional, numerical_names, transform)
    transformed_by_column = {
        name: transformed[:, index] for index, name in enumerate(numerical_names)
    }

    output_dir = Path(args.output_dir)
    panel_dir = output_dir / "per_query"
    panel_dir.mkdir(parents=True, exist_ok=True)
    records = []
    panels = []

    for _, query, predicate, target_band, realized_train in queries:
        query_id = str(query["query_id"])
        column = str(predicate["col"])
        raw_lower, raw_upper = map(float, predicate["values"])
        lower, upper = transformed_bounds(
            [predicate], numerical_names, transform
        )[column]
        width = upper - lower
        if width <= 0:
            continue
        view_lower = lower - args.window_widths * width
        view_upper = upper + args.window_widths * width
        edges = np.linspace(view_lower, view_upper, args.bins + 1)
        values = transformed_by_column[column]
        counts, _ = np.histogram(values, bins=edges)
        mode_index = int(np.argmax(counts))
        mode_lower, mode_upper = edges[mode_index], edges[mode_index + 1]
        mode_center = 0.5 * (mode_lower + mode_upper)
        mode_position, mode_distance = position_and_distance(
            mode_center, lower, upper
        )
        raw_values = pd.to_numeric(unconditional[column], errors="coerce").to_numpy()
        raw_hits = (
            np.isfinite(raw_values)
            & (raw_values >= raw_lower)
            & (raw_values <= raw_upper)
        )
        in_view = (values >= view_lower) & (values <= view_upper)
        query_bin_mask = (edges[:-1] < upper) & (edges[1:] > lower)
        mean_query_bin_count = (
            float(counts[query_bin_mask].mean()) if query_bin_mask.any() else 0.0
        )
        mode_to_query_density_ratio = (
            float(counts[mode_index] / mean_query_bin_count)
            if mean_query_bin_count > 0
            else math.inf
        )
        record = {
            "query_id": query_id,
            "column": column,
            "target_band": target_band,
            "realized_train_selectivity": realized_train,
            "raw_lower": raw_lower,
            "raw_upper": raw_upper,
            "transformed_lower": lower,
            "transformed_upper": upper,
            "transformed_width": width,
            "unconditional_rows": len(unconditional),
            "unconditional_query_hit_rate": float(raw_hits.mean()),
            "unconditional_query_violation_rate": float(1 - raw_hits.mean()),
            "unconditional_rows_in_zoom": int(in_view.sum()),
            "local_mode_bin_lower": mode_lower,
            "local_mode_bin_upper": mode_upper,
            "local_mode_bin_center": mode_center,
            "local_mode_bin_rows": int(counts[mode_index]),
            "local_mode_position": mode_position,
            "local_mode_distance_in_interval_widths": mode_distance,
            "mode_to_mean_query_bin_density_ratio": mode_to_query_density_ratio,
        }
        records.append(record)
        panels.append((record, values, edges))

        fig, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        axis.hist(values, bins=edges, density=True, color="0.35", alpha=0.82)
        axis.axvspan(lower, upper, color="tab:green", alpha=0.23, label="query")
        axis.axvspan(mode_lower, mode_upper, color="tab:red", alpha=0.25, label="local mode bin")
        axis.axvline(lower, color="tab:green", linestyle="--")
        axis.axvline(upper, color="tab:green", linestyle="--")
        axis.set_xlim(view_lower, view_upper)
        axis.set_xlabel(f"{column} in TabDiff quantile-normalized space")
        axis.set_ylabel("Unconditional probability density")
        axis.set_title(
            f"{query_id} | target={target_band:.1%}, realized train={realized_train:.2%}\n"
            f"unconditional hit={raw_hits.mean():.2%}; local mode={mode_position}, "
            f"distance={mode_distance:.2f} interval widths"
        )
        axis.legend(fontsize=8)
        fig.savefig(panel_dir / f"{query_id}.png", dpi=180)
        plt.close(fig)

    summary = pd.DataFrame(records).sort_values(
        ["target_band", "local_mode_distance_in_interval_widths", "query_id"]
    )
    summary.to_csv(output_dir / "arity1_tight_unconditional_density_summary.csv", index=False)

    columns = 3
    rows = int(math.ceil(len(panels) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(6.1 * columns, 4.0 * rows),
        squeeze=False, constrained_layout=True,
    )
    for axis, (record, values, edges) in zip(axes.flat, panels):
        axis.hist(values, bins=edges, density=True, color="0.35", alpha=0.82)
        axis.axvspan(
            record["transformed_lower"], record["transformed_upper"],
            color="tab:green", alpha=0.23,
        )
        axis.axvspan(
            record["local_mode_bin_lower"], record["local_mode_bin_upper"],
            color="tab:red", alpha=0.25,
        )
        axis.set_xlim(edges[0], edges[-1])
        axis.set_title(
            f"{record['query_id']} ({record['column']})\n"
            f"hit={record['unconditional_query_hit_rate']:.2%}; "
            f"mode {record['local_mode_position']} by "
            f"{record['local_mode_distance_in_interval_widths']:.2f} widths",
            fontsize=9,
        )
        axis.set_xlabel("Transformed value")
        axis.set_ylabel("Unconditional density")
    for axis in axes.flat[len(panels):]:
        axis.set_visible(False)
    fig.suptitle(
        "Unconditional FT-periodic density around tight arity-1 numerical queries\n"
        "green=query interval, red=highest-density local histogram bin",
        fontsize=15,
    )
    fig.savefig(output_dir / "arity1_tight_unconditional_density_grid.png", dpi=190)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(f"Selected {len(summary)} tight arity-1 numerical queries")
    print(f"Saved unconditional-density diagnostic to {output_dir}")


if __name__ == "__main__":
    main()
