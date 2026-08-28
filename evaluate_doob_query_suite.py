"""Compare arbitrary conditional generators over a full-arity query suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tabdiff.doob_h_evaluation import (
    raw_constraint_report,
    raw_modality_constraint_report,
)
from tabdiff.metrics import TabMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="LABEL=SAMPLE_DIRECTORY",
        help="Repeat for every trained guide being compared",
    )
    parser.add_argument("--real-data", default="synthetic/shoppers/real.csv")
    parser.add_argument("--info-file", default="data/shoppers/info.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--group-by",
        choices=("target_band", "arity"),
        default="target_band",
    )
    parser.add_argument(
        "--baseline-method",
        default=None,
        help="Optional method label used for paired per-query metric differences",
    )
    return parser.parse_args()


def parse_methods(values: list[str]) -> dict[str, Path]:
    methods = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid method {value!r}; expected LABEL=DIRECTORY")
        label, directory = value.split("=", maxsplit=1)
        label = label.strip()
        if not label or label in methods:
            raise ValueError(f"empty or duplicate method label {label!r}")
        methods[label] = Path(directory)
    return methods


def load_queries(query_dir: Path) -> list[tuple[Path, dict]]:
    queries = []
    for path in sorted(query_dir.glob("qf_*.json")):
        with path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if query.get("accepted", True):
            queries.append((path, query))
    if not queries:
        raise ValueError(f"no accepted queries found in {query_dir}")
    return queries


def density_metrics(reference_path: Path, samples: pd.DataFrame, info: dict) -> dict:
    evaluator = TabMetrics(
        str(reference_path),
        str(reference_path),
        None,
        info,
        torch.device("cpu"),
        metric_list=["density"],
        include_density_diagnostic=False,
    )
    metrics, _ = evaluator.evaluate(samples.copy())
    return {name: float(value) for name, value in metrics.items()}


def confidence_interval(hit_rate: float, count: int) -> tuple[float, float]:
    standard_error = math.sqrt(hit_rate * (1.0 - hit_rate) / count)
    violation = 1.0 - hit_rate
    return (
        max(0.0, violation - 1.96 * standard_error),
        min(1.0, violation + 1.96 * standard_error),
    )


def aggregate(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metrics = [
        "raw_joint_hit_rate",
        "violation_rate",
        "shape",
        "trend",
        "overall",
        "conditional_real_rate",
        "conditional_real_rows",
        "numeric_joint_miss_rate",
        "categorical_joint_miss_rate",
        "numeric_mean_column_miss_rate",
        "categorical_mean_column_miss_rate",
    ]
    grouped = rows.groupby(keys, sort=True, dropna=False)
    means = grouped[metrics].mean().add_suffix("_mean")
    standard_deviations = grouped[metrics].std(ddof=0).add_suffix("_std")
    counts = grouped.size().rename("num_queries")
    return pd.concat((means, standard_deviations, counts), axis=1).reset_index()


def make_plots(grouped: pd.DataFrame, output_dir: Path, group_by: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

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
            axis.fill_between(
                x,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                alpha=0.14,
            )
    for axis, (_, title, ylabel) in zip(axes, definitions):
        if group_by == "target_band":
            axis.set_xscale("log")
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(
            "Target selectivity band" if group_by == "target_band" else "Active predicates"
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"query_suite_by_{group_by}.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    query_dir = Path(args.query_dir)
    real_path = Path(args.real_data)
    info_path = Path(args.info_file)
    methods = parse_methods(args.method)
    if args.baseline_method is not None and args.baseline_method not in methods:
        raise ValueError("baseline-method must match one of the supplied method labels")
    for path in (query_dir, real_path, info_path, *methods.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    queries = load_queries(query_dir)
    real = pd.read_csv(real_path)
    with info_path.open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    output_dir = Path(args.output_dir)
    reference_dir = output_dir / "conditional_real_references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, query in queries:
        query_id = query["query_id"]
        target_band = float(query["target_band"])
        arity = int(query.get("arity", len(query["predicates"])))
        real_report, real_mask = raw_constraint_report(real, query)
        conditional_real = real.loc[real_mask].reset_index(drop=True)
        if len(conditional_real) < 2:
            raise ValueError(f"fewer than two real rows satisfy {query_id}")
        reference_path = reference_dir / f"{query_id}.csv"
        conditional_real.to_csv(reference_path, index=False)

        for label, sample_dir in methods.items():
            samples_path = sample_dir / f"{query_id}.csv"
            if not samples_path.is_file():
                raise FileNotFoundError(samples_path)
            samples = pd.read_csv(samples_path)
            if list(samples.columns) != list(real.columns):
                raise ValueError(f"column mismatch for {samples_path}")
            constraint_report, _ = raw_constraint_report(samples, query)
            modality_report = raw_modality_constraint_report(samples, query)
            metrics = density_metrics(reference_path, samples, info)
            hit_rate = float(constraint_report["joint_hit_rate"])
            ci_low, ci_high = confidence_interval(hit_rate, len(samples))
            rows.append(
                {
                    "method": label,
                    "query_id": query_id,
                    "target_band": target_band,
                    "arity": arity,
                    "num_numeric_constraints": modality_report["numeric"]["num_constraints"],
                    "num_categorical_constraints": modality_report["categorical"]["num_constraints"],
                    "generated_rows": len(samples),
                    "conditional_real_rows": len(conditional_real),
                    "conditional_real_rate": real_report["joint_hit_rate"],
                    "raw_joint_hit_rate": hit_rate,
                    "violation_rate": 1.0 - hit_rate,
                    "numeric_joint_miss_rate": modality_report["numeric"][
                        "joint_miss_rate"
                    ],
                    "categorical_joint_miss_rate": modality_report["categorical"][
                        "joint_miss_rate"
                    ],
                    "numeric_mean_column_miss_rate": modality_report["numeric"][
                        "mean_per_constraint_miss_rate"
                    ],
                    "categorical_mean_column_miss_rate": modality_report[
                        "categorical"
                    ]["mean_per_constraint_miss_rate"],
                    "violation_ci95_low": ci_low,
                    "violation_ci95_high": ci_high,
                    "shape": metrics["density/Shape"],
                    "trend": metrics["density/Trend"],
                    "overall": metrics["density/Overall"],
                    "samples": str(samples_path),
                }
            )

    per_query = pd.DataFrame(rows)
    grouping_column = args.group_by
    grouped = aggregate(per_query, ["method", grouping_column])
    overall = aggregate(per_query, ["method"])
    per_query.to_csv(output_dir / "per_query.csv", index=False)
    grouped_filename = (
        "by_selectivity_band.csv" if grouping_column == "target_band" else "by_arity.csv"
    )
    grouped.to_csv(output_dir / grouped_filename, index=False)
    overall.to_csv(output_dir / "overall.csv", index=False)
    relative_grouped = None
    if args.baseline_method is not None:
        difference_metrics = [
            "raw_joint_hit_rate",
            "violation_rate",
            "numeric_joint_miss_rate",
            "categorical_joint_miss_rate",
            "shape",
            "trend",
            "overall",
        ]
        baseline = per_query[per_query["method"] == args.baseline_method][
            ["query_id", *difference_metrics]
        ].set_index("query_id")
        relative_parts = []
        for method in methods:
            if method == args.baseline_method:
                continue
            selected = per_query[per_query["method"] == method].copy()
            selected = selected.set_index("query_id")
            if set(selected.index) != set(baseline.index):
                raise ValueError(f"{method} and baseline query sets differ")
            for metric in difference_metrics:
                selected[f"delta_{metric}"] = selected[metric] - baseline[metric]
            selected["baseline_method"] = args.baseline_method
            relative_parts.append(selected.reset_index())
        if relative_parts:
            relative = pd.concat(relative_parts, ignore_index=True)
            relative.to_csv(output_dir / "relative_to_baseline_per_query.csv", index=False)
            delta_columns = [f"delta_{metric}" for metric in difference_metrics]
            grouped_delta = relative.groupby(
                ["method", grouping_column], sort=True
            )[delta_columns]
            delta_means = grouped_delta.mean().add_suffix("_mean")
            delta_stds = grouped_delta.std(ddof=0).add_suffix("_std")
            relative_grouped = pd.concat((delta_means, delta_stds), axis=1).reset_index()
            relative_grouped.to_csv(
                output_dir / "relative_to_baseline_grouped.csv", index=False
            )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "query_directory": str(query_dir),
                "num_queries": len(queries),
                "methods": list(methods),
                "overall": overall.to_dict(orient="records"),
                "group_by": grouping_column,
                "grouped": grouped.to_dict(orient="records"),
                "baseline_method": args.baseline_method,
                "relative_to_baseline": (
                    None
                    if relative_grouped is None
                    else relative_grouped.to_dict(orient="records")
                ),
            },
            stream,
            indent=2,
        )
    make_plots(grouped, output_dir, grouping_column)
    print(f"Evaluated {len(queries)} queries for {len(methods)} model(s)")
    print(overall.to_string(index=False))
    print(f"Saved query-suite evaluation to {output_dir}")


if __name__ == "__main__":
    main()
