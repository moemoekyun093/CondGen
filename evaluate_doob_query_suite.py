"""Compare arbitrary conditional generators over a full-arity query suite."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
from tabdiff.query_split import load_query_split
from tabdiff.query_suite_samples import replicate_sample_path


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
        choices=("target_band", "arity", "mean_interval_width"),
        default="target_band",
    )
    parser.add_argument(
        "--query-coordinates",
        default=None,
        help="Exact transformed query coordinates exported by export_query_model_coordinates.py",
    )
    parser.add_argument("--interval-width-bins", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--baseline-method",
        default=None,
        help="Optional method label used for paired per-query metric differences",
    )
    parser.add_argument("--alpha-beta-seed", type=int, default=0)
    parser.add_argument(
        "--sample-seed-base",
        action="append",
        type=int,
        default=[],
        help="Repeat for sampling replicates. The first seed uses legacy direct CSVs.",
    )
    parser.add_argument(
        "--alpha-beta-results",
        default=None,
        help="CSV produced by evaluate_synthcity_alpha_suite.py",
    )
    parser.add_argument(
        "--filtered-min-rows",
        type=int,
        default=50,
        help="Minimum rows required on each side for filtered Shape/Trend",
    )
    parser.add_argument(
        "--reuse-existing-full-metrics",
        action="store_true",
        help="Reuse Shape/Trend/C2ST values already present in output per_query.csv",
    )
    parser.add_argument(
        "--test-supported-only",
        action="store_true",
        help="Evaluate only queries marked as supported by the held-out test split",
    )
    parser.add_argument("--query-split-manifest", default=None)
    parser.add_argument("--query-split", choices=("train", "test"), default=None)
    parser.add_argument(
        "--query-id",
        action="append",
        default=[],
        help="Evaluate only these query ids; repeat for multiple queries",
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


def load_queries(
    query_dir: Path,
    *,
    test_supported_only: bool = False,
    query_ids: set[str] | None = None,
) -> list[tuple[Path, dict]]:
    queries = []
    for path in sorted(query_dir.glob("q*.json")):
        with path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if query.get("accepted", True) and (
            not test_supported_only or query.get("test_supported", False)
        ) and (query_ids is None or query["query_id"] in query_ids):
            queries.append((path, query))
    if not queries:
        raise ValueError(f"no accepted queries found in {query_dir}")
    return queries


def tabular_metrics(reference_path: Path, samples: pd.DataFrame, info: dict) -> dict:
    evaluator = TabMetrics(
        str(reference_path),
        str(reference_path),
        None,
        info,
        torch.device("cpu"),
        metric_list=["density", "c2st", "c2st_xgb"],
        include_density_diagnostic=False,
    )
    metrics, _ = evaluator.evaluate(samples.copy())
    return {name: float(value) for name, value in metrics.items()}


def tabular_metrics_task(task):
    reference_path, samples, info = task
    return tabular_metrics(Path(reference_path), samples, info)


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
        "c2st",
        "c2st_xgb",
        "c2st_xgb_auc",
        "alpha_precision",
        "beta_recall",
        "conditional_real_rate",
        "conditional_real_rows",
        "numeric_joint_miss_rate",
        "categorical_joint_miss_rate",
        "numeric_mean_column_miss_rate",
        "categorical_mean_column_miss_rate",
        "filtered_valid_rows",
        "filtered_reference_rows",
        "filtered_minimum_side_rows",
        "filtered_shape",
        "filtered_trend",
        "filtered_overall",
        "filtered_c2st",
        "filtered_c2st_xgb",
        "filtered_c2st_xgb_auc",
    ]
    grouped = rows.groupby(keys, sort=True, dropna=False)
    means = grouped[metrics].mean().add_suffix("_mean")
    standard_deviations = grouped[metrics].std(ddof=0).add_suffix("_std")
    counts = grouped.size().rename("num_queries")
    filtered_counts = grouped["filtered_shape"].count().rename(
        "num_filtered_queries_available"
    )
    return pd.concat(
        (means, standard_deviations, counts, filtered_counts), axis=1
    ).reset_index()


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

    quality_definitions = (
        ("c2st", "Logistic C2ST similarity"),
        ("c2st_xgb", "XGBoost C2ST similarity"),
        ("alpha_precision", "Alpha Precision"),
        ("beta_recall", "Beta Recall"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharex=True)
    for label in grouped["method"].unique():
        selected = grouped[grouped["method"] == label].sort_values(group_by)
        x = selected[group_by].to_numpy(dtype=float)
        for axis, (metric, _) in zip(axes.flat, quality_definitions):
            mean = selected[f"{metric}_mean"].to_numpy(dtype=float)
            std = selected[f"{metric}_std"].to_numpy(dtype=float)
            axis.plot(x, mean, marker="o", linewidth=2, label=label)
            axis.fill_between(
                x,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.0, 1.0),
                alpha=0.14,
            )
    for axis, (_, title) in zip(axes.flat, quality_definitions):
        if group_by == "target_band":
            axis.set_xscale("log")
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(
            "Target selectivity band" if group_by == "target_band" else "Active predicates"
        )
        axis.set_ylabel("Score (higher is better)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes.flat[0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"quality_metrics_by_{group_by}.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if (args.query_split_manifest is None) != (args.query_split is None):
        raise ValueError(
            "--query-split-manifest and --query-split must be supplied together"
        )
    if args.query_id and args.query_split_manifest is not None:
        raise ValueError("query-id cannot be combined with a query split manifest")
    if args.filtered_min_rows <= 0:
        raise ValueError("filtered-min-rows must be positive")
    if args.interval_width_bins < 2:
        raise ValueError("interval-width-bins must be at least 2")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    query_dir = Path(args.query_dir)
    real_path = Path(args.real_data)
    info_path = Path(args.info_file)
    methods = parse_methods(args.method)
    alpha_beta_lookup = {}
    if args.alpha_beta_results is not None:
        alpha_beta_frame = pd.read_csv(args.alpha_beta_results)
        required = {"method", "query_id", "alpha_precision", "beta_recall"}
        if not required.issubset(alpha_beta_frame.columns):
            raise ValueError("alpha-beta results CSV is missing required columns")
        for row in alpha_beta_frame.itertuples(index=False):
            seed_base = int(getattr(row, "seed_base", 0))
            key = (str(row.method), str(row.query_id), seed_base)
            if key in alpha_beta_lookup:
                raise ValueError(f"duplicate Alpha/Beta result for {key}")
            alpha_beta_lookup[key] = (
                float(row.alpha_precision),
                float(row.beta_recall),
            )
    if args.baseline_method is not None and args.baseline_method not in methods:
        raise ValueError("baseline-method must match one of the supplied method labels")
    for path in (query_dir, real_path, info_path, *methods.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    selected_query_ids = set(args.query_id) if args.query_id else None
    if args.query_split_manifest is not None:
        selected_query_ids = set(
            load_query_split(args.query_split_manifest, args.query_split)
        )
    queries = load_queries(
        query_dir,
        test_supported_only=args.test_supported_only,
        query_ids=selected_query_ids,
    )
    if selected_query_ids is not None and len(queries) != len(selected_query_ids):
        raise ValueError(
            f"query split selects {len(selected_query_ids)} ids but "
            f"{len(queries)} accepted query files were loaded"
        )
    query_coordinates = None
    if args.query_coordinates is not None:
        with Path(args.query_coordinates).open("r", encoding="utf-8") as stream:
            query_coordinates = json.load(stream)["queries"]
        missing_coordinates = {
            query["query_id"] for _, query in queries
        } - set(query_coordinates)
        if missing_coordinates:
            raise ValueError(
                "query coordinate file is missing selected ids: "
                f"{sorted(missing_coordinates)[:5]}"
            )
    elif args.group_by == "mean_interval_width":
        raise ValueError("--query-coordinates is required for mean interval width")
    real = pd.read_csv(real_path)
    with info_path.open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    output_dir = Path(args.output_dir)
    existing_full_metrics = {}
    existing_per_query_path = output_dir / "per_query_seed.csv"
    if args.reuse_existing_full_metrics and existing_per_query_path.is_file():
        existing_frame = pd.read_csv(existing_per_query_path)
        cache_columns = {
            "method", "query_id", "samples", "shape", "trend", "overall",
            "c2st", "c2st_xgb", "c2st_xgb_auc", "alpha_precision", "beta_recall",
        }
        if cache_columns.issubset(existing_frame.columns):
            existing_full_metrics = {
                (
                    str(row.method),
                    str(row.query_id),
                    int(getattr(row, "seed_base", 0)),
                ): row
                for row in existing_frame.itertuples(index=False)
            }
            print(
                f"Loaded {len(existing_full_metrics)} cached full-metric rows from "
                f"{existing_per_query_path}"
            )
        else:
            print("Existing per_query.csv is missing cache columns; recomputing full metrics")
    reference_dir = output_dir / "conditional_real_references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    metric_executor = (
        None if args.workers == 1 else ProcessPoolExecutor(max_workers=args.workers)
    )
    for query_index, (_, query) in enumerate(queries):
        query_id = query["query_id"]
        target_band = float(query["target_band"])
        arity = int(query.get("arity", len(query["predicates"])))
        mean_interval_width = float("nan")
        if query_coordinates is not None:
            coordinate = query_coordinates[query_id]
            active = np.asarray(coordinate["numerical_active"], dtype=bool)
            widths = (
                np.asarray(coordinate["numerical_upper"], dtype=float)
                - np.asarray(coordinate["numerical_lower"], dtype=float)
            )[active]
            if len(widths):
                mean_interval_width = float(widths.mean())
        real_report, real_mask = raw_constraint_report(real, query)
        conditional_real = real.loc[real_mask].reset_index(drop=True)
        reference_metrics_available = len(conditional_real) >= 2
        if not reference_metrics_available:
            print(
                f"Reference-based metrics unavailable for {query_id}: "
                f"conditional test rows={len(conditional_real)}"
            )
        reference_path = reference_dir / f"{query_id}.csv"
        conditional_real.to_csv(reference_path, index=False)

        seed_bases = args.sample_seed_base or [0]
        payloads = {}
        for label, sample_dir in methods.items():
            for seed_index, seed_base in enumerate(seed_bases):
                samples_path = replicate_sample_path(
                    sample_dir, query_id, seed_base, seed_index
                )
                if not samples_path.is_file():
                    raise FileNotFoundError(samples_path)
                samples = pd.read_csv(samples_path)
                if list(samples.columns) != list(real.columns):
                    raise ValueError(f"column mismatch for {samples_path}")
                constraint_report, joint_mask = raw_constraint_report(samples, query)
                modality_report = raw_modality_constraint_report(samples, query)
                payloads[(label, seed_base)] = {
                    "samples_path": samples_path,
                    "samples": samples,
                    "valid_samples": samples.loc[joint_mask].reset_index(drop=True),
                    "constraint_report": constraint_report,
                    "modality_report": modality_report,
                }

        metric_request_keys = []
        metric_request_payloads = []
        for (label, seed_base), payload in payloads.items():
            samples_path = payload["samples_path"]
            cached = existing_full_metrics.get((label, query_id, seed_base))
            cache_matches = cached is not None and str(cached.samples) == str(
                samples_path
            )
            if reference_metrics_available and not cache_matches:
                metric_request_keys.append(("full", label, seed_base))
                metric_request_payloads.append(
                    (str(reference_path), payload["samples"], info)
                )
            minimum_side_rows = min(
                len(conditional_real), len(payload["valid_samples"])
            )
            if minimum_side_rows >= args.filtered_min_rows:
                metric_request_keys.append(("filtered", label, seed_base))
                metric_request_payloads.append(
                    (str(reference_path), payload["valid_samples"], info)
                )
        if metric_executor is None:
            metric_values = map(tabular_metrics_task, metric_request_payloads)
        else:
            metric_values = metric_executor.map(
                tabular_metrics_task, metric_request_payloads
            )
        metric_results = dict(zip(metric_request_keys, metric_values))

        for (label, seed_base), payload in payloads.items():
            samples_path = payload["samples_path"]
            samples = payload["samples"]
            constraint_report = payload["constraint_report"]
            modality_report = payload["modality_report"]
            cached = existing_full_metrics.get((label, query_id, seed_base))
            if not reference_metrics_available:
                metrics = {
                    "density/Shape": float("nan"),
                    "density/Trend": float("nan"),
                    "density/Overall": float("nan"),
                    "c2st": float("nan"),
                    "c2st_xgb": float("nan"),
                    "c2st_xgb_auc": float("nan"),
                }
            elif cached is not None and str(cached.samples) == str(samples_path):
                metrics = {
                    "density/Shape": float(cached.shape),
                    "density/Trend": float(cached.trend),
                    "density/Overall": float(cached.overall),
                    "c2st": float(cached.c2st),
                    "c2st_xgb": float(cached.c2st_xgb),
                    "c2st_xgb_auc": float(cached.c2st_xgb_auc),
                }
            else:
                metrics = metric_results[("full", label, seed_base)]
            alpha_key = (label, query_id, seed_base)
            if args.alpha_beta_results is not None and alpha_key not in alpha_beta_lookup:
                raise ValueError(f"missing official SynthCity result for {alpha_key}")
            if alpha_key in alpha_beta_lookup:
                alpha_precision, beta_recall = alpha_beta_lookup[alpha_key]
            elif cached is not None and str(cached.samples) == str(samples_path):
                alpha_precision = float(cached.alpha_precision)
                beta_recall = float(cached.beta_recall)
            else:
                alpha_precision, beta_recall = float("nan"), float("nan")
            hit_rate = float(constraint_report["joint_hit_rate"])
            ci_low, ci_high = confidence_interval(hit_rate, len(samples))
            valid_samples = payload["valid_samples"]
            minimum_side_rows = min(len(conditional_real), len(valid_samples))
            filtered_available = minimum_side_rows >= args.filtered_min_rows
            if not filtered_available:
                filtered_reliability = "unavailable"
                filtered_shape = filtered_trend = filtered_overall = float("nan")
                filtered_c2st = filtered_c2st_xgb = filtered_c2st_xgb_auc = float(
                    "nan"
                )
                print(
                    f"Filtered metrics unavailable for {label}/{query_id}: "
                    f"valid_generated={len(valid_samples)}, "
                    f"conditional_real={len(conditional_real)}, "
                    f"minimum={args.filtered_min_rows}"
                )
            else:
                filtered_reliability = (
                    "exploratory_below_200_rows"
                    if minimum_side_rows < 200
                    else "preferred_200_plus_rows"
                )
                filtered = metric_results[("filtered", label, seed_base)]
                filtered_shape = filtered["density/Shape"]
                filtered_trend = filtered["density/Trend"]
                filtered_overall = filtered["density/Overall"]
                filtered_c2st = filtered["c2st"]
                filtered_c2st_xgb = filtered["c2st_xgb"]
                filtered_c2st_xgb_auc = filtered["c2st_xgb_auc"]
            rows.append(
                {
                    "method": label,
                    "query_id": query_id,
                    "seed_base": seed_base,
                    "target_band": target_band,
                    "arity": arity,
                    "mean_transformed_interval_width": mean_interval_width,
                    "num_numeric_constraints": modality_report["numeric"]["num_constraints"],
                    "num_categorical_constraints": modality_report["categorical"]["num_constraints"],
                    "generated_rows": len(samples),
                    "conditional_real_rows": len(conditional_real),
                    "reference_metrics_available": reference_metrics_available,
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
                    "filtered_valid_rows": len(payload["valid_samples"]),
                    "filtered_reference_rows": len(conditional_real),
                    "filtered_minimum_side_rows": minimum_side_rows,
                    "filtered_min_rows": args.filtered_min_rows,
                    "filtered_metrics_available": filtered_available,
                    "filtered_reliability": filtered_reliability,
                    "filtered_shape": filtered_shape,
                    "filtered_trend": filtered_trend,
                    "filtered_overall": filtered_overall,
                    "filtered_c2st": filtered_c2st,
                    "filtered_c2st_xgb": filtered_c2st_xgb,
                    "filtered_c2st_xgb_auc": filtered_c2st_xgb_auc,
                    "c2st": metrics["c2st"],
                    "c2st_xgb": metrics["c2st_xgb"],
                    "c2st_xgb_auc": metrics["c2st_xgb_auc"],
                    "alpha_precision": alpha_precision,
                    "beta_recall": beta_recall,
                    "alpha_precision_backend": "synthcity.AlphaPrecision.naive",
                    "samples": str(samples_path),
                }
            )

    if metric_executor is not None:
        metric_executor.shutdown(wait=True, cancel_futures=True)
    per_query_seed = pd.DataFrame(rows)
    print("Feasible-only Shape/Trend reliability:")
    print(
        per_query_seed[["method", "query_id", "filtered_reliability"]]
        ["filtered_reliability"]
        .value_counts()
        .to_string()
    )
    aggregation = {}
    for column in per_query_seed.columns:
        if column in {"method", "query_id", "seed_base"}:
            continue
        if column == "samples":
            aggregation[column] = lambda values: ";".join(map(str, values))
        elif pd.api.types.is_bool_dtype(per_query_seed[column]):
            aggregation[column] = "first"
        elif pd.api.types.is_numeric_dtype(per_query_seed[column]):
            aggregation[column] = "mean"
        else:
            aggregation[column] = "first"
    per_query = (
        per_query_seed.groupby(["method", "query_id"], sort=True, as_index=False)
        .agg(aggregation)
    )
    replicate_metrics = [
        "violation_rate",
        "numeric_joint_miss_rate",
        "categorical_joint_miss_rate",
        "shape",
        "trend",
        "overall",
        "c2st",
        "c2st_xgb",
        "alpha_precision",
        "beta_recall",
    ]
    replicate_standard_deviations = (
        per_query_seed.groupby(["method", "query_id"], sort=True)[replicate_metrics]
        .std(ddof=0)
        .add_suffix("_seed_std")
        .reset_index()
    )
    per_query = per_query.merge(
        replicate_standard_deviations,
        on=["method", "query_id"],
        how="left",
        validate="one_to_one",
    )
    per_query["num_sampling_seeds"] = len(args.sample_seed_base or [0])
    by_mean_interval_width = None
    if query_coordinates is not None:
        query_widths = (
            per_query[["query_id", "mean_transformed_interval_width"]]
            .drop_duplicates("query_id")
            .dropna(subset=["mean_transformed_interval_width"])
        )
        edges = np.unique(
            np.quantile(
                query_widths["mean_transformed_interval_width"],
                np.linspace(0.0, 1.0, args.interval_width_bins + 1),
            )
        )
        per_query["mean_interval_width_bin"] = pd.NA
        per_query["mean_interval_width_bin_midpoint"] = np.nan
        if len(edges) >= 2:
            width_bins = pd.cut(
                per_query["mean_transformed_interval_width"],
                bins=edges,
                include_lowest=True,
                duplicates="drop",
            )
            per_query["mean_interval_width_bin"] = width_bins.astype("string")
            per_query["mean_interval_width_bin_midpoint"] = [
                interval.mid if pd.notna(interval) else np.nan
                for interval in width_bins
            ]
            by_mean_interval_width = aggregate(
                per_query.dropna(subset=["mean_interval_width_bin_midpoint"]),
                ["method", "mean_interval_width_bin_midpoint"],
            )
    grouping_column = (
        "mean_interval_width_bin_midpoint"
        if args.group_by == "mean_interval_width"
        else args.group_by
    )
    by_selectivity_band = aggregate(per_query, ["method", "target_band"])
    by_arity = aggregate(per_query, ["method", "arity"])
    if grouping_column == "target_band":
        grouped = by_selectivity_band
    elif grouping_column == "arity":
        grouped = by_arity
    else:
        if by_mean_interval_width is None:
            raise ValueError("no numerical interval widths are available")
        grouped = by_mean_interval_width
    overall = aggregate(per_query, ["method"])
    sampling_seed_variability = (
        per_query.groupby("method", sort=True)[
            [f"{metric}_seed_std" for metric in replicate_metrics]
        ]
        .mean()
        .reset_index()
    )
    per_query_seed.to_csv(output_dir / "per_query_seed.csv", index=False)
    per_query.to_csv(output_dir / "per_query.csv", index=False)
    by_selectivity_band.to_csv(output_dir / "by_selectivity_band.csv", index=False)
    by_arity.to_csv(output_dir / "by_arity.csv", index=False)
    if by_mean_interval_width is not None:
        by_mean_interval_width.to_csv(
            output_dir / "by_mean_interval_width.csv", index=False
        )
    overall.to_csv(output_dir / "overall.csv", index=False)
    sampling_seed_variability.to_csv(
        output_dir / "sampling_seed_variability.csv", index=False
    )
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
            "filtered_shape",
            "filtered_trend",
            "filtered_overall",
            "filtered_c2st",
            "filtered_c2st_xgb",
            "filtered_c2st_xgb_auc",
            "c2st",
            "c2st_xgb",
            "alpha_precision",
            "beta_recall",
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
                "test_supported_only": args.test_supported_only,
                "query_split_manifest": args.query_split_manifest,
                "query_split": args.query_split,
                "reference_data": str(real_path),
                "num_queries": len(queries),
                "num_sampling_seeds": len(args.sample_seed_base or [0]),
                "sample_seed_bases": args.sample_seed_base or [0],
                "methods": list(methods),
                "overall": overall.to_dict(orient="records"),
                "sampling_seed_variability": sampling_seed_variability.to_dict(
                    orient="records"
                ),
                "group_by": grouping_column,
                "grouped": grouped.to_dict(orient="records"),
                "by_selectivity_band": by_selectivity_band.to_dict(orient="records"),
                "by_arity": by_arity.to_dict(orient="records"),
                "by_mean_interval_width": (
                    None
                    if by_mean_interval_width is None
                    else by_mean_interval_width.to_dict(orient="records")
                ),
                "baseline_method": args.baseline_method,
                "filtered_feasible_evaluation": {
                    "minimum_rows_per_side": args.filtered_min_rows,
                    "uses_all_valid_generated_rows": True,
                    "uses_all_conditional_real_rows": True,
                    "matched_across_methods": False,
                    "recommended_rows_for_trend": 200,
                    "metrics": [
                        "shape",
                        "trend",
                        "overall",
                        "logistic_c2st_similarity",
                        "xgboost_c2st_similarity",
                        "xgboost_c2st_auc",
                    ],
                },
                "alpha_precision_backend": (
                    "synthcity.metrics.eval_statistical.AlphaPrecision"
                    if args.alpha_beta_results is not None
                    else None
                ),
                "relative_to_baseline": (
                    None
                    if relative_grouped is None
                    else relative_grouped.to_dict(orient="records")
                ),
            },
            stream,
            indent=2,
        )
    try:
        from tabdiff.query_suite_plots import make_query_suite_plots

        # Always emit all meaningful views from the same cached evaluations.
        # The primary grouping controls baseline-delta tables, not which plots exist.
        make_query_suite_plots(by_selectivity_band, output_dir, "target_band")
        make_query_suite_plots(by_arity, output_dir, "arity")
        if by_mean_interval_width is not None:
            make_query_suite_plots(
                by_mean_interval_width,
                output_dir,
                "mean_interval_width_bin_midpoint",
            )
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        print(
            "Matplotlib is unavailable; metrics and CSVs were saved. "
            "Run plot_doob_query_suite_results.py in an environment with Matplotlib."
        )
    print(f"Evaluated {len(queries)} queries for {len(methods)} model(s)")
    print(overall.to_string(index=False))
    print(f"Saved query-suite evaluation to {output_dir}")


if __name__ == "__main__":
    main()
