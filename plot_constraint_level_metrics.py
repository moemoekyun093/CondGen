"""Plot Shape, Trend, and violations for arbitrary constraint-sweep models."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from alpha_precision_standalone import (
    alpha_precision_beta_recall_authenticity,
    build_features,
)
from tabdiff.doob_h_evaluation import raw_constraint_report
from tabdiff.metrics import TabMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--method",
        action="append",
        nargs=2,
        metavar=("LABEL", "SAMPLES_GLOB"),
        help="Completed constraint sweep; repeat to compare multiple methods",
    )
    parser.add_argument(
        "--method-params",
        action="append",
        nargs=2,
        metavar=("LABEL", "JSON_OBJECT"),
        help="Optional metadata for a --method label, such as learning rate",
    )
    parser.add_argument(
        "--unconditional",
        nargs=2,
        metavar=("LABEL", "SAMPLES_CSV"),
    )
    parser.add_argument("--unconditional-params", default=None)
    parser.add_argument("--real-data", default=None)
    parser.add_argument("--info-file", default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--alpha-beta-seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def validate_config(config: dict) -> dict:
    series = config.get("series")
    unconditional = config.get("unconditional")
    if not isinstance(series, list) or not series:
        raise ValueError("config.series must be a non-empty list")
    if not isinstance(unconditional, dict):
        raise ValueError("config.unconditional must be an object")
    labels = [str(item.get("label", "")).strip() for item in series]
    labels.append(str(unconditional.get("label", "")).strip())
    if any(not label for label in labels):
        raise ValueError("every series and the unconditional baseline need a label")
    if len(labels) != len(set(labels)):
        raise ValueError("series labels must be unique")
    sample_limit = config.get("sample_limit")
    if sample_limit is not None and int(sample_limit) <= 0:
        raise ValueError("sample_limit must be positive")
    for key in ("real_data", "info_file"):
        if not config.get(key):
            raise ValueError(f"config.{key} is required")
    for item in [*series, unconditional]:
        parameters = item.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"parameters for {item['label']!r} must be an object")
    for item in series:
        if not item.get("samples_glob"):
            raise ValueError(f"samples_glob for {item['label']!r} is required")
    if not unconditional.get("samples"):
        raise ValueError("config.unconditional.samples is required")
    return config


def json_object(text: str, description: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON for {description}: {text!r}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def resolve_config(args: argparse.Namespace) -> dict:
    if args.config:
        with open(args.config, "r", encoding="utf-8") as stream:
            config = json.load(stream)
    else:
        config = {}

    method_parameters = {}
    for label, value in args.method_params or []:
        if label in method_parameters:
            raise ValueError(f"duplicate --method-params for {label!r}")
        method_parameters[label] = json_object(value, f"method {label!r}")
    if args.method:
        config["series"] = [
            {
                "label": label,
                "samples_glob": pattern,
                "parameters": method_parameters.pop(label, {}),
            }
            for label, pattern in args.method
        ]
    elif method_parameters:
        configured = {
            str(item["label"]): item for item in config.get("series", [])
        }
        for label, parameters in method_parameters.items():
            if label not in configured:
                raise ValueError(
                    f"--method-params label {label!r} is absent from config.series"
                )
            configured[label]["parameters"] = parameters
        method_parameters.clear()
    if method_parameters:
        raise ValueError(
            "--method-params labels must exactly match supplied --method labels: "
            + ", ".join(method_parameters)
        )

    if args.unconditional:
        label, samples = args.unconditional
        config["unconditional"] = {
            "label": label,
            "samples": samples,
            "parameters": {},
        }
    if args.unconditional_params is not None:
        if "unconditional" not in config:
            raise ValueError(
                "--unconditional-params requires --unconditional or config.unconditional"
            )
        config["unconditional"]["parameters"] = json_object(
            args.unconditional_params,
            "unconditional baseline",
        )
    for argument, key in (
        (args.real_data, "real_data"),
        (args.info_file, "info_file"),
        (args.sample_limit, "sample_limit"),
        (args.alpha_beta_seed, "alpha_beta_seed"),
        (args.output_dir, "output_dir"),
    ):
        if argument is not None:
            config[key] = argument
    return validate_config(config)


def canonical_columns(query: dict) -> tuple:
    return tuple(
        (
            int(column.get("model_index", position)),
            str(column["name"]),
            None if column.get("raw_lower") is None else float(column["raw_lower"]),
            None if column.get("raw_upper") is None else float(column["raw_upper"]),
        )
        for position, column in enumerate(query.get("columns", []))
    )


def discover_sweep(pattern: str) -> dict[int, tuple[Path, dict]]:
    levels = {}
    for filename in sorted(glob.glob(pattern)):
        samples_path = Path(filename)
        query_path = samples_path.with_suffix(".query.json")
        if not query_path.is_file():
            raise FileNotFoundError(query_path)
        with open(query_path, "r", encoding="utf-8") as stream:
            query = json.load(stream)
        count = len(query.get("columns", []))
        if count <= 0:
            raise ValueError(f"query contains no active constraints: {query_path}")
        if count in levels:
            raise ValueError(
                f"glob {pattern!r} matched multiple CSVs for level {count}"
            )
        levels[count] = (samples_path, query)
    if not levels:
        raise FileNotFoundError(f"no sample CSVs matched {pattern!r}")
    counts = sorted(levels)
    if counts != list(range(min(counts), max(counts) + 1)):
        raise ValueError(f"constraint levels are incomplete for {pattern!r}: {counts}")
    return levels


def evaluate_tabdiff_metrics(
    reference_path: Path,
    samples: pd.DataFrame,
    info: dict,
) -> dict:
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


def evaluate_alpha_beta(
    reference_path: Path,
    samples: pd.DataFrame,
    info: dict,
    seed: int,
) -> tuple[float, float]:
    """Use the repository's exact standalone SynthCity-naive metric port."""
    real = pd.read_csv(reference_path)
    real_features, synthetic_features = build_features(real, samples, info)
    if not np.isfinite(real_features).all() or not np.isfinite(
        synthetic_features
    ).all():
        raise ValueError("Alpha Precision/Beta Recall inputs contain non-finite values")

    random = np.random.RandomState(seed)
    count = min(len(real_features), len(synthetic_features))
    if len(real_features) > count:
        real_features = real_features[
            random.choice(len(real_features), count, replace=False)
        ]
    if len(synthetic_features) > count:
        synthetic_features = synthetic_features[
            random.choice(len(synthetic_features), count, replace=False)
        ]
    alpha_precision, beta_recall, _ = alpha_precision_beta_recall_authenticity(
        real_features,
        synthetic_features,
        seed=seed,
    )
    return alpha_precision, beta_recall


def read_samples(path: Path, columns: list[str], sample_limit: int | None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != columns:
        raise ValueError(f"columns in {path} do not match the real reference")
    if sample_limit is not None:
        if len(frame) < sample_limit:
            raise ValueError(
                f"{path} contains {len(frame)} rows, fewer than sample_limit={sample_limit}"
            )
        frame = frame.iloc[:sample_limit].reset_index(drop=True)
    return frame


def metric_row(
    label: str,
    kind: str,
    parameters: dict,
    level: int,
    samples_path: Path,
    samples: pd.DataFrame,
    query: dict,
    reference_path: Path,
    conditional_real_rows: int,
    info: dict,
    alpha_beta_seed: int,
) -> dict:
    constraint_report, _ = raw_constraint_report(samples, query)
    metrics = evaluate_tabdiff_metrics(reference_path, samples, info)
    alpha_precision, beta_recall = evaluate_alpha_beta(
        reference_path,
        samples,
        info,
        alpha_beta_seed,
    )
    hit_rate = float(constraint_report["joint_hit_rate"])
    standard_error = math.sqrt(hit_rate * (1.0 - hit_rate) / len(samples))
    violation = 1.0 - hit_rate
    return {
        "label": label,
        "kind": kind,
        "parameters": json.dumps(parameters, sort_keys=True),
        "constraint_count": level,
        "active_columns": ",".join(
            str(column["name"]) for column in query["columns"]
        ),
        "generated_rows": len(samples),
        "conditional_real_rows": conditional_real_rows,
        "shape": metrics["density/Shape"],
        "trend": metrics["density/Trend"],
        "overall": metrics["density/Overall"],
        "c2st": metrics["c2st"],
        "c2st_xgb": metrics["c2st_xgb"],
        "c2st_xgb_auc": metrics["c2st_xgb_auc"],
        "alpha_precision": alpha_precision,
        "beta_recall": beta_recall,
        "raw_joint_hit_rate": hit_rate,
        "violation_rate": violation,
        "violation_ci95_low": max(0.0, violation - 1.96 * standard_error),
        "violation_ci95_high": min(1.0, violation + 1.96 * standard_error),
        "samples": str(samples_path),
    }


def save_plot(rows: list[dict], metric: str, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axis = plt.subplots(figsize=(7.4, 4.9))
    labels = list(dict.fromkeys(row["label"] for row in rows))
    for label in labels:
        selected = [row for row in rows if row["label"] == label]
        counts = [row["constraint_count"] for row in selected]
        values = [row[metric] for row in selected]
        linestyle = "--" if selected[0]["kind"] == "unconditional" else "-"
        if metric == "violation_rate":
            lower = [
                row["violation_rate"] - row["violation_ci95_low"]
                for row in selected
            ]
            upper = [
                row["violation_ci95_high"] - row["violation_rate"]
                for row in selected
            ]
            axis.errorbar(
                counts,
                values,
                yerr=[lower, upper],
                marker="o",
                linewidth=2,
                capsize=3,
                linestyle=linestyle,
                label=label,
            )
        else:
            axis.plot(
                counts,
                values,
                marker="o",
                linewidth=2,
                linestyle=linestyle,
                label=label,
            )

    titles = {
        "shape": "Column Shape by constraint level (higher is better)",
        "trend": "Column-pair Trend by constraint level (higher is better)",
        "violation_rate": "Joint constraint violations (lower is better)",
        "c2st": "Logistic C2ST similarity by constraint level (higher is better)",
        "c2st_xgb": "XGBoost C2ST similarity by constraint level (higher is better)",
        "alpha_precision": "Alpha Precision by constraint level (higher is better)",
        "beta_recall": "Beta Recall by constraint level (higher is better)",
    }
    ylabels = {
        "shape": "Shape score",
        "trend": "Trend score",
        "violation_rate": "Joint violation rate",
        "c2st": "C2ST similarity score",
        "c2st_xgb": "C2ST XGBoost similarity score",
        "alpha_precision": "Alpha Precision",
        "beta_recall": "Beta Recall",
    }
    counts = sorted({row["constraint_count"] for row in rows})
    axis.set_title(titles[metric])
    axis.set_xlabel("Number of active numerical constraints")
    axis.set_ylabel(ylabels[metric])
    axis.set_xticks(counts)
    axis.set_ylim(0.0, 1.0)
    if metric == "violation_rate":
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    real_path = Path(config["real_data"])
    info_path = Path(config["info_file"])
    unconditional_path = Path(config["unconditional"]["samples"])
    for path in (real_path, info_path, unconditional_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    series_config = {str(item["label"]): item for item in config["series"]}
    sweeps = {
        label: discover_sweep(str(item["samples_glob"]))
        for label, item in series_config.items()
    }
    first_label = next(iter(sweeps))
    levels = sorted(sweeps[first_label])
    for label, sweep in sweeps.items():
        if sorted(sweep) != levels:
            raise ValueError(
                f"constraint levels differ: {first_label}={levels}, "
                f"{label}={sorted(sweep)}"
            )

    real = pd.read_csv(real_path)
    columns = list(real.columns)
    with open(info_path, "r", encoding="utf-8") as stream:
        info = json.load(stream)
    sample_limit = config.get("sample_limit")
    sample_limit = None if sample_limit is None else int(sample_limit)
    alpha_beta_seed = int(config.get("alpha_beta_seed", 0))
    unconditional = read_samples(unconditional_path, columns, sample_limit)

    output_dir = Path(
        config.get("output_dir")
        or "evaluations/constraint_level_comparison"
    )
    reference_dir = output_dir / "conditional_real_references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for level in levels:
        _, reference_query = sweeps[first_label][level]
        reference_columns = canonical_columns(reference_query)
        for label, sweep in sweeps.items():
            if canonical_columns(sweep[level][1]) != reference_columns:
                raise ValueError(
                    f"{label} uses different constraints at level {level}"
                )

        _, real_mask = raw_constraint_report(real, reference_query)
        conditional_real = real.loc[real_mask].reset_index(drop=True)
        if len(conditional_real) < 2:
            raise ValueError(
                f"fewer than two real rows satisfy constraint level {level}"
            )
        reference_path = reference_dir / f"k{level:02d}.csv"
        conditional_real.to_csv(reference_path, index=False)

        for label, sweep in sweeps.items():
            samples_path, query = sweep[level]
            samples = read_samples(samples_path, columns, sample_limit)
            rows.append(
                metric_row(
                    label,
                    "conditional",
                    series_config[label].get("parameters", {}),
                    level,
                    samples_path,
                    samples,
                    query,
                    reference_path,
                    len(conditional_real),
                    info,
                    alpha_beta_seed,
                )
            )
        rows.append(
            metric_row(
                str(config["unconditional"]["label"]),
                "unconditional",
                config["unconditional"].get("parameters", {}),
                level,
                unconditional_path,
                unconditional,
                reference_query,
                reference_path,
                len(conditional_real),
                info,
                alpha_beta_seed,
            )
        )

    csv_path = output_dir / "constraint_level_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "constraint_level_metrics.json"
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump({"config": config, "results": rows}, stream, indent=2)

    outputs = {
        "shape": output_dir / "shape_by_constraint_level.png",
        "trend": output_dir / "trend_by_constraint_level.png",
        "violation_rate": output_dir / "violation_by_constraint_level.png",
        "c2st": output_dir / "c2st_by_constraint_level.png",
        "c2st_xgb": output_dir / "c2st_xgb_by_constraint_level.png",
        "alpha_precision": output_dir / "alpha_precision_by_constraint_level.png",
        "beta_recall": output_dir / "beta_recall_by_constraint_level.png",
    }
    for metric, output in outputs.items():
        save_plot(rows, metric, output)

    print(f"Evaluated levels {levels}")
    print(f"Series: {', '.join(sweeps)}")
    print(f"Unconditional baseline: {config['unconditional']['label']}")
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    for output in outputs.values():
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
