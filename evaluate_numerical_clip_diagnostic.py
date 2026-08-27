"""Evaluate numerical correction clipping against constraint and quality metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from tabdiff.doob_h_evaluation import (
    raw_constraint_report,
    raw_modality_constraint_report,
)
from tabdiff.metrics import TabMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--real-data", default="synthetic/shoppers/real.csv")
    parser.add_argument("--info-file", default="data/shoppers/info.json")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def density_metrics(reference_path: Path, samples: pd.DataFrame, info: dict) -> dict:
    evaluator = TabMetrics(
        str(reference_path), str(reference_path), None, info, torch.device("cpu"),
        metric_list=["density"], include_density_diagnostic=False,
    )
    metrics, _ = evaluator.evaluate(samples.copy())
    return {name: float(value) for name, value in metrics.items()}


def main() -> None:
    args = parse_args()
    sample_root = Path(args.sample_root)
    query_dir = Path(args.query_dir)
    real_path = Path(args.real_data)
    info_path = Path(args.info_file)
    for path in (sample_root, query_dir, real_path, info_path):
        if not path.exists():
            raise FileNotFoundError(path)
    real = pd.read_csv(real_path)
    with info_path.open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    output_dir = Path(args.output_dir)
    reference_dir = output_dir / "conditional_real_references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    column_rows = []
    time_rows = []
    for cap_dir in sorted(sample_root.glob("cap_*")):
        for samples_path in sorted(cap_dir.glob("qf_*.csv")):
            query_path = query_dir / f"{samples_path.stem}.json"
            guidance_path = samples_path.with_suffix(".guidance.json")
            for path in (query_path, guidance_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            with query_path.open("r", encoding="utf-8") as stream:
                query = json.load(stream)
            with guidance_path.open("r", encoding="utf-8") as stream:
                guidance = json.load(stream)
            samples = pd.read_csv(samples_path)
            report, _ = raw_constraint_report(samples, query)
            modality = raw_modality_constraint_report(samples, query)
            _, real_mask = raw_constraint_report(real, query)
            conditional_real = real.loc[real_mask].reset_index(drop=True)
            reference_path = reference_dir / f"{query['query_id']}.csv"
            if not reference_path.is_file():
                conditional_real.to_csv(reference_path, index=False)
            density = density_metrics(reference_path, samples, info)
            cap = float(guidance["max_correction"])
            rows.append(
                {
                    "max_correction": cap,
                    "query_id": query["query_id"],
                    "target_band": float(query["target_band"]),
                    "overall_clip_rate": float(guidance["overall_clip_rate"]),
                    "full_query_miss_rate": 1.0 - float(report["joint_hit_rate"]),
                    "numeric_joint_miss_rate": modality["numeric"]["joint_miss_rate"],
                    "categorical_joint_miss_rate": modality["categorical"]["joint_miss_rate"],
                    "numeric_mean_column_miss_rate": modality["numeric"]["mean_per_constraint_miss_rate"],
                    "categorical_mean_column_miss_rate": modality["categorical"]["mean_per_constraint_miss_rate"],
                    "shape": density["density/Shape"],
                    "trend": density["density/Trend"],
                    "overall": density["density/Overall"],
                }
            )
            predicate_by_name = {
                predicate["col"]: predicate for predicate in query["predicates"]
            }
            miss_by_name = {
                column["name"]: 1.0 - float(column["hit_rate"])
                for column in report["per_column"]
            }
            for column in guidance["per_column"]:
                name = column["name"]
                predicate = predicate_by_name[name]
                column_rows.append(
                    {
                        "max_correction": cap,
                        "query_id": query["query_id"],
                        "target_band": float(query["target_band"]),
                        "column": name,
                        "raw_lower": float(predicate["values"][0]),
                        "raw_upper": float(predicate["values"][1]),
                        "miss_rate": miss_by_name[name],
                        "clip_rate": float(column["clip_rate"]),
                        "absolute_q90": float(column["absolute_q90"]),
                        "absolute_q99": float(column["absolute_q99"]),
                        "absolute_max": float(column["absolute_max"]),
                    }
                )
            for time_call in guidance["per_time_call"]:
                time_rows.append(
                    {
                        "max_correction": cap,
                        "query_id": query["query_id"],
                        "target_band": float(query["target_band"]),
                        "call_index": int(time_call["call_index"]),
                        "mean_t": float(time_call["mean_t"]),
                        "clip_rate": float(time_call["clip_rate"]),
                        "mean_absolute_correction": float(
                            time_call["mean_absolute_correction"]
                        ),
                        "absolute_q99": float(time_call["absolute_q99"]),
                    }
                )

    if not rows:
        raise ValueError(f"no diagnostic samples found below {sample_root}")
    per_run = pd.DataFrame(rows).sort_values(["max_correction", "target_band"])
    per_column = pd.DataFrame(column_rows).sort_values(
        ["max_correction", "target_band", "column"]
    )
    per_time = pd.DataFrame(time_rows).sort_values(
        ["max_correction", "target_band", "call_index"]
    )
    metrics = [
        "overall_clip_rate", "full_query_miss_rate", "numeric_joint_miss_rate",
        "categorical_joint_miss_rate", "numeric_mean_column_miss_rate",
        "categorical_mean_column_miss_rate", "shape", "trend", "overall",
    ]
    by_cap = per_run.groupby("max_correction")[metrics].agg(["mean", "std"])
    by_cap.columns = [f"{name}_{stat}" for name, stat in by_cap.columns]
    by_cap = by_cap.reset_index()
    per_run.to_csv(output_dir / "per_query_cap.csv", index=False)
    per_column.to_csv(output_dir / "per_numeric_column.csv", index=False)
    per_time.to_csv(output_dir / "per_time_call.csv", index=False)
    by_cap_time = per_time.groupby(["max_correction", "call_index"])[
        ["mean_t", "clip_rate", "mean_absolute_correction", "absolute_q99"]
    ].mean().reset_index()
    by_cap_time.to_csv(output_dir / "by_cap_time.csv", index=False)
    by_cap.to_csv(output_dir / "by_cap.csv", index=False)

    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    for band, selected in per_run.groupby("target_band"):
        selected = selected.sort_values("max_correction")
        label = f"{100 * band:g}%"
        axes[0].plot(selected["max_correction"], selected["overall_clip_rate"], marker="o", label=label)
        axes[1].plot(selected["max_correction"], selected["numeric_joint_miss_rate"], marker="o", label=label)
        axes[2].plot(selected["max_correction"], selected["shape"], marker="o", label=label)
    axes[0].set_title("Raw corrections exceeding cap")
    axes[1].set_title("Numerical joint miss rate")
    axes[2].set_title("Shape score")
    for axis in axes:
        axis.set_xlabel("Numerical correction cap")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
    axes[0].legend(title="Selectivity")
    figure.tight_layout()
    figure.savefig(output_dir / "numerical_clip_diagnostic.png", dpi=180)
    plt.close(figure)
    print(by_cap.to_string(index=False))
    print(f"Saved numerical clipping diagnostic to {output_dir}")


if __name__ == "__main__":
    main()
