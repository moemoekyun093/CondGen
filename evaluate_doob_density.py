"""Evaluate conditional samples with TabDiff's Shape and Trend metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from tabdiff.doob_h_evaluation import raw_constraint_report
from tabdiff.metrics import TabMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", default="shoppers")
    parser.add_argument("--samples", required=True)
    parser.add_argument(
        "--unconditional-samples",
        required=True,
        help="Existing unconditional samples from the matching frozen base model",
    )
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--real-data", default=None)
    parser.add_argument("--info-file", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def density_evaluation(real_path: Path, samples: pd.DataFrame, info: dict):
    evaluator = TabMetrics(
        str(real_path),
        str(real_path),
        None,
        info,
        torch.device("cpu"),
        metric_list=["density"],
        include_density_diagnostic=False,
    )
    return evaluator.evaluate(samples.copy())


def save_extras(output_dir: Path, prefix: str, extras: dict) -> None:
    for name, value in extras.items():
        path = output_dir / f"{prefix}_{name}"
        if isinstance(value, pd.DataFrame):
            value.to_csv(path.with_suffix(".csv"), index=False)
        elif isinstance(value, dict):
            with open(path.with_suffix(".json"), "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2)
        else:
            raise TypeError(f"unsupported metric detail type for {name}: {type(value)}")


def float_metrics(metrics: dict) -> dict:
    return {name: float(value) for name, value in metrics.items()}


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples)
    unconditional_samples_path = Path(args.unconditional_samples)
    real_path = Path(args.real_data or f"synthetic/{args.dataname}/real.csv")
    info_path = Path(args.info_file or f"data/{args.dataname}/info.json")
    output_dir = Path(
        args.output_dir
        or samples_path.parent / f"{samples_path.stem}_evaluation"
    )

    for path in (
        samples_path,
        unconditional_samples_path,
        real_path,
        info_path,
        Path(args.query_file),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    samples = pd.read_csv(samples_path)
    unconditional_samples = pd.read_csv(unconditional_samples_path)
    real = pd.read_csv(real_path)
    with open(info_path, "r", encoding="utf-8") as stream:
        info = json.load(stream)
    with open(args.query_file, "r", encoding="utf-8") as stream:
        query = json.load(stream)

    if list(samples.columns) != list(real.columns):
        raise ValueError("sample and real-table columns differ or are in a different order")
    if list(unconditional_samples.columns) != list(real.columns):
        raise ValueError(
            "unconditional-sample and real-table columns differ or are in a different order"
        )

    generated_constraint_report, _ = raw_constraint_report(samples, query)
    real_constraint_report, real_condition_mask = raw_constraint_report(real, query)
    conditional_real = real.loc[real_condition_mask].reset_index(drop=True)
    if len(conditional_real) < 2:
        raise ValueError("fewer than two real rows satisfy the raw constraint")

    output_dir.mkdir(parents=True, exist_ok=True)
    conditional_real_path = output_dir / "real_conditional_reference.csv"
    conditional_real.to_csv(conditional_real_path, index=False)

    conditional_metrics, conditional_extras = density_evaluation(
        conditional_real_path,
        samples,
        info,
    )
    unconditional_generation_metrics, unconditional_generation_extras = density_evaluation(
        conditional_real_path,
        unconditional_samples,
        info,
    )

    results = {
        "dataname": args.dataname,
        "constraint_id": query.get("constraint_id"),
        "generated_rows": len(samples),
        "unconditional_generated_rows": len(unconditional_samples),
        "unconditional_samples": str(unconditional_samples_path),
        "real_rows": len(real),
        "real_conditional_rows": len(conditional_real),
        "generated_raw_joint_hit_rate": generated_constraint_report["joint_hit_rate"],
        "real_raw_joint_hit_rate": real_constraint_report["joint_hit_rate"],
        "conditional_generation_vs_conditional_real": float_metrics(
            conditional_metrics
        ),
        "unconditional_generation_vs_conditional_real": float_metrics(
            unconditional_generation_metrics
        ),
    }
    with open(output_dir / "density_results.json", "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    save_extras(output_dir, "conditional_generation", conditional_extras)
    save_extras(
        output_dir,
        "unconditional_generation",
        unconditional_generation_extras,
    )

    print(f"Generated raw joint hit rate: {results['generated_raw_joint_hit_rate']:.2%}")
    print(
        f"Conditional real reference: {len(conditional_real)}/{len(real)} rows "
        f"({results['real_raw_joint_hit_rate']:.2%})"
    )
    print("Conditional generation vs conditional real:")
    print(json.dumps(results["conditional_generation_vs_conditional_real"], indent=2))
    print("Unconditional generation vs the same conditional real:")
    print(json.dumps(results["unconditional_generation_vs_conditional_real"], indent=2))
    print(f"Saved density evaluation to {output_dir}")


if __name__ == "__main__":
    main()
