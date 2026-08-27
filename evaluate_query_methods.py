"""Compare Doob and HARPOON against one identical conditional-real table."""

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
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--doob-samples", required=True)
    parser.add_argument("--harpoon-samples", required=True)
    parser.add_argument("--real-data", default=None)
    parser.add_argument("--info-file", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def evaluate_density(reference_path: Path, samples: pd.DataFrame, info: dict) -> dict:
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


def main() -> None:
    args = parse_args()
    query_path = Path(args.query_file)
    real_path = Path(args.real_data or f"synthetic/{args.dataname}/real.csv")
    info_path = Path(args.info_file or f"data/{args.dataname}/info.json")
    method_paths = {
        "doob": Path(args.doob_samples),
        "harpoon": Path(args.harpoon_samples),
    }
    for path in (query_path, real_path, info_path, *method_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    with open(query_path, "r", encoding="utf-8") as stream:
        query = json.load(stream)
    with open(info_path, "r", encoding="utf-8") as stream:
        info = json.load(stream)
    real = pd.read_csv(real_path)
    real_report, real_mask = raw_constraint_report(real, query)
    conditional_real = real.loc[real_mask].reset_index(drop=True)
    if len(conditional_real) < 2:
        raise ValueError("fewer than two real rows satisfy the full query")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditional_reference_path = output_dir / "real_conditional_reference.csv"
    conditional_real.to_csv(conditional_reference_path, index=False)

    rows = []
    results = {
        "dataname": args.dataname,
        "query_id": query.get("query_id"),
        "query_file": str(query_path),
        "real_rows": len(real),
        "conditional_real_rows": len(conditional_real),
        "conditional_real_rate": real_report["joint_hit_rate"],
        "methods": {},
    }
    for method, samples_path in method_paths.items():
        samples = pd.read_csv(samples_path)
        if list(samples.columns) != list(real.columns):
            raise ValueError(
                f"{method} columns differ from the real table or are out of order"
            )
        constraint_report, _ = raw_constraint_report(samples, query)
        density = evaluate_density(conditional_reference_path, samples, info)
        method_result = {
            "samples": str(samples_path),
            "num_rows": len(samples),
            "raw_full_query_hit_rate": constraint_report["joint_hit_rate"],
            "constraint_violation_rate": 1.0 - constraint_report["joint_hit_rate"],
            **density,
        }
        results["methods"][method] = method_result
        rows.append({"method": method, **method_result})

    with open(output_dir / "comparison.json", "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "comparison.csv", index=False)
    print(
        f"Shared conditional real reference: {len(conditional_real)}/{len(real)} "
        f"({real_report['joint_hit_rate']:.2%})"
    )
    print(
        table[
            [
                "method",
                "raw_full_query_hit_rate",
                "constraint_violation_rate",
                "density/Shape",
                "density/Trend",
                "density/Overall",
            ]
        ].to_string(index=False)
    )
    print(f"Saved paired evaluation to {output_dir}")


if __name__ == "__main__":
    main()
