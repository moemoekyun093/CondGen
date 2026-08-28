"""Compare guided HARPOON with exact shared unconditional rows across arity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_doob_query_suite import aggregate, density_metrics, make_plots
from tabdiff.doob_h_evaluation import (
    raw_constraint_report,
    raw_modality_constraint_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--guided-samples", required=True)
    parser.add_argument("--unconditional-source-samples", required=True)
    parser.add_argument("--real-data", default="synthetic/shoppers/real.csv")
    parser.add_argument("--info-file", default="data/shoppers/info.json")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_dir = Path(args.query_dir)
    guided_dir = Path(args.guided_samples)
    unconditional_dir = Path(args.unconditional_source_samples)
    real_path = Path(args.real_data)
    info_path = Path(args.info_file)
    for path in (query_dir, guided_dir, unconditional_dir, real_path, info_path):
        if not path.exists():
            raise FileNotFoundError(path)
    real = pd.read_csv(real_path)
    with info_path.open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    output_dir = Path(args.output_dir)
    reference_dir = output_dir / "conditional_real_references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    queries = []
    for path in sorted(query_dir.glob("qf_*.json")):
        with path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if query.get("accepted", True):
            queries.append(query)
    if not queries:
        raise ValueError("no derived arity queries found")

    unconditional_cache = {}
    rows = []
    for query in queries:
        query_id = query["query_id"]
        source_id = query["source_query_id"]
        arity = int(query["arity"])
        guided_path = guided_dir / f"{query_id}.csv"
        unconditional_path = unconditional_dir / f"{source_id}.csv"
        for path in (guided_path, unconditional_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        guided = pd.read_csv(guided_path)
        if source_id not in unconditional_cache:
            unconditional_cache[source_id] = pd.read_csv(unconditional_path)
        unconditional = unconditional_cache[source_id]
        for frame, path in ((guided, guided_path), (unconditional, unconditional_path)):
            if list(frame.columns) != list(real.columns):
                raise ValueError(f"column mismatch for {path}")

        real_report, real_mask = raw_constraint_report(real, query)
        conditional_real = real.loc[real_mask].reset_index(drop=True)
        if len(conditional_real) < 2:
            raise ValueError(f"fewer than two real rows satisfy {query_id}")
        reference_path = reference_dir / f"{query_id}.csv"
        conditional_real.to_csv(reference_path, index=False)

        for method, samples, samples_path in (
            ("harpoon_guided_eta02", guided, guided_path),
            ("harpoon_unconditional_shared", unconditional, unconditional_path),
        ):
            constraint, _ = raw_constraint_report(samples, query)
            modality = raw_modality_constraint_report(samples, query)
            density = density_metrics(reference_path, samples, info)
            hit_rate = float(constraint["joint_hit_rate"])
            rows.append(
                {
                    "method": method,
                    "query_id": query_id,
                    "source_query_id": source_id,
                    "arity": arity,
                    "num_numeric_constraints": modality["numeric"]["num_constraints"],
                    "num_categorical_constraints": modality["categorical"]["num_constraints"],
                    "generated_rows": len(samples),
                    "conditional_real_rows": len(conditional_real),
                    "conditional_real_rate": real_report["joint_hit_rate"],
                    "raw_joint_hit_rate": hit_rate,
                    "violation_rate": 1.0 - hit_rate,
                    "numeric_joint_miss_rate": modality["numeric"]["joint_miss_rate"],
                    "categorical_joint_miss_rate": modality["categorical"]["joint_miss_rate"],
                    "numeric_mean_column_miss_rate": modality["numeric"]["mean_per_constraint_miss_rate"],
                    "categorical_mean_column_miss_rate": modality["categorical"]["mean_per_constraint_miss_rate"],
                    "shape": density["density/Shape"],
                    "trend": density["density/Trend"],
                    "overall": density["density/Overall"],
                    "samples": str(samples_path),
                }
            )

    per_query = pd.DataFrame(rows)
    unconditional_rows = per_query[
        per_query["method"] == "harpoon_unconditional_shared"
    ]
    monotonicity = []
    for source_id, selected in unconditional_rows.groupby("source_query_id"):
        selected = selected.sort_values("arity")
        rates = selected["violation_rate"].to_list()
        is_monotone = all(left <= right + 1e-12 for left, right in zip(rates, rates[1:]))
        monotonicity.append(
            {
                "source_query_id": source_id,
                "arities": selected["arity"].astype(int).to_list(),
                "violation_rates": rates,
                "nondecreasing": is_monotone,
            }
        )
    if not all(item["nondecreasing"] for item in monotonicity):
        failing = [item["source_query_id"] for item in monotonicity if not item["nondecreasing"]]
        raise RuntimeError(
            "shared unconditional samples violate nested-query monotonicity for "
            f"{failing}; inspect query construction"
        )

    by_arity = aggregate(per_query, ["method", "arity"])
    per_query.to_csv(output_dir / "per_query.csv", index=False)
    by_arity.to_csv(output_dir / "by_arity.csv", index=False)
    with (output_dir / "monotonicity.json").open("w", encoding="utf-8") as stream:
        json.dump(monotonicity, stream, indent=2)
    make_plots(by_arity, output_dir, "arity")
    print("Shared-unconditional monotonicity passed for every source query")
    print(by_arity.to_string(index=False))
    print(f"Saved HARPOON arity ablation to {output_dir}")


if __name__ == "__main__":
    main()
