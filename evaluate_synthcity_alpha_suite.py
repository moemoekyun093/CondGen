"""Evaluate official SynthCity Alpha Precision/Beta Recall for a query suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader

from tabdiff.doob_h_evaluation import raw_constraint_report
from tabdiff.query_split import load_query_split
from tabdiff.query_suite_samples import replicate_sample_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--real-data", required=True)
    parser.add_argument("--info-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-split-manifest", default=None)
    parser.add_argument("--query-split", choices=("train", "test"), default=None)
    parser.add_argument(
        "--sample-seed-base",
        action="append",
        type=int,
        default=[],
        help="Repeat for sampling replicates. The first seed uses legacy direct CSVs.",
    )
    return parser.parse_args()


def parse_methods(values: list[str]) -> dict[str, Path]:
    methods = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid method specification: {value}")
        label, directory = value.split("=", maxsplit=1)
        if not label or label in methods:
            raise ValueError(f"empty or duplicate method label: {label!r}")
        methods[label] = Path(directory)
    return methods


def synthcity_features(
    schema_real: pd.DataFrame,
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    info: dict,
):
    """Match TabDiff preprocessing with categories fitted on the full schema."""
    schema_real = schema_real.copy()
    real = real.copy()
    synthetic = synthetic.copy()
    schema_real.columns = range(len(schema_real.columns))
    real.columns = range(len(real.columns))
    synthetic.columns = range(len(synthetic.columns))
    numerical = list(info["num_col_idx"])
    categorical = list(info["cat_col_idx"])
    target = list(info["target_col_idx"])
    if info["task_type"] == "regression":
        numerical += target
    else:
        categorical += target

    real_num = real[numerical].to_numpy()
    synthetic_num = synthetic[numerical].to_numpy()
    if categorical:
        # Invalid/previously unseen generated labels must not abort a 5-seed
        # suite evaluation. They map to the all-zero block and are therefore
        # still visible to the downstream distributional metric.
        encoder = OneHotEncoder(handle_unknown="ignore")
        schema_cat = schema_real[categorical].to_numpy().astype(str)
        real_cat = real[categorical].to_numpy().astype(str)
        synthetic_cat = synthetic[categorical].to_numpy().astype(str)
        # The conditional subset can omit valid categories. Fit the vocabulary
        # on the full real table, but compute Alpha/Beta using only the filtered
        # conditional rows and generated rows below.
        encoder.fit(schema_cat)
        real_cat = encoder.transform(real_cat)
        synthetic_cat = encoder.transform(synthetic_cat)
        if hasattr(real_cat, "toarray"):
            real_cat = real_cat.toarray()
            synthetic_cat = synthetic_cat.toarray()
    else:
        real_cat = np.zeros((len(real), 0))
        synthetic_cat = np.zeros((len(synthetic), 0))
    return (
        np.concatenate((real_num, real_cat), axis=1).astype(float),
        np.concatenate((synthetic_num, synthetic_cat), axis=1).astype(float),
    )


def evaluate_pair(
    schema_real: pd.DataFrame,
    conditional_real: pd.DataFrame,
    synthetic: pd.DataFrame,
    info: dict,
    seed: int,
) -> tuple[float, float]:
    real_features, synthetic_features = synthcity_features(
        schema_real, conditional_real, synthetic, info
    )
    if not np.isfinite(real_features).all() or not np.isfinite(
        synthetic_features
    ).all():
        raise ValueError("SynthCity AlphaPrecision inputs contain non-finite values")
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
    result = eval_statistical.AlphaPrecision().evaluate(
        GenericDataLoader(pd.DataFrame(real_features)),
        GenericDataLoader(pd.DataFrame(synthetic_features)),
    )
    return (
        float(result["delta_precision_alpha_naive"]),
        float(result["delta_coverage_beta_naive"]),
    )


def main() -> None:
    args = parse_args()
    if (args.query_split_manifest is None) != (args.query_split is None):
        raise ValueError(
            "--query-split-manifest and --query-split must be supplied together"
        )
    query_dir = Path(args.query_dir)
    methods = parse_methods(args.method)
    real = pd.read_csv(args.real_data)
    with open(args.info_file, "r", encoding="utf-8") as stream:
        info = json.load(stream)
    selected_ids = None
    if args.query_split_manifest is not None:
        selected_ids = set(load_query_split(args.query_split_manifest, args.query_split))
    query_paths = sorted(query_dir.glob("q*.json"))
    if not query_paths:
        raise ValueError(f"no structured queries found in {query_dir}")
    seed_bases = args.sample_seed_base or [0]

    evaluator_rows = []
    for query_index, query_path in enumerate(query_paths):
        with query_path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if not query.get("accepted", True):
            continue
        query_id = query["query_id"]
        if selected_ids is not None and query_id not in selected_ids:
            continue
        _, real_mask = raw_constraint_report(real, query)
        conditional_real = real.loc[real_mask].reset_index(drop=True)
        for method, sample_dir in methods.items():
            for seed_index, seed_base in enumerate(seed_bases):
                sample_path = replicate_sample_path(
                    sample_dir, query_id, seed_base, seed_index
                )
                if not sample_path.is_file():
                    raise FileNotFoundError(sample_path)
                samples = pd.read_csv(sample_path)
                alpha_precision, beta_recall = evaluate_pair(
                    real,
                    conditional_real,
                    samples,
                    info,
                    args.seed + query_index + 100000 * seed_index,
                )
                evaluator_rows.append(
                    {
                        "method": method,
                        "query_id": query_id,
                        "seed_base": seed_base,
                        "alpha_precision": alpha_precision,
                        "beta_recall": beta_recall,
                        "backend": "synthcity.metrics.eval_statistical.AlphaPrecision",
                    }
                )
                print(
                    f"{method}/{query_id}/seed_{seed_base}: "
                    f"alpha={alpha_precision:.6f} beta={beta_recall:.6f}"
                )

    if selected_ids is not None:
        found_ids = {row["query_id"] for row in evaluator_rows}
        if found_ids != selected_ids:
            missing = sorted(selected_ids - found_ids)
            raise ValueError(f"query split files missing for ids: {missing[:5]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(evaluator_rows).to_csv(output, index=False)
    print(f"Saved official SynthCity metrics to {output}")


if __name__ == "__main__":
    main()
