#!/usr/bin/env python
"""Sample a frozen TabbyFlow model under structured interval/set queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from tabdiff.baselines.tabbyflow_conditional import (
    ConditionalTabbyFlowVelocity,
    decode_state,
    encode_query,
    load_frozen_model,
    load_schema,
    resolve_transform,
    sample_conditioned_state,
)
from tabdiff.doob_h_evaluation import raw_constraint_report
from tabdiff.query_split import load_query_split
from tabdiff.query_suite_samples import replicate_sample_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--transform-file", default=None)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--query-split-manifest", default=None)
    parser.add_argument("--query-split", choices=("train", "test"), default="test")
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument("--data-dir", default="data/shoppers")
    parser.add_argument("--info-file", default="data/shoppers/info.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed-bases", default="10000")
    parser.add_argument("--bundle-index", type=int, default=0)
    parser.add_argument("--bundle-count", type=int, default=1)
    parser.add_argument("--solver", choices=("dopri5", "euler", "heun"), default="dopri5")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_queries(args: argparse.Namespace) -> list[tuple[Path, dict]]:
    requested = set(args.query_id) if args.query_id else None
    if args.query_split_manifest:
        split_ids = set(load_query_split(args.query_split_manifest, args.query_split))
        requested = split_ids if requested is None else requested & split_ids
    queries = []
    for path in sorted(Path(args.query_dir).glob("q*.json")):
        with path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if not query.get("accepted", True):
            continue
        if requested is not None and query["query_id"] not in requested:
            continue
        queries.append((path, query))
    if not queries:
        raise ValueError("no accepted queries selected")
    return queries


def main() -> None:
    args = parse_args()
    if args.bundle_count <= 0 or not 0 <= args.bundle_index < args.bundle_count:
        raise ValueError("bundle index/count are inconsistent")
    seeds = [int(value) for value in args.seed_bases.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seed-bases selected no seeds")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading frozen TabbyFlow once from {args.run_dir}", flush=True)
    model, config, checkpoint = load_frozen_model(
        args.run_dir, checkpoint=args.checkpoint, device=args.device
    )
    transform, transform_path = resolve_transform(args.run_dir, args.transform_file)
    schema = load_schema(args.data_dir, args.info_file)
    queries = load_queries(args)
    units = [(seed_index, query_index) for seed_index in range(len(seeds)) for query_index in range(len(queries))]
    assigned = units[args.bundle_index :: args.bundle_count]
    print(
        f"Checkpoint={checkpoint}; transform={transform_path}; solver={args.solver}; "
        f"queries={len(queries)}; seeds={seeds}; bundle={args.bundle_index + 1}/"
        f"{args.bundle_count}; assigned={len(assigned)}",
        flush=True,
    )

    sampled = reused = 0
    for seed_index, query_index in assigned:
        _, query = queries[query_index]
        query_id = str(query["query_id"])
        seed_base = seeds[seed_index]
        output = replicate_sample_path(output_dir, query_id, seed_base, seed_index)
        if output.is_file():
            print(f"Reuse {output}", flush=True)
            reused += 1
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        sample_seed = seed_base + query_index
        encoded = encode_query(
            query,
            schema,
            transform,
            d_numerical=int(config["d_num"]),
            categories=config["categories"],
        )
        velocity = ConditionalTabbyFlowVelocity(model, encoded).to(args.device).eval()
        state = sample_conditioned_state(
            velocity,
            args.num_samples,
            batch_size=args.batch_size,
            seed=sample_seed,
            solver=args.solver,
            steps=args.steps,
        )
        frame = decode_state(state, schema, transform, config["categories"])
        report, _ = raw_constraint_report(frame, query)
        frame.to_csv(output, index=False)
        with output.with_suffix(".query.json").open("w", encoding="utf-8") as stream:
            json.dump(query, stream, indent=2)
        diagnostic = {
            "method": "conditional_tabbyflow_analytic_endpoint",
            "checkpoint": str(checkpoint.resolve()),
            "transform": str(transform_path.resolve()),
            "solver": args.solver,
            "steps": args.steps if args.solver != "dopri5" else None,
            "sample_seed": sample_seed,
            **report,
        }
        with output.with_suffix(".constraints.json").open("w", encoding="utf-8") as stream:
            json.dump(diagnostic, stream, indent=2)
        print(
            f"Saved {output}: hit={100.0 * report['joint_hit_rate']:.2f}%",
            flush=True,
        )
        sampled += 1
    print(f"Bundle complete: sampled={sampled}, reused={reused}", flush=True)


if __name__ == "__main__":
    main()
