"""Persistently sample many HARPOON-style TabDiff queries with one model load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sample_harpoon_style_tabdiff import sample_query, save_query_result
from tabdiff.doob_h_runtime import load_doob_runtime, resolve_base_checkpoint
from tabdiff.doob_query_suite import load_structured_query_suite
from tabdiff.query_split import load_query_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", default="shoppers")
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--query-split-manifest", default=None)
    parser.add_argument("--query-split", choices=("train", "test"), default="test")
    parser.add_argument("--test-supported-only", action="store_true")
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-bases", default="10000")
    parser.add_argument("--bundle-index", type=int, required=True)
    parser.add_argument("--bundle-count", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def selected_query_paths(args: argparse.Namespace) -> list[Path]:
    selected_ids = None
    if args.query_split_manifest:
        selected_ids = set(load_query_split(args.query_split_manifest, args.query_split))
    paths = []
    for path in sorted(Path(args.query_dir).glob("q*.json")):
        with path.open("r", encoding="utf-8") as stream:
            specification = json.load(stream)
        if not specification.get("accepted", True):
            continue
        if selected_ids is not None and specification["query_id"] not in selected_ids:
            continue
        if args.test_supported_only and not specification.get("test_supported", False):
            continue
        paths.append(path)
    if not paths:
        raise ValueError("no queries selected")
    if selected_ids is not None and not args.test_supported_only and len(paths) != len(selected_ids):
        raise ValueError(
            f"selected {len(selected_ids)} query ids but found {len(paths)} accepted files"
        )
    return paths


def main() -> None:
    args = parse_args()
    if args.bundle_count <= 0 or not 0 <= args.bundle_index < args.bundle_count:
        raise ValueError("invalid bundle index/count")
    if args.num_samples <= 0 or args.batch_size <= 0 or args.num_timesteps < 2:
        raise ValueError("invalid sampling size or timestep count")
    if args.eta <= 0:
        raise ValueError("eta must be positive")
    seeds = [int(value) for value in args.seed_bases.split(",") if value]
    paths = selected_query_paths(args)
    units = [
        (seed_index, query_index)
        for seed_index in range(len(seeds))
        for query_index in range(len(paths))
    ]
    assigned = units[args.bundle_index :: args.bundle_count]
    output_dir = Path(args.output_dir)
    missing = []
    reused = 0
    for seed_index, query_index in assigned:
        query_id = paths[query_index].stem
        if seed_index == 0:
            output = output_dir / f"{query_id}.csv"
        else:
            output = output_dir / f"seed_{seeds[seed_index]}" / f"{query_id}.csv"
        if output.is_file():
            reused += 1
        else:
            missing.append((seed_index, query_index, output))
    print(
        f"Persistent HARPOON-style worker: assigned={len(assigned)} "
        f"missing={len(missing)} reused={reused}"
    )
    if not missing:
        return

    device = torch.device(args.device)
    base_checkpoint = resolve_base_checkpoint(args.dataname, args.base_ckpt, None)
    print(f"Loading frozen TabDiff once: {base_checkpoint}")
    runtime = load_doob_runtime(args.dataname, base_checkpoint, device)
    runtime.diffusion.num_timesteps = args.num_timesteps
    needed_ids = [paths[query_index].stem for _, query_index, _ in missing]
    queries = load_structured_query_suite(
        args.query_dir,
        runtime,
        query_ids=needed_ids,
    )
    by_id = {query.query_id: query for query in queries}

    sampled = 0
    for seed_index, query_index, output in missing:
        query_id = paths[query_index].stem
        query = by_id[query_id]
        seed = seeds[seed_index] + query_index
        print(f"Sample: query={query_id} seed={seed} -> {output}", flush=True)
        frame = sample_query(
            runtime,
            query,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            eta=args.eta,
            seed=seed,
        )
        hit_rate = save_query_result(
            frame,
            query,
            output,
            eta=args.eta,
            num_timesteps=args.num_timesteps,
            base_checkpoint=base_checkpoint,
        )
        sampled += 1
        print(f"Saved {output}; raw hit={hit_rate:.2%}", flush=True)
    print(f"Bundle complete: sampled={sampled}, reused={reused}")


if __name__ == "__main__":
    main()
