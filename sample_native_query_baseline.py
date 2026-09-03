"""Sample one arbitrary query with a trained native baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tabdiff.baseline_data import load_baseline_table
from tabdiff.baselines.native_query import sample_diffputer_repaint, sample_great
from tabdiff.doob_h_evaluation import raw_constraint_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("diffputer", "great"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataname")
    parser.add_argument("--train-data")
    parser.add_argument("--test-data")
    parser.add_argument("--info-file")
    parser.add_argument("--harpoon-root", default="baselines/harpoon")
    parser.add_argument("--great-root", default="baselines/great")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--great-max-length",
        type=int,
        default=512,
        help="Maximum total prompt-plus-output token length for GReaT",
    )
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = load_baseline_table(
        dataname=args.dataname,
        train_data=args.train_data,
        test_data=args.test_data,
        info_file=args.info_file,
    )
    with Path(args.query_file).open("r", encoding="utf-8") as stream:
        query = json.load(stream)
    model_path = Path(args.model_path)
    if args.method == "diffputer":
        checkpoint = model_path / "model.pt" if model_path.is_dir() else model_path
        frame, metadata = sample_diffputer_repaint(
            table=table,
            query=query,
            checkpoint=checkpoint,
            harpoon_root=Path(args.harpoon_root),
            num_rows=args.num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
    else:
        directory = model_path / "model" if (model_path / "model").is_dir() else model_path
        frame, metadata = sample_great(
            table=table,
            query=query,
            model_dir=directory,
            great_root=Path(args.great_root),
            num_rows=args.num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            max_length=args.great_max_length,
        )
    report, _ = raw_constraint_report(frame, query)
    metadata["constraint_report"] = report
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {output}")
    print(f"Raw full-query hit rate: {100 * report['joint_hit_rate']:.2f}%")


if __name__ == "__main__":
    main()
