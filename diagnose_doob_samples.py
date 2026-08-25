"""Re-evaluate an existing conditional CSV without generating new samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tabdiff.doob_h_evaluation import raw_constraint_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples_path = Path(args.samples)
    with open(args.query_file, "r", encoding="utf-8") as stream:
        query = json.load(stream)
    frame = pd.read_csv(samples_path)
    report, _ = raw_constraint_report(frame, query)

    output = Path(args.output or samples_path.with_suffix(".raw_diagnostic.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print(f"Existing samples: {samples_path}")
    print(f"Rows satisfying every raw constraint: {report['rows_satisfying']}/{len(frame)}")
    print(f"Raw-space joint hit rate: {report['joint_hit_rate']:.2%}")
    print(f"Saved raw diagnostic to {output}")


if __name__ == "__main__":
    main()
