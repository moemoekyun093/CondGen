"""Render query-suite plots from an already-computed grouped CSV."""

import argparse
from pathlib import Path

import pandas as pd

from tabdiff.query_suite_plots import make_query_suite_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-by", choices=("target_band", "arity"), default="target_band")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    filename = "by_selectivity_band.csv" if args.group_by == "target_band" else "by_arity.csv"
    grouped_path = output_dir / filename
    if not grouped_path.is_file():
        raise FileNotFoundError(grouped_path)
    make_query_suite_plots(pd.read_csv(grouped_path), output_dir, args.group_by)
    print(f"Saved query-suite plots to {output_dir}")


if __name__ == "__main__":
    main()
