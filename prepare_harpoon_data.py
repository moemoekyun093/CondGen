"""Expose any processed TabDiff table to the vendored HARPOON code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--harpoon-root", default="baselines/harpoon")
    parser.add_argument("--train-data")
    parser.add_argument("--test-data")
    parser.add_argument("--info-file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path("data") / args.dataname
    train_path = Path(args.train_data or source_dir / "train.csv")
    test_path = Path(args.test_data or source_dir / "test.csv")
    info_path = Path(args.info_file or source_dir / "info.json")
    for path in (train_path, test_path, info_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if list(train.columns) != list(test.columns):
        raise ValueError("TabDiff train/test columns do not match")
    with open(info_path, "r", encoding="utf-8") as stream:
        info = json.load(stream)

    numerical = list(info["num_col_idx"])
    categorical = list(info["cat_col_idx"])
    target = list(info["target_col_idx"])
    if info["task_type"] == "regression":
        numerical = target + numerical
    else:
        categorical = target + categorical
    modeled = numerical + categorical
    expected = list(range(len(train.columns)))
    if sorted(modeled) != expected:
        raise ValueError(
            "HARPOON adapter expects numerical, categorical, and target indices "
            "to cover the full table exactly once"
        )

    harpoon_root = Path(args.harpoon_root)
    destination = harpoon_root / "datasets" / args.dataname
    destination.mkdir(parents=True, exist_ok=True)
    train.to_csv(destination / "train.csv", index=False)
    test.to_csv(destination / "test.csv", index=False)
    pd.concat([train, test], ignore_index=True).to_csv(
        destination / "data.csv",
        index=False,
    )

    harpoon_info = {
        "name": args.dataname,
        "task_type": info["task_type"],
        "num_col_idx": numerical,
        "cat_col_idx": categorical,
        "target_col_idx": target,
        "column_names": list(train.columns),
        "source": "TabDiff processed train/test CSVs",
    }
    harpoon_info_dir = harpoon_root / "datasets" / "Info"
    harpoon_info_dir.mkdir(parents=True, exist_ok=True)
    with open(
        harpoon_info_dir / f"{args.dataname}.json",
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(harpoon_info, stream, indent=2)

    print(f"Prepared HARPOON dataset at {destination}")
    print(f"Rows: train={len(train)}, test={len(test)}")
    print(f"Numerical columns: {len(numerical)}")
    print(f"Categorical columns including target: {len(categorical)}")


if __name__ == "__main__":
    main()
