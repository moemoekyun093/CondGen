"""Reuse completed samples whose sidecar describes the exact target query."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--source-glob", action="append", default=[])
    return parser.parse_args()


def number(value: Any) -> float | None:
    return None if value is None else round(float(value), 12)


def signature(specification: dict) -> tuple:
    """Canonicalize legacy column boxes and structured predicates alike."""
    entries = []
    if "predicates" in specification:
        for predicate in specification["predicates"]:
            modality = str(predicate["modality"])
            name = str(predicate["col"])
            operation = str(predicate["op"])
            values = predicate.get("values", [])
            if modality == "numeric" and operation == "between":
                payload = tuple(number(value) for value in values)
            elif modality == "categorical" and operation == "in":
                payload = tuple(sorted(str(value) for value in values))
            else:
                payload = tuple(str(value) for value in values)
            entries.append((modality, name, operation, payload))
    elif "columns" in specification:
        for column in specification["columns"]:
            entries.append(
                (
                    "numeric",
                    str(column["name"]),
                    "between",
                    (number(column.get("raw_lower")), number(column.get("raw_upper"))),
                )
            )
    else:
        raise ValueError("query sidecar has neither predicates nor columns")
    return tuple(sorted(entries))


def completed_sample(path: Path) -> bool:
    return path.is_file() and path.with_suffix(".constraints.json").is_file()


def link(source: Path, target: Path) -> None:
    target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))


def main() -> None:
    args = parse_args()
    query_dir = Path(args.query_dir)
    target_dir = Path(args.target_dir)
    if not query_dir.is_dir():
        raise FileNotFoundError(query_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Include already completed files in the destination. This lets a rerun
    # reuse one sample for another permutation having the same active subset.
    source_paths = set(target_dir.glob("*.csv"))
    for pattern in args.source_glob:
        source_paths.update(Path(value) for value in glob.glob(pattern))

    by_signature: dict[tuple, Path] = {}
    for sample in sorted(source_paths):
        sidecar = sample.with_suffix(".query.json")
        if not completed_sample(sample) or not sidecar.is_file():
            continue
        try:
            with sidecar.open("r", encoding="utf-8") as stream:
                query_signature = signature(json.load(stream))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        by_signature.setdefault(query_signature, sample)

    reused = 0
    already_complete = 0
    for query_path in sorted(query_dir.glob("qf_*.json")):
        sample_target = target_dir / f"{query_path.stem}.csv"
        if completed_sample(sample_target):
            already_complete += 1
            continue
        with query_path.open("r", encoding="utf-8") as stream:
            query_signature = signature(json.load(stream))
        source = by_signature.get(query_signature)
        if source is None:
            continue
        source_constraints = source.with_suffix(".constraints.json")
        if sample_target.exists() or sample_target.is_symlink():
            # Preserve incomplete user files; the sampler can diagnose them.
            continue
        link(source, sample_target)
        link(source_constraints, sample_target.with_suffix(".constraints.json"))
        link(query_path, sample_target.with_suffix(".query.json"))
        reused += 1

    print(
        f"Sample cache: existing={already_complete}, linked={reused}, "
        f"unresolved={len(list(query_dir.glob('qf_*.json'))) - already_complete - reused}"
    )


if __name__ == "__main__":
    main()
