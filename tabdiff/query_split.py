"""Utilities for reproducible train/test splits over query definitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


SPLIT_NAMES = ("train", "test")


def query_id_digest(query_ids) -> str:
    payload = "\n".join(sorted(str(query_id) for query_id in query_ids)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_query_fingerprint(query: dict) -> str:
    """Hash the constraint semantics while ignoring query ids and metadata."""
    predicates = []
    for predicate in query["predicates"]:
        values = predicate["values"]
        if predicate["modality"] == "categorical":
            values = sorted(str(value) for value in values)
        else:
            values = [float(value) for value in values]
        predicates.append(
            {
                "col": str(predicate["col"]),
                "modality": str(predicate["modality"]),
                "op": str(predicate["op"]),
                "values": values,
            }
        )
    payload = json.dumps(
        sorted(predicates, key=lambda value: value["col"]),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_query_split(path: str | Path, split: str) -> list[str]:
    """Load one disjoint query-id partition and validate the manifest."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"query split must be one of {SPLIT_NAMES}, got {split!r}")
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    partitions = manifest.get("partitions", {})
    missing = [name for name in SPLIT_NAMES if name not in partitions]
    if missing:
        raise ValueError(f"query split manifest is missing partitions: {missing}")
    train_ids = [str(value) for value in partitions["train"]]
    test_ids = [str(value) for value in partitions["test"]]
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise ValueError("query split manifest contains duplicate query ids")
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"query split partitions overlap: {sorted(overlap)[:5]}")
    expected_digest = manifest.get("source_query_ids_sha256")
    if expected_digest is not None:
        observed_digest = query_id_digest([*train_ids, *test_ids])
        if observed_digest != expected_digest:
            raise ValueError("query split manifest source digest does not match partitions")
    selected = train_ids if split == "train" else test_ids
    if not selected:
        raise ValueError(f"query split partition {split!r} is empty")
    return selected
