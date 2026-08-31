"""Shared paths for independently seeded query-suite samples."""

from __future__ import annotations

from pathlib import Path


def replicate_sample_path(
    method_directory: str | Path,
    query_id: str,
    seed_base: int,
    seed_index: int,
) -> Path:
    """Return one replicate path while preserving the legacy first-seed layout."""
    method_directory = Path(method_directory)
    if seed_index == 0:
        return method_directory / f"{query_id}.csv"
    return method_directory / f"seed_{int(seed_base)}" / f"{query_id}.csv"
