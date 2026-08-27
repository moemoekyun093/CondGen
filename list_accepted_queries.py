"""Print accepted structured-query paths in deterministic order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_dir")
    args = parser.parse_args()
    query_dir = Path(args.query_dir)
    if not query_dir.is_dir():
        raise FileNotFoundError(query_dir)
    found = 0
    for path in sorted(query_dir.glob("qf_*.json")):
        with path.open("r", encoding="utf-8") as stream:
            query = json.load(stream)
        if query.get("accepted", True):
            print(path)
            found += 1
    if found == 0:
        raise ValueError(f"no accepted queries found in {query_dir}")


if __name__ == "__main__":
    main()
