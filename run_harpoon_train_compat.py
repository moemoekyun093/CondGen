"""Run upstream HARPOON training with a PyTorch scheduler compatibility shim."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: run_harpoon_train_compat.py UPSTREAM_SCRIPT [arguments ...]"
        )
    upstream_script = Path(sys.argv[1]).resolve()
    if not upstream_script.is_file():
        raise FileNotFoundError(upstream_script)

    original_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

    def compatible_scheduler(*args, **kwargs):
        kwargs.pop("verbose", None)
        return original_scheduler(*args, **kwargs)

    torch.optim.lr_scheduler.ReduceLROnPlateau = compatible_scheduler
    sys.argv = [str(upstream_script), *sys.argv[2:]]
    runpy.run_path(str(upstream_script), run_name="__main__")


if __name__ == "__main__":
    main()
