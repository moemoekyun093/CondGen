"""Small compatibility loader for the pinned upstream HARPOON checkout."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path


def load_harpoon(harpoon_root: Path):
    """Return upstream dataset/model/diffusion objects without editing upstream."""
    root = harpoon_root.resolve()
    sys.path.insert(0, str(root))

    import dataset as harpoon_dataset
    from model import MLPDiffusion
    from utils import calc_diffusion_hyperparams

    encoder = harpoon_dataset.OneHotEncoder
    if "sparse_output" not in inspect.signature(encoder).parameters:
        # HARPOON uses the new sklearn name; TabDiff environments may still use
        # the pre-1.2 ``sparse`` spelling.
        def compatible_one_hot_encoder(*, sparse_output=False, **kwargs):
            return encoder(sparse=sparse_output, **kwargs)

        harpoon_dataset.OneHotEncoder = compatible_one_hot_encoder

    return harpoon_dataset.Preprocessor, MLPDiffusion, calc_diffusion_hyperparams
