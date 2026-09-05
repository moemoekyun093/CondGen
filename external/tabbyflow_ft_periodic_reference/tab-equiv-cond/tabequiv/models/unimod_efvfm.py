"""ef-vfm wrapper: UniModMLP + ExpVFM.mixed_loss (fully constructible).

Baseline classes (read-only, via use_baseline("ef-vfm")):
* ``ef_vfm.modules.main_modules.UniModMLP`` — same graph as TabDiff's, but the FFN
  activation defaults to GELU (TabDiff hard-codes ReLU) and categories are passed
  WITHOUT a mask class (``ef_vfm/main.py:154``).
* ``ef_vfm.models.flow_model.ExpVFM``       — mixed_loss = MVG NLL on numericals +
  per-column categorical NLL; one-hots the int-coded categoricals internally.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from tabequiv.paths import use_baseline
from tabequiv.repro import seed_everything

BASELINE = "ef-vfm"

UNIMOD_KW = dict(num_layers=2, d_token=4, n_head=1, factor=32, bias=True,
                 dim_t=1024, use_mlp=True)


SYMMETRIC_DENOISER = "ft_periodic"


def build_unimod_efvfm(d_num: int = 3, categories=(4, 5), seed: int = 0,
                       denoiser: str = "original", ft_kw: dict | None = None) -> torch.nn.Module:
    """ExpVFM (the OFFICIAL TabbyFlow) at seeded random init, owning as ``_vf_fn`` either its
    UniModMLP (``denoiser="original"``) or AA_code's symmetric ft_periodic core
    (``denoiser="ft_periodic"``).  ExpVFM only needs ``_vf_fn(x_num, x_cat, t) -> (mu, logits)``
    plus ``.categories`` / ``.d_numerical`` (flow_model.py:63,73; Velocity:165) -- the core has
    all three, categories WITHOUT a mask class, t in [0, 1]: a drop-in, as in TabDiff."""
    categories = [int(c) for c in categories]
    with use_baseline("ef-vfm"):
        from ef_vfm.models.flow_model import ExpVFM
        from ef_vfm.modules.main_modules import UniModMLP
        seed_everything(seed)
        if denoiser == "original":
            net = UniModMLP(d_numerical=d_num, categories=categories, **UNIMOD_KW)
        elif denoiser == "ft_periodic":
            from tabequiv.models.ft_periodic import FT_PERIODIC_KW, UniModMLPFTPeriodic
            kw = dict(FT_PERIODIC_KW); kw.update(ft_kw or {})
            net = UniModMLPFTPeriodic(d_numerical=d_num, categories=categories, **kw)
        else:
            raise ValueError(f"unknown denoiser {denoiser!r}")
        model = ExpVFM(num_classes=np.array(categories),
                       num_numerical_features=d_num, vf_fn=net, device="cpu")
    return model
build = build_unimod_efvfm


def build_symmetric(d_num: int = 3, categories=(4, 5), seed: int = 0,
                    ft_kw: dict | None = None) -> torch.nn.Module:
    return build_unimod_efvfm(d_num, categories, seed=seed, denoiser=SYMMETRIC_DENOISER, ft_kw=ft_kw)


def make_batch(d_num: int = 3, categories=(4, 5), batch_size: int = 8,
               seed: int = 1) -> dict:
    categories = [int(c) for c in categories]
    seed_everything(seed)
    x_num = torch.randn(batch_size, d_num)
    x_cat = torch.stack([torch.arange(batch_size) % k for k in categories], dim=1)
    t = torch.rand(batch_size)
    return {"x_num": x_num, "x_cat": x_cat, "t": t}


def forward_fn(model: torch.nn.Module, batch: dict):
    """UniModMLP forward on clean one-hots (no mask class in ef-vfm)."""
    cats = [int(k) for k in model.num_classes]
    x_cat_oh = torch.cat(
        [F.one_hot(batch["x_cat"][:, i], num_classes=k) for i, k in enumerate(cats)],
        dim=1).float()
    x_num_pred, x_cat_pred = model._vf_fn(batch["x_num"], x_cat_oh, batch["t"])
    return x_num_pred, x_cat_pred


def loss_fn(model: torch.nn.Module, batch: dict) -> dict:
    x = torch.cat([batch["x_num"], batch["x_cat"].float()], dim=1)
    d_loss, c_loss = model.mixed_loss(x)
    return {"d_loss": d_loss, "c_loss": c_loss, "total": d_loss + c_loss}
