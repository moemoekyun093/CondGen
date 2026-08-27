"""Shared loading and recovery helpers for fixed-query Doob guidance."""

from __future__ import annotations

import glob
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.modules.main_modules import Model, UniModMLP, UniModMLP_Original, UniModMLP_TabNet
from utils_train import TabDiffDataset


@dataclass
class DoobRuntime:
    dataset: TabDiffDataset
    info: Dict[str, Any]
    config: Dict[str, Any]
    diffusion: UnifiedCtimeDiffusion
    checkpoint_path: str


def frozen_ft_tokenizer(runtime: DoobRuntime):
    """Return the frozen tokenizer used by an FT-periodic base denoiser."""
    model = runtime.diffusion._denoise_fn
    backbone = model.denoise_fn_D
    if hasattr(backbone, "denoise_fn_F"):
        backbone = backbone.denoise_fn_F
    tokenizer = getattr(backbone, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "d_token"):
        raise ValueError(
            "structured query conditioning currently requires an FT-periodic "
            "base denoiser with a reusable tokenizer"
        )
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    return tokenizer


def resolve_base_checkpoint(
    dataname: str,
    checkpoint_path: str | None,
    experiment_name: str = "learnable_schedule",
) -> str:
    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"base checkpoint does not exist: {checkpoint_path}")
        return checkpoint_path

    pattern = f"tabdiff/ckpt/{dataname}/{experiment_name}/best_ema_model*.pt"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"no base checkpoint matched {pattern}; pass --base-ckpt explicitly"
        )
    return matches[0]


def _torch_load(path: str, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def infer_denoiser_type(state_dict: Dict[str, Any]) -> str | None:
    """Infer legacy/current backbone type from checkpoint parameter names."""
    keys = state_dict.keys()
    if any("denoise_fn_F.encoder." in key for key in keys):
        return "original"
    if any("denoise_fn_F.tabnet_steps." in key for key in keys):
        return "tabnet"
    if any("denoise_fn_F.blocks." in key for key in keys):
        return "ft_periodic"
    return None


def load_doob_runtime(
    dataname: str,
    checkpoint_path: str,
    device: torch.device,
    require_numerical_only: bool = False,
) -> DoobRuntime:
    data_dir = f"data/{dataname}"
    info_path = os.path.join(data_dir, "info.json")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(
            f"processed dataset not found at {data_dir}; run process_dataset.py first"
        )
    with open(info_path, "r", encoding="utf-8") as stream:
        info = json.load(stream)

    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.pkl")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"checkpoint config does not exist: {config_path}")
    with open(config_path, "rb") as stream:
        config = pickle.load(stream)

    dataset = TabDiffDataset(
        dataname,
        data_dir,
        info,
        isTrain=True,
        dequant_dist=config["data"]["dequant_dist"],
        int_dequant_factor=config["data"]["int_dequant_factor"],
    )
    d_numerical = dataset.d_numerical
    categories = np.asarray(dataset.categories)
    if require_numerical_only and len(categories) != 0:
        raise ValueError(
            "this first Doob experiment is numerical-only; use a dataset with no "
            "categorical columns (the default is news_nocat)"
        )

    state = _torch_load(checkpoint_path, device)
    model_config = dict(config["unimodmlp_params"])
    model_config["d_numerical"] = d_numerical
    model_config["categories"] = (categories + 1).tolist()
    configured_type = model_config.get("denoiser_type")
    inferred_type = infer_denoiser_type(state["denoise_fn"])
    denoiser_type = inferred_type or configured_type or "ft_periodic"
    if inferred_type is not None and configured_type not in (None, inferred_type):
        print(
            "Warning: checkpoint architecture "
            f"({inferred_type}) overrides config denoiser_type ({configured_type})"
        )
    model_config["denoiser_type"] = denoiser_type
    denoiser_classes = {
        "original": UniModMLP_Original,
        "ft_periodic": UniModMLP,
        "tabnet": UniModMLP_TabNet,
    }
    if denoiser_type not in denoiser_classes:
        raise ValueError(f"unknown denoiser_type in base config: {denoiser_type}")

    print(f"Loading base checkpoint as denoiser_type={denoiser_type}")
    backbone = denoiser_classes[denoiser_type](**model_config)
    denoiser = Model(backbone, **config["diffusion_params"]["edm_params"]).to(device)
    denoiser.load_state_dict(state["denoise_fn"])

    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=denoiser,
        y_only_model=None,
        **config["diffusion_params"],
        device=device,
    ).to(device)
    if "num_schedule" in state:
        diffusion.num_schedule.load_state_dict(state["num_schedule"])
    if "cat_schedule" in state:
        diffusion.cat_schedule.load_state_dict(state["cat_schedule"])

    diffusion.eval()
    for parameter in diffusion.parameters():
        parameter.requires_grad_(False)

    return DoobRuntime(
        dataset=dataset,
        info=info,
        config=config,
        diffusion=diffusion,
        checkpoint_path=str(Path(checkpoint_path).resolve()),
    )
