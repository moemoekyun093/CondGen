#!/usr/bin/env python
"""Per-checkpoint evaluation: full metric union on val AND test, 3 sampling seeds.

For every checkpoint saved by a training run:
  * load the weights (EMA where the family uses EMA, else raw);
  * for each split in {val, test}, and each sampling seed in {0,1,2}, generate a table
    of the SAME size as that split's reference set, using the family's OWN sampler
    (``tabequiv.sampler_equiv`` -- read-only baseline entry points, already validated);
  * score it with the full union: ``suite.ALL_METRICS`` (which includes synthcity) plus
    the two split-requiring DCR variants (``dcr_ratio``, ``dcr_quantile``);
  * time every stage.

Output: tidy long-format ``checkpoint_metrics.csv`` (step, split, metric, seed, value)
plus a mean/std rollup, and ``timing.json`` merged with the training timing.

Reporting rule: an eval that raises is recorded with ``value=NaN`` and its error kept in
``eval_errors.csv`` -- never silently dropped, since a missing row would otherwise read
as a metric that was never attempted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from tabequiv.eval import suite  # noqa: E402
from tabequiv.eval.dcr_variants import DcrCloserToTrainRatio, QuantileDcr  # noqa: E402
from tabequiv.repro import seed_everything  # noqa: E402
from tabequiv.splits import load_frame, load_split  # noqa: E402

SAMPLING_SEEDS = (0, 1, 2)
_ONLY_CKPT: str | None = None
_REAL_FRAME: dict = {}          # dataset -> full real frame, for categorical levels
SPLITS = ("val", "test")


_TRANSFORM_CACHE = {}


def _decimals(src, cap: int = 10) -> int:
    """Decimal places actually used by a real column (so decoding can snap to it).

    Exact match required: ``np.allclose`` with a default tolerance accepts a rounding
    that is not bit-identical, which left 268.416661 against a source 268.41666 and
    made every such row count as a mismatch in the KS statistic.
    """
    v = src.dropna().to_numpy(dtype=float)
    for d in range(cap + 1):
        if np.array_equal(v, np.round(v, d)):
            return d
    return cap


def _fitted_transforms(dataset: str, cfg: dict):
    """The TRAIN-fitted num/cat transforms, from tab-ddpm's own dataset pipeline.

    Samplers emit rows in NORMALIZED space (quantile-transformed numericals, encoded
    categoricals). The baselines invert that before writing a table --
    ``D.num_transform.inverse_transform`` / ``D.cat_transform.inverse_transform``
    (``tab-ddpm/scripts/sample.py:133,138``); tabsyn does the same via
    ``num_inverse``/``cat_inverse``. Skipping it left generated ``Administrative`` with
    mean 2452.9 against a real 2.31 -- a ~1000x scale error that scored BELOW gaussian
    noise on Shape (0.10-0.28 vs a 0.286 noise floor).

    We rebuild the same ``Dataset`` the trainer used, so the transforms are the exact
    objects fitted on our frozen train split.
    """
    key = (dataset, json.dumps(cfg.get("native", {}).get("T_dict", {}), sort_keys=True))
    if key in _TRANSFORM_CACHE:
        return _TRANSFORM_CACHE[key]
    from tabequiv.paths import use_baseline
    from tabequiv.train.run_tabddpm import NATIVE, _patch_quantile_transformer
    from tabequiv.train.stage import stage

    _patch_quantile_transformer()
    data_dir = stage(dataset)
    with use_baseline("tab-ddpm"):
        import lib
        from utils_train import make_dataset
        T = lib.Transformations(**NATIVE["T_dict"])
        D = make_dataset(str(data_dir), T,
                         num_classes=NATIVE["model_params"]["num_classes"],
                         is_y_cond=NATIVE["model_params"]["is_y_cond"],
                         change_val=False)
    _TRANSFORM_CACHE[key] = D
    return D


def _decode(state, d_num, categories, columns, info, dataset=None, cfg=None):
    """Sampler state -> DataFrame in the reference frame's column order.

    Applies the TRAIN-fitted inverse transforms so the emitted table is on the ORIGINAL
    data scale -- without this the metrics compare normalized samples against raw real
    data and every score is meaningless.
    """
    x = state.detach().cpu().float().numpy()
    cats = [int(k) for k in categories]
    num = x[:, :d_num]
    codes = np.rint(x[:, d_num:d_num + len(cats)]).astype(int)
    for j, k in enumerate(cats):
        codes[:, j] = np.clip(codes[:, j], 0, max(k - 1, 0))
    num_idx = list(info["num_col_idx"])
    cat_idx = list(info["cat_col_idx"])
    tgt = list(info["target_col_idx"])
    if info.get("task_type") == "regression":
        num_idx = sorted(set(num_idx) | set(tgt))
    else:
        cat_idx = sorted(set(cat_idx) | set(tgt))
    # Invert with the SAME scheme the run trained under -- they differ by runner:
    #   run_generic  -> plain z-score (norm_mean/norm_std recorded in config.json)
    #   run_tabddpm  -> tab-ddpm's fitted quantile transform
    # Using the wrong one is silent and severe: real data round-tripped through the
    # mismatched inverse scored Shape 0.564 instead of ~0.98.
    cfg = cfg or {}
    if dataset is not None and num.shape[1]:
        if cfg.get("norm_scheme") == "zscore":
            mu = np.asarray(cfg["norm_mean"], dtype=np.float64)
            sd = np.asarray(cfg["norm_std"], dtype=np.float64)
            k = min(num.shape[1], len(mu))
            num = num[:, :k] * sd[:k] + mu[:k]
        elif cfg.get("norm_scheme") == "quantile" and cfg.get("norm_file"):
            # run_generic --norm quantile: invert with the TRAIN-fitted transformer the
            # run persisted (the lineage's native preprocessing; see run_generic.load_tensors)
            import joblib
            qt = joblib.load(cfg["norm_file"])
            k = min(num.shape[1], qt.n_features_in_)
            num = qt.inverse_transform(num[:, :k])
        else:
            try:
                D = _fitted_transforms(dataset, cfg)
                if getattr(D, "num_transform", None) is not None:
                    k = D.X_num["train"].shape[1]
                    num = D.num_transform.inverse_transform(num[:, :k])
            except Exception as exc:          # never silently emit normalized rows
                raise RuntimeError(
                    f"inverse transform failed for {dataset}: "
                    f"{type(exc).__name__}: {exc}")

    out = pd.DataFrame(index=range(len(x)), columns=columns, dtype=object)
    real = _REAL_FRAME.get(dataset)
    for j, i in enumerate(num_idx[:num.shape[1]]):
        col = num[:, j]
        # Restore the column's ORIGINAL precision/dtype. Inverting a z-score returns
        # 2.999999992 where the source held 3.0, and an integer-valued column emitted
        # as float compares unequal in every marginal test -- Shape read 0.399 on
        # perfectly round-tripped REAL data purely from this.
        if real is not None:
            src = pd.to_numeric(real[columns[i]], errors="coerce").dropna()
            if len(src) and (src % 1 == 0).all():
                col = np.clip(np.rint(col), float(src.min()), float(src.max()))
                col = col.astype(np.int64)
            else:
                # Snap to the source column's own decimal precision. A z-score inverse
                # turns an exact 0.0 into -4e-07, which destroys zero-inflated columns:
                # Informational_Duration is 80.5% zeros in real data but 0% after
                # decoding, dropping its 1-KS to 0.195 and the whole Shape to 0.73.
                # Snap onto the source column's OWN value grid. Rounding to a decimal
                # count is not enough: BounceRates/ExitRates repeat a small set of
                # values, and a 6e-09 residue breaks those exact ties -- KS is
                # tie-sensitive, so 1-KS fell to 0.95 on columns that are otherwise
                # numerically identical. Mapping each decoded value to the nearest
                # value the real column actually takes restores the ties exactly.
                dec = _decimals(src)
                col = np.round(col, dec) + 0.0       # +0.0 normalises -0.0
                uniq = np.unique(src.to_numpy(dtype=float))
                if len(uniq) <= 200_000:      # cover full-frame uniques (ExitRates has 4777)
                    idx = np.searchsorted(uniq, col).clip(0, len(uniq) - 1)
                    left = np.maximum(idx - 1, 0)
                    pick = np.where(np.abs(col - uniq[left]) <= np.abs(col - uniq[idx]),
                                    left, idx)
                    near = uniq[pick]
                    tol = np.maximum(np.abs(col), 1.0) * 1e-6
                    col = np.where(np.abs(col - near) <= tol, near, col)
                # Snap near-zero residue to EXACT zero. A z-score inverse leaves
                # -4e-07 where the source held 0.0; on a zero-inflated column that
                # is catastrophic -- Informational_Duration is 80.5% zeros in real
                # data but 0% after decoding, taking its 1-KS to 0.195 and Shape to
                # 0.73. The tolerance is scaled to the column, not absolute.
                if (src == 0).any():
                    scale = float(np.abs(src[src != 0]).min()) if (src != 0).any() else 1.0
                    col = np.where(np.abs(col) < scale * 0.5, 0.0, col)
        out[columns[i]] = col
    # categorical codes map back to the ORIGINAL level strings, in the same sorted order
    # the staging used, so a code is the level it names rather than a bare integer.
    for j, i in enumerate(cat_idx[:codes.shape[1]]):
        levels = sorted(set(_REAL_FRAME[dataset].iloc[:, i].astype(str))) \
            if dataset in _REAL_FRAME else None
        if levels:
            vals = [levels[min(c, len(levels) - 1)] for c in codes[:, j]]
            # Cast back to the real column's dtype: bool columns (Weekend, Revenue)
            # otherwise emerge as the strings 'True'/'False' and never match.
            # Cast back to the REAL column's dtype. Levels are carried as strings, but
            # SDMetrics' TVComplement compares values directly: an int64 real column
            # against object '2' matches nothing and scores ~3e-09 -- four such columns
            # dragged Shape from ~0.98 to 0.775 while a manual TVD check (which casts
            # both sides to str) read a perfect 1.0000 and hid it.
            src_col = real[columns[i]] if real is not None else None
            if src_col is not None and pd.api.types.is_bool_dtype(src_col):
                out[columns[i]] = [v == "True" for v in vals]
            elif src_col is not None and pd.api.types.is_integer_dtype(src_col):
                out[columns[i]] = pd.Series(vals).astype(np.int64).to_numpy()
            elif src_col is not None and pd.api.types.is_float_dtype(src_col):
                out[columns[i]] = pd.Series(vals).astype(float).to_numpy()
            else:
                out[columns[i]] = vals
        else:
            out[columns[i]] = codes[:, j].astype(str)
    return out.infer_objects()


_VAE_CACHE: dict = {}


def _load_frozen_vae(dataset: str, cfg: dict, device: str):
    """The TRAINED TabSyn-VAE this dataset's latent runs were encoded with.

    Mirrors ``tabequiv/train/run_generic.py:_encode_latents`` line for line: same
    checkpoint (the final ``step_*.pt`` of ``TabSyn-VAE__original/seed0``), same
    ``ablate.build("tabsyn_vae", ...)`` construction, EMA-else-raw weights, hard error on
    missing keys. Using any OTHER decoder (e.g. a random-init one) would decode the
    latent into noise while looking like a valid table.
    """
    key = (dataset, str(device), cfg.get("vae_checkpoint"))     # a run may pin its own frozen VAE
    if key in _VAE_CACHE:
        return _VAE_CACHE[key]
    from tabequiv import ablate
    from tabequiv.train.run_generic import _move_stray_tensors

    ck = (REPO / "experiments" / "backbone_training" / dataset /
          "TabSyn-VAE__original" / "seed0" / "checkpoints")
    ckpts = sorted(ck.glob("step_*.pt"))
    if cfg.get("vae_checkpoint"):                 # native TabSyn/TabSynFlow runs record the exact VAE they encoded with
        ckpts = [Path(cfg["vae_checkpoint"])]
    if not ckpts:
        raise RuntimeError(f"latent eval needs the trained TabSyn-VAE; none in {ck}")
    blob = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    dev = torch.device(device)
    vae = ablate.build("tabsyn_vae", cfg["d_num"], cfg["categories"],
                       seed=cfg.get("seed", 0)).to(dev)
    _move_stray_tensors(vae, dev)
    sd = blob.get("ema") or blob.get("raw")
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"VAE checkpoint mismatch: {len(missing)} missing keys")
    vae.eval()
    print(f"  latent decoder: frozen VAE {ckpts[-1].name} "
          f"({len(unexpected)} unexpected keys ignored)", flush=True)
    _VAE_CACHE[key] = vae
    return vae


def _to_decode_form(family: str, state: torch.Tensor, dataset: str, cfg: dict,
                    device: str) -> torch.Tensor:
    """Raw sampler state (any family) -> the ``[x_num | cat codes]`` form ``_decode`` reads.

    The samplers do NOT share an output layout (``sampler_equiv.generate``):
      unimod / tabddpm -> ``[x_num | codes]``          (already the right form)
      tabvvfm          -> ``[x_num | ONE-HOT blocks]``  (77 wide on shoppers, not 18)
      latent           -> flat VAE latent              (72 wide)
    ``sampler_equiv.state_to_table`` already normalises all three to ``(x_num, codes)``;
    this eval path previously called it for NONE of them and fed the raw state to
    ``_decode`` -- which silently read one-hot columns (tabvvfm, Shape ~0.62) and latent
    dims (TabSyn/TabSynFlow, Shape ~0.34) as if they were table columns. Both defects
    reproduce fox's recorded numbers exactly, i.e. they pre-date the Triton migration.
    """
    from tabequiv.sampler_equiv import LATENT_FAMILIES, state_to_table
    vae = _load_frozen_vae(dataset, cfg, device) if family in LATENT_FAMILIES else None
    state = state.to(torch.device(device))
    if family in LATENT_FAMILIES and cfg.get("latent_mean") is not None:
        # native TabSyn / TabSynFlow runs train on (z - mean) / scale (tabsyn/main.py, tabsynflow/main.py);
        # their samplers un-normalise before decoding (sample.py: x * 2 + mean) -- so must we.
        state = state * float(cfg.get("latent_scale", 2.0)) + torch.as_tensor(cfg["latent_mean"], dtype=state.dtype, device=state.device)
    x_num, codes = state_to_table(family, state, cfg["d_num"], cfg["categories"], vae=vae)
    return torch.cat([x_num.float(), codes.float()], dim=1)


def _latent_to_state(z: torch.Tensor, dataset: str, cfg: dict, device: str) -> torch.Tensor:
    """Flat VAE latent ``[B, n_cols*d_token]`` -> sampler-state ``[B, d_num + m]``.

    THE BUG THIS FIXES (found 2026-08-25, Triton): ``evaluate_run`` handed the raw
    latent straight to :func:`_decode`, which reads columns 0..d_num as numericals and
    the next m as categorical codes. For TabSyn / TabSynFlow that is a 72-wide latent
    being read as 18 table columns -- every value was noise, and both families scored
    Shape ~0.34 against a published ~0.985. The decode helper the repo already had
    (``sampler_equiv.decode_latent``) was simply never called from the eval path.

    The VAE emits numericals in the z-score space it was trained in
    (``run_generic.load_tensors``), which is the same space every ``run_generic`` run
    records in ``config.json`` as ``norm_mean``/``norm_std`` -- so the ordinary
    ``norm_scheme == "zscore"`` inverse in :func:`_decode` applies unchanged.
    """
    from tabequiv.sampler_equiv import decode_latent
    vae = _load_frozen_vae(dataset, cfg, device)
    if cfg.get("latent_mean") is not None:      # native TabSyn/TabSynFlow train on (z - mean) / 2
        z = z * 2 + torch.as_tensor(cfg["latent_mean"], dtype=z.dtype, device=z.device)
    x_num, codes = decode_latent(vae, z.to(torch.device(device)), cfg["d_num"],
                                 cfg["categories"])
    return torch.cat([x_num.float(), codes.float()], dim=1)


def evaluate_run(run_dir: Path, dataset: str, family: str, device: str = "cuda",
                 max_checkpoints: int | None = None, shard: int | None = None,
                 n_shards: int | None = None) -> dict:
    """Evaluate a run's checkpoints.

    SHARDING (``shard``/``n_shards``): checkpoints are independent, so a run can be
    split across parallel Slurm array tasks. Task ``i`` of ``n`` takes checkpoints
    ``i::n`` (strided, NOT contiguous, so every shard spans the whole training
    trajectory and they finish in comparable time -- early checkpoints sample faster
    than late ones for some families).

    A sharded call writes to ``_shards/shard_<i>_of_<n>.csv`` instead of the run's
    top-level CSVs, because every shard would otherwise write the SAME filenames and
    silently clobber its siblings. Merge with ``--merge-shards``.
    """
    from tabequiv.sampler_equiv import generate      # family dispatch, read-only

    df, info = load_frame(dataset)
    _REAL_FRAME[dataset] = df
    sp = load_split(dataset)
    columns = list(df.columns)
    refs = {s: df.iloc[sp[s]].reset_index(drop=True) for s in SPLITS}
    train_ref = df.iloc[sp["train"]].reset_index(drop=True)

    cfg = json.loads((run_dir / "config.json").read_text())
    ckpts = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    if _ONLY_CKPT:
        ckpts = [run_dir / "checkpoints" / _ONLY_CKPT]
    if max_checkpoints:
        ckpts = ckpts[:max_checkpoints]
    all_n = len(ckpts)
    if n_shards and n_shards > 1:
        if shard is None or not (0 <= shard < n_shards):
            raise ValueError(f"shard {shard} out of range for n_shards={n_shards}")
        ckpts = ckpts[shard::n_shards]
        print(f"  shard {shard}/{n_shards}: {len(ckpts)} of {all_n} checkpoints "
              f"({[int(c.stem.split('_')[1]) for c in ckpts][:4]}...)", flush=True)

    rows, errors, timings = [], [], []
    metrics = list(suite.ALL_METRICS)
    for ck in ckpts:
        try: step = int(ck.stem.split("_")[1])
        except ValueError: step = int(torch.load(ck, map_location="cpu", weights_only=False).get("step", 0))
        t_ck = time.time()
        for split in SPLITS:
            ref = refs[split]
            n_rows = len(ref)
            for sd in SAMPLING_SEEDS:
                t0 = time.time()
                try:
                    # Build FIRST, then seed. ``_load_model`` -> ``ablate.build`` calls
                    # ``seed_everything(cfg['seed'])`` internally (ablate.py:419,
                    # models/*.py:32), which WIPES the sampling seed if set beforehand.
                    # That made all three "independent" sample sets byte-identical --
                    # every metric had std=0.0 across seeds, so the mean+/-std the pilot
                    # reports was meaningless.
                    mdl = _load_model(ck, cfg, device)
                    seed_everything(sd)
                    for _attempt in range(4):
                        try:
                            state = generate(family, mdl, n_rows=n_rows, d_num=cfg["d_num"],
                                             categories=cfg["categories"])
                            break
                        except (AssertionError, RuntimeError) as _exc:   # ef-vfm dopri5 underflow: redraw the noise
                            if _attempt == 3: raise
                            seed_everything(sd + 100_000 * (_attempt + 1))
                    # every family -> [x_num | codes] (latent via the TRAINED VAE,
                    # tabvvfm via per-block argmax, unimod/tabddpm pass-through)
                    state = _to_decode_form(family, state, dataset, cfg, device)
                    syn = _decode(state, cfg["d_num"], cfg["categories"], columns,
                                  info, dataset=dataset, cfg=cfg)
                    t_sample = time.time() - t0
                    t1 = time.time()
                    res = suite.evaluate_frames(ref, syn, info, device=device,
                                                metrics=metrics, skip_on_error=True)
                    for M in (DcrCloserToTrainRatio, QuantileDcr):
                        res.update(M()(ref, syn, info, device=device,
                                       train=train_ref, holdout=ref))
                    t_eval = time.time() - t1
                    for k, v in res.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            rows.append({"step": step, "split": split, "metric": k,
                                         "seed": sd, "value": float(v)})
                    timings.append({"step": step, "split": split, "seed": sd,
                                    "seconds_sample": round(t_sample, 2),
                                    "seconds_eval": round(t_eval, 2)})
                except Exception as exc:
                    errors.append({"step": step, "split": split, "seed": sd,
                                   "error": f"{type(exc).__name__}: {exc}",
                                   "traceback": traceback.format_exc()[-800:]})
                    rows.append({"step": step, "split": split,
                                 "metric": "FAILED", "seed": sd, "value": float("nan")})
        print(f"  ckpt {step}: {time.time()-t_ck:.1f}s", flush=True)

    long = pd.DataFrame(rows)
    sharded = bool(n_shards and n_shards > 1)
    if sharded:
        out_dir = run_dir / "_shards"
        out_dir.mkdir(exist_ok=True)
        long.to_csv(out_dir / f"shard_{shard}_of_{n_shards}.csv", index=False)
        if errors:
            pd.DataFrame(errors).to_csv(
                out_dir / f"errors_{shard}_of_{n_shards}.csv", index=False)
        if len(tdf := pd.DataFrame(timings)):
            tdf.to_csv(out_dir / f"timing_{shard}_of_{n_shards}.csv", index=False)
        print(f"  shard {shard}/{n_shards} -> {out_dir} "
              f"({len(long)} rows, {len(errors)} errors)", flush=True)
        return {"run": str(run_dir), "shard": shard, "n_shards": n_shards,
                "n_checkpoints": len(ckpts), "n_rows": len(long),
                "n_errors": len(errors)}
    long.to_csv(run_dir / "checkpoint_metrics.csv", index=False)
    if len(long):
        roll = (long.dropna(subset=["value"])
                    .groupby(["step", "split", "metric"])["value"]
                    .agg(["mean", "std", "count"]).reset_index())
        roll.to_csv(run_dir / "checkpoint_metrics_rollup.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(run_dir / "eval_errors.csv", index=False)
    tdf = pd.DataFrame(timings)
    timing = json.loads((run_dir / "timing.json").read_text()) \
        if (run_dir / "timing.json").exists() else {}
    if len(tdf):
        per_ck = tdf.groupby("step")[["seconds_sample", "seconds_eval"]].sum().sum(axis=1)
        timing.update({
            "n_checkpoints_evaluated": int(tdf.step.nunique()),
            "eval_seconds_total": float(tdf[["seconds_sample", "seconds_eval"]].sum().sum()),
            "eval_seconds_per_checkpoint_median": float(per_ck.median()),
            "eval_seconds_per_pass_median": float(tdf.seconds_eval.median()),
            "sample_seconds_per_pass_median": float(tdf.seconds_sample.median()),
        })
        (run_dir / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        tdf.to_csv(run_dir / "eval_timing.csv", index=False)
    return {"run": str(run_dir), "n_checkpoints": len(ckpts),
            "n_rows": len(long), "n_errors": len(errors)}


def _build_native(cfg: dict, device: str = "cpu"):
    """Construct the family's model at the architecture the run actually trained.

    Built DIRECTLY on ``device``. ``GaussianMultinomialDiffusion`` binds its schedule
    tensors at construction (``gaussian_multinomial_diffsuion.py:89-151``) and several
    of them -- ``posterior_mean_coef1/2``, ``posterior_log_variance_clipped`` -- are
    plain attributes rather than registered buffers, so a later ``.to(device)`` silently
    leaves them behind and sampling dies with a cuda/cpu mismatch. Constructing on the
    target device is the only way to get them all onto the GPU.
    """
    from tabequiv.paths import use_baseline

    fam = cfg["family_key"]
    if fam == "tabddpm":
        native = cfg["native"]
        mp = json.loads(json.dumps(native["model_params"]))
        K = [int(k) for k in cfg["categories"]]
        mp["d_in"] = int(sum(K) + cfg["d_num"])
        with use_baseline("tab-ddpm"):
            from tab_ddpm import GaussianMultinomialDiffusion
            from utils_train import get_model
            denoise = get_model("mlp", mp, cfg["d_num"], category_sizes=K)
            diff = GaussianMultinomialDiffusion(
                num_classes=np.array(K), num_numerical_features=cfg["d_num"],
                denoise_fn=denoise,
                gaussian_loss_type=native["gaussian_loss_type"],
                num_timesteps=native["num_timesteps"],
                scheduler=native["scheduler"], device=torch.device(device))
        if cfg.get("variant") == "symmetric":
            from tabequiv.ablate import wrap_symmetric
            wrap_symmetric(diff, "tabddpm", seed=cfg.get("seed", 0))
        return diff.to(device)
    # Every non-TabDDPM family was trained through ``ablate.build`` in run_generic.py,
    # so the same call reproduces the exact architecture. ``_move_stray_tensors`` is
    # required for the same reason it is at training time: TabDiff's ``mask_index`` and
    # friends are plain attributes that ``.to(device)`` silently skips.
    from tabequiv import ablate
    # NEW-PROTOCOL runs record which TabDiff denoiser they trained ("original" or
    # "ft_periodic" = AA_code's symmetric core). They are built through the family
    # wrapper's ``denoiser=`` knob, NOT through the retired tabequiv.ablate cores.
    if fam in ("tabsyn_diffusion", "tabflow_net") and cfg.get("denoiser"):
        import importlib
        _fam = importlib.import_module(f"tabequiv.models.{fam}")
        return _fam.build(cfg["d_num"], cfg["categories"], seed=cfg.get("seed", 0),
                          denoiser=cfg["denoiser"], ft_kw=cfg.get("ft_kw")).to(device)
    if fam == "unimod_efvfm" and cfg.get("denoiser"):
        from tabequiv.models import unimod_efvfm as _fam
        from tabequiv.train.run_generic import _move_stray_tensors
        m = _fam.build(cfg["d_num"], cfg["categories"], seed=cfg.get("seed", 0),
                       denoiser=cfg["denoiser"], ft_kw=cfg.get("ft_kw")).to(device)
        _move_stray_tensors(m, device)          # ExpVFM keeps plain-attribute tensors on its build device
        tgt = torch.device(device)              # ...and samples its noise on ``self.device`` (flow_model.py:90-95)
        for mod in m.modules():
            if hasattr(mod, "device"):
                try: mod.device = tgt
                except Exception: pass
        return m
    if fam == "tabvvfm" and cfg.get("denoiser"):
        from tabequiv.models import tabvvfm as _fam
        return _fam.build(cfg["d_num"], cfg["categories"], seed=cfg.get("seed", 0),
                          denoiser=cfg["denoiser"], ft_kw=cfg.get("ft_kw")).to(device)
    if fam == "unimod_tabdiff" and cfg.get("denoiser"):
        from tabequiv.models import unimod_tabdiff as _fam
        _real_device = torch.device
        def _forced(*a, **k):
            if a and isinstance(a[0], str) and a[0].startswith("cpu"):
                return _real_device(device)
            return _real_device(*a, **k)
        torch.device = _forced
        try:
            model = _fam.build(cfg["d_num"], cfg["categories"], seed=cfg.get("seed", 0),
                               denoiser=cfg["denoiser"], ft_kw=cfg.get("ft_kw"),
                               learnable_schedule=bool(cfg.get("learnable_schedule", False)))
        finally:
            torch.device = _real_device
        model = model.to(device)
        tgt = torch.device(device)
        for mod in model.modules():
            if hasattr(mod, "device") and not isinstance(getattr(mod, "device"), property):
                try: mod.device = tgt
                except Exception: pass
        return model
    # Build ON the target device. Our wrappers hardcode ``device=torch.device("cpu")``
    # (e.g. tabequiv/models/unimod_tabdiff.py:61) because they were written for the
    # no-training golden-master phase, and the baselines bind schedule tensors at
    # CONSTRUCTION -- TabDiff's sampler then mixes cpu/cuda inside q_xt/edm_update no
    # matter what we move afterwards. Redirecting torch.device during the build makes
    # the baseline construct everything on the GPU in the first place.
    _real_device = torch.device

    def _forced(*a, **k):
        if a and isinstance(a[0], str) and a[0].startswith("cpu"):
            return _real_device(device)
        return _real_device(*a, **k)

    torch.device = _forced
    try:
        model = ablate.build(fam, cfg["d_num"], cfg["categories"],
                             seed=cfg.get("seed", 0),
                             symmetric_core=(cfg.get("variant") == "symmetric"))
    finally:
        torch.device = _real_device
    model = model.to(device)
    # Some baselines keep the target device as a PLAIN ATTRIBUTE and read it back when
    # sampling -- ef-vfm's ``sample()`` does ``dev = self.device`` then
    # ``torch.randn(..., device=dev)`` (ef-vfm/ef_vfm/models/flow_model.py:90,95). Our
    # wrappers construct with device="cpu", so that attribute stays "cpu" even after
    # .to(cuda) and the sampler mixes devices. Rewrite it everywhere it appears.
    tgt = torch.device(device)
    for mod in model.modules():
        if hasattr(mod, "device") and not isinstance(
                getattr(mod, "device"), (property,)):
            try:
                mod.device = tgt
            except Exception:
                pass
    return model


def _load_model(ck: Path, cfg: dict, device: str):
    """Rebuild the family's model and load this checkpoint's weights (EMA preferred).

    Sampling runs on ``device`` (GPU by default). See :func:`_build_native` for why the
    model must be CONSTRUCTED there rather than moved afterwards.
    """
    blob = torch.load(ck, map_location=device, weights_only=False)
    # Rebuild with the TRAINING architecture from config.json, not the golden-master
    # toy one. tabequiv/models/tabddpm.py hardcodes d_layers [64,64] for its bit-exact
    # tests, whereas training used the native shoppers config [1024,2048,2048,1024];
    # loading across that mismatch (strict=False) would have silently evaluated an
    # untrained model.
    model = _build_native(cfg, device)
    sd = blob.get("ema") or blob.get("raw")
    # The checkpoint holds the DENOISER's weights; the wrapper exposes it at
    # ``_denoise_fn``. Load strictly onto that submodule -- a silent strict=False
    # failure would evaluate an untrained model and look like a real metric.
    # Which module the checkpoint's keys belong to differs by runner:
    #   run_tabddpm  saves ``diffusion._denoise_fn.state_dict()`` -> denoiser-scoped
    #   run_generic  saves the WHOLE model's state_dict()        -> model-scoped
    # Pick by inspecting the keys rather than assuming, so a prefix mismatch can never
    # be papered over (it previously produced 190 missing / 190 unexpected per ckpt).
    target = model
    if not any(k.startswith("_denoise_fn.") for k in sd) and hasattr(model, "_denoise_fn"):
        target = model._denoise_fn
    missing, unexpected = target.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch for {ck.name}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected "
            f"(first missing: {list(missing)[:3]})")
    # Move AFTER loading: load_state_dict copies the checkpoint's (CPU) tensors in, so
    # relocating stray non-buffer attributes beforehand is undone by the load.
    from tabequiv.train.run_generic import _move_stray_tensors
    model = model.to(device)
    _move_stray_tensors(model, torch.device(device))
    return model.eval()


def merge_shards(run_dir: Path) -> dict:
    """Concatenate ``_shards/*.csv`` into the run's canonical top-level outputs.

    Fails loudly on a MISSING shard rather than silently emitting a partial curve: a
    gap in the checkpoint series would look like a real result. Every shard file names
    its own index, so the expected set is recoverable from any one of them.
    """
    sd = run_dir / "_shards"
    parts = sorted(sd.glob("shard_*_of_*.csv"))
    if not parts:
        raise FileNotFoundError(f"no shard files in {sd}")
    n_shards = int(parts[0].stem.split("_of_")[1])
    have = {int(p.stem.split("_")[1]) for p in parts}
    missing = set(range(n_shards)) - have
    if missing:
        raise RuntimeError(
            f"{run_dir.name}: shards {sorted(missing)} missing of {n_shards} -- "
            f"refusing to merge a partial curve (re-run those array tasks)")

    long = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    long = long.sort_values(["step", "split", "metric", "seed"]).reset_index(drop=True)
    long.to_csv(run_dir / "checkpoint_metrics.csv", index=False)
    roll = (long.dropna(subset=["value"])
                .groupby(["step", "split", "metric"])["value"]
                .agg(["mean", "std", "count"]).reset_index())
    roll.to_csv(run_dir / "checkpoint_metrics_rollup.csv", index=False)

    errs = sorted(sd.glob("errors_*_of_*.csv"))
    if errs:
        pd.concat([pd.read_csv(p) for p in errs], ignore_index=True).to_csv(
            run_dir / "eval_errors.csv", index=False)
    tims = sorted(sd.glob("timing_*_of_*.csv"))
    if tims:
        tdf = pd.concat([pd.read_csv(p) for p in tims], ignore_index=True)
        tdf.to_csv(run_dir / "eval_timing.csv", index=False)
        timing = json.loads((run_dir / "timing.json").read_text()) \
            if (run_dir / "timing.json").exists() else {}
        per_ck = tdf.groupby("step")[["seconds_sample", "seconds_eval"]].sum().sum(axis=1)
        timing.update({
            "n_checkpoints_evaluated": int(tdf.step.nunique()),
            "eval_seconds_total": float(tdf[["seconds_sample", "seconds_eval"]].sum().sum()),
            "eval_seconds_per_checkpoint_median": float(per_ck.median()),
            "eval_seconds_per_pass_median": float(tdf.seconds_eval.median()),
            "sample_seconds_per_pass_median": float(tdf.seconds_sample.median()),
            "n_shards": n_shards,
        })
        (run_dir / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    return {"run": str(run_dir), "merged_shards": n_shards, "n_rows": len(long),
            "n_checkpoints": int(long.step.nunique()),
            "n_errors": sum(len(pd.read_csv(p)) for p in errs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="shoppers")
    ap.add_argument("--family", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-checkpoints", type=int, default=None)
    ap.add_argument("--n-sampling-seeds", type=int, default=None,
                    help="override SAMPLING_SEEDS with range(N) (train once, generate many)")
    ap.add_argument("--only", default=None, help="evaluate only this checkpoint filename, e.g. best_ema.pt")
    ap.add_argument("--shard", type=int, default=None,
                    help="this task's index in [0, n-shards)")
    ap.add_argument("--n-shards", type=int, default=None,
                    help="split the checkpoints across N parallel tasks")
    ap.add_argument("--merge-shards", action="store_true",
                    help="concatenate _shards/*.csv into the canonical outputs and exit")
    a = ap.parse_args()
    global SAMPLING_SEEDS
    if a.n_sampling_seeds:
        SAMPLING_SEEDS = tuple(range(a.n_sampling_seeds))
    if a.only:
        global _ONLY_CKPT; _ONLY_CKPT = a.only
    if a.merge_shards:
        print(json.dumps(merge_shards(Path(a.run_dir)), indent=2))
        return 0
    r = evaluate_run(Path(a.run_dir), a.dataset, a.family, a.device, a.max_checkpoints,
                     shard=a.shard, n_shards=a.n_shards)
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
