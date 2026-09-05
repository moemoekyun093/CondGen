#!/usr/bin/env python
"""Generic trainer for the 5 non-TabDDPM neural families (+ TabSyn-VAE).

TabDDPM has its own runner because tab-ddpm ships a self-contained ``Trainer`` class we
can subclass. The remaining families' native trainers are entangled with their CLI and
data pipelines, so here we drive each family's **native loss function** -- the wrapper's
``loss_fn``, which calls the baseline's own objective read-only -- with its native
optimizer settings.

RECIPE stays native per family (lr, architecture, batch, optimizer, EMA); only the
BUDGET is the uniform 10,000-epoch pilot budget. Both are recorded in config.json.

This is a deliberate, stated deviation: the optimizer LOOP is ours, the LOSS and the
MODEL are the baselines'. TabDDPM (the family with a directly reusable native trainer)
therefore runs its true native loop, and is the reference point for whether this
generic loop behaves comparably.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from tabequiv.repro import seed_everything  # noqa: E402
from tabequiv.splits import load_frame, load_split  # noqa: E402
from tabequiv.train.stage import stage  # noqa: E402

EPOCH_BUDGET = 10_000

#: Native recipe per family, from each baseline's own config.
#: lr/batch: TabDiff tabdiff_configs.toml:45-49; tabsyn vae/main.py + tabsyn/main.py;
#: tabular-flow-matching mirrors tabsyn; ef-vfm mirrors TabDiff.
NATIVE = {
    "unimod_tabdiff":   {"lr": 1e-3,  "batch_size": 4096, "wd": 0.0,  "ema": 0.997},
    "unimod_efvfm":     {"lr": 1e-3,  "batch_size": 4096, "wd": 0.0,  "ema": 0.997},
    "tabvvfm":          {"lr": 1e-3,  "batch_size": 4096, "wd": 0.0,  "ema": None},
    "tabsyn_diffusion": {"lr": 1e-3,  "batch_size": 4096, "wd": 0.0,  "ema": None},
    "tabflow_net":      {"lr": 1e-3,  "batch_size": 4096, "wd": 0.0,  "ema": None},
    "tabsyn_vae":       {"lr": 1e-3,  "batch_size": 4096, "wd": 0.0,  "ema": None},
}
CANON = {"unimod_tabdiff": "TabDiff", "unimod_efvfm": "TabbyFlow-orig",
         "tabvvfm": "TabbyFlow-reimpl", "tabsyn_diffusion": "TabSyn",
         "tabflow_net": "TabSynFlow", "tabsyn_vae": "TabSyn-VAE"}


def load_tensors(dataset: str, norm: str = "zscore"):
    """-> per-split (x_num, x_cat_codes) float/long tensors + schema.

    ``norm``:
      "zscore"   -- per-column (x - mean) / std, TRAIN statistics. The pilot's original
                    choice; NOT what any baseline does.
      "quantile" -- the NATIVE preprocessing of the whole tabsyn lineage AND tabpc:
                    ``QuantileTransformer(output_distribution="normal")`` fit on TRAIN
                    (``baselines/tabsyn/utils_train.py:28`` sets
                    ``T_dict["normalization"]="quantile"``; ``src/data.py:221`` sizes it
                    ``n_quantiles=max(min(n_train//30,1000),10)``;
                    tabpc ``df_quantile_normalizer.py:44``). MEASURED reason to prefer it
                    (Triton, 2026-08-25, TabDiff on shoppers): under z-score a Gaussian
                    diffusion smears zero-inflated columns -- BounceRates 45% exact zeros
                    real vs 5% synthetic, KS 0.283 -- and an XGBoost discriminator reaches
                    AUC 0.90 (xgb-detect 0.20 vs published 0.74). The quantile map sends
                    the zero mass to one point that the inverse restores exactly.
    The fitted transformer is returned under ``out["_qt"]`` so the run can persist it;
    eval MUST invert with the same object (see eval_checkpoints._decode).
    """
    df, info = load_frame(dataset)
    sp = load_split(dataset)
    num_idx = list(info["num_col_idx"])
    cat_idx = list(info["cat_col_idx"])
    tgt = list(info["target_col_idx"])
    if info.get("task_type") == "regression":
        num_idx = sorted(set(num_idx) | set(tgt))
    else:
        cat_idx = sorted(set(cat_idx) | set(tgt))
    cols = list(df.columns)

    levels = {i: {v: k for k, v in enumerate(sorted(set(df[cols[i]].astype(str))))}
              for i in cat_idx}
    out = {}
    for split, idx in sp.items():
        sub = df.iloc[idx]
        xn = sub.iloc[:, num_idx].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        xn = np.nan_to_num(xn)
        mu, sd = xn.mean(0), xn.std(0)
        if split == "train":
            out["_norm"] = (mu, np.where(sd > 0, sd, 1.0))
            if norm == "quantile":
                from sklearn.preprocessing import QuantileTransformer
                out["_qt"] = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=max(min(len(xn) // 30, 1000), 10),
                    subsample=int(1e9), random_state=0).fit(xn)
        if norm == "quantile":
            xn = out["_qt"].transform(xn) if xn.shape[1] else xn
        else:
            mu, sd = out["_norm"]
            xn = (xn - mu) / sd
        xc = np.stack([sub[cols[i]].astype(str).map(levels[i]).to_numpy()
                       for i in cat_idx], axis=1) if cat_idx else np.zeros((len(sub), 0))
        out[split] = (torch.tensor(xn, dtype=torch.float32),
                      torch.tensor(xc, dtype=torch.long))
    cats = [len(levels[i]) for i in cat_idx]
    return out, len(num_idx), cats, info


def _encode_latents(data, d_num, cats, dev, seed, dataset="shoppers"):
    """Encode every split through the FROZEN TabSyn-VAE -> per-split latent tensors.

    TabSyn and TabSynFlow are latent models: they train on the VAE's embedding, not on
    raw columns. The VAE is trained once and frozen (this pilot's ``TabSyn-VAE__original``
    run), so its final checkpoint is loaded here. Feeding these families random noise
    instead -- which the toy ``make_batch`` does for golden-master tests -- would train
    them on nothing.
    """
    from tabequiv import ablate
    # The VAE is PER-DATASET: its Reconstructor has one head per categorical column, so
    # a shoppers VAE (8 cat cols) cannot be loaded against magic (1 cat col). Hardcoding
    # "shoppers" here made every magic latent run fail on a Reconstructor size mismatch.
    ck = (REPO / "experiments" / "backbone_training" / dataset /
          "TabSyn-VAE__original" / "seed0" / "checkpoints")
    ckpts = sorted(ck.glob("step_*.pt"))
    if not ckpts:
        raise RuntimeError(
            "TabSyn-VAE must be trained before the latent families "
            f"(no checkpoints in {ck}). Run --family tabsyn_vae --dataset "
            f"{dataset} first.")
    blob = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    vae = ablate.build("tabsyn_vae", d_num, cats, seed=seed).to(dev)
    _move_stray_tensors(vae, dev)
    sd = blob.get("ema") or blob.get("raw")
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"VAE checkpoint mismatch: {len(missing)} missing keys")
    vae.eval()
    out = {}
    with torch.no_grad():
        for split in ("train", "val", "test"):
            xn, xc = data[split]
            zs = []
            for i in range(0, len(xn), 4096):
                tok = vae.VAE.Tokenizer(xn[i:i+4096].to(dev), xc[i:i+4096].to(dev))
                mu = vae.VAE.encoder_mu(tok)
                # Drop the CLS token: the tokenizer emits [CLS | one token per column]
                # (n_cols+1 tokens), but the latent heads are sized for n_cols columns
                # -- latent_dim() is (d_num + len(categories)) * TOKEN_DIM. Keeping CLS
                # gives 76 vs the expected 72 on shoppers and fails the first matmul.
                zs.append(mu[:, 1:, :].reshape(len(tok), -1).cpu())
            out[split] = torch.cat(zs)
    print(f"  latents: {tuple(out['train'].shape)} from frozen VAE "
          f"{ckpts[-1].name}", flush=True)
    return out


def _move_stray_tensors(model, dev, _depth=0):
    """Move plain-attribute tensors onto ``dev``.

    Several baselines keep device-bound tensors as ORDINARY ATTRIBUTES rather than
    registered buffers, so ``nn.Module.to()`` silently skips them and the forward pass
    mixes cuda and cpu tensors. Concretely: TabDiff's ``self.mask_index``
    (``TabDiff/tabdiff/models/unified_ctime_diffusion.py:44``) and TabDDPM's
    ``posterior_mean_coef1/2``. Our wrappers construct with ``device=cpu``
    (``tabequiv/models/unimod_tabdiff.py:61``) because they were written for the
    no-training golden-master phase.

    Without this the loss still RUNS -- it just runs those ops on CPU, which is why the
    process was seen burning 211 cores at 0% GPU. Walk the module tree and relocate any
    stray tensor.
    """
    if _depth > 6:
        return model
    for name, val in list(vars(model).items()):
        if isinstance(val, torch.Tensor) and val.device != dev:
            try:
                setattr(model, name, val.to(dev))
            except Exception:
                pass
    for child in model.children():
        _move_stray_tensors(child, dev, _depth + 1)
    for val in vars(model).values():
        if isinstance(val, torch.nn.Module):
            _move_stray_tensors(val, dev, _depth + 1)
    return model



def require_gpu(device: str) -> torch.device:
    """FAIL LOUDLY if a GPU was requested but is unavailable.

    Without this an invalid ``CUDA_VISIBLE_DEVICES`` (e.g. index 5 on a 5-GPU host,
    whose valid indices are 0-4) makes ``torch.cuda.is_available()`` False, and the run
    silently falls back to CPU -- observed burning 211 cores at 0% GPU. A training run
    that quietly runs 100x slower on the wrong device is worse than one that crashes.
    """
    if device.startswith("cpu"):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {device} requested but torch.cuda.is_available() is False "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')!r}). "
            f"Refusing to fall back to CPU silently.")
    return torch.device(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(NATIVE))
    ap.add_argument("--variant", choices=("original", "symmetric"), default="original")
    ap.add_argument("--denoiser", choices=("original", "ft_periodic"), default=None,
                    help="NEW PROTOCOL: swap the family's data-space net for AA_code's symmetric "
                         "ft_periodic core via the family's own build(denoiser=...) knob "
                         "(unimod_tabdiff, tabvvfm). Bypasses the retired --variant symmetric path.")
    ap.add_argument("--ft-kw", default=None, help="JSON overrides for FT_PERIODIC_KW (with --denoiser)")
    ap.add_argument("--dataset", default="shoppers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCH_BUDGET)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--norm", choices=("zscore", "quantile"), default="quantile",
                    help="numerical normalisation. DEFAULT 'quantile' = the baselines' NATIVE "
                         "preprocessing (decided 2026-08-25 after it moved TabDiff xgb-detect "
                         "0.20 -> 0.76 vs published 0.74). 'zscore' = the pilot's earlier, "
                         "non-native choice, kept for reproducing those runs.")
    ap.add_argument("--out", default=None, help="override the run directory (experiments)")
    ap.add_argument("--lr", type=float, default=None,
                    help="EXPERIMENT ONLY: override the native learning rate (recorded in config.json)")
    ap.add_argument("--grad-clip", type=float, default=None,
                    help="EXPERIMENT ONLY: clip global grad-norm to this value (recorded in config.json)")
    args = ap.parse_args()

    rec = NATIVE[args.family]
    fam_name = CANON[args.family]
    out = Path(args.out) if args.out else (
        REPO / "experiments" / "backbone_training" / args.dataset /
        f"{fam_name}__{args.variant}" / f"seed{args.seed}")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    stage(args.dataset)

    # Cap intra-op threads: if anything DOES fall back to CPU, it must not consume the
    # whole shared machine (observed: 211 cores at 0% GPU before the stray-tensor fix).
    torch.set_num_threads(8)
    dev = require_gpu(args.device)
    data, d_num, cats, info = load_tensors(args.dataset, norm=args.norm)
    norm_file = None
    if args.norm == "quantile":
        import joblib
        norm_file = out / "num_transform.joblib"
        joblib.dump(data["_qt"], norm_file)
    n_train = len(data["train"][0])
    spe = max(1, -(-n_train // rec["batch_size"]))
    steps = args.steps or args.epochs * spe

    seed_everything(args.seed)
    from tabequiv import ablate
    from tabequiv.models import get_family
    ft_kw = json.loads(args.ft_kw) if args.ft_kw else None
    if args.denoiser:
        if args.variant == "symmetric":
            raise SystemExit("--denoiser and --variant symmetric are mutually exclusive (the ablate cores are retired)")
        model = get_family(args.family).build(d_num, cats, seed=args.seed,
                                              denoiser=args.denoiser, ft_kw=ft_kw).to(dev)
    else:
        model = ablate.build(args.family, d_num, cats, seed=args.seed,
                             symmetric_core=(args.variant == "symmetric")).to(dev)
    _move_stray_tensors(model, dev)
    fam = get_family(args.family)
    lr = args.lr if args.lr is not None else rec["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=rec["wd"])
    ema = None
    if rec["ema"]:
        from tabequiv.harness import EMA
        ema = EMA(model, beta=rec["ema"], update_after_step=0, update_every=1)

    # Per-family batch contract. The wrappers do NOT share one signature:
    #   x_num/x_cat  -> unimod_tabdiff, unimod_efvfm, tabsyn_vae
    #   x1, t        -> tabvvfm (concat [x_cont | one-hots]), tabflow_net (VAE latent)
    #   z, t, sigma  -> tabsyn_diffusion (VAE latent)
    # The two LATENT families consume the frozen TabSyn-VAE embedding, so real latents
    # are encoded once here rather than fed random noise.
    latent = None
    if args.family in ("tabflow_net", "tabsyn_diffusion"):
        latent = _encode_latents(data, d_num, cats, dev, args.seed, args.dataset)

    def batch(split, bs, gen):
        xn, xc = data[split]
        i = torch.randint(0, len(xn), (min(bs, len(xn)),), generator=gen)
        xn_b, xc_b = xn[i].to(dev), xc[i].to(dev)
        t = torch.rand(len(i), device=dev)
        if args.family in ("tabflow_net", "tabsyn_diffusion"):
            z = latent[split][i].to(dev)
            if args.family == "tabsyn_diffusion":
                return {"z": z, "t": t, "sigma": (t * 1.2 - 1.2).exp()}
            return {"x1": z, "t": t}
        if args.family == "tabvvfm":
            oh = torch.cat([torch.nn.functional.one_hot(xc_b[:, j], num_classes=k)
                            for j, k in enumerate(cats)], dim=1).float()
            return {"x1": torch.cat([xn_b, oh], dim=1), "t": t}
        return {"x_num": xn_b, "x_cat": xc_b}

    g = torch.Generator().manual_seed(args.seed)
    gv = torch.Generator().manual_seed(args.seed + 777)
    rows, t0 = [], time.time()
    model.train()
    for step in range(1, steps + 1):
        loss = fam.loss_fn(model, batch("train", rec["batch_size"], g))["total"]
        opt.zero_grad(); loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if ema is not None:
            ema.update(model)
        if step % args.ckpt_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([
                    float(fam.loss_fn(model, batch("val", rec["batch_size"], gv))["total"])
                    for _ in range(3)]))
            model.train()
            src = ema.ema_model if ema is not None else model
            torch.save({"step": step, "ema" if ema is not None else "raw":
                        src.state_dict()}, out / "checkpoints" / f"step_{step:07d}.pt")
            rows.append({"step": step, "epoch": round(step / spe, 3),
                         "train_loss": float(loss.item()), "val_loss": vl})
            pd.DataFrame(rows).to_csv(out / "train_loss.csv", index=False)
            print(f"  step {step}/{steps} train={loss.item():.4f} val={vl:.4f}", flush=True)

    train_secs = time.time() - t0
    ck = sorted((out / "checkpoints").glob("*.pt"))
    sizes = [p.stat().st_size for p in ck]
    (out / "storage.json").write_text(json.dumps({
        "n_checkpoints": len(ck),
        "bytes_per_checkpoint_mean": int(np.mean(sizes)) if sizes else 0,
        "total_bytes": int(sum(sizes)),
        "total_mb": round(sum(sizes) / 1e6, 2)}, indent=2) + "\n")
    (out / "timing.json").write_text(json.dumps({
        "train_seconds": round(train_secs, 1), "steps": steps,
        "host": os.uname().nodename,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}, indent=2) + "\n")
    (out / "config.json").write_text(json.dumps({
        "family": fam_name, "family_key": args.family, "variant": args.variant,
        "denoiser": args.denoiser, "ft_kw": ft_kw,          # None = the family's official net
        "dataset": args.dataset, "seed": args.seed,
        "d_num": d_num, "categories": cats,
        "native_recipe": rec, "epoch_budget": args.epochs,
        "overrides": {"lr": args.lr, "grad_clip": args.grad_clip},   # None = native
        "steps_per_epoch": spe, "steps_run": steps,
        "loop_note": "native loss + native hyperparameters; optimizer loop is ours "
                     "(the baselines' own loops are CLI/data-pipeline bound)",
        # The z-score stats this run normalized with. Eval MUST invert with these --
        # inverting with tab-ddpm's quantile transform instead scored real data at
        # Shape 0.564 (vs ~0.98 correct), which looked like a model failure.
        "norm_scheme": args.norm,
        "norm_file": str(norm_file) if norm_file else None,
        "norm_mean": [float(v) for v in data["_norm"][0]],
        "norm_std": [float(v) for v in data["_norm"][1]]},
        indent=2) + "\n")
    print(f"DONE {len(ck)} checkpoints, {sum(sizes)/1e6:.1f} MB, "
          f"train {train_secs/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
