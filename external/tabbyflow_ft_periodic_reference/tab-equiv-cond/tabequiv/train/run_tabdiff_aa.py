#!/usr/bin/env python
"""TabDiff (original or AA's ft_periodic core) trained with AA_code's/TabDiff's OWN recipe,
inside our framework, on data90.  Purpose: show that OUR port reproduces AA_code's numbers.

Mirrors ``AA_code/TabDiff/tabdiff/trainer.py`` (verified 2026-08-25) step for step:
  * data: quantile-normalised numericals (TRAIN-fit), ordinal categoricals; the model
    sees ``x = [x_num | cat codes]`` exactly as their DataLoader feeds it
    (``batch_size=4096, shuffle=True``); training set = ``trainval`` (their 90% train)
    unless ``--train-on core`` (our 81% core).
  * EPOCH loop (their ``steps`` are epochs): loss = d_loss + closs_weight * c_loss with
    ``closs_weight = c_lambda*(1 - epoch/epochs)`` ("anneal"), AdamW(lr 1e-3, wd 0),
    ReduceLROnPlateau(factor .9, patience 50) stepped on the epoch's mean train loss,
    EMA(0.997) of ALL parameters (denoiser + learnable noise schedules) once per epoch.
  * selection: after epoch 4000, the lowest UNWEIGHTED (d + c) train loss of the EMA
    model (recomputed on the full training set every epoch, as they do) ->
    ``checkpoints/best_ema.pt``; also best raw, every-2000-epoch and final checkpoints.
  * protocol (user, 2026-08-25): TRAIN ONCE, then GENERATE MANY -- evaluation is done
    afterwards by ``eval_checkpoints.py`` with N sampling seeds on the selected
    checkpoint (``--n-sampling-seeds 20``) and by ``scripts/score_vs_train_ref.py``.

Noise schedules: per-column LEARNABLE (AA's default run), like theirs.
Deviations from AA_code are ONLY: 10k epochs (theirs 8k) and running in this framework.
Everything is written to ``config.json`` so eval rebuilds the exact architecture.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from tabequiv.harness import EMA                              # noqa: E402
from tabequiv.models import unimod_tabdiff as fam            # noqa: E402
from tabequiv.models.ft_periodic import FT_PERIODIC_KW        # noqa: E402
from tabequiv.repro import seed_everything                    # noqa: E402
from tabequiv.splits import load_frame                        # noqa: E402
from tabequiv.splits90 import load_split90                    # noqa: E402
from tabequiv.train.run_generic import _move_stray_tensors, require_gpu  # noqa: E402

# ef-vfm (official TabbyFlow) adds to the same toml: max_grad_norm = 1.0 and a 100-epoch linear LR warmup
# (ef_vfm/configs/ef_vfm_configs.toml:28-29; trainer._run_step clips, warmup replaces the plateau step).
FAMILY_TWEAKS = {"unimod_tabdiff": dict(max_grad_norm=0.0, warmup_epochs=0),
                 "unimod_efvfm": dict(max_grad_norm=1.0, warmup_epochs=100)}
AA = dict(epochs=10_000, lr=1e-3, weight_decay=0.0, batch_size=4096, ema_decay=0.997,
          lr_factor=0.90, lr_patience=50, c_lambda=1.0, d_lambda=1.0,
          closs_weight_schedule="anneal", select_after_epoch=4000, ckpt_every_epochs=2000)


def load_tensors_aa(dataset: str, train_on: str):
    """Quantile-normalise (fit on the TRAINING set), ordinal-code categoricals."""
    from sklearn.preprocessing import QuantileTransformer
    df, info = load_frame(dataset)
    sp = load_split90(dataset)
    num_idx = list(info["num_col_idx"]); cat_idx = list(info["cat_col_idx"]); tgt = list(info["target_col_idx"])
    if info.get("task_type") == "regression": num_idx = sorted(set(num_idx) | set(tgt))
    else: cat_idx = sorted(set(cat_idx) | set(tgt))
    cols = list(df.columns)
    levels = {i: {v: k for k, v in enumerate(sorted(set(df[cols[i]].astype(str))))} for i in cat_idx}
    tr = df.iloc[sp[train_on]]
    xn_tr = np.nan_to_num(tr.iloc[:, num_idx].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64))
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=max(min(len(xn_tr) // 30, 1000), 10),
                             subsample=int(1e9), random_state=0).fit(xn_tr)
    out = {}
    for split in ("train", "val", "test", "trainval"):
        sub = df.iloc[sp[split]]
        xn = np.nan_to_num(sub.iloc[:, num_idx].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64))
        xn = qt.transform(xn) if xn.shape[1] else xn
        xc = (np.stack([sub[cols[i]].astype(str).map(levels[i]).to_numpy() for i in cat_idx], 1)
              if cat_idx else np.zeros((len(sub), 0)))
        out[split] = (torch.tensor(xn, dtype=torch.float32), torch.tensor(xc, dtype=torch.long))
    return out, qt, len(num_idx), [len(levels[i]) for i in cat_idx], info


def mixed(model, xn, xc):
    d, c = model.mixed_loss(torch.cat([xn, xc.float()], 1))
    return d, c


@torch.no_grad()
def full_loss(model, xn, xc, bs, dev):
    """Unweighted mean (d + c) over a whole split -- AA's compute_loss()."""
    model.eval(); dsum = csum = 0.0; n = 0
    for i in range(0, len(xn), bs):
        d, c = mixed(model, xn[i:i+bs].to(dev), xc[i:i+bs].to(dev)); m = len(xn[i:i+bs])
        dsum += float(d) * m; csum += float(c) * m; n += m
    model.train(); return dsum / n, csum / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="shoppers")
    ap.add_argument("--family", choices=("unimod_tabdiff", "unimod_efvfm"), default="unimod_tabdiff",
                    help="unimod_efvfm = the OFFICIAL TabbyFlow (ef-vfm): same trainer lineage and toml as TabDiff")
    ap.add_argument("--denoiser", choices=("original", "ft_periodic"), default="original")
    ap.add_argument("--train-on", choices=("trainval", "train"), default="trainval",
                    help="trainval = AA's 90%% training set (default); train = our 81%% core")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=AA["epochs"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import importlib
    fam = importlib.import_module(f"tabequiv.models.{a.family}")
    canon = {"unimod_tabdiff": "TabDiff", "unimod_efvfm": "TabbyFlow"}[a.family]
    tw = FAMILY_TWEAKS[a.family]
    tag = f"{canon}__{a.denoiser}"
    out = Path(a.out) if a.out else (REPO / "experiments" / "backbone_training" / a.dataset / f"{tag}__aa" / f"seed{a.seed}")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8); dev = require_gpu(a.device)

    data, qt, d_num, cats, info = load_tensors_aa(a.dataset, a.train_on)
    import joblib; joblib.dump(qt, out / "num_transform.joblib")
    xn_tr, xc_tr = data[a.train_on]
    n_train = len(xn_tr); bs = AA["batch_size"]; spe = -(-n_train // bs)

    seed_everything(a.seed)
    ft_kw = dict(FT_PERIODIC_KW) if a.denoiser == "ft_periodic" else None
    # AA/TabDiff default run = per-column LEARNABLE noise schedules (tabdiff/main.py:236-238)
    build_kw = dict(learnable_schedule=True) if a.family == "unimod_tabdiff" else {}
    model = fam.build(d_num, cats, seed=a.seed, denoiser=a.denoiser, ft_kw=ft_kw, **build_kw).to(dev)
    _move_stray_tensors(model, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=AA["lr"], weight_decay=AA["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=AA["lr_factor"], patience=AA["lr_patience"])
    ema = EMA(model, beta=AA["ema_decay"], update_after_step=0, update_every=1)

    def save(name, state, epoch, extra=None):
        torch.save({"step": epoch, "epoch": epoch, "ema": ema.ema_model.state_dict(), "raw": model.state_dict(),
                    **(extra or {})}, out / "checkpoints" / name)

    g = torch.Generator().manual_seed(a.seed)
    rows, best_raw, best_ema, t0 = [], np.inf, np.inf, time.time()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{tag} on {a.dataset} ({a.train_on}, n={n_train}, {spe} steps/epoch, {n_params:,} params)", flush=True)
    model.train()
    for epoch in range(a.epochs):
        cw = AA["c_lambda"] * (1 - epoch / a.epochs) if AA["closs_weight_schedule"] == "anneal" else AA["c_lambda"]
        perm = torch.randperm(n_train, generator=g); dsum = csum = 0.0
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]; xn, xc = xn_tr[idx].to(dev), xc_tr[idx].to(dev)
            d, c = mixed(model, xn, xc); loss = AA["d_lambda"] * d + cw * c
            opt.zero_grad(); loss.backward()
            if tw["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tw["max_grad_norm"])
            opt.step()
            dsum += float(d) * len(idx); csum += float(c) * len(idx)
        mloss, gloss = dsum / n_train, csum / n_train; total = mloss + gloss
        if not np.isfinite(gloss): print("NaN in loss -- stopping like AA does"); break
        if tw["warmup_epochs"] > 0 and (epoch + 1) <= tw["warmup_epochs"]:
            for pg in opt.param_groups: pg["lr"] = AA["lr"] * (epoch + 1) / tw["warmup_epochs"]
        else:
            sched.step(total)
        ema.update(model)
        ep = epoch + 1; row = {"epoch": ep, "lr": opt.param_groups[0]["lr"], "closs_weight": cw,
                               "train_d": mloss, "train_c": gloss, "train_total": total}
        if ep > AA["select_after_epoch"]:
            if total < best_raw: best_raw = total; save("best_raw.pt", None, ep, {"train_total": total})
            ed, ec = full_loss(ema.ema_model, xn_tr, xc_tr, bs, dev); etot = ed + ec
            row.update({"ema_d": ed, "ema_c": ec, "ema_total": etot})
            if etot < best_ema: best_ema = etot; save("best_ema.pt", None, ep, {"ema_total": etot})
        if ep % AA["ckpt_every_epochs"] == 0 or ep == a.epochs:
            save(f"step_{ep:07d}.pt", None, ep)
            vd, vc = full_loss(ema.ema_model, *data["val"], bs, dev); row.update({"val_ema_total": vd + vc})
            et = f"{row['ema_total']:.4f}" if "ema_total" in row else f"n/a(select>{AA['select_after_epoch']})"
            print(f"  epoch {ep}/{a.epochs} train={total:.4f} ema_train={et} "
                  f"val_ema={vd+vc:.4f} lr={row['lr']:.2e} best_ema={best_ema:.4f}", flush=True)
        rows.append(row)
        if ep % 500 == 0: pd.DataFrame(rows).to_csv(out / "train_loss.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "train_loss.csv", index=False)

    (out / "config.json").write_text(json.dumps({
        "family": canon, "family_key": a.family, "variant": a.denoiser,
        "family_tweaks": tw,
        "denoiser": a.denoiser, "ft_kw": ft_kw, "learnable_schedule": a.family == "unimod_tabdiff",
        "dataset": a.dataset, "seed": a.seed,
        "d_num": d_num, "categories": cats, "protocol": "aa_faithful", "train_on": a.train_on,
        "n_train": n_train, "recipe": AA, "epochs_run": ep, "best_raw_train_loss": best_raw,
        "best_ema_train_loss": best_ema, "norm_scheme": "quantile",
        "norm_file": str(out / "num_transform.joblib"), "selected_checkpoint": "checkpoints/best_ema.pt"},
        indent=2) + "\n")
    ck = sorted((out / "checkpoints").glob("*.pt")); sizes = [p.stat().st_size for p in ck]
    (out / "storage.json").write_text(json.dumps({"n_checkpoints": len(ck), "total_mb": round(sum(sizes) / 1e6, 2)}, indent=2) + "\n")
    (out / "timing.json").write_text(json.dumps({"train_seconds": round(time.time() - t0, 1), "epochs": ep,
                                                  "host": os.uname().nodename}, indent=2) + "\n")
    print(f"DONE {len(ck)} checkpoints, best_ema_train_loss={best_ema:.4f}, {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
