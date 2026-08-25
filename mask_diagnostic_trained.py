"""
Mask diagnostic for the TabNet-style TabDiff denoiser.

WHAT THIS ANSWERS: "Are the sparsemax masks learning anything meaningful, or are they
degenerate?" It loads a trained checkpoint, runs a validation batch through the denoiser
at several noise levels, extracts the per-step masks, and reports FOUR things, each tied
to a specific failure mode:

  1. SPARSITY        - are masks actually sparse (not collapsed to uniform/dense)?
  2. SPECIALIZATION  - do different steps select DIFFERENT features (prior_scale working)?
  3. COVERAGE        - is every feature selected by SOME step (no permanently-dead columns)?
  4. INSTANCE-DEPENDENCE - do masks vary across samples (data-driven), or are they fixed
                           regardless of input (which means selection is learning nothing)?

The last one is the sharpest "is it meaningful" test: a mask that ignores its input and
picks the same features for every row isn't doing instance-wise selection at all -- it has
collapsed to a static feature subset, which is strictly worse than dense processing.

USAGE (from the TabDiff repo root, so imports resolve):
    python mask_diagnostic_trained.py --dataname news --exp_name <exact_folder_name>
    # or point directly at a checkpoint:
    python mask_diagnostic_trained.py --dataname news --ckpt_path ckpt/news/<exp>/best_ema_model_....pt
"""
import argparse
import glob
import json
import os
import pickle
from copy import deepcopy

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# Mask extraction: mirrors UniModMLP.forward but returns the per-step masks.
# This must stay in sync with your actual UniModMLP.forward in tabdiff/modules/main_modules.py
# --------------------------------------------------------------------------------------
def forward_with_masks(raw_model, x_num_t, x_cat_oh, timesteps):
    """
    x_num_t:  (B, d_num)  already-noised numeric (or clean, for a clean probe)
    x_cat_oh: (B, sum(cardinality+1))  ONE-HOT categoricals incl. mask slot
    timesteps: (B,)
    """
    e = raw_model.tokenizer(x_num_t, x_cat_oh)
    x = e[:, 1:, :]

    emb = raw_model.map_noise(timesteps)
    emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
    t_emb = raw_model.time_embed(emb)

    B, Fn, _ = x.shape
    prior_scale = torch.ones(B, Fn, device=x.device)
    masks = []
    for step in raw_model.tabnet_steps:
        _, x, mask, prior_scale = step(x, prior_scale, t_emb)
        masks.append(mask.detach())
    return masks


def _get_raw_denoiser(diffusion):
    """Walk Model -> Precond -> UniModMLP. Confirmed chain from main.py."""
    m = diffusion._denoise_fn
    for attr in ("denoise_fn_D", "denoise_fn_F"):
        if hasattr(m, attr):
            m = getattr(m, attr)
    return m if hasattr(m, "tabnet_steps") else None


# --------------------------------------------------------------------------------------
# The diagnostics themselves
# --------------------------------------------------------------------------------------
def analyze_masks(masks, feature_names=None, eps=1e-6):
    """
    masks: list of (B, F) tensors (one per step), all from the SAME batch.
    Returns a dict of raw numbers plus prints an interpreted verdict.
    """
    n_steps = len(masks)
    B, Fn = masks[0].shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(Fn)]

    binar = [(m > eps).float() for m in masks]

    # 1. SPARSITY: effective # features selected per step (entropy-based, robust)
    eff_counts = []
    frac_nonzero = []
    for m in masks:
        frac_nonzero.append((m > eps).float().mean().item())
        p = m / (m.sum(dim=-1, keepdim=True) + 1e-8)
        ent = -(p * (p + 1e-8).log()).sum(dim=-1)
        eff_counts.append(ent.exp().mean().item())

    # 2. SPECIALIZATION: mean pairwise Jaccard between step masks (lower = more specialized)
    jac = []
    for i in range(n_steps):
        for j in range(i + 1, n_steps):
            a, b = binar[i], binar[j]
            inter = (a * b).sum(-1)
            union = ((a + b) > 0).float().sum(-1).clamp(min=1)
            jac.append((inter / union).mean().item())
    mean_jaccard = float(np.mean(jac)) if jac else 0.0

    # 3. COVERAGE: per-feature selection frequency across all (step, sample) pairs
    stacked = torch.stack(binar, 0)  # (n_steps, B, F)
    sel_freq = stacked.mean(dim=(0, 1)).cpu().numpy()  # (F,)
    dead = [feature_names[i] for i in range(Fn) if sel_freq[i] < 0.02]
    always = [feature_names[i] for i in range(Fn) if sel_freq[i] > 0.98]

    # 4. INSTANCE-DEPENDENCE: does the mask vary across samples, or is it static?
    #    For each step, measure how much the per-sample binary mask deviates from the
    #    step's most-common pattern. ~0 variation = static/degenerate selection.
    instance_var = []
    for b in binar:  # (B, F)
        # fraction of features whose selection is NOT unanimous across the batch
        col_mean = b.mean(dim=0)  # (F,) how often each feature is selected in this step
        # a feature is "input-dependent" if it's selected for some rows but not others
        undecided = ((col_mean > 0.02) & (col_mean < 0.98)).float().mean().item()
        instance_var.append(undecided)
    mean_instance_var = float(np.mean(instance_var))

    # ---------------- print raw table ----------------
    print(f"\n{'Step':>5} {'%nonzero':>10} {'eff#feats':>11} {'input-dep frac':>15}")
    print("-" * 45)
    for i in range(n_steps):
        print(f"{i:>5} {frac_nonzero[i]*100:>9.1f}% {eff_counts[i]:>11.2f} {instance_var[i]:>15.2f}")

    print(f"\nMean pairwise Jaccard across steps : {mean_jaccard:.3f}")
    print(f"Mean input-dependence fraction     : {mean_instance_var:.3f}")
    print(f"Dead features (selected <2%)       : {len(dead)}/{Fn}"
          + (f"  {dead}" if dead else ""))
    print(f"Always-on features (>98%)          : {len(always)}/{Fn}"
          + (f"  {always}" if always else ""))

    # ---------------- interpreted verdict ----------------
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    verdicts = []

    # sparsity check
    avg_eff = np.mean(eff_counts)
    if avg_eff > 0.8 * Fn:
        verdicts.append(("SPARSITY", "FAIL", f"masks are near-dense (eff {avg_eff:.1f}/{Fn}) "
                         "- sparsemax not actually sparse; logits likely too flat/large."))
    elif avg_eff < 1.2:
        verdicts.append(("SPARSITY", "WARN", f"masks near-degenerate (~1 feature/step). "
                         "Over-sparse; gamma may be too aggressive."))
    else:
        verdicts.append(("SPARSITY", "OK", f"healthy sparsity (~{avg_eff:.1f} of {Fn} feats/step)."))

    # specialization check
    if mean_jaccard > 0.6:
        verdicts.append(("SPECIALIZATION", "FAIL", f"steps highly redundant (Jaccard {mean_jaccard:.2f}) "
                         "- prior_scale not pushing steps apart; extra steps wasted."))
    elif mean_jaccard > 0.35:
        verdicts.append(("SPECIALIZATION", "WARN", f"moderate step overlap (Jaccard {mean_jaccard:.2f})."))
    else:
        verdicts.append(("SPECIALIZATION", "OK", f"steps specialize on different features (Jaccard {mean_jaccard:.2f})."))

    # coverage check
    if len(dead) > 0.25 * Fn:
        verdicts.append(("COVERAGE", "FAIL", f"{len(dead)}/{Fn} features NEVER selected "
                         "- those columns can't be reconstructed well. Lower gamma or raise n_steps."))
    elif len(dead) > 0:
        verdicts.append(("COVERAGE", "WARN", f"{len(dead)} features rarely/never selected: {dead}."))
    else:
        verdicts.append(("COVERAGE", "OK", "every feature is selected by some step."))

    # instance-dependence check -- THE key "is it meaningful" test
    if mean_instance_var < 0.05:
        verdicts.append(("INSTANCE-DEP", "FAIL", f"masks barely vary across samples ({mean_instance_var:.2f}) "
                         "- selection is STATIC, ignoring input. This is the degenerate case: "
                         "the mask learned a fixed feature subset, not instance-wise selection. "
                         "Strictly worse than dense processing."))
    elif mean_instance_var < 0.15:
        verdicts.append(("INSTANCE-DEP", "WARN", f"masks weakly input-dependent ({mean_instance_var:.2f})."))
    else:
        verdicts.append(("INSTANCE-DEP", "OK", f"masks vary with input ({mean_instance_var:.2f}) "
                         "- genuine instance-wise selection."))

    for name, status, msg in verdicts:
        flag = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
        print(f"{flag} {name}: {msg}")

    n_fail = sum(1 for _, s, _ in verdicts if s == "FAIL")
    n_warn = sum(1 for _, s, _ in verdicts if s == "WARN")
    print("\n" + "-" * 60)
    if n_fail == 0 and n_warn == 0:
        print("SUMMARY: Masks look healthy and meaningful. If quality still lags,")
        print("the bottleneck is likely capacity (scale n_steps/d_token) or interaction")
        print("modeling (encoder-hybrid), NOT the mask mechanism.")
    elif n_fail == 0:
        print(f"SUMMARY: Masks mostly OK ({n_warn} warnings). Address warnings before scaling.")
    else:
        print(f"SUMMARY: {n_fail} FAILURE(S) in the mask mechanism. Fix these FIRST --")
        print("scaling n_steps/d_token will NOT help until the masks are healthy.")
        print("Typical fixes: lower gamma (~1.2-1.3), add/raise an entropy regularizer on")
        print("the masks, or reconsider sparsemax vs a softer sigmoid gate.")
    print("-" * 60)

    return {
        "eff_counts": eff_counts,
        "mean_jaccard": mean_jaccard,
        "mean_instance_var": mean_instance_var,
        "sel_freq": dict(zip(feature_names, sel_freq.round(3).tolist())),
        "dead_features": dead,
    }


# --------------------------------------------------------------------------------------
# Checkpoint loading -- reconstructs the model exactly as main.py test-mode does
# --------------------------------------------------------------------------------------
def load_trained_diffusion(dataname, exp_name=None, ckpt_path=None, device="cpu"):
    # (imports here so the diagnostics functions above can be reused without the full repo)
    from tabdiff.modules.main_modules import UniModMLP, Model
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from utils_train import TabDiffDataset

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    if ckpt_path is None:
        assert exp_name is not None, "provide --exp_name or --ckpt_path"
        ckpt_arr = glob.glob(f"{curr_dir}/tabdiff/ckpt/{dataname}/{exp_name}/best_ema_model*")
        assert ckpt_arr, f"no best_ema_model* found under tabdiff/ckpt/{dataname}/{exp_name}"
        ckpt_path = ckpt_arr[0]
    print(f"Loading checkpoint: {ckpt_path}")

    # frozen config saved next to the ckpt during training
    config_path = os.path.join(os.path.dirname(ckpt_path), "config.pkl")
    with open(config_path, "rb") as f:
        raw_config = pickle.load(f)

    data_dir = f"data/{dataname}"
    with open(f"data/{dataname}/info.json") as f:
        info = json.load(f)

    val_data = TabDiffDataset(
        dataname, data_dir, info, y_only=False, isTrain=False,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )
    d_numerical, categories = val_data.d_numerical, val_data.categories

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories, num_numerical_features=d_numerical,
        denoise_fn=model, y_only_model=None,
        **raw_config["diffusion_params"], device=device,
    )
    state = torch.load(ckpt_path, map_location=device)
    diffusion._denoise_fn.load_state_dict(state["denoise_fn"])
    diffusion.to(device)
    diffusion.eval()

    return diffusion, val_data, d_numerical, categories, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataname", required=True)
    ap.add_argument("--exp_name", default=None)
    ap.add_argument("--ckpt_path", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_probe", type=int, default=512, help="rows to probe")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu"

    diffusion, val_data, d_numerical, categories, info = load_trained_diffusion(
        args.dataname, args.exp_name, args.ckpt_path, device
    )
    raw_model = _get_raw_denoiser(diffusion)
    assert raw_model is not None, "denoiser has no tabnet_steps -- is this the TabNet variant?"

    # probe batch: raw dataset tensors
    X = val_data.X[:args.n_probe].to(device)
    x_num = X[:, :d_numerical].float()
    x_cat = X[:, d_numerical:].long()   # integer indices, as stored

    feature_names = ([f"num_{i}" for i in range(d_numerical)]
                     + [f"cat_{i}" for i in range(len(categories))])

    for sigma in [0.1, 0.5, 1.0, 2.0]:
        print("\n" + "#" * 60)
        print(f"# NOISE LEVEL sigma = {sigma}")
        print("#" * 60)

        b = x_num.shape[0]
        sig = torch.full((b, 1), float(sigma), device=device)

        # numeric forward noise: x + sigma * eps  (matches mixed_loss)
        x_num_t = x_num + torch.randn_like(x_num) * sig if x_num.shape[1] > 0 else x_num

        # categorical: convert to the one-hot (incl. mask slot) the denoiser expects.
        # Use the clean one-hot (no absorbing noise) so masks reflect feature CONTENT,
        # not which columns happen to be masked out this draw. If you'd rather see the
        # noised distribution, use diffusion.q_xt(...) as in mixed_loss instead.
        if x_cat.shape[1] > 0:
            x_cat_oh = diffusion.to_one_hot(x_cat).to(x_num_t.dtype)
        else:
            x_cat_oh = x_cat.float()

        # t corresponding to this sigma is only needed for the timestep embedding;
        # use the schedule inverse so t_emb matches what the model saw in training.
        t = torch.full((b,), float(sigma), device=device)  # approximate; see note below

        with torch.no_grad():
            masks = forward_with_masks(raw_model, x_num_t, x_cat_oh, t)
        analyze_masks(masks, feature_names=feature_names)


if __name__ == "__main__":
    main()