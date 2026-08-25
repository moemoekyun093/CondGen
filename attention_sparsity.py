"""
Attention sparsity diagnostic for the FT-Transformer TabDiff denoiser.

Measures attention CONCENTRATION over feature tokens across noise levels, for one or
more datasets, and prints a cross-dataset summary.

  - entropy H(a) of each attention row (over F feature keys). Low H = concentrated.
  - effective support exp(H) = "how many features is this query effectively attending to?"
      exp(H) = F  <=> uniform (dense);   exp(H) = 1 <=> one feature (maximally sparse)

USAGE (from TabDiff repo root):
    # all datasets, seed 0:
    python attention_sparsity.py --seed 0 --gpu 0
    # a subset:
    python attention_sparsity.py --datasets news diabetes --seed 0 --gpu 0
"""
import argparse
import glob
import math
import os
import pickle
import json

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Attention capture
# --------------------------------------------------------------------------------------
class AttentionCapture:
    """
    Hooks each FTBlock's MultiheadAttention and recomputes the attention probability
    matrix from the captured inputs using the module's own W_q / W_k and head reshaping
    (the module doesn't return its weights). Reproduces the module's forward exactly.
    """
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.attn_probs = []

    def _make_hook(self):
        def hook(module, inputs, output):
            x_q = inputs[0]
            x_kv = inputs[1] if len(inputs) > 1 else inputs[0]
            q = module.W_q(x_q)
            k = module.W_k(x_kv)
            B, F_tok, D = q.shape
            n_heads = module.n_heads
            d_head = D // n_heads

            def split(t):
                b, n, d = t.shape
                return t.reshape(b, n, n_heads, d_head).transpose(1, 2).reshape(b * n_heads, n, d_head)

            qh, kh = split(q), split(k)
            logits = qh @ kh.transpose(1, 2) / math.sqrt(d_head)
            probs = F.softmax(logits, dim=-1).reshape(B, n_heads, F_tok, F_tok)
            self.attn_probs.append(probs.detach().cpu())
        return hook

    def __enter__(self):
        for blk in self.model.blocks:
            self.handles.append(blk.attention.register_forward_hook(self._make_hook()))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self):
        self.attn_probs = []


# --------------------------------------------------------------------------------------
# Concentration metrics
# --------------------------------------------------------------------------------------
def entropy(p, eps=1e-12):
    return -(p * (p + eps).log()).sum(-1)


def gini(p):
    p_sorted, _ = p.sort(dim=-1)
    n = p.shape[-1]
    idx = torch.arange(1, n + 1, dtype=p.dtype, device=p.device)
    return ((2 * idx - n - 1) * p_sorted).sum(-1) / (n * p_sorted.sum(-1) + 1e-12)


def topk_mass(p, k):
    v, _ = p.topk(k, dim=-1)
    return v.sum(-1)


def analyze(probs, n_features, topk=(1, 3, 5)):
    B, H, Fq, Fk = probs.shape
    flat = probs.reshape(-1, Fk)
    H_ent = entropy(flat)
    eff_support = H_ent.exp()
    g = gini(flat)
    out = {
        "entropy": H_ent.mean().item(),
        "eff_support": eff_support.mean().item(),
        "eff_support_frac": (eff_support.mean() / Fk).item(),
        "gini": g.mean().item(),
    }
    for k in topk:
        if k <= Fk:
            out[f"top{k}_mass"] = topk_mass(flat, k).mean().item()
    return out


# --------------------------------------------------------------------------------------
# Per-dataset analysis
# --------------------------------------------------------------------------------------
def analyze_one_dataset(dataname, ckpt_path, device, args):
    """Full analysis for one (dataset, checkpoint). Returns summary dict or None if skipped."""
    ckpt_dir = os.path.dirname(ckpt_path)
    with open(os.path.join(ckpt_dir, "config.pkl"), "rb") as f:
        raw_config = pickle.load(f)

    from tabdiff.modules.main_modules import UniModMLP

    dt = raw_config["unimodmlp_params"].get("denoiser_type", "ft_periodic")
    if dt != "ft_periodic":
        print(f"[skip] {dataname}: denoiser_type='{dt}' (not ft_periodic)")
        return None

    model = UniModMLP(**raw_config["unimodmlp_params"]).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    sd = state["denoise_fn"] if "denoise_fn" in state else state
    cleaned = {k.replace("denoise_fn_D.denoise_fn_F.", "").replace("denoise_fn_D.", ""): v
               for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)

    # ---- data (working loader) ----
    from utils_train import TabDiffDataset
    with open(f"data/{dataname}/info.json", "r") as f:
        data_info = json.load(f)
    ds = TabDiffDataset(
        dataname, f"data/{dataname}", data_info, isTrain=False,
        dequant_dist=raw_config["data"].get("dequant_dist", "none"),
        int_dequant_factor=raw_config["data"].get("int_dequant_factor", 0),
    )
    n = min(args.n_samples, len(ds))
    X = torch.stack([ds[i] for i in range(n)]).to(device)
    d_numerical = ds.d_numerical

    # IMPORTANT: use the MODEL's categorical widths, not the dataset's.
    # TabDiff's tokenizer is built for categories+mask; raw_config carries the widths the
    # model actually expects. Using ds.categories caused +1 / high-cardinality mismatches.
    model_categories = list(raw_config["unimodmlp_params"]["categories"])
    n_cat = len(model_categories)
    n_features = d_numerical + n_cat

    x_num = X[:, :d_numerical].float()
    x_cat_int = X[:, d_numerical:d_numerical + n_cat].long()

    # Clamp each categorical index into its model width [0, width-1] so out-of-range /
    # non-zero-based encodings (e.g. diabetes's high-cardinality diagnosis codes) can't
    # overflow the one-hot. This slightly mis-encodes any out-of-range values, but it
    # affects only the offending columns and lets the attention analysis proceed.
    cat_cols = []
    for i in range(n_cat):
        w = model_categories[i]
        col = x_cat_int[:, i].clamp(0, w - 1)
        cat_cols.append(F.one_hot(col, w).float())
    x_cat_oh = torch.cat(cat_cols, dim=-1).to(device) if n_cat > 0 else torch.zeros(n, 0, device=device)

    n_layers = len(model.blocks)
    print("\n" + "=" * 78)
    print(f"DATASET: {dataname}   (F={n_features} features, {n_layers} layers, "
          f"uniform eff_support={n_features})")
    print(f"  checkpoint: {os.path.basename(ckpt_path)}   n_samples={n}")
    # flag any columns whose raw indices exceeded the model width (were clamped)
    for i in range(n_cat):
        raw_max = int(x_cat_int[:, i].max())
        if raw_max >= model_categories[i]:
            print(f"  [caveat] cat col {i}: raw max index {raw_max} >= model width "
                  f"{model_categories[i]} -> clamped (encoding approximate for this column)")
    print("=" * 78)

    per_sigma = {}
    with AttentionCapture(model) as cap:
        for sigma in args.sigmas:
            cap.reset()
            x_noisy = x_num + torch.randn_like(x_num) * sigma
            t = torch.full((n,), float(sigma), device=device)
            with torch.no_grad():
                model(x_noisy, x_cat_oh, t)

            layer_fracs, layer_gini = [], []
            for probs in cap.attn_probs:
                m = analyze(probs, n_features)
                layer_fracs.append(m["eff_support_frac"])
                layer_gini.append(m["gini"])
            mean_frac = float(np.mean(layer_fracs))
            mean_gini = float(np.mean(layer_gini))
            per_sigma[sigma] = mean_frac
            print(f"  sigma={sigma:<5} mean eff-support frac = {mean_frac:.3f}  "
                  f"(~{mean_frac*n_features:.1f}/{n_features} feats)   gini={mean_gini:.3f}")

    return {"n_features": n_features, "per_sigma": per_sigma}
# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["adult", "default", "shoppers", "magic", "beijing", "diabetes", "news"])
    ap.add_argument("--seed", type=int, default=0, help="which trained seed to analyze")
    ap.add_argument("--exp_prefix", default="ft_periodic", help="exp_name prefix before _seed{N}")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=512)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0])
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu")

    results = {}
    for dataname in args.datasets:
        exp_name = f"{args.exp_prefix}_seed{args.seed}"
        ckpt_dir = f"tabdiff/ckpt/{dataname}/{exp_name}"
        # try the exact prefix first
        matches = sorted(glob.glob(f"{ckpt_dir}/best_ema_model_*.pt"))
        # fall back to ANY ft_periodic exp folder for this seed (handles news's L6_d128 naming)
        if not matches:
            matches = sorted(glob.glob(f"tabdiff/ckpt/{dataname}/*ft_periodic*_seed{args.seed}/best_ema_model_*.pt"))
        # last resort: model_*.pt (non-best, still wrapped) under either naming
        if not matches:
            matches = sorted(glob.glob(f"{ckpt_dir}/model_*.pt"))
        if not matches:
            matches = sorted(glob.glob(f"tabdiff/ckpt/{dataname}/*ft_periodic*_seed{args.seed}/model_*.pt"))
        if not matches:
            print(f"\n[skip] {dataname}: no checkpoint found for seed {args.seed}")
            continue
        try:
            res = analyze_one_dataset(dataname, matches[0], device, args)
            if res is not None:
                results[dataname] = res
        except Exception as e:
            print(f"\n[error] {dataname}: {type(e).__name__}: {e}")

    # ---- cross-dataset summary ----
    print("\n" + "#" * 78)
    print("# CROSS-DATASET SUMMARY: mean effective-support fraction (frac of F)")
    print("#   low = concentrated/selective ; ~1.0 = dense/uniform")
    print("#" * 78)
    header = f"{'dataset':<12} {'F':>4} | " + " ".join(f"s={s:<5}" for s in args.sigmas)
    print(header)
    print("-" * len(header))
    for dataname, res in results.items():
        row = f"{dataname:<12} {res['n_features']:>4} | "
        row += " ".join(f"{res['per_sigma'].get(s, float('nan')):<7.3f}" for s in args.sigmas)
        print(row)
    print("-" * len(header))
    print("\nRead across each row: does frac rise with sigma (input-adaptive selection —")
    print("concentrate when input is informative, spread when it's noise) or stay flat")
    print("(fixed pattern)? Compare low-sigma frac across datasets: lower = more selective.")


if __name__ == "__main__":
    main()