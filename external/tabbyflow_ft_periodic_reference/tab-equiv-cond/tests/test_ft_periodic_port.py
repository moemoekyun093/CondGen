"""The vendored AA_code ``ft_periodic`` denoiser: bit-exact vs its source, and symmetric.

Four claims, each a test:
  1. BIT-EXACT (CPU): under the same seed, ``tabequiv.models.ft_periodic.UniModMLPFTPeriodic``
     and AA_code's ``tabdiff.modules.main_modules.UniModMLP`` have identical state_dicts,
     identical forward outputs, identical loss and identical gradients (``torch.equal``).
  2. BIT-EXACT through the FULL wrapper: ``build(denoiser="ft_periodic")`` and an AA-net
     inside the pristine ``Model``/``UnifiedCtimeDiffusion`` give the same ``loss_fn``.
  3. PARAM COUNT at the shoppers schema equals the author's 1,469,611 (L6/d128/h8/f4).
  4. EQUIVARIANCE: whole-column permutation with per-column params moved ->
     E ~ float epsilon (float32 <= 1e-5, float64 <= 1e-12); the control that does NOT
     move per-column params is O(1).
Plus: the default build is still the pristine ``UniModMLP`` (nothing golden-tested moved).

AA_code and the pristine baseline share the top-level package name ``tabdiff``; the AA
classes are imported with ``sys.modules`` purged before/after and only the class
objects are kept, so the two never coexist in ``sys.modules``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tabequiv.models import unimod_tabdiff as fam
from tabequiv.models.ft_periodic import (AA_CODE_ROOT, FT_PERIODIC_KW,
                                         UniModMLPFTPeriodic)
from tabequiv.paths import purge_baseline_modules, use_baseline
from tabequiv.perturb import (block_permutation_indices, column_permutation,
                              permute_batch, permute_model_params)
from tabequiv.repro import seed_everything

AA = Path(AA_CODE_ROOT)
requires_aa = pytest.mark.skipif(not (AA / "tabdiff" / "modules" / "main_modules.py").exists(),
                                 reason=f"AA_code not available at {AA}")

# shoppers schema (data/shoppers/info.json): 10 numericals, 8 categoricals incl. target
SHOPPERS_D_NUM, SHOPPERS_CATS = 10, [2, 10, 8, 13, 9, 20, 3, 2]
SMALL_D_NUM, SMALL_CATS = 3, [4, 5]
SMALL_KW = dict(FT_PERIODIC_KW, num_layers=2, d_token=32, n_head=4)   # fast but same code paths


def aa_classes():
    """AA_code's UniModMLP (ft_periodic) class object, imported in isolation."""
    purge_baseline_modules()
    sys.path.insert(0, str(AA))
    try:
        from tabdiff.modules.main_modules import UniModMLP as AA_UniModMLP  # noqa: N814
    finally:
        sys.path.remove(str(AA))
        purge_baseline_modules()
    return AA_UniModMLP


def onehot_with_mask(x_codes: torch.Tensor, cats) -> torch.Tensor:
    """[B, m] codes -> [B, sum(K+1)] one-hot in the diffusion wrapper's layout."""
    return torch.cat([torch.nn.functional.one_hot(x_codes[:, j], num_classes=k + 1).float()
                      for j, k in enumerate(cats)], dim=1)


def inputs(d_num, cats, b=16, seed=1):
    seed_everything(seed)
    x_num = torch.randn(b, d_num)
    codes = torch.stack([torch.randint(0, k, (b,)) for k in cats], dim=1)
    t = torch.rand(b)
    return x_num, codes, onehot_with_mask(codes, cats), t


# ---------------------------------------------------------------------------
@requires_aa
def test_bit_exact_vs_aa_code_net():
    AA_UniModMLP = aa_classes()
    cats_m = [k + 1 for k in SHOPPERS_CATS]
    seed_everything(0); aa = AA_UniModMLP(SHOPPERS_D_NUM, cats_m, **SMALL_KW)
    seed_everything(0); ours = UniModMLPFTPeriodic(SHOPPERS_D_NUM, cats_m, **SMALL_KW)

    sa, so = aa.state_dict(), ours.state_dict()
    assert list(sa) == list(so), "state_dict keys/order differ -> not a verbatim port"
    for k in sa:
        assert torch.equal(sa[k], so[k]), f"param {k} differs at init (RNG order changed?)"

    x_num, _, x_cat, t = inputs(SHOPPERS_D_NUM, SHOPPERS_CATS)
    out_a = aa(x_num, x_cat, t); out_o = ours(x_num, x_cat, t)
    assert torch.equal(out_a[0], out_o[0]) and torch.equal(out_a[1], out_o[1])

    la = out_a[0].pow(2).mean() + out_a[1].pow(2).mean(); la.backward()
    lo = out_o[0].pow(2).mean() + out_o[1].pow(2).mean(); lo.backward()
    assert torch.equal(la, lo)
    for (ka, pa), (ko, po) in zip(aa.named_parameters(), ours.named_parameters()):
        assert ka == ko and torch.equal(pa.grad, po.grad), f"grad differs: {ka}"


@requires_aa
def test_bit_exact_through_full_wrapper():
    """Same loss from build(denoiser='ft_periodic') and from the AA net inside the
    pristine Model/UnifiedCtimeDiffusion (which are byte-identical in both repos)."""
    AA_UniModMLP = aa_classes()
    ours = fam.build(SMALL_D_NUM, SMALL_CATS, seed=0, denoiser="ft_periodic", ft_kw=SMALL_KW)
    with use_baseline("tabdiff"):
        from tabdiff.modules.main_modules import Model
        from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
        seed_everything(0)
        net = AA_UniModMLP(d_numerical=SMALL_D_NUM,
                           categories=[c + 1 for c in SMALL_CATS], **SMALL_KW)
        ref = UnifiedCtimeDiffusion(
            num_classes=np.array(SMALL_CATS), num_numerical_features=SMALL_D_NUM,
            denoise_fn=Model(net, **fam.EDM_PARAMS), y_only_model=None,
            edm_params=dict(fam.EDM_PARAMS), noise_dist_params=dict(fam.NOISE_DIST_PARAMS),
            noise_schedule_params=dict(fam.NOISE_SCHEDULE_PARAMS),
            sampler_params=dict(fam.SAMPLER_PARAMS), device=torch.device("cpu"),
            **fam.DIFFUSION_KW)
    batch = fam.make_batch(SMALL_D_NUM, SMALL_CATS, batch_size=8, seed=1)
    seed_everything(2); lo = fam.loss_fn(ours, batch)["total"]
    seed_everything(2); lr = fam.loss_fn(ref, batch)["total"]
    assert torch.equal(lo, lr), f"wrapper loss differs: {lo.item()} vs {lr.item()}"


def test_param_count_matches_author():
    net = UniModMLPFTPeriodic(SHOPPERS_D_NUM, [k + 1 for k in SHOPPERS_CATS], **FT_PERIODIC_KW)
    assert sum(p.numel() for p in net.parameters()) == 1_469_611


def test_default_build_is_still_pristine():
    m = fam.build(SMALL_D_NUM, SMALL_CATS, seed=0)
    net = m._denoise_fn.denoise_fn_D.denoise_fn_F
    assert type(net).__name__ == "UniModMLP" and hasattr(net, "mlp"), \
        "default build must remain the golden-tested pristine UniModMLP"
    m2 = fam.build(SMALL_D_NUM, SMALL_CATS, seed=0, denoiser="ft_periodic", ft_kw=SMALL_KW)
    assert isinstance(m2._denoise_fn.denoise_fn_D.denoise_fn_F, UniModMLPFTPeriodic)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
def test_whole_column_equivariance(dtype, tol):
    d_num, cats = SHOPPERS_D_NUM, SHOPPERS_CATS
    model = fam.build(d_num, cats, seed=0, denoiser="ft_periodic", ft_kw=SMALL_KW)
    perm = column_permutation(d_num, cats, seed=3)
    perm_num, perm_cat = perm
    model_p = permute_model_params("unimod_tabdiff", model, perm)
    net, net_p = (m._denoise_fn.denoise_fn_D.denoise_fn_F.to(dtype).eval()
                  for m in (model, model_p))

    x_num, codes, _, t = inputs(d_num, cats)
    x_num_p, codes_p = permute_batch(x_num, codes, perm)
    cats_p = [cats[int(j)] for j in perm_cat]
    x_cat = onehot_with_mask(codes, cats).to(dtype); x_cat_p = onehot_with_mask(codes_p, cats_p).to(dtype)
    x_num, x_num_p, t = x_num.to(dtype), x_num_p.to(dtype), t.to(dtype)

    with torch.no_grad():
        n0, c0 = net(x_num, x_cat, t)
        n1, c1 = net_p(x_num_p, x_cat_p, t)
    # f(Px) must equal P f(x): compare the permuted-run outputs against the original
    # outputs re-ordered into the permuted layout
    idx = block_permutation_indices([k + 1 for k in cats], perm_cat)
    pn = torch.as_tensor(np.asarray(perm_num))
    e_num = (n1 - n0[:, pn]).norm() / n0.norm()
    e_cat = (c1 - c0[:, idx]).norm() / c0.norm()
    assert e_num <= tol and e_cat <= tol, f"NOT equivariant: E_num={e_num:.3e} E_cat={e_cat:.3e}"

    # CONTROL: build at the permuted schema but do NOT move per-column params -> must break
    ctrl = fam.build(d_num, cats_p, seed=0, denoiser="ft_periodic", ft_kw=SMALL_KW)
    with torch.no_grad():
        nc, _ = ctrl._denoise_fn.denoise_fn_D.denoise_fn_F.to(dtype).eval()(x_num_p, x_cat_p, t)
    assert (nc - n0[:, pn]).norm() / n0.norm() > 1e-2, "control did not break -> test is vacuous"
