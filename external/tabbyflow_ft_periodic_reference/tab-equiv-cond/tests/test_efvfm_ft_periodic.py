"""AA_code's symmetric core inside the OFFICIAL TabbyFlow (ef-vfm ExpVFM): pristine control arm,
contract, loss, sampler, whole-column equivariance."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tabequiv.models import unimod_efvfm as fam
from tabequiv.models.ft_periodic import UniModMLPFTPeriodic
from tabequiv.perturb import block_permutation_indices, column_permutation, permute_batch, permute_model_params

SMALL_KW = dict(num_layers=2, d_token=32, n_head=4, factor=2, n_frequencies=8)


def test_original_build_is_pristine():
    from tabequiv.paths import use_baseline
    from tabequiv.repro import seed_everything
    from tabequiv.models.unimod_efvfm import UNIMOD_KW
    d_num, cats = 3, [4, 5]
    ours = fam.build(d_num, cats, seed=0)
    with use_baseline("ef-vfm"):
        from ef_vfm.models.flow_model import ExpVFM
        from ef_vfm.modules.main_modules import UniModMLP
        seed_everything(0)
        ref = ExpVFM(num_classes=np.array(cats), num_numerical_features=d_num,
                     vf_fn=UniModMLP(d_numerical=d_num, categories=cats, **UNIMOD_KW), device="cpu")
    a, b = ours.state_dict(), ref.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_contract_loss_backward_sample():
    d_num, cats = 3, [4, 5]
    m = fam.build(d_num, cats, seed=0, denoiser="ft_periodic", ft_kw=SMALL_KW)
    assert isinstance(m._vf_fn, UniModMLPFTPeriodic)
    b = fam.make_batch(d_num, cats, batch_size=16)
    d, c = fam.forward_fn(m, b)
    assert d.shape == (16, d_num) and c.shape == (16, sum(cats))
    loss = fam.loss_fn(m, b)["total"]
    assert torch.isfinite(loss); loss.backward()
    assert any(p.grad is not None for p in m._vf_fn.parameters())
    with torch.no_grad():
        x = m.sample(8)                                        # ExpVFM's own ODE sampler via Velocity(_vf_fn)
    x = x[0] if isinstance(x, (tuple, list)) else x
    assert torch.isfinite(torch.as_tensor(x).float()).all()


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
def test_whole_column_equivariance(dtype, tol):
    d_num, cats = 3, [4, 5, 3]
    torch.set_default_dtype(dtype)
    try:
        m = fam.build(d_num, cats, seed=0, denoiser="ft_periodic", ft_kw=SMALL_KW).to(dtype)
        perm = column_permutation(d_num, cats, seed=3); perm_num, perm_cat = perm
        m_p = permute_model_params("unimod_efvfm", m, perm).to(dtype)
        b = fam.make_batch(d_num, cats, batch_size=16)
        x_num, codes = b["x_num"].to(dtype), b["x_cat"]
        t = b["t"].to(dtype)
        oh = torch.cat([F.one_hot(codes[:, i], k).to(dtype) for i, k in enumerate(cats)], 1)
        x_num_p, codes_p = permute_batch(x_num, codes, perm)
        cats_p = [cats[int(j)] for j in perm_cat]
        oh_p = torch.cat([F.one_hot(codes_p[:, i], k).to(dtype) for i, k in enumerate(cats_p)], 1)
        with torch.no_grad():
            n0, c0 = m._vf_fn(x_num, oh, t)
            n1, c1 = m_p._vf_fn(x_num_p, oh_p, t)
        pn = torch.as_tensor(np.asarray(perm_num)); idx = block_permutation_indices(cats, perm_cat)
        e_num = (n1 - n0[:, pn]).norm() / n0.norm(); e_cat = (c1 - c0[:, idx]).norm() / c0.norm()
        assert e_num <= tol and e_cat <= tol, f"NOT equivariant: E_num={e_num:.3e} E_cat={e_cat:.3e}"
        ctrl = fam.build(d_num, cats_p, seed=0, denoiser="ft_periodic", ft_kw=SMALL_KW).to(dtype)
        with torch.no_grad():
            nc, _ = ctrl._vf_fn(x_num_p, oh_p, t)
        assert (nc - n0[:, pn]).norm() / n0.norm() > 1e-2, "control did not break"
    finally:
        torch.set_default_dtype(torch.float32)
