"""END-TO-END SAMPLER equivariance: does a whole-column permutation commute with
each family's OWN generation loop, and does that show up in the eval metrics?

This is Part B of the metric-equivariance study, and it is also the long-outstanding
*generation*-side equivariance probe: prompt-4/task-1 measured per-LAYER error under
one forward pass; here the permutation is pushed through the family's real sampler
(50-1000 denoising / ODE steps, stochastic categorical resampling, the frozen VAE
decode for the latent families) all the way to a table.

Protocol (per family x core-variant x dataset x init seed x permutation)
-----------------------------------------------------------------------
1. ``table_orig``  — run the family's own sampler at fixed seeded init under an RNG
   RECORDER, then decode the sampler state to a table.
2. ``table_perm``  — permute the model (``perturb.permute_model_params`` /
   ``perturb_latent.permute_model_params_task1`` / ``ablate.permute_symmetric``),
   REPLAY the recorded noise draws permuted along their column axes (common random
   numbers), run the same sampler, then UN-permute the generated columns back to the
   original order -> ``table_perm_unpermuted``.
3. **sampler-E** = ``|| table_perm_unpermuted - table_orig || / || table_orig ||`` in
   float64 on the generated values (numericals as-is, categoricals as one-hot).  This
   is the load-bearing signal: it is a direct algebraic identity check with no fitted
   model anywhere in it, so unlike every discriminator metric it CANNOT saturate.

Expected: ~float noise for a symmetric core, O(1) for the untouched asymmetric core.

Where the permutation acts, per structural type (docs/arch_map.md, findings_task1.md)
-------------------------------------------------------------------------------------
* **unimod** (``unimod_tabdiff``, ``unimod_efvfm``) — raw columns; the sampler emits
  ``[x_num | cat codes]`` directly (``TabDiff/tabdiff/models/unified_ctime_diffusion.py:215``
  ``sample_all``; ``ef-vfm/ef_vfm/models/flow_model.py:113`` ``sample_all``).
* **concat** (``tabddpm``, ``tabvvfm``) — raw columns on a flat
  ``[x_num | one-hot blocks]`` state (``tab-ddpm/tab_ddpm/gaussian_multinomial_diffsuion.py:966``
  ``sample_all``; ``tabular-flow-matching/baselines/tabvvfm/flow_matching.py:129``
  ``CondVF.decode_t0_t1``).
* **latent** (``tabsyn_diffusion``, ``tabflow_net``) — the model never sees columns;
  P acts on the per-column token blocks of the flattened frozen-VAE latent
  (``tabsyn/tabsyn/diffusion_utils.py:22`` ``sample``;
  ``tabular-flow-matching/tabsynflow/fm_utils.py:255`` ``CondVF.decode_t0_t1``).
  The latent is decoded to a table through the frozen tabsyn VAE decoder following
  ``tabsyn/tabsyn/latent_utils.py:70`` ``split_num_cat_target`` — reshape to
  ``[B, n_cols, d_token]``, decode, ``argmax`` each categorical head.  The VAE itself
  is column-equivariant (verified in docs/findings.md), so it neither creates nor
  hides a violation; it is held FIXED (same seeded init) across both runs and simply
  permuted alongside the columns.

NO TRAINING anywhere: every model is at fixed seeded random init.  NO train/val/test
splitting: the FULL real dataset is the metric reference (``eval.io.load_real``).

Two deviations from the shared probe helpers, both mechanical and both scoped to this
module so ``tabequiv/probe.py`` and ``tabequiv/eval/`` stay untouched — see
:class:`SamplerRngReplay` and :class:`TabDiffGumbelAligner`.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from tabequiv import ablate
from tabequiv.eval.base import column_kinds
from tabequiv.paths import use_baseline
from tabequiv.perturb import column_permutation, permute_model_params
from tabequiv.perturb_latent import (LatentRngReplay, LatentSchema, TOKEN_DIM,
                                     latent_block_indices,
                                     permute_model_params_task1,
                                     tabvvfm_flat_indices)
from tabequiv.probe import PermAligner, RngReplay, Schema
from tabequiv.repro import seed_everything

__all__ = [
    "FAMILIES", "STRUCTURAL_TYPE", "DATASETS",
    "SamplerRngReplay", "LatentSamplerRngReplay", "TabDiffSamplerRngReplay",
    "TabDiffGumbelAligner",
    "dataset_schema", "make_schema", "make_replay", "build_model", "permuted_model",
    "generate", "unpermute_state", "state_to_table", "sampler_error",
    "run_cell", "CellResult",
]

#: every neural family, in a fixed order
FAMILIES = ("unimod_tabdiff", "unimod_efvfm", "tabddpm", "tabvvfm",
            "tabsyn_diffusion", "tabflow_net")

#: structural type -> which permutation index math and decode path applies
STRUCTURAL_TYPE = {
    "unimod_tabdiff": "unimod",
    "unimod_efvfm": "unimod",
    "tabddpm": "concat",
    "tabvvfm": "concat",
    "tabsyn_diffusion": "latent",
    "tabflow_net": "latent",
}

LATENT_FAMILIES = tuple(f for f, t in STRUCTURAL_TYPE.items() if t == "latent")

DATASETS = ("default", "news")

#: Step count for the ONE sampler that takes it as a call argument:
#: ``tabsyn_diffusion`` (``tabsyn/tabsyn/diffusion_utils.py:22``, ``num_steps=``).
#: Every other family fixes its schedule at CONSTRUCTION and ignores this value —
#: TabDiff ``num_timesteps=50`` (``tabequiv/models/unimod_tabdiff.py:37``), tabddpm
#: ``num_timesteps=1000`` (``tabequiv/models/tabddpm.py:31``), tabvvfm / tabflow_net
#: ``CondVF.n_steps=100`` with euler, ef-vfm adaptive ``dopri5``.  So this knob trades
#: wall time only for tabsyn_diffusion; the ACTUAL per-family step counts are the
#: constructor ones listed above.  That is fine here: equivariance is STRUCTURAL, so
#: the verdict cannot depend on the step count — a violation present at step 1 only
#: compounds over more steps.
SAMPLER_STEPS = 50


# ---------------------------------------------------------------------------
# RNG replay fixes (scoped here; tabequiv/probe.py is left untouched)
# ---------------------------------------------------------------------------

class SamplerRngReplay(RngReplay):
    """:class:`tabequiv.probe.RngReplay` with numpy-integer-safe shape derivation.

    ``probe.py`` derives the requested shape of a ``torch.randn(*sizes)`` call with
    ``tuple(a for a in args if isinstance(a, int))`` (``tabequiv/probe.py:216``).
    ef-vfm's sampler calls ``torch.randn(num_samples, d_in)`` where
    ``d_in = self.num_numerical_features + sum(self.num_classes)`` and
    ``num_classes`` is a **numpy array** (``ef-vfm/ef_vfm/models/flow_model.py:95``),
    so ``d_in`` is ``np.int64`` — which is not an ``int`` — and the second dimension
    is silently dropped, making the replay assert with a shape mismatch.  Accepting
    ``np.integer`` fixes it; behaviour is otherwise identical.

    Reported upstream rather than patched in place: ``probe.py`` is shared with the
    prompt-4 / task-1 runners and is not this task's file to edit.
    """

    def _make_patch(self, fname, orig):
        base = super()._make_patch(fname, orig)
        if fname.endswith("_like") or fname == "randint":
            return base

        def patched(*args, **kwargs):
            if not self.replaying:
                return self._record(fname, orig(*args, **kwargs))
            if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)):
                shape = tuple(int(s) for s in args[0])
            else:
                shape = tuple(int(a) for a in args
                              if isinstance(a, (int, np.integer)))
            return self._replay(fname, shape,
                                kwargs.get("dtype", torch.get_default_dtype()),
                                kwargs.get("device", "cpu"))
        return patched


class LatentSamplerRngReplay(LatentRngReplay, SamplerRngReplay):
    """Latent-aware aligner (:class:`LatentRngReplay`) + the numpy-int shape fix."""


class TabDiffGumbelAligner(PermAligner):
    """Aligner that pins TabDiff's padded ``[B, m, K_max]`` gumbel draw to the
    COLUMN axis.

    ``_sample_categorical`` draws ``torch.rand_like(q_xs)`` on the padded
    class-probability tensor ``q_xs`` of shape ``[B, m, max(K)+1]``
    (``TabDiff/tabdiff/models/unified_ctime_diffusion.py:379-383``, built at
    ``:364-370``).  The generic size-based axis search tries composites first, so
    whenever ``max(K)+1`` happens to equal one of the composite sizes — on
    ``default`` it equals ``n_cols`` — it permutes the CLASS axis instead of the
    column axis, and the replayed draws no longer correspond column-for-column.
    A 3-D tensor whose axis 1 is ``m`` is unambiguously ``[B, m, classes]`` here, so
    it is aligned along axis 1 with ``perm_cat``; the class axis of a single column
    is unchanged when the whole column moves.
    """

    def align(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 3 and t.shape[1] == self.schema.m:
            return t.index_select(1, self.index["cat"].to(t.device))
        return super().align(t)


class TabDiffSamplerRngReplay(SamplerRngReplay):
    """:class:`SamplerRngReplay` using :class:`TabDiffGumbelAligner`."""

    def __init__(self, schema, perm=None, recorded=None):
        super().__init__(schema, perm=perm, recorded=recorded)
        if self.replaying:
            self.aligner = TabDiffGumbelAligner(schema, perm)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def dataset_schema(info: dict, real: pd.DataFrame):
    """``(d_num, categories, num_idx, cat_idx)`` of the FULL table.

    The model schema is the whole table (target folded in by task type, exactly as
    ``eval.base.column_kinds`` does it), not the staged feature matrices — the
    sampler must emit every column ``info['column_names']`` names so the generated
    frame can be written and evaluated against the full real dataset.
    Cardinalities are the observed level counts of the FULL real table.
    """
    num_idx, cat_idx = column_kinds(info)
    cols = list(info["column_names"])
    categories = [int(real[cols[i]].astype(str).nunique()) for i in cat_idx]
    return len(num_idx), categories, num_idx, cat_idx


def make_schema(family: str, d_num: int, categories):
    """Probe :class:`Schema` for ``family`` (latent families get the flat-latent axis)."""
    if family in LATENT_FAMILIES:
        return LatentSchema(d_num, categories)
    return Schema(d_num, categories,
                  mask_class=(family == "unimod_tabdiff"))


def make_replay(family: str, schema, perm=None, recorded=None):
    """The right recorder/replayer for ``family`` (see the classes above)."""
    if family in LATENT_FAMILIES:
        return LatentSamplerRngReplay(schema, perm=perm, recorded=recorded)
    if family == "unimod_tabdiff":
        return TabDiffSamplerRngReplay(schema, perm=perm, recorded=recorded)
    return SamplerRngReplay(schema, perm=perm, recorded=recorded)


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------

def build_model(family: str, d_num: int, categories, seed: int,
                symmetric: bool = False) -> torch.nn.Module:
    """Fixed seeded random init.  ``symmetric=False`` is the untouched golden path."""
    with contextlib.redirect_stdout(io.StringIO()):
        model = ablate.build(family, int(d_num), [int(k) for k in categories],
                             seed=seed, symmetric_core=symmetric)
    model.eval()
    return model


def permuted_model(family: str, model: torch.nn.Module, perm, d_num, categories,
                   symmetric: bool = False) -> torch.nn.Module:
    """NEW model at the permuted schema — REUSES the existing permutation helpers."""
    with contextlib.redirect_stdout(io.StringIO()):
        if symmetric:
            if family in LATENT_FAMILIES:
                out = ablate.permute_symmetric(
                    family, model, perm, d_num=int(d_num),
                    categories=[int(k) for k in categories])
            else:
                out = ablate.permute_symmetric(family, model, perm)
        elif STRUCTURAL_TYPE[family] == "unimod" or family == "tabddpm":
            out = permute_model_params(family, model, perm)
        else:
            out = permute_model_params_task1(family, model, perm, int(d_num),
                                             [int(k) for k in categories])
    out.eval()
    return out


# ---------------------------------------------------------------------------
# frozen tabsyn VAE decoder (latent families only)
# ---------------------------------------------------------------------------

def build_vae(d_num: int, categories, seed: int = 7) -> torch.nn.Module:
    """Frozen tabsyn VAE at fixed seeded init — the latent families' decoder.

    Both latent pipelines train the diffusion/flow on a CACHED ``train_z`` produced
    by an already-trained, frozen VAE (``tabsyn/tabsyn/latent_utils.py:19-26``), so
    in a no-training study any FIXED tokenizer/decoder is a faithful stand-in: what
    matters is the per-column block LAYOUT, which is identical, and the VAE is itself
    column-equivariant (docs/findings.md).  The same seed is used for both runs of a
    cell, and the decoder is permuted alongside the columns, so the decode step is
    part of the equivariance identity rather than an uncontrolled extra.
    """
    from tabequiv.models import tabsyn_vae as vae_fam
    with contextlib.redirect_stdout(io.StringIO()):
        vae = vae_fam.build(int(d_num), [int(k) for k in categories], seed=seed)
    vae.eval()
    return vae


def decode_latent(vae: torch.nn.Module, z: torch.Tensor, d_num: int, categories):
    """Flat latent -> ``(x_num, x_cat codes)`` via the frozen VAE decoder.

    Mirrors ``tabsyn/tabsyn/latent_utils.py:70`` ``split_num_cat_target``: reshape the
    flat latent to ``[B, n_cols, d_token]``, run the decoder + Reconstructor, and take
    ``argmax`` over each per-column categorical head.  ``split_num_cat_target`` itself
    is not called because it additionally applies the TRAINED preprocessor's
    ``num_inverse``/``cat_inverse`` (built by ``utils_train.preprocess``), which do not
    exist in a no-training run; the inverse transforms are column-wise bijections and
    so are equivariant by construction, i.e. they cannot change the verdict.
    """
    n_cols = int(d_num) + len(categories)
    with torch.no_grad():
        h = vae.VAE.decoder(z.reshape(z.shape[0], n_cols, TOKEN_DIM))
        recon_num, recon_cat = vae.Reconstructor(h)
        codes = torch.stack([logits.argmax(dim=-1) for logits in recon_cat], dim=1)
    return recon_num, codes


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def generate(family: str, model: torch.nn.Module, n_rows: int, d_num: int,
             categories, steps: int = SAMPLER_STEPS) -> torch.Tensor:
    """Run the family's OWN sampler.  Returns the raw sampler STATE, un-decoded.

    * unimod / tabddpm -> ``[B, d_num + m]`` = ``[x_num | cat codes]``
    * tabvvfm          -> ``[B, d_num + sum(K)]`` = ``[x_cont | one-hot blocks]``
    * latent           -> ``[B, n_cols * d_token]`` flat latent

    Every entry point is imported READ-ONLY through ``tabequiv.paths.use_baseline``;
    no baseline file is touched.  ``device`` is passed explicitly to tabsyn's
    ``sample`` because it defaults to a hardcoded ``'cuda:0'``
    (``tabsyn/tabsyn/diffusion_utils.py:22``).
    """
    cats = [int(k) for k in categories]
    n_cols = int(d_num) + len(cats)
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        if family == "unimod_tabdiff":
            # TabDiff/tabdiff/models/unified_ctime_diffusion.py:215
            return model.sample_all(n_rows, n_rows)
        if family == "unimod_efvfm":
            # ef-vfm/ef_vfm/models/flow_model.py:113
            return model.sample_all(n_rows, n_rows)
        if family == "tabddpm":
            # tab-ddpm/tab_ddpm/gaussian_multinomial_diffsuion.py:966
            x, _ = model.sample_all(n_rows, n_rows, torch.tensor([1.0]), ddim=False)
            return x
        if family == "tabvvfm":
            # tabular-flow-matching/baselines/tabvvfm/flow_matching.py:129
            with use_baseline("tabular-flow-matching"):
                dev = next(model.parameters()).device
                x0 = torch.randn(n_rows, int(d_num) + sum(cats), device=dev)
                return model.decode_t0_t1(x0, 0.0, 1.0, method="euler")
        if family == "tabflow_net":
            # tabular-flow-matching/tabsynflow/fm_utils.py:255
            with use_baseline("tabular-flow-matching"):
                dev = next(model.parameters()).device
                x0 = torch.randn(n_rows, n_cols * TOKEN_DIM, device=dev)
                return model.decode_t0_t1(x0, 0.0, 1.0, method="euler")
        if family == "tabsyn_diffusion":
            # tabsyn/tabsyn/diffusion_utils.py:22 — device passed EXPLICITLY
            with use_baseline("tabsyn"):
                from tabsyn.diffusion_utils import sample as tabsyn_sample
                # device follows the MODEL (tabsyn's sample() otherwise defaults to a
                # hardcoded 'cuda:0' -- tabsyn/tabsyn/diffusion_utils.py:22). Part B ran
                # everything on CPU; training/eval here runs on GPU, so pass it through.
                return tabsyn_sample(model.denoise_fn_D, n_rows,
                                     n_cols * TOKEN_DIM, num_steps=steps,
                                     device=str(next(model.parameters()).device))
    raise ValueError(f"unknown family {family!r}")


def unpermute_state(family: str, state: torch.Tensor, perm, d_num, categories
                    ) -> torch.Tensor:
    """Map a permuted-run sampler state back into the ORIGINAL column order.

    Index math is REUSED from the existing helpers, per structural type:
    ``perturb_latent.latent_block_indices`` (latent),
    ``perturb_latent.tabvvfm_flat_indices`` (tabvvfm), and the plain
    ``[perm_num | d_num + perm_cat]`` column vector for the code-emitting samplers.
    """
    perm_num, perm_cat = np.asarray(perm[0]), np.asarray(perm[1])
    cats = [int(k) for k in categories]
    if family in LATENT_FAMILIES:
        idx = latent_block_indices(int(d_num), len(cats), perm)
    elif family == "tabvvfm":
        idx = tabvvfm_flat_indices(int(d_num), cats, perm)
    else:
        idx = torch.as_tensor(
            np.concatenate([perm_num, int(d_num) + perm_cat]).astype(np.int64))
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(len(idx), dtype=idx.dtype)
    return state.index_select(-1, inv.to(state.device))


def state_to_table(family: str, state: torch.Tensor, d_num, categories,
                   vae: torch.nn.Module | None = None):
    """Sampler state (already in ORIGINAL column order) -> ``(x_num, cat codes)``."""
    cats = [int(k) for k in categories]
    d_num = int(d_num)
    if family in LATENT_FAMILIES:
        assert vae is not None, "latent families need the frozen VAE decoder"
        return decode_latent(vae, state, d_num, cats)
    if family == "tabvvfm":
        x_num = state[:, :d_num]
        codes, off = [], d_num
        for k in cats:
            codes.append(state[:, off:off + k].argmax(dim=-1))
            off += k
        return x_num, torch.stack(codes, dim=1)
    # unimod / tabddpm: the sampler already emits [x_num | cat codes]
    return state[:, :d_num], state[:, d_num:].round().long()


# ---------------------------------------------------------------------------
# sampler-E
# ---------------------------------------------------------------------------

def sampler_error(x_num_a: torch.Tensor, codes_a: torch.Tensor,
                  x_num_b: torch.Tensor, codes_b: torch.Tensor,
                  categories) -> float:
    """``|| a - b || / || b ||`` in float64 over generated values.

    Numericals enter as-is; categoricals enter as ONE-HOT, so a single differing code
    contributes a fixed distance regardless of how the levels happen to be numbered
    (comparing raw integer codes would make the error depend on level ordering, which
    carries no meaning).  ``b`` is the original-run table, i.e. the denominator.
    """
    import torch.nn.functional as F

    cats = [int(k) for k in categories]

    def flat(x_num, codes):
        parts = [x_num.double()]
        for j, k in enumerate(cats):
            parts.append(F.one_hot(codes[:, j].long().clamp(0, k - 1),
                                   num_classes=k).double())
        return torch.cat(parts, dim=1)

    a, b = flat(x_num_a, codes_a), flat(x_num_b, codes_b)
    denom = torch.linalg.vector_norm(b).item()
    num = torch.linalg.vector_norm(a - b).item()
    return num / denom if denom > 0 else num


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    """One (family, dataset, seed, variant) cell."""
    family: str
    dataset: str
    init_seed: int
    variant: str                    # "asymmetric" | "symmetric"
    n_rows: int
    n_perms: int
    sampler_steps: int
    E_values: list                  # sampler-E per permutation
    table_orig: pd.DataFrame
    table_perm: pd.DataFrame        # from the FIRST permutation, un-permuted
    perm_seed_first: int

    @property
    def E_median(self) -> float:
        return float(np.median(self.E_values))

    @property
    def E_max(self) -> float:
        return float(np.max(self.E_values))

    @property
    def E_min(self) -> float:
        return float(np.min(self.E_values))


def _frame(info: dict, real: pd.DataFrame, num_idx, cat_idx,
           x_num: torch.Tensor, codes: torch.Tensor) -> pd.DataFrame:
    """Assemble a generated table in the ORIGINAL ``info['column_names']`` order.

    Numericals are written as float64; each categorical code is mapped back to the
    corresponding real level (sorted level order, the same order the cardinalities
    were counted in), so the frame round-trips through ``eval.io`` with the contract
    dtypes and the metrics see real levels rather than integers.
    """
    cols = list(info["column_names"])
    out = {}
    x_num = x_num.detach().double().cpu().numpy()
    codes = codes.detach().long().cpu().numpy()
    for j, i in enumerate(num_idx):
        out[cols[i]] = x_num[:, j]
    for j, i in enumerate(cat_idx):
        levels = sorted(real[cols[i]].astype(str).unique())
        c = np.clip(codes[:, j], 0, len(levels) - 1)
        out[cols[i]] = np.asarray(levels, dtype=object)[c]
    return pd.DataFrame(out, columns=cols)


def run_cell(family: str, dataset: str, info: dict, real: pd.DataFrame,
             init_seed: int, symmetric: bool, n_rows: int, n_perms: int,
             steps: int = SAMPLER_STEPS, perm_seed0: int = 10_000,
             gen_seed: int = 5) -> CellResult:
    """Generate once, then re-generate under ``n_perms`` permutations; -> sampler-E.

    The original run is done ONCE and its RNG draws recorded; every permuted run
    replays those same draws permuted along their column axes (common random
    numbers), which is what makes the comparison an equivariance identity check
    rather than a two-sample noise comparison.
    """
    d_num, categories, num_idx, cat_idx = dataset_schema(info, real)
    schema = make_schema(family, d_num, categories)
    assert not schema.ambiguous, (
        f"{family}/{dataset}: ambiguous schema axis sizes {schema.ambiguous}")

    model = build_model(family, d_num, categories, init_seed, symmetric=symmetric)
    vae = build_vae(d_num, categories) if family in LATENT_FAMILIES else None

    rec = make_replay(family, schema)
    seed_everything(gen_seed)
    with rec:
        state_orig = generate(family, model, n_rows, d_num, categories, steps)
    x_num_o, codes_o = state_to_table(family, state_orig, d_num, categories, vae)
    table_orig = _frame(info, real, num_idx, cat_idx, x_num_o, codes_o)

    E_values, table_perm = [], None
    for p in range(n_perms):
        perm = column_permutation(d_num, categories, seed=perm_seed0 + p)
        model_p = permuted_model(family, model, perm, d_num, categories,
                                 symmetric=symmetric)

        seed_everything(gen_seed)
        with make_replay(family, schema, perm, rec.recorded):
            state_p = generate(family, model_p, n_rows, d_num, categories, steps)

        # UN-PERMUTE FIRST, then decode with the ORIGINAL-order VAE.  The permuted
        # run's latent is in permuted column-block order, so un-permuting it puts it
        # back in the order the original decoder expects; that decoder is therefore
        # the matching one and no permuted VAE is needed.  Decoding with a
        # P-permuted VAE and un-permuting the resulting columns afterwards gives the
        # same table — the VAE is column-equivariant (docs/findings.md, "fully
        # equivariant") — so this ordering costs a model rebuild per permutation
        # without changing any number.
        state_pu = unpermute_state(family, state_p, perm, d_num, categories)
        x_num_p, codes_p = state_to_table(family, state_pu, d_num, categories, vae)
        E_values.append(sampler_error(x_num_p, codes_p, x_num_o, codes_o, categories))
        if p == 0:
            table_perm = _frame(info, real, num_idx, cat_idx, x_num_p, codes_p)
        del model_p

    return CellResult(
        family=family, dataset=dataset, init_seed=init_seed,
        variant="symmetric" if symmetric else "asymmetric",
        n_rows=n_rows, n_perms=n_perms, sampler_steps=steps,
        E_values=E_values, table_orig=table_orig, table_perm=table_perm,
        perm_seed_first=perm_seed0)
