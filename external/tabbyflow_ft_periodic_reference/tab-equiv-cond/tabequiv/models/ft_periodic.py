"""VENDORED copy of AA_code's ``ft_periodic`` TabDiff denoiser (the "symmetric" core).

PROVENANCE. Copied VERBATIM (class bodies unchanged, only renamed at the top level) from
``/scratch/work/royi1/code/AA_code/TabDiff/tabdiff/modules/{main_modules,transformer}.py``
on 2026-08-25. That repo is agrawaa4's fork of upstream TabDiff at
``5ecdb3356261aea72716cc9a779f31d7ad083bf4`` (2025-06-02) with an UNCOMMITTED working tree;
the two files above are the only denoiser-side changes vs the pristine clone at
``baselines/TabDiff`` (``Precond``, ``Model``, ``UnifiedCtimeDiffusion`` and the noise
schedules are byte-identical there, which is what makes this net a drop-in for
``tabequiv.models.unimod_tabdiff.build``).

WHY IT IS HERE. The architecture is a per-column FT-Transformer denoiser: a periodic
(PLR, Gorishniy 2022) numerical tokenizer, L pre-norm attention blocks over the column
tokens with the timestep added at every layer, and per-column reconstruction heads. It
deletes TabDiff's flatten->MLPDiffusion->unflatten bottleneck -- the ONLY column-order
breaker in TabDiff (docs/arch_map.md) -- so it is whole-column permutation-EQUIVARIANT by
construction (probed: E_num 4.7e-16, E_cat 5.8e-16 in float64). Trained by its author
for 5 seeds x 6 datasets it beats the pristine TabDiff on Shape AND Trend on 5/6 datasets
(tie on magic) with ~7x fewer parameters; their pristine ``original`` reproduces the
published TabDiff numbers within ~0.004 (memory: aa-code-comparison).

CONTRACT. ``UniModMLPFTPeriodic.forward(x_num, x_cat, timesteps) -> (x_num_pred,
x_cat_logits)`` exactly like ``tabdiff.modules.main_modules.UniModMLP``; ``x_cat`` is the
one-hot (with mask class) the diffusion wrapper feeds. ``dim_t``/``use_mlp`` are accepted
and ignored (kept so the pristine ``UNIMOD_KW`` can be passed through).

BIT-EXACTNESS. Parameter creation ORDER is preserved so that, under the same seed, this
module and AA_code's class produce identical ``state_dict``s and identical outputs.
``tests/test_ft_periodic_port.py`` asserts that with ``torch.equal`` on CPU.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

#: the author's 5-seed configuration (train5.sh: NUM_LAYERS=6 D_TOKEN=128 N_HEAD=8 FACTOR=4)
FT_PERIODIC_KW = dict(num_layers=6, d_token=128, n_head=8, factor=4, bias=True,
                      n_frequencies=48, freq_sigma=0.05)
#: their toml default (L4/d64), ~0.3M params
FT_PERIODIC_SMALL_KW = dict(num_layers=4, d_token=64, n_head=8, factor=4, bias=True,
                            n_frequencies=48, freq_sigma=0.05)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class PeriodicTokenizer(nn.Module):
    """
    FT-Transformer tokenizer with PERIODIC (PLR) numerical embeddings
    (Gorishniy et al. 2022). Drop-in for Tokenizer: same output structure
    [CLS, num_0..num_{d-1}, cat_0..cat_{m-1}] of shape (B, 1+d+m, D).
    """
    def __init__(self, d_numerical, categories, d_token, bias, n_frequencies=48, freq_sigma=0.05):
        super().__init__()
        self.d_numerical = d_numerical
        self.d_token = d_token

        self.cls = nn.Parameter(Tensor(1, d_token))
        nn.init.kaiming_uniform_(self.cls, a=math.sqrt(5))

        if d_numerical > 0:
            self.freqs = nn.Parameter(torch.randn(d_numerical, n_frequencies) * freq_sigma)
            self.num_linear = nn.Parameter(Tensor(d_numerical, 2 * n_frequencies, d_token))
            self.num_bias = nn.Parameter(Tensor(d_numerical, d_token))
            nn.init.kaiming_uniform_(self.num_linear, a=math.sqrt(5))
            nn.init.zeros_(self.num_bias)

        if categories is None:
            self.category_offsets = None
        else:
            category_offsets = torch.tensor([0] + list(categories[:-1])).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.cat_weight = nn.Parameter(Tensor(sum(categories), d_token))
            nn.init.kaiming_uniform_(self.cat_weight, a=math.sqrt(5))

        d_bias = d_numerical + (0 if categories is None else len(categories))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self):
        n = 1 + self.d_numerical
        if self.category_offsets is not None:
            n += len(self.category_offsets)
        return n

    def _embed_numeric(self, x_num):
        v = 2 * math.pi * self.freqs[None] * x_num[:, :, None]
        emb = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
        out = torch.einsum('bfk,fkd->bfd', emb, self.num_linear) + self.num_bias[None]
        return F.relu(out)

    def forward(self, x_num, x_cat):
        x_some = x_num if x_cat is None else x_cat
        B = len(x_some)

        tokens = [self.cls[None].expand(B, 1, self.d_token)]

        if x_num is not None and self.d_numerical > 0:
            tokens.append(self._embed_numeric(x_num))

        if x_cat is not None and self.category_offsets is not None:
            cat_tokens = []
            ends = torch.cat([self.category_offsets[1:],
                              torch.tensor([x_cat.shape[1]], device=x_cat.device)])
            for start, end in zip(self.category_offsets, ends):
                if start < end:
                    cat_tokens.append(x_cat[:, start:end].unsqueeze(1) @ self.cat_weight[start:end][None])
            if cat_tokens:
                tokens.append(torch.cat(cat_tokens, dim=1))

        x = torch.cat(tokens, dim=1)

        if self.bias is not None:
            bias = torch.cat([torch.zeros(1, self.bias.shape[1], device=x.device), self.bias])
            x = x + bias[None]

        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout, initialization = 'kaiming'):

        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv, key_compression = None, value_compression = None):

        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        a = q @ k.transpose(1, 2)
        b = math.sqrt(d_head_key)
        attention = F.softmax(a/b , dim=-1)

        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)

        return x


class FTBlock(nn.Module):
    """
    One PreNorm FT-Transformer layer: multi-head attention over ALL feature tokens,
    then a token-wise FFN. Timestep conditioning is added to the token stream before
    each sublayer so every layer denoises noise-level-aware.
    """
    def __init__(self, d_token, n_heads, d_ffn_factor=4, attention_dropout=0.0, ffn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attention = MultiheadAttention(d_token, n_heads, attention_dropout)
        self.norm2 = nn.LayerNorm(d_token)
        d_hidden = int(d_token * d_ffn_factor)
        self.linear0 = nn.Linear(d_token, d_hidden)
        self.linear1 = nn.Linear(d_hidden, d_token)
        self.act = nn.GELU()
        self.ffn_dropout = ffn_dropout
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(d_token, d_token))

    def forward(self, x, t_emb):
        t = self.t_proj(t_emb).unsqueeze(1)          # (B, 1, D), broadcast over tokens

        # PreNorm attention sublayer + residual
        h = self.norm1(x + t)
        h = self.attention(h, h)
        x = x + h

        # PreNorm FFN sublayer + residual
        h = self.norm2(x)
        h = self.linear1(F.dropout(self.act(self.linear0(h)), self.ffn_dropout, self.training))
        x = x + h
        return x


class Reconstructor(nn.Module):
    def __init__(self, d_numerical, categories, d_token):
        super(Reconstructor, self).__init__()

        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token

        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()

        for d in categories:
            recon = nn.Linear(d_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h):
        h_num  = h[:, :self.d_numerical]
        h_cat  = h[:, self.d_numerical:]

        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        recon_x_cat = []

        for i, recon in enumerate(self.cat_recons):

            recon_x_cat.append(recon(h_cat[:, i]))

        return recon_x_num, recon_x_cat


class UniModMLPFTPeriodic(nn.Module):
    """
    ACTIVE DENOISER: FT-Transformer.
        Input:
            x_num: [bs, d_numerical]
            x_cat: [bs, len(categories)]  (one-hot, as fed by the diffusion wrapper)
        Output:
            x_num_pred: [bs, d_numerical], predicted mean for numerical data
            x_cat_pred: [bs, sum(categories)], UNORMALIZED logits for categorical data
    Dense per-feature attention over all tokens at every layer (no sparse masking,
    no flatten-MLP bottleneck). Timestep injected into every layer.
    """
    def __init__(
            self, d_numerical, categories, num_layers, d_token,
            n_head = 8, factor = 4, bias = True, dim_t=512, use_mlp=True, **kwargs
        ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories
        self.n_features = d_numerical + len(categories)
        self.tokenizer = PeriodicTokenizer(
            d_numerical, categories, d_token, bias=bias,
            n_frequencies=kwargs.get('n_frequencies', 48),
            freq_sigma=kwargs.get('freq_sigma', 0.05),
        )
        self.map_noise = PositionalEmbedding(num_channels=d_token)
        self.time_embed = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.SiLU(),
            nn.Linear(d_token, d_token)
        )
        self.blocks = nn.ModuleList([
            FTBlock(d_token, n_head, d_ffn_factor=factor)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_token)
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)
        self.model = nn.ModuleList([self.tokenizer, self.time_embed, self.blocks, self.detokenizer])

    def forward(self, x_num, x_cat, timesteps):
        e = self.tokenizer(x_num, x_cat)
        x = e[:, 1:, :]                              # ignore CLS token
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        t_emb = self.time_embed(emb)                   # (B, d_token)
        for blk in self.blocks:
            x = blk(x, t_emb)
        x = self.final_norm(x)
        x_num_pred, x_cat_pred = self.detokenizer(x)
        x_cat_pred = torch.cat(x_cat_pred, dim=-1) if len(x_cat_pred)>0 else torch.zeros_like(x_cat).to(x_num_pred.dtype)
        return x_num_pred, x_cat_pred


#: where the verbatim source lives (tests import it read-only to prove bit-exactness)
AA_CODE_ROOT = "/scratch/work/royi1/code/AA_code/TabDiff"
AA_CODE_COMMIT = "5ecdb3356261aea72716cc9a779f31d7ad083bf4 (+ uncommitted denoiser changes)"


# ---------------------------------------------------------------------------------------
# TabbyFlow (tabvvfm) adapter -- the SAME vendored core plugged into TabbyFlow's ``CondVF``.
#
# TabbyFlow's data-space net contract (``baselines/tabvvfm/networks.py::Net``,
# ``flow_matching.py::TabFlowMatching.loss`` / ``CondVF.forward_for_ode``):
#   forward(t, x) with t in [0, 1] (shape [B]) and x = [x_cont | one-hot blocks] (noisy
#   interpolant), returning theta_all = [numeric means | per-column categorical LOGITS] --
#   Gaussian NLL on the first block, cross-entropy per one-hot block, softmax per block in
#   the ODE.  That is exactly ``UniModMLPFTPeriodic.forward(x_num, x_cat, t)``'s
#   ``(x_num_pred, x_cat_pred logits)`` pair, and its tokenizer embeds categoricals by a
#   block-wise matmul (so the soft/noisy one-hot is fine, as with TabDiff's x_cat_t_soft);
#   TabDiff also feeds the core t in [0, 1], so no time rescaling.  No mask class here.
# ---------------------------------------------------------------------------------------
class FTPeriodicVFMNet(nn.Module):
    """``Net``-compatible wrapper: ``forward(t, x) -> theta_all`` via AA's symmetric core."""

    def __init__(self, d_numerical: int, categories, **ft_kw):
        super().__init__()
        self.d_numerical = int(d_numerical)
        self.categories = [int(k) for k in categories]
        kw = dict(FT_PERIODIC_KW); kw.update(ft_kw)
        self.core = UniModMLPFTPeriodic(d_numerical=self.d_numerical,
                                        categories=self.categories, **kw)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1).expand(x.shape[0]) if t.numel() == 1 else t.reshape(-1)
        x_num = x[:, :self.d_numerical]
        x_cat = x[:, self.d_numerical:]
        num_pred, cat_pred = self.core(x_num, x_cat if x_cat.shape[1] > 0 else None, t)
        parts = [num_pred] if self.d_numerical > 0 else []
        if x_cat.shape[1] > 0:
            parts.append(cat_pred)
        return torch.cat(parts, dim=-1)
