from typing import Callable, Union

from tabdiff.modules.transformer import Reconstructor, Tokenizer, Transformer, TabNetDenoiseStep, MultiheadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import math
from torch import Tensor

ModuleType = Union[str, Callable[..., nn.Module]]


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


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

class MLPDiffusion(nn.Module):
    """
    Original TabDiff flatten-then-dense-MLP bottleneck. KEPT FOR REFERENCE.
    Used by the original UniModMLP design (encoder -> flatten -> MLPDiffusion -> decoder).
    Not used by the active FT-Transformer UniModMLP.
    """
    def __init__(self, d_in, dim_t = 512, use_mlp=True):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        ) if use_mlp else nn.Linear(dim_t, d_in)

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

        self.use_mlp = use_mlp

    def forward(self, x, timesteps):
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        emb = self.time_embed(emb)

        x = self.proj(x) + emb
        return self.mlp(x)


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

# class UniModMLP(nn.Module):
#     """
#         Input:
#             x_num: [bs, d_numerical]
#             x_cat: [bs, len(categories)]
#         Output:
#             x_num_pred: [bs, d_numerical], the predicted mean for numerical data
#             x_cat_pred: [bs, sum(categories)], the predicted UNORMALIZED logits for categorical data
#     """
#     def __init__(
#             self, d_numerical, categories, num_layers, d_token,
#             n_head = 1, factor = 4, bias = True, dim_t=512, use_mlp=True, **kwargs
#         ):
#         super().__init__()
#         self.d_numerical = d_numerical
#         self.categories = categories

#         self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias = bias)
#         self.encoder = Transformer(num_layers, d_token, n_head, d_token, factor)
#         d_in = d_token * (d_numerical + len(categories))
#         self.mlp = MLPDiffusion(d_in, dim_t=dim_t, use_mlp=use_mlp)
#         self.decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
#         self.detokenizer = Reconstructor(d_numerical, categories, d_token)
        
#         self.model = nn.ModuleList([self.tokenizer, self.encoder, self.mlp, self.decoder, self.detokenizer])

#     def forward(self, x_num, x_cat, timesteps):
#         e = self.tokenizer(x_num, x_cat)
#         decoder_input = e[:, 1:, :]        # ignore the first CLS token. 
#         y = self.encoder(decoder_input)
#         pred_y = self.mlp(y.reshape(y.shape[0], -1), timesteps)
#         pred_e = self.decoder(pred_y.reshape(*y.shape))
#         x_num_pred, x_cat_pred = self.detokenizer(pred_e)
#         x_cat_pred = torch.cat(x_cat_pred, dim=-1) if len(x_cat_pred)>0 else torch.zeros_like(x_cat).to(x_num_pred.dtype)

#         return x_num_pred, x_cat_pred
class UniModMLP_Original(nn.Module):
    """
    ORIGINAL TabDiff denoiser: encoder Transformer -> flatten -> MLPDiffusion -> decoder Transformer.
    The architecture that the pre-FT checkpoints (e.g. learnable_schedule) were trained with.
    Kept so those checkpoints load, and so it can serve as the baseline for the 5-seed comparison.
    """
    def __init__(
            self, d_numerical, categories, num_layers, d_token,
            n_head = 1, factor = 4, bias = True, dim_t=512, use_mlp=True, **kwargs
        ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories

        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias = bias)
        self.encoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        d_in = d_token * (d_numerical + len(categories))
        self.mlp = MLPDiffusion(d_in, dim_t=dim_t, use_mlp=use_mlp)
        self.decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)

        self.model = nn.ModuleList([self.tokenizer, self.encoder, self.mlp, self.decoder, self.detokenizer])

    def forward(self, x_num, x_cat, timesteps):
        e = self.tokenizer(x_num, x_cat)
        decoder_input = e[:, 1:, :]        # ignore the first CLS token
        y = self.encoder(decoder_input)
        pred_y = self.mlp(y.reshape(y.shape[0], -1), timesteps)
        pred_e = self.decoder(pred_y.reshape(*y.shape))
        x_num_pred, x_cat_pred = self.detokenizer(pred_e)
        x_cat_pred = torch.cat(x_cat_pred, dim=-1) if len(x_cat_pred)>0 else torch.zeros_like(x_cat).to(x_num_pred.dtype)
        return x_num_pred, x_cat_pred


class UniModMLP(nn.Module):
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

        # self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias = bias)

        self.tokenizer = PeriodicTokenizer(
            d_numerical, categories, d_token, bias=bias,
            n_frequencies=kwargs.get('n_frequencies', 48),
            freq_sigma=kwargs.get('freq_sigma', 0.05),
        )

        # timestep embedding -> d_token, fed into every layer
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


class UniModMLP_TabNet(nn.Module):
    """
    REFERENCE ONLY: TabNet-style sparse-selection denoiser. Not the active denoiser
    (main.py imports UniModMLP, the FT-Transformer version).

    Known limitation (see mask diagnostics): the sparsemax masks collapse after step 0
    (steps 1+ go dead, many features never selected), because sparse instance-wise
    selection is a prediction/interpretability bias, not a generative one -- a denoiser
    must reconstruct every feature, so masking most of them out fights the objective.
    Retained to document the approach and to allow re-testing with mask regularization
    (self.last_masks is exposed for the diffusion-side entropy/coverage regularizer).
    """
    def __init__(
            self, d_numerical, categories, num_layers, d_token,
            n_head = 1, factor = 4, bias = True, dim_t=512, use_mlp=True,
            n_steps = 4, gamma = 1.5, **kwargs
        ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories
        self.n_features = d_numerical + len(categories)

        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias = bias)

        self.map_noise = PositionalEmbedding(num_channels=d_token)
        self.time_embed = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.SiLU(),
            nn.Linear(d_token, d_token)
        )

        self.tabnet_steps = nn.ModuleList([
            TabNetDenoiseStep(d_token, self.n_features, n_head, gamma=gamma)
            for _ in range(n_steps)
        ])

        self.detokenizer = Reconstructor(d_numerical, categories, d_token)

        self.model = nn.ModuleList([self.tokenizer, self.time_embed, self.tabnet_steps, self.detokenizer])

    def forward(self, x_num, x_cat, timesteps):
        e = self.tokenizer(x_num, x_cat)
        x = e[:, 1:, :]                              # ignore CLS, as before

        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        t_emb = self.time_embed(emb)                   # (B, d_token)

        B, Fn, _ = x.shape
        prior_scale = torch.ones(B, Fn, device=x.device)
        agg = torch.zeros_like(x)
        masks = []                                     # kept for mask regularizer / diagnostics

        for step in self.tabnet_steps:
            decision, x, mask, prior_scale = step(x, prior_scale, t_emb)
            agg = agg + decision
            masks.append(mask)

        self.last_masks = masks                         # side-channel for _mask_regularization

        x_num_pred, x_cat_pred = self.detokenizer(agg)
        x_cat_pred = torch.cat(x_cat_pred, dim=-1) if len(x_cat_pred)>0 else torch.zeros_like(x_cat).to(x_num_pred.dtype)

        return x_num_pred, x_cat_pred


class Precond(nn.Module):
    def __init__(self,
        denoise_fn,
        sigma_data = 0.5,              # Expected standard deviation of the training data.
        net_conditioning = "sigma",
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.net_conditioning = net_conditioning
        self.denoise_fn_F = denoise_fn

    def forward(self, x_num, x_cat, t, sigma):

        x_num = x_num.to(torch.float32)

        sigma = sigma.to(torch.float32)
        assert sigma.ndim == 2
        if sigma.dim() > 1: # if learnable column-wise noise schedule, sigma conditioning is set to the defaults schedule of rho=7
            sigma_cond = (0.002 ** (1/7) + t * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
        else:
            sigma_cond = sigma 
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma_cond.log() / 4

        x_in = c_in * x_num
        if self.net_conditioning == "sigma":
            F_x, x_cat_pred = self.denoise_fn_F(x_in, x_cat, c_noise.flatten())
        elif self.net_conditioning == "t":
            F_x, x_cat_pred = self.denoise_fn_F(x_in, x_cat, t)

        assert F_x.dtype == dtype
        D_x = c_skip * x_num + c_out * F_x.to(torch.float32)
        
        return D_x, x_cat_pred
    

class Model(nn.Module):
    def __init__(
            self, denoise_fn,
            sigma_data=0.5, 
            precond=False, 
            net_conditioning="sigma",
            **kwargs
        ):
        super().__init__()
        self.precond = precond
        if precond:
            self.denoise_fn_D = Precond(
                denoise_fn,
                sigma_data=sigma_data,
                net_conditioning=net_conditioning
            )
        else:
            self.denoise_fn_D = denoise_fn

    def forward(self, x_num, x_cat, t, sigma=None):
        if self.precond:
            return self.denoise_fn_D(x_num, x_cat, t, sigma)
        else:
            return self.denoise_fn_D(x_num, x_cat, t)