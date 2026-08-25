"""Lightweight numerical Doob h-transform guide for a fixed terminal event.

The guide predicts the denoiser-space correction

    delta_D(x_t, t) = sigma(t)^2 * grad_x log h(t, x_t),

where h(t, x_t) = P(X_0 in B | X_t=x_t).  Predicting ``delta_D`` instead of
the raw score is a numerically stable reparameterization of the score-matching
objective in Section 5 of the project note.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from torch import Tensor, nn

from tabdiff.modules.main_modules import FTBlock, PeriodicTokenizer, PositionalEmbedding


@dataclass(frozen=True)
class NumericalBoxQuery:
    """A single fixed hyper-rectangle over every normalized numerical column."""

    lower: Tensor
    upper: Tensor

    def __post_init__(self) -> None:
        if self.lower.ndim != 1 or self.upper.ndim != 1:
            raise ValueError("query bounds must be one-dimensional")
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper query bounds must have the same shape")
        if not torch.all(self.lower <= self.upper):
            raise ValueError("every lower bound must be <= its upper bound")

    @classmethod
    def from_quantiles(
        cls,
        x_num: Tensor,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> "NumericalBoxQuery":
        if x_num.ndim != 2 or x_num.shape[1] == 0:
            raise ValueError("x_num must be a non-empty [rows, numerical_columns] tensor")
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
        return cls(
            lower=torch.quantile(x_num, lower_quantile, dim=0),
            upper=torch.quantile(x_num, upper_quantile, dim=0),
        )

    def contains(self, x_num: Tensor) -> Tensor:
        """Return one boolean per row; every numerical column must be in range."""
        if x_num.ndim != 2 or x_num.shape[1] != self.lower.numel():
            raise ValueError(
                f"expected [rows, {self.lower.numel()}] numerical input, got {tuple(x_num.shape)}"
            )
        lower = self.lower.to(device=x_num.device, dtype=x_num.dtype)
        upper = self.upper.to(device=x_num.device, dtype=x_num.dtype)
        return ((x_num >= lower) & (x_num <= upper)).all(dim=1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower": self.lower.detach().cpu().tolist(),
            "upper": self.upper.detach().cpu().tolist(),
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "NumericalBoxQuery":
        return cls(
            lower=torch.tensor(state["lower"], dtype=torch.float32),
            upper=torch.tensor(state["upper"], dtype=torch.float32),
        )


class NumericalDoobHGuide(nn.Module):
    """Small FT-periodic network parameterized only by the full ``x_t`` and time.

    The fixed query is deliberately *not* an input.  One trained guide therefore
    represents one terminal event B, which is the requested first experiment.
    """

    def __init__(
        self,
        d_numerical: int,
        categories: Sequence[int] | None = None,
        d_token: int = 32,
        num_layers: int = 2,
        n_head: int = 4,
        factor: float = 2.0,
        n_frequencies: int = 16,
        freq_sigma: float = 0.05,
    ) -> None:
        super().__init__()
        if d_numerical <= 0:
            raise ValueError("the numerical Doob guide needs at least one numerical column")
        if d_token < 4 or d_token % 2 != 0 or d_token % n_head != 0:
            raise ValueError("d_token must be even, at least 4, and divisible by n_head")
        if num_layers <= 0 or n_frequencies <= 0 or factor <= 0:
            raise ValueError("num_layers, n_frequencies, and factor must be positive")

        categories = list(categories or [])
        self.d_numerical = d_numerical
        self.categories = categories
        self.d_categorical_one_hot = sum(categories)
        self._config = {
            "d_numerical": d_numerical,
            "categories": categories,
            "d_token": d_token,
            "num_layers": num_layers,
            "n_head": n_head,
            "factor": factor,
            "n_frequencies": n_frequencies,
            "freq_sigma": freq_sigma,
        }
        self.tokenizer = PeriodicTokenizer(
            d_numerical=d_numerical,
            categories=categories if categories else None,
            d_token=d_token,
            bias=True,
            n_frequencies=n_frequencies,
            freq_sigma=freq_sigma,
        )
        self.map_time = PositionalEmbedding(num_channels=d_token)
        self.time_embed = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.SiLU(),
            nn.Linear(d_token, d_token),
        )
        self.blocks = nn.ModuleList(
            [FTBlock(d_token, n_head, d_ffn_factor=factor) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_token)
        self.correction_head = nn.Linear(d_token, 1)

        # With a fresh guide, sampling is exactly the unconditional sampler.
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def forward(self, x_num_t: Tensor, x_cat_t: Tensor | None, t: Tensor) -> Tensor:
        if x_num_t.ndim != 2 or x_num_t.shape[1] != self.d_numerical:
            raise ValueError(
                f"expected [batch, {self.d_numerical}] numerical input, got {tuple(x_num_t.shape)}"
            )
        if self.categories:
            if x_cat_t is None or x_cat_t.shape != (
                x_num_t.shape[0],
                self.d_categorical_one_hot,
            ):
                raise ValueError(
                    "expected categorical one-hot input with shape "
                    f"({x_num_t.shape[0]}, {self.d_categorical_one_hot})"
                )
        else:
            x_cat_t = None

        t = t.reshape(-1).to(device=x_num_t.device, dtype=x_num_t.dtype)
        if t.numel() == 1:
            t = t.expand(x_num_t.shape[0])
        if t.numel() != x_num_t.shape[0]:
            raise ValueError("time must be scalar or have one value per input row")

        tokens = self.tokenizer(x_num_t, x_cat_t)[:, 1:, :]
        t_emb = self.map_time(t)
        t_emb = t_emb.reshape(t_emb.shape[0], 2, -1).flip(1).reshape_as(t_emb)
        t_emb = self.time_embed(t_emb)
        for block in self.blocks:
            tokens = block(tokens, t_emb)
        numeric_tokens = self.final_norm(tokens)[:, : self.d_numerical]
        return self.correction_head(numeric_tokens).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return dict(self._config)
