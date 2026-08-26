"""Lightweight Doob guides for mixed tabular states.

The current path uses separately parameterized FT-periodic networks for the
numerical h-score and categorical scalar log-h, both conditioned on the whole
mixed noisy state, time, and optional active-column mask. Legacy single-guide
checkpoints remain loadable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from torch import Tensor, nn

from tabdiff.modules.main_modules import FTBlock, PeriodicTokenizer, PositionalEmbedding


def categorical_log_h_ratios(
    guide: nn.Module,
    x_num_t: Tensor,
    x_cat_t: Tensor,
    t: Tensor,
    num_classes: Sequence[int],
    mask_index: Tensor,
    to_one_hot,
    query_active_mask: Tensor | None = None,
    candidate_batch_size: int = 65536,
) -> Tensor:
    """Evaluate ``log h(child) - log h(current)`` for categorical children.

    The helper is differentiable for guide training and is reused under
    ``no_grad`` by the sampler, keeping both paths identical.
    """
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    b, n_columns = x_cat_t.shape
    max_classes_with_mask = int(mask_index.max().item()) + 1
    log_ratios = x_num_t.new_zeros((b, n_columns, max_classes_with_mask))
    current_one_hot = to_one_hot(x_cat_t).to(x_num_t.dtype)
    current_log_h = guide.log_h(
        x_num_t,
        current_one_hot,
        t,
        query_active_mask=query_active_mask,
    )

    for column, class_count_value in enumerate(num_classes):
        class_count = int(class_count_value)
        masked_rows = torch.nonzero(
            x_cat_t[:, column] == mask_index[column],
            as_tuple=False,
        ).flatten()
        if masked_rows.numel() == 0:
            continue

        repeated_rows = masked_rows.repeat_interleave(class_count)
        candidate_categories = x_cat_t[repeated_rows].clone()
        candidate_categories[:, column] = torch.arange(
            class_count,
            device=x_cat_t.device,
        ).repeat(masked_rows.numel())

        child_log_h_parts = []
        for start in range(0, len(repeated_rows), candidate_batch_size):
            stop = start + candidate_batch_size
            rows = repeated_rows[start:stop]
            child_one_hot = to_one_hot(candidate_categories[start:stop]).to(
                x_num_t.dtype
            )
            child_log_h_parts.append(
                guide.log_h(
                    x_num_t[rows],
                    child_one_hot,
                    t[rows],
                    query_active_mask=(
                        None
                        if query_active_mask is None
                        else query_active_mask[rows]
                    ),
                )
            )
        child_log_h = torch.cat(child_log_h_parts)
        column_ratios = (
            child_log_h - current_log_h[repeated_rows]
        ).reshape(masked_rows.numel(), class_count)
        log_ratios[
            masked_rows[:, None],
            column,
            torch.arange(class_count, device=x_cat_t.device)[None, :],
        ] = column_ratios
    return log_ratios


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

    def _active_mask(self, x_num: Tensor, active_mask: Tensor | None) -> Tensor:
        if active_mask is None:
            return torch.ones_like(x_num, dtype=torch.bool)
        active_mask = torch.as_tensor(active_mask, device=x_num.device)
        if active_mask.ndim == 1:
            if active_mask.numel() != self.lower.numel():
                raise ValueError(
                    f"expected {self.lower.numel()} query-mask entries, "
                    f"got {active_mask.numel()}"
                )
            active_mask = active_mask[None, :].expand(x_num.shape[0], -1)
        if active_mask.shape != x_num.shape:
            raise ValueError(
                f"expected query mask with shape {tuple(x_num.shape)}, "
                f"got {tuple(active_mask.shape)}"
            )
        return active_mask.bool()

    def contains(self, x_num: Tensor, active_mask: Tensor | None = None) -> Tensor:
        """Return whether every active numerical interval contains each row."""
        if x_num.ndim != 2 or x_num.shape[1] != self.lower.numel():
            raise ValueError(
                f"expected [rows, {self.lower.numel()}] numerical input, got {tuple(x_num.shape)}"
            )
        lower = self.lower.to(device=x_num.device, dtype=x_num.dtype)
        upper = self.upper.to(device=x_num.device, dtype=x_num.dtype)
        in_range = (x_num >= lower) & (x_num <= upper)
        active = self._active_mask(x_num, active_mask)
        return (in_range | ~active).all(dim=1)

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
    """Small FT-periodic guide over ``x_t``, time, and an active-column mask.

    The interval endpoints remain fixed and are deliberately not network inputs.
    When mask conditioning is enabled, the binary value ``a_j`` is embedded into
    numerical token ``j`` so one guide represents all subsets of that fixed box.
    The default remains disabled so older fixed-query checkpoints still load.
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
        query_mask_conditioning: bool = False,
        scalar_h_gradient: bool = False,
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
            "query_mask_conditioning": query_mask_conditioning,
            "scalar_h_gradient": scalar_h_gradient,
        }
        self.query_mask_conditioning = bool(query_mask_conditioning)
        self.scalar_h_gradient = bool(scalar_h_gradient)
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
        self.query_active_embed = None
        if self.query_mask_conditioning:
            self.query_active_embed = nn.Sequential(
                nn.Linear(1, d_token),
                nn.SiLU(),
                nn.Linear(d_token, d_token),
            )
        self.blocks = nn.ModuleList(
            [FTBlock(d_token, n_head, d_ffn_factor=factor) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_token)
        self.correction_head = None
        if not self.scalar_h_gradient:
            # Backward-compatible path for earlier two-head checkpoints.
            self.correction_head = nn.Linear(d_token, 1)
        self.h_logit_head = nn.Linear(d_token, 1)

        if self.correction_head is not None:
            nn.init.zeros_(self.correction_head.weight)
            nn.init.zeros_(self.correction_head.bias)
        nn.init.zeros_(self.h_logit_head.weight)
        nn.init.zeros_(self.h_logit_head.bias)

    def _encode(
        self,
        x_num_t: Tensor,
        x_cat_t: Tensor | None,
        t: Tensor,
        query_active_mask: Tensor | None = None,
    ) -> Tensor:
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
        if self.query_mask_conditioning:
            if query_active_mask is None:
                query_active_mask = x_num_t.new_ones(
                    (x_num_t.shape[0], self.d_numerical)
                )
            else:
                query_active_mask = torch.as_tensor(
                    query_active_mask,
                    device=x_num_t.device,
                    dtype=x_num_t.dtype,
                )
                if query_active_mask.ndim == 1:
                    query_active_mask = query_active_mask[None, :].expand(
                        x_num_t.shape[0], -1
                    )
                if query_active_mask.shape != (
                    x_num_t.shape[0],
                    self.d_numerical,
                ):
                    raise ValueError(
                        "query_active_mask must have shape "
                        f"({x_num_t.shape[0]}, {self.d_numerical})"
                    )
            active_embedding = self.query_active_embed(
                query_active_mask.unsqueeze(-1)
            )
            tokens = torch.cat(
                (
                    tokens[:, : self.d_numerical] + active_embedding,
                    tokens[:, self.d_numerical :],
                ),
                dim=1,
            )
        t_emb = self.map_time(t)
        t_emb = t_emb.reshape(t_emb.shape[0], 2, -1).flip(1).reshape_as(t_emb)
        t_emb = self.time_embed(t_emb)
        for block in self.blocks:
            tokens = block(tokens, t_emb)
        return self.final_norm(tokens)

    def h_logit(
        self,
        x_num_t: Tensor,
        x_cat_t: Tensor | None,
        t: Tensor,
        query_active_mask: Tensor | None = None,
    ) -> Tensor:
        """Return the scalar logit of the joint harmonic function."""
        tokens = self._encode(x_num_t, x_cat_t, t, query_active_mask)
        return self.h_logit_head(tokens.mean(dim=1)).squeeze(-1)

    def grad_log_h(
        self,
        x_num_t: Tensor,
        x_cat_t: Tensor | None,
        t: Tensor,
        query_active_mask: Tensor | None = None,
        create_graph: bool = False,
    ) -> Tensor:
        """Differentiate the same scalar ``log h`` used by categorical ratios."""
        with torch.enable_grad():
            x_for_grad = x_num_t.detach().requires_grad_(True)
            log_h_value = self.log_h(
                x_for_grad,
                x_cat_t,
                t,
                query_active_mask=query_active_mask,
            )
            gradient = torch.autograd.grad(
                log_h_value.sum(),
                x_for_grad,
                create_graph=create_graph,
            )[0]
        return gradient if create_graph else gradient.detach()

    def forward_with_log_h(
        self,
        x_num_t: Tensor,
        x_cat_t: Tensor | None,
        t: Tensor,
        query_active_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return raw Gaussian ``grad log h`` and scalar ``h`` logit."""
        h_logit = self.h_logit(x_num_t, x_cat_t, t, query_active_mask)
        if self.scalar_h_gradient:
            correction = self.grad_log_h(
                x_num_t,
                x_cat_t,
                t,
                query_active_mask=query_active_mask,
            )
        else:
            tokens = self._encode(x_num_t, x_cat_t, t, query_active_mask)
            correction = self.correction_head(
                tokens[:, : self.d_numerical]
            ).squeeze(-1)
        return correction, h_logit

    def forward(
        self,
        x_num_t: Tensor,
        x_cat_t: Tensor | None,
        t: Tensor,
        query_active_mask: Tensor | None = None,
    ) -> Tensor:
        if self.scalar_h_gradient:
            return self.grad_log_h(x_num_t, x_cat_t, t, query_active_mask)
        tokens = self._encode(x_num_t, x_cat_t, t, query_active_mask)
        return self.correction_head(tokens[:, : self.d_numerical]).squeeze(-1)

    def log_h(
        self,
        x_num_t: Tensor,
        x_cat_t: Tensor | None,
        t: Tensor,
        query_active_mask: Tensor | None = None,
    ) -> Tensor:
        """Return the shared scalar log-potential used for Doob ratios.

        New Section 5 checkpoints train only derivatives of this potential and
        therefore use the raw scalar. Legacy BCE checkpoints retain their
        probability interpretation through ``logsigmoid``.
        """
        h_logit = self.h_logit(x_num_t, x_cat_t, t, query_active_mask)
        if self.scalar_h_gradient:
            return h_logit
        return torch.nn.functional.logsigmoid(h_logit)

    def config_dict(self) -> Dict[str, Any]:
        return dict(self._config)


class NumericalHScoreGuide(NumericalDoobHGuide):
    """FT-periodic vector guide for ``sigma^2 grad_x log h``.

    This network has its own backbone and only a numerical token-wise output
    head. It consumes the complete mixed noisy state and time.
    """

    def __init__(self, **kwargs) -> None:
        kwargs = dict(kwargs)
        kwargs["scalar_h_gradient"] = False
        super().__init__(**kwargs)
        self.h_logit_head = None


class CategoricalHTransformGuide(NumericalDoobHGuide):
    """FT-periodic scalar ``log h`` guide for categorical Doob ratios.

    This is a separate network from :class:`NumericalHScoreGuide`, with the
    same backbone architecture but independent parameters and a scalar head.
    """

    def __init__(self, **kwargs) -> None:
        kwargs = dict(kwargs)
        kwargs["scalar_h_gradient"] = False
        super().__init__(**kwargs)
        self.correction_head = None

    def forward(
        self,
        x_num_t: Tensor,
        x_cat_t: Tensor | None,
        t: Tensor,
        query_active_mask: Tensor | None = None,
    ) -> Tensor:
        return self.log_h(x_num_t, x_cat_t, t, query_active_mask)

    def log_h(
        self,
        x_num_t: Tensor,
        x_cat_t: Tensor | None,
        t: Tensor,
        query_active_mask: Tensor | None = None,
    ) -> Tensor:
        return self.h_logit(x_num_t, x_cat_t, t, query_active_mask)
