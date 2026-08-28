"""Small, model-agnostic pieces of HARPOON-style test-time guidance."""

from __future__ import annotations

import torch
from torch import Tensor


def interval_relu_loss(
    clean_numerical: Tensor,
    lower: Tensor,
    upper: Tensor,
    active: Tensor,
) -> Tensor:
    """Per-row squared-ReLU loss for active closed numerical intervals."""
    below = torch.relu(lower - clean_numerical)
    above = torch.relu(clean_numerical - upper)
    return ((below.square() + above.square()) * active).sum(dim=1)


def categorical_set_loss(
    raw_logits: Tensor,
    allowed: Tensor,
    active: Tensor,
    category_counts: list[int],
) -> Tensor:
    """Per-row penalty on probability mass outside each active allowed set."""
    losses = raw_logits.new_zeros(raw_logits.shape[0])
    logit_offset = 0
    allowed_offset = 0
    for column, count in enumerate(category_counts):
        # TabDiff's denoiser emits K real-category logits plus one MASK logit.
        logits = raw_logits[:, logit_offset : logit_offset + count]
        allowed_column = allowed[:, allowed_offset : allowed_offset + count]
        allowed_mass = (logits.softmax(dim=1) * allowed_column).sum(dim=1)
        # This is the categorical analogue of a one-sided feasibility loss.
        violation = torch.relu(1.0 - allowed_mass)
        losses = losses + active[:, column] * violation.square()
        logit_offset += count + 1
        allowed_offset += count
    return losses
