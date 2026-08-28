import unittest

import torch

from tabdiff.models.harpoon_style import (
    categorical_set_loss,
    interval_relu_loss,
)


class ConstraintLossTest(unittest.TestCase):
    def test_interval_loss_uses_only_active_columns(self):
        clean = torch.tensor([[0.0, 100.0], [3.0, -100.0]])
        lower = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        upper = torch.tensor([[2.0, 1.0], [2.0, 1.0]])
        active = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        expected = torch.tensor([1.0, 1.0])
        torch.testing.assert_close(
            interval_relu_loss(clean, lower, upper, active), expected
        )

    def test_categorical_set_loss_ignores_mask_logits(self):
        # Two columns: K=2 and K=3, each followed by its MASK logit.
        raw_logits = torch.tensor(
            [[5.0, -5.0, 100.0, -5.0, 5.0, -5.0, 100.0]]
        )
        allowed = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0]])
        active = torch.tensor([[1.0, 1.0]])
        loss = categorical_set_loss(raw_logits, allowed, active, [2, 3])
        self.assertLess(float(loss), 1e-6)

    def test_inactive_categorical_column_has_no_loss(self):
        raw_logits = torch.tensor([[-5.0, 5.0, 0.0]])
        allowed = torch.tensor([[1.0, 0.0]])
        active = torch.tensor([[0.0]])
        torch.testing.assert_close(
            categorical_set_loss(raw_logits, allowed, active, [2]),
            torch.zeros(1),
        )


if __name__ == "__main__":
    unittest.main()
