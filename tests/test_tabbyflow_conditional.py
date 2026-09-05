import math
import unittest

import torch
import torch.nn as nn

from tabdiff.baselines.tabbyflow_conditional import (
    ConditionalTabbyFlowVelocity,
    EncodedQuery,
    truncated_normal_mean,
)


class ToyEndpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.d_numerical = 2
        self.categories = [3, 2]

    def forward(self, x_num, x_cat, timesteps):
        mean = torch.tensor([[0.0, 3.0]], device=x_num.device).expand(len(x_num), -1)
        logits = torch.tensor([[0.0, 1.0, 2.0, 2.0, 0.0]], device=x_num.device)
        return mean, logits.expand(len(x_num), -1)


def query(active_num=(True, False), active_cat=(True, False)):
    return EncodedQuery(
        query_id="q",
        specification={},
        numerical_lower=torch.tensor([0.0, -10.0]),
        numerical_upper=torch.tensor([10.0, 10.0]),
        numerical_active=torch.tensor(active_num),
        categorical_allowed=(torch.tensor([True, False, True]), torch.tensor([True, True])),
        categorical_active=torch.tensor(active_cat),
    )


class TruncatedNormalTest(unittest.TestCase):
    def test_half_normal_mean(self):
        result = truncated_normal_mean(
            torch.tensor([[0.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([[0.0]]),
            torch.tensor([[10.0]]),
        )
        self.assertAlmostEqual(result.item(), math.sqrt(2.0 / math.pi), places=5)

    def test_result_is_inside_narrow_interval(self):
        lower = torch.tensor([[-1.0]])
        upper = torch.tensor([[-0.9]])
        result = truncated_normal_mean(
            torch.tensor([[20.0]]),
            torch.tensor([[1.0]]),
            lower,
            upper,
        )
        self.assertGreaterEqual(result.item(), lower.item())
        self.assertLessEqual(result.item(), upper.item())


class ConditionalVelocityTest(unittest.TestCase):
    def test_allowed_set_masks_and_renormalizes(self):
        model = ToyEndpointModel()
        velocity = ConditionalTabbyFlowVelocity(model, query())
        _, probabilities = velocity.conditioned_predictions(
            torch.tensor(0.5), torch.zeros(4, 7)
        )
        expected = torch.softmax(torch.tensor([0.0, 2.0]), dim=0)
        self.assertTrue(torch.allclose(probabilities[0][:, [0, 2]], expected.expand(4, -1)))
        self.assertTrue(torch.equal(probabilities[0][:, 1], torch.zeros(4)))

    def test_inactive_columns_keep_unconditional_prediction(self):
        model = ToyEndpointModel()
        velocity = ConditionalTabbyFlowVelocity(
            model, query(active_num=(False, False), active_cat=(False, False))
        )
        mean, probabilities = velocity.conditioned_predictions(
            torch.tensor(0.5), torch.zeros(2, 7)
        )
        self.assertTrue(torch.equal(mean, torch.tensor([[0.0, 3.0], [0.0, 3.0]])))
        self.assertTrue(
            torch.allclose(probabilities[0][0], torch.softmax(torch.tensor([0.0, 1.0, 2.0]), 0))
        )


if __name__ == "__main__":
    unittest.main()
