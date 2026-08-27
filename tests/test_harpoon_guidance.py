import json
import tempfile
import unittest
from pathlib import Path

import torch

from sample_harpoon_fixed_box import (
    load_query_specs,
    parse_bound_specs,
    summed_squared_relu_loss,
)
from sample_harpoon_full_query import (
    categorical_allowed_set_loss,
    full_query_guidance_loss,
)


class HarpoonConstraintTest(unittest.TestCase):
    def test_default_is_paper_shoppers_range_task(self):
        self.assertEqual(
            parse_bound_specs([], []),
            [
                {
                    "name": "Administrative",
                    "raw_lower": 4.0,
                    "raw_upper": None,
                }
            ],
        )

    def test_new_test_time_conjunction(self):
        self.assertEqual(
            parse_bound_specs(
                ["Administrative=5", "PageValues=10"],
                ["Administrative=8"],
            ),
            [
                {
                    "name": "Administrative",
                    "raw_lower": 5.0,
                    "raw_upper": 8.0,
                },
                {
                    "name": "PageValues",
                    "raw_lower": 10.0,
                    "raw_upper": None,
                },
            ],
        )

    def test_rejects_inverted_interval(self):
        with self.assertRaisesRegex(ValueError, "lower bound exceeds"):
            parse_bound_specs(["Administrative=5"], ["Administrative=4"])

    def test_selects_nested_query_columns_in_requested_order(self):
        payload = {
            "columns": [
                {
                    "model_index": 0,
                    "name": "Administrative",
                    "raw_lower": 0.0,
                    "raw_upper": 6.0,
                },
                {
                    "model_index": 1,
                    "name": "Administrative_Duration",
                    "raw_lower": 0.0,
                    "raw_upper": 178.2,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            selected = load_query_specs(str(path), "1,0")
        self.assertEqual(
            [spec["name"] for spec in selected],
            ["Administrative_Duration", "Administrative"],
        )

    def test_and_guidance_adds_squared_relu_losses(self):
        values = torch.tensor([[0.0, 5.0], [2.0, 9.0]])
        lower = torch.tensor([1.0, 4.0])
        upper = torch.tensor([3.0, 7.0])
        loss = summed_squared_relu_loss(values, lower, upper)
        torch.testing.assert_close(loss, torch.tensor([1.0, 4.0]))

    def test_uses_nearest_allowed_one_hot_target(self):
        block = torch.tensor([[0.8, 0.1, 0.1], [0.0, 0.1, 0.9]])
        allowed = torch.tensor([0, 2])
        actual = categorical_allowed_set_loss(block, allowed)
        torch.testing.assert_close(actual, torch.tensor([0.4, 0.2]))

    def test_full_and_adds_numerical_and_categorical_terms(self):
        x0_hat = torch.tensor([[2.0, 0.8, 0.1, 0.1]])
        numerical_indices = torch.tensor([0])
        lower = torch.tensor([0.0])
        upper = torch.tensor([1.0])
        categorical = [(1, 4, torch.tensor([0, 2]))]
        actual = full_query_guidance_loss(
            x0_hat,
            numerical_indices,
            lower,
            upper,
            categorical,
        )
        # Numeric upper violation: (2 - 1)^2 = 1; categorical min L1 = 0.4.
        torch.testing.assert_close(actual, torch.tensor([1.4]))


if __name__ == "__main__":
    unittest.main()
