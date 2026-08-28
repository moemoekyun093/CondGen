import unittest

import numpy as np
import pandas as pd
import torch

from tabdiff.doob_query_masking import (
    eligible_indices_for_predicate_mask,
    mask_query_kwargs,
    masked_query_specification,
    parse_predicate_mask,
    predicate_hit_matrix,
    sample_predicate_mask,
)


class QueryMaskingTest(unittest.TestCase):
    def setUp(self):
        self.specification = {
            "query_id": "q1",
            "target_band": 0.01,
            "predicates": [
                {"col": "age", "modality": "numeric", "op": "between", "values": [20, 40]},
                {"col": "Month", "modality": "categorical", "op": "in", "values": ["May"]},
            ],
        }

    def test_anchor_and_random_masks(self):
        active, kind = sample_predicate_mask(
            4,
            device=torch.device("cpu"),
            random_active_probability=0.5,
            all_active_probability=1.0,
            all_inactive_probability=0.0,
        )
        self.assertEqual(kind, "all_active")
        self.assertTrue(active.all())
        inactive, kind = sample_predicate_mask(
            4,
            device=torch.device("cpu"),
            random_active_probability=0.5,
            all_active_probability=0.0,
            all_inactive_probability=1.0,
        )
        self.assertEqual(kind, "all_inactive")
        self.assertFalse(inactive.any())

    def test_partial_support_uses_only_active_predicates(self):
        frame = pd.DataFrame({"age": [25, 50, 30], "Month": ["May", "May", "Jun"]})
        hits = predicate_hit_matrix(frame, self.specification)
        core = np.array([0, 1, 2])
        age_only = eligible_indices_for_predicate_mask(
            hits, core, torch.tensor([True, False])
        )
        self.assertEqual(age_only.tolist(), [0, 2])
        none = eligible_indices_for_predicate_mask(
            hits, core, torch.tensor([False, False])
        )
        self.assertEqual(none.tolist(), [0, 1, 2])

    def test_mask_maps_specification_order_to_model_order(self):
        kwargs = {
            "query_lower": torch.zeros(2),
            "query_upper": torch.ones(2),
            "query_numerical_active": torch.ones(2),
            "query_categorical_allowed": torch.ones(3),
            "query_categorical_active": torch.ones(1),
        }
        masked = mask_query_kwargs(
            kwargs,
            self.specification,
            torch.tensor([False, True]),
            numerical_names=["income", "age"],
            categorical_names=["Month"],
        )
        self.assertEqual(masked["query_numerical_active"].tolist(), [0.0, 0.0])
        self.assertEqual(masked["query_categorical_active"].tolist(), [1.0])

    def test_sampling_mask_accepts_column_names(self):
        mask = parse_predicate_mask(
            self.specification,
            active_columns="Month",
            predicate_mask=None,
            device=torch.device("cpu"),
        )
        self.assertEqual(mask.tolist(), [False, True])
        masked = masked_query_specification(self.specification, mask)
        self.assertEqual(masked["active_columns"], ["Month"])
        self.assertEqual(masked["arity"], 1)


if __name__ == "__main__":
    unittest.main()
