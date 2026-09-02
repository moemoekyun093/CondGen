import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tabdiff.baseline_data import load_baseline_table, query_categorical_observations


class BaselineTableTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        train = pd.DataFrame(
            {"cat": ["a", "a", "b"], "num": [1.0, 2.0, 3.0], "target": [0, 1, 0]}
        )
        test = pd.DataFrame(
            {"cat": ["c"], "num": [4.0], "target": [1]}
        )
        info = {
            "num_col_idx": [1],
            "cat_col_idx": [0],
            "target_col_idx": [2],
            "task_type": "binclass",
        }
        train.to_csv(root / "train.csv", index=False)
        test.to_csv(root / "test.csv", index=False)
        (root / "info.json").write_text(json.dumps(info))
        self.table = load_baseline_table(
            train_data=root / "train.csv",
            test_data=root / "test.csv",
            info_file=root / "info.json",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_round_trip_and_test_schema_category(self):
        encoded = self.table.encode(self.table.train)
        decoded = self.table.decode(encoded)
        self.assertEqual(list(decoded.columns), list(self.table.train.columns))
        self.assertEqual(decoded["cat"].tolist(), ["a", "a", "b"])
        cat_index = self.table.categorical_columns.index("cat")
        self.assertIn("c", set(self.table.encoder.categories_[cat_index]))

    def test_query_set_is_restricted_and_numeric_is_missing(self):
        query = {
            "predicates": [
                {"col": "num", "modality": "numeric", "op": "between", "values": [1, 2]},
                {"col": "cat", "modality": "categorical", "op": "in", "values": ["b"]},
            ]
        }
        observed, metadata = query_categorical_observations(
            self.table, query, 20, np.random.default_rng(7)
        )
        self.assertTrue(observed["num"].isna().all())
        self.assertEqual(set(observed["cat"]), {"b"})
        self.assertEqual(metadata["numerical_interval_adapter"], "left_missing_unconditional")

    def test_unknown_allowed_set_fails(self):
        query = {
            "predicates": [
                {"col": "cat", "modality": "categorical", "op": "in", "values": ["z"]}
            ]
        }
        with self.assertRaisesRegex(ValueError, "no allowed value"):
            query_categorical_observations(self.table, query, 2, np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()
