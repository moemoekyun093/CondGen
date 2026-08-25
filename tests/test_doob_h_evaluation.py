import unittest

import pandas as pd

from tabdiff.doob_h_evaluation import raw_constraint_report


class RawConstraintEvaluationTest(unittest.TestCase):
    def test_joint_and_per_column_rates_use_raw_values(self):
        frame = pd.DataFrame(
            {
                "a": [0.0, 1.0, 2.0],
                "special": [0.0, 0.0, 0.2],
            }
        )
        query = {
            "constraint_id": "test",
            "columns": [
                {"name": "a", "raw_lower": 0.0, "raw_upper": 1.0},
                {"name": "special", "raw_lower": 0.0, "raw_upper": 0.0},
            ],
        }

        report, mask = raw_constraint_report(frame, query)

        self.assertEqual(mask.tolist(), [True, True, False])
        self.assertAlmostEqual(report["joint_hit_rate"], 2 / 3)
        self.assertAlmostEqual(report["per_column"][0]["hit_rate"], 2 / 3)
        self.assertAlmostEqual(report["per_column"][1]["hit_rate"], 2 / 3)

    def test_small_float_error_at_boundary_is_tolerated(self):
        frame = pd.DataFrame({"special": [1e-9]})
        query = {
            "columns": [
                {"name": "special", "raw_lower": 0.0, "raw_upper": 0.0},
            ],
        }
        report, _ = raw_constraint_report(frame, query)
        self.assertEqual(report["joint_hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
