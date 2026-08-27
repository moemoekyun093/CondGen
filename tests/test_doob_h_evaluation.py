import unittest

import pandas as pd

from tabdiff.doob_h_evaluation import (
    compare_correlation_matrices,
    raw_constraint_report,
    raw_modality_constraint_report,
)


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

    def test_one_sided_paper_inequality(self):
        frame = pd.DataFrame({"Administrative": [3.0, 4.0, 5.0]})
        query = {
            "columns": [
                {
                    "name": "Administrative",
                    "raw_lower": 4.0,
                    "raw_upper": None,
                }
            ]
        }
        report, mask = raw_constraint_report(frame, query)
        self.assertEqual(mask.tolist(), [False, True, True])
        self.assertAlmostEqual(report["joint_hit_rate"], 2 / 3)

    def test_full_arity_predicates_support_intervals_and_sets(self):
        frame = pd.DataFrame(
            {
                "age": [20.0, 30.0, 40.0],
                "group": ["a", "b", "c"],
            }
        )
        query = {
            "query_id": "mixed",
            "predicates": [
                {
                    "col": "age",
                    "modality": "numeric",
                    "op": "between",
                    "values": [20.0, 35.0],
                },
                {
                    "col": "group",
                    "modality": "categorical",
                    "op": "in",
                    "values": ["a", "c"],
                },
            ],
        }
        report, mask = raw_constraint_report(frame, query)
        self.assertEqual(mask.tolist(), [True, False, False])
        self.assertEqual(report["constraint_id"], "mixed")
        self.assertAlmostEqual(report["per_column"][1]["hit_rate"], 2 / 3)

    def test_modality_miss_rates_separate_joint_and_column_averages(self):
        frame = pd.DataFrame(
            {
                "n1": [0.0, 2.0, 0.0],
                "n2": [0.0, 0.0, 2.0],
                "cat": ["a", "b", "a"],
            }
        )
        query = {
            "predicates": [
                {"col": "n1", "modality": "numeric", "op": "between", "values": [0, 1]},
                {"col": "n2", "modality": "numeric", "op": "between", "values": [0, 1]},
                {"col": "cat", "modality": "categorical", "op": "in", "values": ["a"]},
            ]
        }
        report = raw_modality_constraint_report(frame, query)
        self.assertEqual(report["numeric"]["num_constraints"], 2)
        self.assertAlmostEqual(report["numeric"]["joint_miss_rate"], 2 / 3)
        self.assertAlmostEqual(
            report["numeric"]["mean_per_constraint_miss_rate"], 1 / 3
        )
        self.assertAlmostEqual(report["categorical"]["joint_miss_rate"], 1 / 3)


class CorrelationComparisonTest(unittest.TestCase):
    def test_reports_structure_similarity_and_changed_pairs(self):
        left = pd.DataFrame(
            [[1.0, 0.2, -0.1], [0.2, 1.0, 0.4], [-0.1, 0.4, 1.0]],
            columns=["a", "b", "c"],
            index=["a", "b", "c"],
        )
        right = pd.DataFrame(
            [[1.0, 0.3, -0.1], [0.3, 1.0, 0.1], [-0.1, 0.1, 1.0]],
            columns=["a", "b", "c"],
            index=["a", "b", "c"],
        )
        report = compare_correlation_matrices(left, right)
        self.assertEqual(report["compared_pairs"], 3)
        self.assertAlmostEqual(report["mean_absolute_correlation_change"], 0.4 / 3)
        self.assertEqual(
            {report["top_absolute_changes"][0]["column_1"],
             report["top_absolute_changes"][0]["column_2"]},
            {"b", "c"},
        )


if __name__ == "__main__":
    unittest.main()
