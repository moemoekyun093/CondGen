import unittest

from sample_harpoon_fixed_box import parse_bound_specs


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


if __name__ == "__main__":
    unittest.main()
