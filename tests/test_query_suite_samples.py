import unittest
from pathlib import Path

from tabdiff.query_suite_samples import replicate_sample_path


class ReplicateSamplePathTest(unittest.TestCase):
    def test_first_seed_preserves_legacy_direct_layout(self):
        root = Path("samples/method")
        self.assertEqual(
            replicate_sample_path(root, "query_1", 10000, 0),
            root / "query_1.csv",
        )

    def test_later_seeds_use_explicit_directories(self):
        root = Path("samples/method")
        self.assertEqual(
            replicate_sample_path(root, "query_1", 30000, 2),
            root / "seed_30000" / "query_1.csv",
        )


if __name__ == "__main__":
    unittest.main()
