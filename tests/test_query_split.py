import json
import tempfile
import unittest
from pathlib import Path

from tabdiff.query_split import (
    canonical_query_fingerprint,
    load_query_split,
    query_id_digest,
)


class QuerySplitTest(unittest.TestCase):
    def write_manifest(self, root, train, test, *, digest=None):
        path = Path(root) / "split.json"
        path.write_text(
            json.dumps(
                {
                    "source_query_ids_sha256": (
                        query_id_digest([*train, *test]) if digest is None else digest
                    ),
                    "partitions": {"train": train, "test": test},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_loads_disjoint_partitions(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_manifest(root, ["q1", "q2"], ["q3"])
            self.assertEqual(load_query_split(path, "train"), ["q1", "q2"])
            self.assertEqual(load_query_split(path, "test"), ["q3"])

    def test_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_manifest(root, ["q1"], ["q1"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_query_split(path, "train")

    def test_rejects_tampered_digest(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.write_manifest(root, ["q1"], ["q2"], digest="not-a-digest")
            with self.assertRaisesRegex(ValueError, "digest"):
                load_query_split(path, "test")

    def test_fingerprint_ignores_predicate_and_allowed_set_order(self):
        left = {
            "predicates": [
                {"col": "b", "modality": "categorical", "op": "in", "values": ["y", "x"]},
                {"col": "a", "modality": "numeric", "op": "between", "values": [1, 2]},
            ]
        }
        right = {
            "predicates": [
                {"col": "a", "modality": "numeric", "op": "between", "values": [1, 2]},
                {"col": "b", "modality": "categorical", "op": "in", "values": ["x", "y"]},
            ]
        }
        self.assertEqual(
            canonical_query_fingerprint(left),
            canonical_query_fingerprint(right),
        )


if __name__ == "__main__":
    unittest.main()
