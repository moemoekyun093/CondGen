import random
import unittest

from tabdiff.doob_query_curriculum import (
    QueryCurriculumSampler,
    parse_bucket_probabilities,
    resolution_bucket,
)


class DummyQuery:
    def __init__(self, query_id, target_band, *, train_selectivity=None, realized_band=None):
        self.query_id = query_id
        self.specification = {"target_band": target_band}
        if train_selectivity is not None:
            self.specification["selectivity"] = {"train": train_selectivity}
        if realized_band is not None:
            self.specification["realized_band"] = realized_band


class QueryCurriculumTest(unittest.TestCase):
    def test_selectivity_bucket_boundaries(self):
        kwargs = {"tight_max_band": 0.01, "broad_min_band": 0.1}
        self.assertEqual(resolution_bucket(0.005, **kwargs), "tight")
        self.assertEqual(resolution_bucket(0.01, **kwargs), "tight")
        self.assertEqual(resolution_bucket(0.02, **kwargs), "medium")
        self.assertEqual(resolution_bucket(0.05, **kwargs), "medium")
        self.assertEqual(resolution_bucket(0.1, **kwargs), "broad")
        self.assertEqual(resolution_bucket(0.4, **kwargs), "broad")

    def test_probabilities_are_normalized(self):
        self.assertEqual(parse_bucket_probabilities("7,2.5,0.5"), (0.7, 0.25, 0.05))

    def test_schedule_reaches_final_mixture(self):
        queries = [
            DummyQuery("broad", 0.4),
            DummyQuery("medium", 0.05),
            DummyQuery("tight", 0.005),
        ]
        sampler = QueryCurriculumSampler(
            queries,
            total_steps=12,
            warmup_steps=2,
            transition_steps=4,
            warmup_probabilities=(0.7, 0.25, 0.05),
            final_probabilities=(0.25, 0.35, 0.4),
            tight_max_band=0.01,
            broad_min_band=0.1,
        )
        self.assertEqual(sampler.phase_and_probabilities(1)[0], "warmup")
        self.assertEqual(sampler.phase_and_probabilities(4)[0], "transition")
        phase, probabilities = sampler.phase_and_probabilities(7)
        self.assertEqual(phase, "final_mixture")
        self.assertEqual(probabilities, (0.25, 0.35, 0.4))
        query, bucket, band, phase = sampler.sample(12, random.Random(0))
        self.assertEqual(query.specification["target_band"], band)
        self.assertEqual(phase, "final_mixture")
        self.assertIn(bucket, ("broad", "medium", "tight"))

    def test_single_available_bucket_is_renormalized(self):
        sampler = QueryCurriculumSampler(
            [DummyQuery("only", 0.005)],
            total_steps=3,
            warmup_steps=1,
            transition_steps=1,
            warmup_probabilities=(0.7, 0.25, 0.05),
            final_probabilities=(0.25, 0.35, 0.4),
            tight_max_band=0.01,
            broad_min_band=0.1,
        )
        self.assertEqual(sampler.phase_and_probabilities(1)[1], (0.0, 0.0, 1.0))

    def test_realized_selectivity_overrides_nominal_target(self):
        query = DummyQuery(
            "nominally-tight-but-realized-broad",
            0.005,
            train_selectivity=0.268,
            realized_band="25-50%",
        )
        sampler = QueryCurriculumSampler(
            [query],
            total_steps=3,
            warmup_steps=1,
            transition_steps=1,
            warmup_probabilities=(0.5, 0.3, 0.2),
            final_probabilities=(0.15, 0.25, 0.6),
            tight_max_band=0.01,
            broad_min_band=0.1,
            selectivity_source="realized_train",
        )
        self.assertEqual(sampler.summary()["broad"], {"25-50%": 1})
        sampled, bucket, band, _ = sampler.sample(1, random.Random(0))
        self.assertIs(sampled, query)
        self.assertEqual(bucket, "broad")
        self.assertEqual(band, "25-50%")

    def test_multi_query_sample_is_distinct(self):
        queries = [
            DummyQuery(f"broad-{index}", 0.4) for index in range(4)
        ] + [
            DummyQuery(f"tight-{index}", 0.005) for index in range(4)
        ]
        sampler = QueryCurriculumSampler(
            queries,
            total_steps=3,
            warmup_steps=1,
            transition_steps=1,
            warmup_probabilities=(0.5, 0.0, 0.5),
            final_probabilities=(0.5, 0.0, 0.5),
            tight_max_band=0.01,
            broad_min_band=0.1,
        )
        selected = sampler.sample_distinct(1, random.Random(0), 6)
        query_ids = [query.query_id for query, _, _, _ in selected]
        self.assertEqual(len(query_ids), 6)
        self.assertEqual(len(set(query_ids)), 6)

    def test_multi_query_sample_rejects_oversized_request(self):
        sampler = QueryCurriculumSampler(
            [DummyQuery("only", 0.005)],
            total_steps=3,
            warmup_steps=1,
            transition_steps=1,
            warmup_probabilities=(0.0, 0.0, 1.0),
            final_probabilities=(0.0, 0.0, 1.0),
            tight_max_band=0.01,
            broad_min_band=0.1,
        )
        with self.assertRaisesRegex(ValueError, "distinct queries"):
            sampler.sample_distinct(1, random.Random(0), 2)


if __name__ == "__main__":
    unittest.main()
