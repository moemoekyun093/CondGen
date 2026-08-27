import random
import unittest

from tabdiff.doob_query_curriculum import (
    QueryCurriculumSampler,
    parse_bucket_probabilities,
    resolution_bucket,
)


class DummyQuery:
    def __init__(self, query_id, target_band):
        self.query_id = query_id
        self.specification = {"target_band": target_band}


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


if __name__ == "__main__":
    unittest.main()
