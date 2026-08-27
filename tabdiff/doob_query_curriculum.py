"""Selectivity-stratified curriculum sampling for structured Doob queries."""

from __future__ import annotations

from collections import defaultdict


BUCKET_ORDER = ("broad", "medium", "tight")


def parse_bucket_probabilities(text: str) -> tuple[float, float, float]:
    """Parse broad,medium,tight probabilities from a command-line value."""
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise ValueError(
            "bucket probabilities must be comma-separated numbers in "
            "broad,medium,tight order"
        ) from error
    if len(values) != 3 or any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError(
            "bucket probabilities require three non-negative values with positive sum"
        )
    total = sum(values)
    return tuple(value / total for value in values)


def resolution_bucket(
    target_band: float,
    *,
    tight_max_band: float,
    broad_min_band: float,
) -> str:
    if target_band <= tight_max_band:
        return "tight"
    if target_band >= broad_min_band:
        return "broad"
    return "medium"


class QueryCurriculumSampler:
    """Sample bucket, then selectivity band, then query within that band."""

    def __init__(
        self,
        queries,
        *,
        total_steps: int,
        warmup_steps: int,
        transition_steps: int,
        warmup_probabilities: tuple[float, float, float],
        final_probabilities: tuple[float, float, float],
        tight_max_band: float,
        broad_min_band: float,
    ):
        if total_steps <= 0 or warmup_steps < 0 or transition_steps < 0:
            raise ValueError("curriculum step counts are invalid")
        if warmup_steps + transition_steps >= total_steps:
            raise ValueError(
                "curriculum must leave at least one optimizer step for the final mixture"
            )
        if tight_max_band >= broad_min_band:
            raise ValueError("tight-max-band must be smaller than broad-min-band")
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.transition_steps = transition_steps
        self.warmup_probabilities = warmup_probabilities
        self.final_probabilities = final_probabilities
        self.tight_max_band = tight_max_band
        self.broad_min_band = broad_min_band
        self.by_bucket = {
            bucket: defaultdict(list) for bucket in BUCKET_ORDER
        }
        for query in queries:
            band = float(query.specification["target_band"])
            bucket = resolution_bucket(
                band,
                tight_max_band=tight_max_band,
                broad_min_band=broad_min_band,
            )
            self.by_bucket[bucket][band].append(query)
        if not any(self.by_bucket[bucket] for bucket in BUCKET_ORDER):
            raise ValueError("cannot construct a curriculum from an empty query suite")

    @property
    def final_phase_start(self) -> int:
        return self.warmup_steps + self.transition_steps + 1

    def phase_and_probabilities(self, step: int) -> tuple[str, tuple[float, ...]]:
        if step < 1 or step > self.total_steps:
            raise ValueError("step is outside the configured curriculum")
        if step <= self.warmup_steps:
            phase = "warmup"
            probabilities = self.warmup_probabilities
        elif step <= self.warmup_steps + self.transition_steps:
            phase = "transition"
            fraction = (step - self.warmup_steps) / self.transition_steps
            probabilities = tuple(
                (1.0 - fraction) * warm + fraction * final
                for warm, final in zip(
                    self.warmup_probabilities, self.final_probabilities
                )
            )
        else:
            phase = "final_mixture"
            probabilities = self.final_probabilities

        available = tuple(bool(self.by_bucket[bucket]) for bucket in BUCKET_ORDER)
        masked = tuple(
            probability if is_available else 0.0
            for probability, is_available in zip(probabilities, available)
        )
        total = sum(masked)
        if total <= 0:
            raise ValueError(
                f"phase {phase} assigns zero probability to every available query bucket"
            )
        return phase, tuple(value / total for value in masked)

    def sample(self, step: int, rng):
        phase, probabilities = self.phase_and_probabilities(step)
        bucket = rng.choices(BUCKET_ORDER, weights=probabilities, k=1)[0]
        bands = sorted(self.by_bucket[bucket])
        band = rng.choice(bands)
        query = rng.choice(self.by_bucket[bucket][band])
        return query, bucket, band, phase

    def summary(self) -> dict:
        return {
            bucket: {
                str(band): len(queries)
                for band, queries in sorted(self.by_bucket[bucket].items())
            }
            for bucket in BUCKET_ORDER
        }
