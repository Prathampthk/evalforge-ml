"""Paired percentile bootstrap; queries, not repeated timings, are sampling units."""

from __future__ import annotations

import numpy as np


def paired_bootstrap(left: dict[str, float], right: dict[str, float], samples: int, seed: int) -> dict:
    if not left or set(left) != set(right):
        raise ValueError("Paired comparison requires identical nonempty query sets")
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    differences = np.array([right[q] - left[q] for q in sorted(left)], dtype=float)
    if not np.all(np.isfinite(differences)):
        raise ValueError("Nonfinite metric values")
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    # Bounded batch allocation for larger user datasets.
    for start in range(0, samples, 100):
        count = min(100, samples - start)
        means[start : start + count] = rng.choice(differences, (count, len(differences)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "difference_right_minus_left": float(differences.mean()),
        "ci95": [float(low), float(high)],
        "query_count": len(differences),
        "resamples": samples,
        "seed": seed,
        "method": "paired query percentile bootstrap; unadjusted exploratory 95% CI, not a superiority test",
    }
