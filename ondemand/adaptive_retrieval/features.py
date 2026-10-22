"""Transparent features over a ranked retrieval-score curve."""

from __future__ import annotations

import numpy as np


TOP_CUTOFFS = (1, 5, 10, 20, 50, 100)
FEATURE_NAMES = (
    *(f"score_mean_top_{k}" for k in TOP_CUTOFFS),
    "score_mean_tail",
    "score_std",
    "score_range",
    "gap_at_1",
    "gap_at_5",
    "gap_at_10",
    "gap_at_20",
    "gap_at_50",
    "max_gap",
    "max_gap_rank",
    "slope_top_10",
    "slope_top_20",
    "slope_top_50",
    "slope_top_100",
    "softmax_entropy",
)


def extract_score_features(scores: list[float] | np.ndarray) -> dict[str, float]:
    """Summarize the already-ranked final retrieval scores.

    The six ``score_mean_top_*`` values are cumulative prefix means, not raw
    point scores. ``score_mean_tail`` is the mean of the last quarter of the
    supplied ranking. Slopes are least-squares slopes against 1-based rank.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("scores must be a one-dimensional array with at least two values")
    if not np.isfinite(values).all():
        raise ValueError("scores contain NaN or infinite values")
    if np.any(values[1:] > values[:-1] + 1e-6):
        raise ValueError("scores must be ordered from highest to lowest (ties within 1e-6 are allowed)")

    n = values.size
    result: dict[str, float] = {}
    for k in TOP_CUTOFFS:
        end = min(k, n)
        result[f"score_mean_top_{k}"] = float(values[:end].mean())

    tail_size = max(1, int(np.ceil(n * 0.25)))
    result["score_mean_tail"] = float(values[-tail_size:].mean())
    result["score_std"] = float(values.std())
    result["score_range"] = float(values.max() - values.min())

    for k in (1, 5, 10, 20, 50):
        result[f"gap_at_{k}"] = float(values[k - 1] - values[k]) if k < n else 0.0

    gaps = values[:-1] - values[1:]
    gap_index = int(np.argmax(gaps))
    result["max_gap"] = float(gaps[gap_index])
    result["max_gap_rank"] = float((gap_index + 1) / n)

    ranks = np.arange(1, n + 1, dtype=np.float64)
    for k in (10, 20, 50, 100):
        end = min(k, n)
        result[f"slope_top_{k}"] = float(np.polyfit(ranks[:end], values[:end], 1)[0]) if end > 1 else 0.0

    shifted = values - values.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    result["softmax_entropy"] = float(-(probabilities * np.log(probabilities + 1e-15)).sum())

    return result
