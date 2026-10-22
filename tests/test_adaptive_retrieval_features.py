import numpy as np
import pytest

from ondemand.adaptive_retrieval.features import FEATURE_NAMES, extract_score_features


def test_score_curve_features_are_stable_and_documented():
    scores = np.linspace(1.0, 0.0, 100)
    features = extract_score_features(scores)

    assert tuple(features) == FEATURE_NAMES
    assert len(features) == 21
    assert features["score_mean_top_1"] == pytest.approx(1.0)
    assert features["score_mean_top_5"] == pytest.approx(scores[:5].mean())
    assert features["score_mean_top_100"] == pytest.approx(scores.mean())
    assert features["gap_at_1"] == pytest.approx(scores[0] - scores[1])
    assert features["score_mean_tail"] == pytest.approx(scores[-25:].mean())
    assert all(np.isfinite(value) for value in features.values())


def test_score_curve_features_reject_non_monotone_values():
    with pytest.raises(ValueError, match="ordered"):
        extract_score_features([0.9, 0.8, 0.81])


def test_score_curve_features_allow_numerical_tie_noise():
    features = extract_score_features([0.9, 0.8000001, 0.8000002])
    assert features["max_gap"] >= 0
