import numpy as np
import pytest

from ondemand.adaptive_retrieval.features import FEATURE_NAMES, extract_score_features
from ondemand.adaptive_retrieval.inference import AdaptiveKPredictor, calibrate_k


class FixedRegressor:
    def __init__(self, prediction):
        self.prediction = prediction
        self.seen = None

    def predict(self, values):
        self.seen = np.asarray(values)
        return np.asarray([self.prediction])


def _policy(*, scaling="none"):
    return {
        "cap": 100,
        "features": list(FEATURE_NAMES),
        "feature_scaling": scaling,
        "scale": 1.5,
        "margin": 5.0,
    }


def test_calibration_matches_training_integer_and_cap_rule():
    assert calibrate_k(20.0, scale=1.5, margin=5.0, cap=100) == 35
    assert calibrate_k(-100.0, scale=1.5, margin=5.0, cap=100) == 1
    assert calibrate_k(100.0, scale=1.5, margin=5.0, cap=100) == 100
    with pytest.raises(ValueError, match="positive"):
        calibrate_k(1.0, scale=1.0, margin=0.0, cap=0)


def test_predictor_uses_ranked_final_score_curve_without_query_or_gold_inputs():
    model = FixedRegressor(prediction=20.0)
    predictor = AdaptiveKPredictor(model, _policy())
    scores = np.linspace(1.0, 0.0, 100)

    assert predictor.predict_k(scores) == 35
    assert model.seen.shape == (1, len(FEATURE_NAMES))
    assert np.isfinite(model.seen).all()


def test_standard_scaler_is_applied_using_saved_training_statistics():
    scaler = {
        "features": list(FEATURE_NAMES),
        "mean": [1.0] * len(FEATURE_NAMES),
        "scale": [2.0] * len(FEATURE_NAMES),
    }
    model = FixedRegressor(prediction=20.0)
    predictor = AdaptiveKPredictor(model, _policy(scaling="standard"), scaler)
    scores = np.linspace(1.0, 0.0, 100)

    predictor.predict_k(scores)

    features = extract_score_features(scores)
    raw = np.asarray([[features[name] for name in FEATURE_NAMES]])
    assert np.allclose(model.seen, (raw - 1.0) / 2.0)


def test_predictor_rejects_policy_with_a_different_feature_contract():
    policy = _policy()
    policy["features"] = list(reversed(FEATURE_NAMES))
    with pytest.raises(ValueError, match="feature order"):
        AdaptiveKPredictor(FixedRegressor(20.0), policy)
