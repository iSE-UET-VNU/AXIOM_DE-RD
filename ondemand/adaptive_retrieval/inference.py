"""Load a trained Adaptive-K artifact and predict K from a ranked score curve.

This is the small boundary intended for a future connection to the adaptive
processor router. It consumes the final retrieval scores in rank order; it
does not need query IDs, labels, qrels, or gold page IDs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from xgboost import XGBRegressor

from .features import FEATURE_NAMES, extract_score_features


def calibrate_k(raw_prediction: float, *, scale: float, margin: float, cap: int) -> int:
    """Apply the saved affine calibration and the same integer/cap rule as training."""
    if cap < 1:
        raise ValueError(f"cap must be positive, got {cap}")
    calibrated = float(raw_prediction) * float(scale) + float(margin)
    if not np.isfinite(calibrated):
        raise ValueError("model produced a non-finite K prediction")
    return int(np.clip(np.rint(calibrated), 1, cap))


class AdaptiveKPredictor:
    """Inference wrapper for ``xgboost_adaptive_k.json`` and its policy JSON.

    ``ranked_scores`` must be the final fused scores in the exact ranking order
    produced by the matching retriever. Do not pass BM25/dense component scores
    independently or sort the list here.
    """

    def __init__(self, model: XGBRegressor, policy: dict, scaler: dict | None = None):
        if tuple(policy.get("features", ())) != FEATURE_NAMES:
            raise ValueError("deployment policy feature order does not match this package")
        self.cap = int(policy["cap"])
        if self.cap < 1:
            raise ValueError("deployment policy cap must be positive")
        self.scale = float(policy["scale"])
        self.margin = float(policy["margin"])
        self.scaling = policy.get("feature_scaling", "none")
        if self.scaling not in {"none", "standard"}:
            raise ValueError(f"unsupported feature scaling method: {self.scaling!r}")
        self.model = model
        self.scaler = scaler

        if self.scaling == "standard":
            if not isinstance(scaler, dict):
                raise ValueError("standard scaling is declared but no scaler artifact was supplied")
            if tuple(scaler.get("features", ())) != FEATURE_NAMES:
                raise ValueError("scaler feature order does not match this package")
            self.mean = np.asarray(scaler.get("mean"), dtype=np.float64)
            self.scale_values = np.asarray(scaler.get("scale"), dtype=np.float64)
            if (self.mean.shape != (len(FEATURE_NAMES),)
                    or self.scale_values.shape != (len(FEATURE_NAMES),)
                    or not np.isfinite(self.mean).all()
                    or not np.isfinite(self.scale_values).all()
                    or np.any(self.scale_values <= 0)):
                raise ValueError("invalid scaler mean/scale arrays")
        else:
            self.mean = None
            self.scale_values = None

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "AdaptiveKPredictor":
        """Load model, deployment policy, and optional standard-scaler JSON."""
        artifact_dir = Path(artifact_dir)
        policy_path = artifact_dir / "deployment_policy.json"
        model_path = artifact_dir / "xgboost_adaptive_k.json"
        if not policy_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"expected deployment_policy.json and xgboost_adaptive_k.json in {artifact_dir}")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        scaler = None
        scaler_name = policy.get("feature_scaler_artifact")
        if scaler_name:
            scaler_path = artifact_dir / scaler_name
            scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
        model = XGBRegressor()
        model.load_model(model_path)
        return cls(model, policy, scaler)

    def predict_k(self, ranked_scores: Iterable[float]) -> int:
        """Predict an integer page budget from one ranked final-score curve."""
        features = extract_score_features(list(ranked_scores))
        # Training matrices are float32; use the same dtype to avoid threshold
        # flips from avoidable float64-vs-float32 rounding at tree splits.
        values = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=np.float32)
        if self.scaling == "standard":
            values = ((values - self.mean) / self.scale_values).astype(np.float32, copy=False)
        raw_prediction = float(self.model.predict(values)[0])
        return calibrate_k(raw_prediction, scale=self.scale, margin=self.margin, cap=self.cap)

    def predict_many(self, ranked_score_curves: Iterable[Iterable[float]]) -> list[int]:
        """Predict K for each query's final scores, preserving input order."""
        return [self.predict_k(scores) for scores in ranked_score_curves]
