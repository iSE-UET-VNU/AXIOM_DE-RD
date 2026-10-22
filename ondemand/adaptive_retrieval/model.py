"""Nested source-stratified OOF training and calibration for continuous K."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .features import FEATURE_NAMES


XGB_PARAMETERS = {
    "n_estimators": 193,
    "max_depth": 2,
    "learning_rate": 0.03,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 5.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_jobs": 4,
    "verbosity": 0,
}


def _new_model(seed: int) -> XGBRegressor:
    return XGBRegressor(**XGB_PARAMETERS, random_state=seed)


def _new_scaler(method: str) -> StandardScaler | None:
    if method == "none":
        return None
    if method == "standard":
        return StandardScaler()
    raise ValueError(f"unsupported feature scaling method: {method}")


def _transform_fold(
    train_x: np.ndarray,
    valid_x: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, StandardScaler | None]:
    scaler = _new_scaler(method)
    if scaler is None:
        return train_x, valid_x, None
    train_scaled = scaler.fit_transform(train_x).astype(np.float32, copy=False)
    valid_scaled = scaler.transform(valid_x).astype(np.float32, copy=False)
    return train_scaled, valid_scaled, scaler


def _integer_k(predictions: np.ndarray, scale: float, margin: float, cap: int) -> np.ndarray:
    return np.clip(np.rint(predictions * scale + margin), 1, cap).astype(int)


def _row_metrics(k_values: np.ndarray, gold_ranks: list[dict[str, int | None]]) -> dict[str, float]:
    recalls: list[float] = []
    precisions: list[float] = []
    f1s: list[float] = []
    ndcgs: list[float] = []
    for k, rank_map in zip(k_values, gold_ranks):
        ranks = [int(rank) for rank in rank_map.values() if rank is not None]
        gold_count = len(rank_map)
        hits = sum(rank <= int(k) for rank in ranks)
        recall = hits / max(gold_count, 1)
        precision = hits / max(int(k), 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        dcg = sum(1.0 / np.log2(rank + 1) for rank in ranks if rank <= int(k))
        ideal_n = min(gold_count, int(k))
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_n + 1))
        ndcg = dcg / idcg if idcg else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        ndcgs.append(ndcg)
    return {
        "mean_k": float(np.mean(k_values)),
        "median_k": float(np.median(k_values)),
        "p90_k": float(np.percentile(k_values, 90)),
        "macro_recall": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
        "macro_f1": float(np.mean(f1s)),
        "macro_ndcg_at_chosen_k": float(np.mean(ndcgs)),
    }


def _choose_calibration(
    predictions: np.ndarray,
    gold_ranks: list[dict[str, int | None]],
    cap: int,
    max_mean_k: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    scales = np.round(np.arange(0.60, 1.5001, 0.05), 2)
    margins = np.arange(-20, 60.01, 5)
    for scale in scales:
        for margin in margins:
            chosen = _integer_k(predictions, float(scale), float(margin), cap)
            metrics = _row_metrics(chosen, gold_ranks)
            candidates.append({"scale": float(scale), "margin": float(margin), **metrics})

    under_budget = [row for row in candidates if row["mean_k"] < max_mean_k]
    if under_budget:
        # Maximise recall first; among ties, use the smaller page budget,
        # then the better nDCG as a stable tie-break.
        best = max(under_budget, key=lambda row: (
            row["macro_recall"], -row["mean_k"], row["macro_ndcg_at_chosen_k"], -row["scale"]
        ))
        best["constraint_met"] = True
    else:
        best = min(candidates, key=lambda row: (row["mean_k"], -row["macro_recall"]))
        best["constraint_met"] = False
    best["calibration_objective"] = f"max macro recall subject to mean K < {max_mean_k:g}"
    return best


def _matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, int | None]]]:
    x = np.asarray([[row["features"][name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    y = np.asarray([row["target_k"] for row in rows], dtype=np.float32)
    sources = np.asarray([row["source"] for row in rows])
    gold_ranks = [row["gold_ranks"] for row in rows]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("feature matrix or target contains NaN/Inf")
    return x, y, sources, gold_ranks


def _inner_oof(
    x: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    seed: int,
    scaling: str = "none",
) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    output = np.full(len(y), np.nan, dtype=np.float64)
    for fold, (train_indices, valid_indices) in enumerate(splitter.split(x, sources)):
        model = _new_model(seed + fold)
        train_x, valid_x, _ = _transform_fold(x[train_indices], x[valid_indices], scaling)
        model.fit(train_x, y[train_indices])
        output[valid_indices] = model.predict(valid_x)
    if not np.isfinite(output).all():
        raise RuntimeError("inner OOF prediction is incomplete")
    return output


def _bootstrap_ci(values: np.ndarray, seed: int = 42, samples: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    distribution = values[indices].mean(axis=1)
    return [float(np.percentile(distribution, 2.5)), float(np.percentile(distribution, 97.5))]


def train_evaluate(
    rows: list[dict[str, Any]],
    cap: int = 100,
    max_mean_k: float = 60.0,
    output_dir: Path | None = None,
    ranking_settings: dict[str, Any] | None = None,
    scaling: str = "none",
) -> dict[str, Any]:
    """Run nested source-stratified OOF and fit a final deployable regressor."""
    if len(rows) != 220:
        raise ValueError(f"expected 220 queries for the canonical comparison, got {len(rows)}")
    x, y, sources, gold_ranks = _matrix(rows)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    chosen_ks = np.zeros(len(rows), dtype=np.int32)
    fold_reports: list[dict[str, Any]] = []
    feature_gains: list[dict[str, float]] = []

    for fold, (train_indices, test_indices) in enumerate(outer.split(x, sources), start=1):
        inner_predictions = _inner_oof(
            x[train_indices], y[train_indices], sources[train_indices], 100 + fold, scaling
        )
        inner_calibration = _choose_calibration(
            inner_predictions,
            [gold_ranks[i] for i in train_indices],
            cap,
            max_mean_k,
        )
        model = _new_model(1000 + fold)
        train_x, test_x, _ = _transform_fold(x[train_indices], x[test_indices], scaling)
        model.fit(train_x, y[train_indices])
        train_predictions = model.predict(train_x).astype(np.float64)
        fold_predictions = model.predict(test_x).astype(np.float64)
        fold_k = _integer_k(fold_predictions, inner_calibration["scale"], inner_calibration["margin"], cap)
        predictions[test_indices] = fold_predictions
        chosen_ks[test_indices] = fold_k
        metrics = _row_metrics(fold_k, [gold_ranks[i] for i in test_indices])
        gain_values = model.get_booster().get_score(importance_type="gain")
        feature_gains.append({
            name: float(gain_values.get(f"f{i}", 0.0))
            for i, name in enumerate(FEATURE_NAMES)
        })
        fold_reports.append({
            "fold": fold,
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "train_target_mean": float(y[train_indices].mean()),
            "test_target_mean": float(y[test_indices].mean()),
            "train_rmse": float(np.sqrt(np.mean((train_predictions - y[train_indices]) ** 2))),
            "train_mae": float(np.mean(np.abs(train_predictions - y[train_indices]))),
            "test_rmse": float(np.sqrt(np.mean((fold_predictions - y[test_indices]) ** 2))),
            "test_mae": float(np.mean(np.abs(fold_predictions - y[test_indices]))),
            "calibration": inner_calibration,
            "test_metrics": metrics,
        })
        print(f"[adaptive-retrieval] outer fold {fold}/5: mean K={metrics['mean_k']:.2f}, "
              f"recall={metrics['macro_recall'] * 100:.2f}%")

    if not np.isfinite(predictions).all() or np.any(chosen_ks < 1):
        raise RuntimeError("outer OOF predictions are incomplete")

    all_metrics = _row_metrics(chosen_ks, gold_ranks)
    fixed_baselines = {
        str(k): _row_metrics(np.full(len(rows), k, dtype=np.int32), gold_ranks)
        for k in (60, cap)
    }
    per_source: dict[str, dict[str, float]] = {}
    for source in sorted(set(sources)):
        indices = np.flatnonzero(sources == source)
        per_source[str(source)] = _row_metrics(chosen_ks[indices], [gold_ranks[i] for i in indices])
    candidate_recall = np.asarray([
        sum(rank is not None and int(rank) <= cap for rank in rank_map.values()) / max(len(rank_map), 1)
        for rank_map in gold_ranks
    ], dtype=np.float64)
    per_query = []
    for i, row in enumerate(rows):
        per_query.append({
            "qid": row["qid"],
            "source": row["source"],
            "target_k": int(y[i]),
            "reachable_at_cap": bool(row["reachable_at_cap"]),
            "gold_count": int(row["gold_count"]),
            "candidate_recall_at_cap": float(candidate_recall[i]),
            "prediction_raw": float(predictions[i]),
            "prediction_k_oof": int(chosen_ks[i]),
            "residual": float(predictions[i] - y[i]),
            "absolute_error": float(abs(predictions[i] - y[i])),
            "features": row["features"],
        })

    # Fit the deployable model and its deployment calibration from OOF
    # predictions. These are artifacts for later integration, not the source
    # of the nested-OOF headline numbers above.
    deployment_oof = _inner_oof(x, y, sources, 777, scaling)
    deployment_calibration = _choose_calibration(deployment_oof, gold_ranks, cap, max_mean_k)
    final_model = _new_model(4242)
    final_x, _, final_scaler = _transform_fold(x, x, scaling)
    final_model.fit(final_x, y)

    report = {
        "title": "Adaptive retrieval continuous-K transfer to adaptive-router ranking",
        "status": "exploratory_nested_oof",
        "protocol": {
            "queries": len(rows),
            "ranking": ranking_settings or {},
            "outer_folds": 5,
            "inner_calibration_folds": 4,
            "folds": "StratifiedKFold by source, shuffle=true, outer random_state=42",
            "feature_count": len(FEATURE_NAMES),
            "features": list(FEATURE_NAMES),
            "label": "smallest rank containing all gold pages, else cap for an unreachable query",
            "cap": cap,
            "calibration_objective": f"max macro page recall subject to mean K < {max_mean_k:g}",
            "gold_use": "training labels and evaluation only; not features",
            "model": "XGBRegressor",
            "feature_scaling": {
                "method": scaling,
                "fit_scope": "training partition only within every inner and outer fold",
                "deploy_transform": "(feature - mean) / scale" if scaling == "standard" else "identity",
            },
            "xgboost_version": xgboost.__version__,
            "parameters": XGB_PARAMETERS,
        },
        "label_distribution": {
            "mean": float(y.mean()),
            "median": float(np.median(y)),
            "p90": float(np.percentile(y, 90)),
            "reachable_count": int(sum(row["reachable_at_cap"] for row in rows)),
            "unreachable_count": int(sum(not row["reachable_at_cap"] for row in rows)),
            "by_source": {
                str(source): {
                    "count": int(np.sum(sources == source)),
                    "reachable": int(sum(rows[i]["reachable_at_cap"] for i in np.flatnonzero(sources == source))),
                    "mean_target_k": float(np.mean(y[sources == source])),
                }
                for source in sorted(set(sources))
            },
        },
        "metrics": all_metrics,
        "regression_diagnostics": {
            "oof_rmse": float(np.sqrt(np.mean((predictions - y) ** 2))),
            "oof_mae": float(np.mean(np.abs(predictions - y))),
        },
        "confidence_intervals_95": {
            "mean_k": _bootstrap_ci(chosen_ks.astype(np.float64), 42),
            "macro_recall": _bootstrap_ci(np.asarray([
                _row_metrics(chosen_ks[[i]], [gold_ranks[i]])["macro_recall"] for i in range(len(rows))
            ]), 43),
        },
        "candidate_recall_at_cap": float(candidate_recall.mean()),
        "fixed_baselines": fixed_baselines,
        "per_source": per_source,
        "folds": fold_reports,
        "feature_gain_mean": {
            name: float(np.mean([fold[name] for fold in feature_gains]))
            for name in FEATURE_NAMES
        },
        "deployment_calibration_from_full_data_oof": deployment_calibration,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / "oof_predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_query), encoding="utf-8"
        )
        table_path = output_dir / "training_table.csv"
        fieldnames = [
            "qid", "source", "target_k", "reachable_at_cap", "gold_count",
            "candidate_recall_at_cap", "prediction_raw", "prediction_k_oof", "residual", "absolute_error",
            *FEATURE_NAMES,
        ]
        with table_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_query:
                writer.writerow({**{key: row[key] for key in fieldnames if key in row}, **row["features"]})
        final_model.save_model(output_dir / "xgboost_adaptive_k.json")
        if final_scaler is not None:
            (output_dir / "feature_scaler.json").write_text(json.dumps({
                "method": scaling,
                "features": list(FEATURE_NAMES),
                "mean": final_scaler.mean_.tolist(),
                "scale": final_scaler.scale_.tolist(),
                "variance": final_scaler.var_.tolist(),
                "training_rows": int(len(x)),
                "note": "Fit on all rows only for the final deployable model; OOF evaluation uses fold-specific scalers.",
            }, indent=2), encoding="utf-8")
        (output_dir / "deployment_policy.json").write_text(json.dumps({
            "cap": cap,
            "features": list(FEATURE_NAMES),
            "feature_scaling": scaling,
            "feature_scaler_artifact": "feature_scaler.json" if final_scaler is not None else None,
            "scale": deployment_calibration["scale"],
            "margin": deployment_calibration["margin"],
            "training_oof_metrics": deployment_calibration,
            "note": "Fit on all 220 rows; use only after separately held-out validation before production integration.",
        }, indent=2), encoding="utf-8")
    return report
