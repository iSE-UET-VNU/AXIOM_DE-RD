"""Unified, leak-aware runner for the PP-OCRv5 Adaptive-K research plan.

Examples
--------
Build/audit the frozen dataset and run the regression feature ablation::

    python research/experiments/run_xgboost_adaptive_k_optimization.py --stage features

Run nested hyper-parameter selection (64 samples per outer fold by default)::

    python research/experiments/run_xgboost_adaptive_k_optimization.py --stage tuning

All selection that affects an outer-test prediction is performed on inner OOF
predictions.  Outer OOF results remain an evaluation, never a calibration set.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "research" / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "research" / "experiments"))

from adaptive_k_optimization_core import (  # noqa: E402
    CAP, SEED, FrozenData, bootstrap_ci, build_query_features,
    build_score_feature_families, combine_families, load_frozen_data,
    metric_summary, recall_at, target_arrays, validate_data,
    _band_features, _point_features, _shape_features,
)

DEFAULT_OUTPUT = ROOT / "data" / "work" / "xgboost_adaptive_k_optimization_20261016"
SUCCESS_RECALL = 0.87
SUCCESS_MEAN_K = 60.0

FEATURE_SPECS: dict[str, tuple[str, ...]] = {
    "baseline21": ("baseline21",),
    "clean": ("clean",),
    "point": ("point",),
    "local": ("local",),
    "band": ("band",),
    "shape": ("shape",),
    "modality": ("modality",),
    "point_band_shape": ("point", "band", "shape"),
    "score_all": ("point", "local", "band", "shape"),
    "score_modality": ("point", "local", "band", "shape", "modality"),
    "score_modality_query": ("point", "local", "band", "shape", "modality", "query"),
}


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def fixed_folds(sources: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(len(sources)), sources))


def calibration_grid() -> list[tuple[float, int]]:
    return [(round(scale, 2), margin) for scale in np.arange(0.60, 1.501, 0.05) for margin in range(-20, 61, 5)]


def choices_from_prediction(prediction: np.ndarray, scale: float, margin: int) -> np.ndarray:
    return np.clip(np.ceil(scale * np.asarray(prediction) + margin), 1, CAP).astype(np.int16)


def choose_calibration(data: FrozenData, indices: np.ndarray, prediction: np.ndarray) -> tuple[float, int, dict[str, Any]]:
    candidates = []
    for scale, margin in calibration_grid():
        choices = choices_from_prediction(prediction, scale, margin)
        # metric_summary expects a global choice vector; evaluate directly here.
        recalls = np.asarray([recall_at(data.rows[int(i)], int(k)) for i, k in zip(indices, choices)])
        item = {"scale": scale, "margin": margin, "mean_k": float(np.mean(choices)), "macro_recall": float(np.mean(recalls))}
        candidates.append(item)
    feasible = [item for item in candidates if item["mean_k"] < SUCCESS_MEAN_K]
    if feasible:
        best = max(feasible, key=lambda item: (item["macro_recall"], -item["mean_k"]))
    else:
        best = min(candidates, key=lambda item: (item["mean_k"], -item["macro_recall"]))
    return float(best["scale"]), int(best["margin"]), best


def base_xgb_params(seed: int) -> dict[str, Any]:
    return {
        "n_estimators": 1000, "max_depth": 2, "learning_rate": 0.05,
        "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85,
        "reg_lambda": 5.0, "reg_alpha": 0.0, "random_state": seed,
        "n_jobs": 4, "tree_method": "hist", "verbosity": 0,
    }


def make_regressor(params: Mapping[str, Any], objective: str, seed: int) -> tuple[XGBRegressor, bool]:
    merged = base_xgb_params(seed); merged.update(params)
    is_log = objective == "log_squared"
    if objective in ("squared", "log_squared"):
        merged["objective"] = "reg:squarederror"
    elif objective.startswith("asymmetric_"):
        under_weight = float(objective.rsplit("_", 1)[1])
        def asymmetric_loss(labels: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            residual = predictions - labels
            local_weight = np.where(residual < 0, under_weight, 1.0)
            return 2.0 * local_weight * residual, 2.0 * local_weight
        merged["objective"] = asymmetric_loss
    elif objective == "pseudo_huber":
        merged["objective"] = "reg:pseudohubererror"
    elif objective.startswith("quantile_"):
        merged["objective"] = "reg:quantileerror"
        merged["quantile_alpha"] = float(objective.rsplit("_", 1)[1]) / 100
    else:
        raise ValueError(objective)
    return XGBRegressor(**merged), is_log


@dataclass
class FitResult:
    prediction: np.ndarray
    best_iteration: int
    train_loss: list[float]
    valid_loss: list[float]
    importance: np.ndarray


def fit_xgb(
    x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray,
    params: Mapping[str, Any], objective: str, seed: int,
    sample_weight: np.ndarray | None = None,
) -> FitResult:
    model, is_log = make_regressor(params, objective, seed)
    train_target = np.log1p(y_train) if is_log else y_train
    valid_target = np.log1p(y_valid) if is_log else y_valid
    model.set_params(early_stopping_rounds=50, eval_metric="rmse")
    model.fit(x_train, train_target, sample_weight=sample_weight, eval_set=[(x_train, train_target), (x_valid, valid_target)], verbose=False)
    prediction = np.asarray(model.predict(x_valid), dtype=float)
    if is_log:
        prediction = np.expm1(prediction)
    history = model.evals_result()
    metric_name = next(iter(history["validation_0"]))
    return FitResult(
        prediction=np.clip(prediction, 1, CAP), best_iteration=int(getattr(model, "best_iteration", len(history["validation_1"][metric_name]) - 1)),
        train_loss=[float(v) for v in history["validation_0"][metric_name]],
        valid_loss=[float(v) for v in history["validation_1"][metric_name]],
        importance=np.asarray(model.feature_importances_, dtype=float),
    )


def inner_oof(
    matrix: np.ndarray, target: np.ndarray, sources: np.ndarray, outer_train: np.ndarray,
    params: Mapping[str, Any], objective: str, outer_fold: int,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    prediction = np.zeros(len(outer_train), dtype=float)
    iterations: list[int] = []
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 100 + outer_fold)
    for inner_fold, (train_pos, valid_pos) in enumerate(splitter.split(matrix[outer_train], sources[outer_train])):
        train_idx, valid_idx = outer_train[train_pos], outer_train[valid_pos]
        result = fit_xgb(matrix[train_idx], target[train_idx], matrix[valid_idx], target[valid_idx], params, objective, SEED + 1000 * outer_fold + inner_fold, None if weights is None else weights[train_idx])
        prediction[valid_pos] = result.prediction
        iterations.append(result.best_iteration + 1)
    return prediction, iterations


def nested_regression(
    data: FrozenData, matrix: np.ndarray, feature_names: list[str], target: np.ndarray,
    sources: np.ndarray, params_by_fold: Sequence[Mapping[str, Any]] | None = None,
    objective: str = "squared", weights: np.ndarray | None = None,
    name: str = "model",
) -> dict[str, Any]:
    folds = fixed_folds(sources)
    raw = np.zeros(len(data.rows), dtype=float)
    choices = np.zeros(len(data.rows), dtype=np.int16)
    fold_rows = []
    histories = []
    importances = []
    for outer_fold, (train_idx, test_idx) in enumerate(folds):
        params = dict((params_by_fold or [{} for _ in folds])[outer_fold])
        inner_prediction, best_rounds = inner_oof(matrix, target, sources, train_idx, params, objective, outer_fold, weights)
        scale, margin, inner_metric = choose_calibration(data, train_idx, inner_prediction)
        final_params = dict(params)
        final_params["n_estimators"] = int(np.clip(np.median(best_rounds), 10, 1000))
        # A train-only validation split determines early stopping; outer test is untouched.
        fit_idx, stop_idx = train_test_split(train_idx, test_size=0.20, random_state=SEED + outer_fold, stratify=sources[train_idx])
        fitted = fit_xgb(matrix[fit_idx], target[fit_idx], matrix[stop_idx], target[stop_idx], final_params, objective, SEED + outer_fold, None if weights is None else weights[fit_idx])
        final_params["n_estimators"] = fitted.best_iteration + 1
        model, is_log = make_regressor(final_params, objective, SEED + outer_fold)
        fit_target = np.log1p(target[train_idx]) if is_log else target[train_idx]
        model.fit(matrix[train_idx], fit_target, sample_weight=None if weights is None else weights[train_idx], verbose=False)
        prediction = np.asarray(model.predict(matrix[test_idx]), dtype=float)
        if is_log: prediction = np.expm1(prediction)
        prediction = np.clip(prediction, 1, CAP)
        raw[test_idx] = prediction
        choices[test_idx] = choices_from_prediction(prediction, scale, margin)
        importances.append(np.asarray(model.feature_importances_, dtype=float))
        histories.append({"fold": outer_fold, "train_loss": fitted.train_loss, "validation_loss": fitted.valid_loss})
        fold_rows.append({"fold": outer_fold, "train_queries": len(train_idx), "test_queries": len(test_idx), "scale": scale, "margin": margin, "inner_mean_k": inner_metric["mean_k"], "inner_recall": inner_metric["macro_recall"], "trees": final_params["n_estimators"], "params": params})
    summary = metric_summary(data.rows, choices)
    per_source = {}
    for source in sorted(set(sources.tolist())):
        per_source[source] = {k: v for k, v in metric_summary(data.rows, choices, np.flatnonzero(sources == source)).items() if k != "recalls"}
    target_easy = target <= 10
    predicted_easy = choices <= 10
    easy_precision = float(np.mean(target_easy[predicted_easy])) if np.any(predicted_easy) else 0.0
    easy_coverage = float(np.mean(predicted_easy[target_easy])) if np.any(target_easy) else 0.0
    mean_importance = np.mean(np.vstack(importances), axis=0)
    stability = np.std(np.vstack(importances), axis=0)
    feature_importance = sorted([{"feature": feature, "gain_importance_mean": float(importance), "gain_importance_std": float(std)} for feature, importance, std in zip(feature_names, mean_importance, stability)], key=lambda row: -row["gain_importance_mean"])
    return {
        "name": name, "objective": objective, "summary": {k: v for k, v in summary.items() if k != "recalls"},
        "bootstrap_ci": bootstrap_ci(data.rows, choices), "per_source": per_source,
        "safe_easy_precision": easy_precision, "easy_coverage": easy_coverage,
        "target_mae": float(np.mean(np.abs(raw - target))), "folds": fold_rows,
        "raw_prediction": raw, "choices": choices, "per_query_recall": summary["recalls"],
        "histories": histories, "feature_importance": feature_importance,
    }


def model_score(result: Mapping[str, Any]) -> tuple[int, float, float]:
    summary = result["summary"]
    return (int(summary["mean_k"] < SUCCESS_MEAN_K), float(summary["macro_recall"]), -float(summary["mean_k"]))


def prepare(output: Path) -> tuple[FrozenData, dict[str, np.ndarray], dict[str, list[str]], dict[str, np.ndarray], np.ndarray]:
    data = load_frozen_data()
    checks = validate_data(data)
    families, names = build_score_feature_families(data)
    query_matrix, query_names = build_query_features(data)
    families["query"] = query_matrix; names["query"] = query_names
    targets = target_arrays(data.rows)
    sources = np.asarray([str(row["source"]) for row in data.rows])
    folds = fixed_folds(sources)
    output.mkdir(parents=True, exist_ok=True)
    json_dump(output / "protocol.json", {**data.protocol, "outer_folds": 5, "inner_folds": 4, "seed": SEED, "success": {"mean_k_strictly_below": SUCCESS_MEAN_K, "macro_recall_at_least": SUCCESS_RECALL}, "checks": checks})
    fold_of = np.empty(len(data.rows), dtype=int)
    for fold, (_, test) in enumerate(folds): fold_of[test] = fold
    write_csv(output / "folds.csv", [{"qid": row["qid"], "source": row["source"], "outer_fold": int(fold_of[i])} for i, row in enumerate(data.rows)])
    dictionary = []
    for family, feature_names in names.items():
        for feature in feature_names:
            dictionary.append({"family": family, "feature": feature})
    write_csv(output / "features" / "feature_dictionary.csv", dictionary)
    all_matrix, all_names = combine_families(families, names, ("point", "local", "band", "shape", "modality", "query"))
    table = []
    for i, row in enumerate(data.rows):
        record: dict[str, Any] = {"qid": row["qid"], "source": row["source"], "query": row["query"], "outer_fold": int(fold_of[i])}
        record.update({key: float(value[i]) if value.dtype.kind == "f" else int(value[i]) for key, value in targets.items()})
        record.update({name: float(value) for name, value in zip(all_names, all_matrix[i])})
        table.append(record)
    write_csv(output / "features" / "training_data.csv", table)
    correlations = []
    for family, matrix in families.items():
        corr = np.corrcoef(matrix, rowvar=False) if matrix.shape[1] > 1 else np.ones((1, 1))
        upper = np.abs(corr[np.triu_indices_from(corr, k=1)]) if matrix.shape[1] > 1 else np.array([])
        correlations.append({"family": family, "features": matrix.shape[1], "mean_abs_pair_correlation": float(np.nanmean(upper)) if len(upper) else 0.0, "pairs_abs_ge_0_95": int(np.nansum(upper >= 0.95)), "constant_features": int(np.sum(np.std(matrix, axis=0) < 1e-12))})
    write_csv(output / "features" / "family_correlation_summary.csv", correlations)
    return data, families, names, targets, sources


def save_result(output: Path, stage: str, result: Mapping[str, Any], data: FrozenData, target: np.ndarray) -> None:
    serializable = {key: value for key, value in result.items() if key not in ("raw_prediction", "choices", "per_query_recall")}
    json_dump(output / stage / f"{result['name']}.json", serializable)
    rows = []
    for i, row in enumerate(data.rows):
        rows.append({"qid": row["qid"], "source": row["source"], "query": row["query"], "target_k": float(target[i]), "raw_prediction": float(result["raw_prediction"][i]), "chosen_k": int(result["choices"][i]), "recall": float(result["per_query_recall"][i])})
    write_csv(output / stage / f"{result['name']}_oof.csv", rows)


def run_features(output: Path, data: FrozenData, families: Mapping[str, np.ndarray], names: Mapping[str, list[str]], targets: Mapping[str, np.ndarray], sources: np.ndarray) -> list[dict[str, Any]]:
    results = []
    target = targets["k100"]
    for spec_name, keys in FEATURE_SPECS.items():
        print(f"[features] {spec_name}", flush=True)
        matrix, feature_names = combine_families(families, names, keys)
        result = nested_regression(data, matrix, feature_names, target, sources, objective="squared", name=spec_name)
        save_result(output, "features", result, data, target); results.append(result)
    table = [{"model": result["name"], **result["summary"], "safe_easy_precision": result["safe_easy_precision"], "easy_coverage": result["easy_coverage"], "target_mae": result["target_mae"]} for result in results]
    write_csv(output / "features" / "ablation.csv", sorted(table, key=lambda row: (-row["macro_recall"], row["mean_k"])))
    json_dump(output / "features" / "summary.json", table)
    return results


def tuning_space(samples: int) -> list[dict[str, Any]]:
    distribution = {
        "max_depth": [1, 2, 3, 4], "learning_rate": [0.02, 0.05, 0.10],
        "min_child_weight": [1, 3, 5, 10], "subsample": [0.70, 0.85, 1.0],
        "colsample_bytree": [0.70, 0.85, 1.0], "reg_lambda": [1.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.5, 2.0],
    }
    return list(ParameterSampler(distribution, n_iter=samples, random_state=SEED))


def select_params_nested(data: FrozenData, matrix: np.ndarray, target: np.ndarray, sources: np.ndarray, samples: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chosen, audit = [], []
    candidates = tuning_space(samples)
    for outer_fold, (train_idx, _) in enumerate(fixed_folds(sources)):
        scored = []
        for number, params in enumerate(candidates):
            prediction, iterations = inner_oof(matrix, target, sources, train_idx, params, "squared", outer_fold)
            scale, margin, metric = choose_calibration(data, train_idx, prediction)
            scored.append({"candidate": number, "params": params, "scale": scale, "margin": margin, **metric, "median_trees": int(np.median(iterations))})
        feasible = [item for item in scored if item["mean_k"] < SUCCESS_MEAN_K]
        winner = max(feasible or scored, key=lambda item: (item["macro_recall"], -item["mean_k"]))
        chosen.append(winner["params"]); audit.append({"outer_fold": outer_fold, "winner": winner, "candidates": scored})
        print(f"[tuning] fold={outer_fold} recall={winner['macro_recall']:.4f} meanK={winner['mean_k']:.2f}", flush=True)
    return chosen, audit


def run_tuning(output: Path, data: FrozenData, families: Mapping[str, np.ndarray], names: Mapping[str, list[str]], targets: Mapping[str, np.ndarray], sources: np.ndarray, feature_spec: str, samples: int) -> dict[str, Any]:
    matrix, feature_names = combine_families(families, names, FEATURE_SPECS[feature_spec])
    target = targets["k100"]
    params_by_fold, audit = select_params_nested(data, matrix, target, sources, samples)
    json_dump(output / "tuning" / "inner_search.json", audit)
    result = nested_regression(data, matrix, feature_names, target, sources, params_by_fold=params_by_fold, name=f"xgb_tuned_{feature_spec}")
    save_result(output, "tuning", result, data, target)
    return result


def loss_weights(target: np.ndarray, censored: np.ndarray, easy_multiplier: float, capped_multiplier: float, balanced: bool = False) -> np.ndarray:
    weights = np.ones(len(target), dtype=float)
    weights[target <= 10] *= easy_multiplier
    weights[censored] *= capped_multiplier
    if balanced:
        bins = np.digitize(target, [10, 20, 50, 100])
        counts = Counter(bins.tolist())
        weights *= np.asarray([len(target) / (len(counts) * counts[int(value)]) for value in bins])
    return weights / np.mean(weights)


def run_loss(output: Path, data: FrozenData, families: Mapping[str, np.ndarray], names: Mapping[str, list[str]], targets: Mapping[str, np.ndarray], sources: np.ndarray, feature_spec: str, loss_filter: str | None = None) -> list[dict[str, Any]]:
    matrix, feature_names = combine_families(families, names, FEATURE_SPECS[feature_spec])
    results = []
    configs = [(name, {}) for name in ("squared", "log_squared", "pseudo_huber", "quantile_50", "quantile_70", "quantile_80", "quantile_90", "asymmetric_1", "asymmetric_2", "asymmetric_4")]
    configs += [("squared", {"easy": easy, "capped": capped, "balanced": balanced}) for easy in (2.0, 4.0) for capped in (0.5, 1.0) for balanced in (False, True)]
    for target_name in ("k80", "k90", "k100"):
        target = targets[target_name]
        censored = targets[target_name.replace("k", "censored")]
        for index, (objective, weight_config) in enumerate(configs):
            suffix = objective if not weight_config else f"squared_easy{weight_config['easy']:g}_cap{weight_config['capped']:g}_bal{int(weight_config['balanced'])}"
            name = f"{target_name}_{suffix}"
            if loss_filter and loss_filter not in name:
                continue
            print(f"[loss] {name}", flush=True)
            weights = None if not weight_config else loss_weights(target, censored, weight_config["easy"], weight_config["capped"], weight_config["balanced"])
            result = nested_regression(data, matrix, feature_names, target, sources, objective=objective, weights=weights, name=name)
            save_result(output, "loss", result, data, target); results.append(result)
    table = [{"model": result["name"], **result["summary"], "safe_easy_precision": result["safe_easy_precision"], "easy_coverage": result["easy_coverage"]} for result in results]
    screen_path = output / "loss" / "screen.csv"
    if loss_filter and screen_path.exists():
        with screen_path.open(encoding="utf-8-sig") as handle:
            previous = list(csv.DictReader(handle))
        merged = {str(row["model"]): row for row in previous}
        merged.update({str(row["model"]): row for row in table})
        table = list(merged.values())
    write_csv(screen_path, sorted(table, key=lambda row: (-float(row["macro_recall"]), float(row["mean_k"]))))
    return results


def nearest_neighbor_features(data: FrozenData, fit_idx: np.ndarray, transform_idx: np.ndarray, target: np.ndarray, k: int = 10, leave_self_out: bool = False) -> tuple[np.ndarray, list[str]]:
    fit_vectors = data.query_vectors[fit_idx]
    transform_vectors = data.query_vectors[transform_idx]
    similarity = transform_vectors @ fit_vectors.T
    if leave_self_out:
        positions = {int(index): pos for pos, index in enumerate(fit_idx)}
        for row_pos, global_index in enumerate(transform_idx):
            if int(global_index) in positions: similarity[row_pos, positions[int(global_index)]] = -np.inf
    order = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    names = ["neighbor_sim_max", "neighbor_sim_mean", "neighbor_sim_gap", "neighbor_target_mean", "neighbor_target_median", "neighbor_target_std", "neighbor_target_q80", "neighbor_target_q90", "neighbor_easy_fraction", "neighbor_hard_fraction", "neighbor_curve_distance", "neighbor_top10_overlap", "neighbor_top20_overlap", "neighbor_same_source_fraction"]
    matrix = []
    for local_row, neighbor_positions in enumerate(order):
        global_neighbors = fit_idx[neighbor_positions]
        sims = similarity[local_row, neighbor_positions]
        values = target[global_neighbors]
        query_index = int(transform_idx[local_row])
        curve = np.asarray(data.rows[query_index]["scores"], dtype=float)
        distances = [float(np.mean(np.abs(curve - np.asarray(data.rows[int(i)]["scores"], dtype=float)))) for i in global_neighbors]
        top10 = set(data.rows[query_index]["ranked"][:10]); top20 = set(data.rows[query_index]["ranked"][:20])
        overlap10 = [len(top10 & set(data.rows[int(i)]["ranked"][:10])) / 10 for i in global_neighbors]
        overlap20 = [len(top20 & set(data.rows[int(i)]["ranked"][:20])) / 20 for i in global_neighbors]
        source = data.rows[query_index]["source"]
        matrix.append([float(sims[0]), float(np.mean(sims)), float(sims[0] - sims[1]), float(np.mean(values)), float(np.median(values)), float(np.std(values)), float(np.quantile(values, 0.8)), float(np.quantile(values, 0.9)), float(np.mean(values <= 10)), float(np.mean(values > 100)), float(np.mean(distances)), float(np.mean(overlap10)), float(np.mean(overlap20)), float(np.mean([data.rows[int(i)]["source"] == source for i in global_neighbors]))])
    return np.asarray(matrix, dtype=np.float32), names


def run_neighbor(output: Path, data: FrozenData, families: Mapping[str, np.ndarray], names: Mapping[str, list[str]], targets: Mapping[str, np.ndarray], sources: np.ndarray, feature_spec: str) -> dict[str, Any]:
    """Nested neighbor experiment; every target aggregate comes from fitting rows."""
    base, base_names = combine_families(families, names, FEATURE_SPECS[feature_spec])
    target = targets["k100"]
    raw = np.zeros(len(data.rows)); choices = np.zeros(len(data.rows), dtype=np.int16)
    fold_rows, importances = [], []
    for outer_fold, (outer_train, outer_test) in enumerate(fixed_folds(sources)):
        splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 100 + outer_fold)
        inner_prediction = np.zeros(len(outer_train))
        for inner_fold, (train_pos, valid_pos) in enumerate(splitter.split(base[outer_train], sources[outer_train])):
            train_idx, valid_idx = outer_train[train_pos], outer_train[valid_pos]
            train_neighbor, neighbor_names = nearest_neighbor_features(data, train_idx, train_idx, target, leave_self_out=True)
            valid_neighbor, _ = nearest_neighbor_features(data, train_idx, valid_idx, target)
            result = fit_xgb(np.hstack([base[train_idx], train_neighbor]), target[train_idx], np.hstack([base[valid_idx], valid_neighbor]), target[valid_idx], {}, "squared", SEED + outer_fold * 1000 + inner_fold)
            inner_prediction[valid_pos] = result.prediction
        scale, margin, inner_metric = choose_calibration(data, outer_train, inner_prediction)
        train_neighbor, neighbor_names = nearest_neighbor_features(data, outer_train, outer_train, target, leave_self_out=True)
        test_neighbor, _ = nearest_neighbor_features(data, outer_train, outer_test, target)
        train_matrix = np.hstack([base[outer_train], train_neighbor])
        test_matrix = np.hstack([base[outer_test], test_neighbor])
        fit_pos, stop_pos = train_test_split(np.arange(len(outer_train)), test_size=.20, random_state=SEED + outer_fold, stratify=sources[outer_train])
        stopped = fit_xgb(train_matrix[fit_pos], target[outer_train[fit_pos]], train_matrix[stop_pos], target[outer_train[stop_pos]], {}, "squared", SEED + outer_fold)
        final_params = {"n_estimators": stopped.best_iteration + 1}
        model, _ = make_regressor(final_params, "squared", SEED + outer_fold)
        model.fit(train_matrix, target[outer_train], verbose=False)
        prediction = np.clip(model.predict(test_matrix), 1, CAP)
        raw[outer_test] = prediction; choices[outer_test] = choices_from_prediction(prediction, scale, margin)
        importances.append(np.asarray(model.feature_importances_, dtype=float))
        fold_rows.append({"fold": outer_fold, "scale": scale, "margin": margin, "inner_mean_k": inner_metric["mean_k"], "inner_recall": inner_metric["macro_recall"]})
    summary = metric_summary(data.rows, choices)
    target_easy, predicted_easy = target <= 10, choices <= 10
    per_source = {source: {key: value for key, value in metric_summary(data.rows, choices, np.flatnonzero(sources == source)).items() if key != "recalls"} for source in sorted(set(sources.tolist()))}
    # Explicit upper-bound diagnostic only: neighbour gold IDs never enter primary features or ranking.
    memory_coverages = np.zeros(len(data.rows))
    for _, (train_idx, test_idx) in enumerate(fixed_folds(sources)):
        similarity = data.query_vectors[test_idx] @ data.query_vectors[train_idx].T
        neighbors = np.argsort(-similarity, axis=1)[:, :10]
        for position, query_index in enumerate(test_idx):
            memory: set[str] = set()
            for neighbor in train_idx[neighbors[position]]: memory.update(data.rows[int(neighbor)]["gold"])
            gold = set(data.rows[int(query_index)]["gold"])
            memory_coverages[int(query_index)] = len(memory & gold) / max(len(gold), 1)
    payload = {"name": f"neighbor_{feature_spec}", "objective": "squared", "summary": {k: v for k, v in summary.items() if k != "recalls"}, "bootstrap_ci": bootstrap_ci(data.rows, choices), "per_source": per_source, "safe_easy_precision": float(np.mean(target_easy[predicted_easy])) if np.any(predicted_easy) else 0.0, "easy_coverage": float(np.mean(predicted_easy[target_easy])) if np.any(target_easy) else 0.0, "target_mae": float(np.mean(np.abs(raw - target))), "folds": fold_rows, "raw_prediction": raw, "choices": choices, "per_query_recall": summary["recalls"], "feature_names": base_names + neighbor_names, "leakage_control": "Inner valid/test target aggregates use inner/outer training neighbours only; training features are leave-one-out. Gold-page memory is diagnostic only.", "gold_memory_upper_bound": {"neighbors": 10, "macro_gold_page_coverage": float(np.mean(memory_coverages)), "queries_with_any_gold_overlap": float(np.mean(memory_coverages > 0))}}
    save_result(output, "neighbor", payload, data, target)
    return payload


def fit_binary(matrix: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> np.ndarray:
    positives = max(int(np.sum(labels[train_idx] == 1)), 1)
    negatives = max(int(np.sum(labels[train_idx] == 0)), 1)
    model = XGBClassifier(
        n_estimators=350, max_depth=2, learning_rate=0.05, min_child_weight=3,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=5,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=negatives / positives, random_state=seed,
        n_jobs=4, tree_method="hist",
    )
    model.fit(matrix[train_idx], labels[train_idx])
    return np.asarray(model.predict_proba(matrix[test_idx])[:, 1], dtype=float)


def choose_tier_policy(data: FrozenData, indices: np.ndarray, easy_probability: np.ndarray, hard_probability: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for easy_threshold in np.arange(0.50, 0.951, 0.05):
        for hard_threshold in np.arange(0.30, 0.851, 0.05):
            for easy_k in (1, 5, 10, 20):
                for base_k in (40, 50, 60, 70):
                    for hard_k in (100, 150, 200):
                        choices = np.full(len(indices), base_k, dtype=np.int16)
                        choices[hard_probability >= hard_threshold] = hard_k
                        # Easy takes precedence only at high confidence; conflicting rows are not easy exits.
                        easy_mask = (easy_probability >= easy_threshold) & (hard_probability < hard_threshold)
                        choices[easy_mask] = easy_k
                        mean_k = float(np.mean(choices))
                        if mean_k >= SUCCESS_MEAN_K:
                            continue
                        recall = float(np.mean([recall_at(data.rows[int(index)], int(k)) for index, k in zip(indices, choices)]))
                        candidate = {"easy_threshold": float(easy_threshold), "hard_threshold": float(hard_threshold), "easy_k": easy_k, "base_k": base_k, "hard_k": hard_k, "mean_k": mean_k, "macro_recall": recall, "easy_exits": int(np.sum(easy_mask)), "hard_routes": int(np.sum(hard_probability >= hard_threshold))}
                        if best is None or (recall, -mean_k) > (best["macro_recall"], -best["mean_k"]):
                            best = candidate
    if best is None:
        raise RuntimeError("No tier policy satisfies the strict mean-K budget")
    return best


def apply_tier_policy(easy_probability: np.ndarray, hard_probability: np.ndarray, policy: Mapping[str, Any]) -> np.ndarray:
    choices = np.full(len(easy_probability), int(policy["base_k"]), dtype=np.int16)
    hard_mask = hard_probability >= float(policy["hard_threshold"])
    choices[hard_mask] = int(policy["hard_k"])
    choices[(easy_probability >= float(policy["easy_threshold"])) & ~hard_mask] = int(policy["easy_k"])
    return choices


def run_safe_policy(output: Path, data: FrozenData, families: Mapping[str, np.ndarray], names: Mapping[str, list[str]], targets: Mapping[str, np.ndarray], sources: np.ndarray, feature_spec: str) -> dict[str, Any]:
    matrix, _ = combine_families(families, names, FEATURE_SPECS[feature_spec])
    target = targets["k100"]
    easy_label = (target <= 10).astype(np.int8)
    hard_label = ((target > 70) | targets["censored100"]).astype(np.int8)
    easy_oof = np.zeros(len(data.rows)); hard_oof = np.zeros(len(data.rows)); choices = np.zeros(len(data.rows), dtype=np.int16)
    fold_rows = []
    for outer_fold, (outer_train, outer_test) in enumerate(fixed_folds(sources)):
        inner_easy = np.zeros(len(outer_train)); inner_hard = np.zeros(len(outer_train))
        splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 100 + outer_fold)
        for inner_fold, (train_pos, valid_pos) in enumerate(splitter.split(matrix[outer_train], sources[outer_train])):
            train_idx, valid_idx = outer_train[train_pos], outer_train[valid_pos]
            inner_easy[valid_pos] = fit_binary(matrix, easy_label, train_idx, valid_idx, SEED + outer_fold * 1000 + inner_fold)
            inner_hard[valid_pos] = fit_binary(matrix, hard_label, train_idx, valid_idx, SEED + 10000 + outer_fold * 1000 + inner_fold)
        policy = choose_tier_policy(data, outer_train, inner_easy, inner_hard)
        easy_oof[outer_test] = fit_binary(matrix, easy_label, outer_train, outer_test, SEED + outer_fold)
        hard_oof[outer_test] = fit_binary(matrix, hard_label, outer_train, outer_test, SEED + 10000 + outer_fold)
        choices[outer_test] = apply_tier_policy(easy_oof[outer_test], hard_oof[outer_test], policy)
        fold_rows.append({"fold": outer_fold, **policy})
    summary = metric_summary(data.rows, choices)
    predicted_easy = choices <= 10
    payload = {
        "name": f"safe_easy_hard_{feature_spec}", "summary": {k: v for k, v in summary.items() if k != "recalls"},
        "bootstrap_ci": bootstrap_ci(data.rows, choices), "folds": fold_rows,
        "safe_easy_precision": float(np.mean(easy_label[predicted_easy])) if np.any(predicted_easy) else 0.0,
        "easy_coverage": float(np.mean(predicted_easy[easy_label == 1])) if np.any(easy_label) else 0.0,
        "easy_probability": easy_oof, "hard_probability": hard_oof,
        "raw_prediction": easy_oof * 10 + hard_oof * 190, "choices": choices, "per_query_recall": summary["recalls"],
        "histories": [], "feature_importance": [],
    }
    save_result(output, "policy", payload, data, target)
    return payload


def fit_reachable_regression(matrix: np.ndarray, target: np.ndarray, censored: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> np.ndarray:
    reachable_train = train_idx[~censored[train_idx]]
    model = XGBRegressor(n_estimators=350, max_depth=2, learning_rate=.05, min_child_weight=3, subsample=.85, colsample_bytree=.85, reg_lambda=5, objective="reg:squarederror", random_state=seed, n_jobs=4, tree_method="hist")
    model.fit(matrix[reachable_train], target[reachable_train])
    return np.clip(model.predict(matrix[test_idx]), 1, CAP)


def run_censoring(output: Path, data: FrozenData, families: Mapping[str, np.ndarray], names: Mapping[str, list[str]], targets: Mapping[str, np.ndarray], sources: np.ndarray, feature_spec: str) -> dict[str, Any]:
    matrix, _ = combine_families(families, names, FEATURE_SPECS[feature_spec]); target = targets["k100"]; censored = targets["censored100"]
    raw = np.zeros(len(data.rows)); choices = np.zeros(len(data.rows), dtype=np.int16); probabilities = np.zeros(len(data.rows)); fold_rows = []
    for outer_fold, (outer_train, outer_test) in enumerate(fixed_folds(sources)):
        inner_raw = np.zeros(len(outer_train)); splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 100 + outer_fold)
        for inner_fold, (train_pos, valid_pos) in enumerate(splitter.split(matrix[outer_train], sources[outer_train])):
            train_idx, valid_idx = outer_train[train_pos], outer_train[valid_pos]
            probability = fit_binary(matrix, censored.astype(np.int8), train_idx, valid_idx, SEED + inner_fold)
            reachable_prediction = fit_reachable_regression(matrix, target, censored, train_idx, valid_idx, SEED + 1000 + inner_fold)
            inner_raw[valid_pos] = reachable_prediction + probability * (CAP - reachable_prediction)
        scale, margin, metric = choose_calibration(data, outer_train, inner_raw)
        probabilities[outer_test] = fit_binary(matrix, censored.astype(np.int8), outer_train, outer_test, SEED + outer_fold)
        reachable_prediction = fit_reachable_regression(matrix, target, censored, outer_train, outer_test, SEED + 1000 + outer_fold)
        raw[outer_test] = reachable_prediction + probabilities[outer_test] * (CAP - reachable_prediction)
        choices[outer_test] = choices_from_prediction(raw[outer_test], scale, margin)
        fold_rows.append({"fold": outer_fold, "scale": scale, "margin": margin, **metric})
    summary = metric_summary(data.rows, choices); predicted_easy = choices <= 10; target_easy = target <= 10
    payload = {"name": f"two_part_censored_{feature_spec}", "summary": {k: v for k, v in summary.items() if k != "recalls"}, "bootstrap_ci": bootstrap_ci(data.rows, choices), "folds": fold_rows, "safe_easy_precision": float(np.mean(target_easy[predicted_easy])) if np.any(predicted_easy) else 0.0, "easy_coverage": float(np.mean(predicted_easy[target_easy])) if np.any(target_easy) else 0.0, "unreachable_probability": probabilities, "raw_prediction": raw, "choices": choices, "per_query_recall": summary["recalls"], "histories": [], "feature_importance": []}
    save_result(output, "censoring", payload, data, target)
    return payload


def jitter_curve_features(data: FrozenData, indices: np.ndarray, target: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """At most two monotone score-curve jitters for rows in below-median bins."""
    rng = np.random.default_rng(seed)
    bins = np.digitize(target[indices], [10, 20, 50, 100])
    counts = Counter(bins.tolist()); median_count = float(np.median(list(counts.values())))
    synthetic_x, synthetic_y = [], []
    for global_index, bin_id in zip(indices, bins):
        if counts[int(bin_id)] >= median_count:
            continue
        base = np.asarray(data.rows[int(global_index)]["scores"], dtype=float)
        for _ in range(2):
            noisy = np.clip(base * (1.0 + rng.normal(0.0, 0.01, size=len(base))), 0.0, None)
            noisy = np.minimum.accumulate(noisy)  # preserve a non-increasing retrieval curve
            row_values: list[float] = []
            for builder in (_point_features, _band_features, _shape_features):
                row_values.extend(builder(noisy, "fused")[1])
            synthetic_x.append(row_values); synthetic_y.append(float(target[int(global_index)]))
    width = len(_point_features(np.asarray(data.rows[0]["scores"], dtype=float), "fused")[1]) + len(_band_features(np.asarray(data.rows[0]["scores"], dtype=float), "fused")[1]) + len(_shape_features(np.asarray(data.rows[0]["scores"], dtype=float), "fused")[1])
    return (np.asarray(synthetic_x, dtype=np.float32).reshape(-1, width), np.asarray(synthetic_y, dtype=np.float32))


def run_augmentation(output: Path, data: FrozenData, families: Mapping[str, np.ndarray], names: Mapping[str, list[str]], targets: Mapping[str, np.ndarray], sources: np.ndarray) -> dict[str, Any]:
    matrix, _ = combine_families(families, names, FEATURE_SPECS["point_band_shape"]); target = targets["k100"]
    raw = np.zeros(len(data.rows)); choices = np.zeros(len(data.rows), dtype=np.int16); folds_out = []
    for outer_fold, (outer_train, outer_test) in enumerate(fixed_folds(sources)):
        inner_prediction = np.zeros(len(outer_train)); splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 100 + outer_fold)
        for inner_fold, (train_pos, valid_pos) in enumerate(splitter.split(matrix[outer_train], sources[outer_train])):
            train_idx, valid_idx = outer_train[train_pos], outer_train[valid_pos]
            synth_x, synth_y = jitter_curve_features(data, train_idx, target, SEED + outer_fold * 100 + inner_fold)
            model, _ = make_regressor({"n_estimators": 350}, "squared", SEED + inner_fold)
            model.fit(np.vstack([matrix[train_idx], synth_x]), np.concatenate([target[train_idx], synth_y]), verbose=False)
            inner_prediction[valid_pos] = np.clip(model.predict(matrix[valid_idx]), 1, CAP)
        scale, margin, metric = choose_calibration(data, outer_train, inner_prediction)
        synth_x, synth_y = jitter_curve_features(data, outer_train, target, SEED + outer_fold)
        model, _ = make_regressor({"n_estimators": 350}, "squared", SEED + outer_fold)
        model.fit(np.vstack([matrix[outer_train], synth_x]), np.concatenate([target[outer_train], synth_y]), verbose=False)
        raw[outer_test] = np.clip(model.predict(matrix[outer_test]), 1, CAP); choices[outer_test] = choices_from_prediction(raw[outer_test], scale, margin)
        folds_out.append({"fold": outer_fold, "synthetic_rows": len(synth_y), "max_per_rare_row": 2, "jitter_std": .01, "scale": scale, "margin": margin, **metric})
    summary = metric_summary(data.rows, choices)
    payload = {"name": "monotone_curve_jitter", "summary": {k: v for k, v in summary.items() if k != "recalls"}, "bootstrap_ci": bootstrap_ci(data.rows, choices), "folds": folds_out, "diagnostic_only": True, "raw_prediction": raw, "choices": choices, "per_query_recall": summary["recalls"], "histories": [], "feature_importance": []}
    save_result(output, "augmentation", payload, data, target)
    return payload


def classification_policy(probabilities: np.ndarray, representatives: np.ndarray, scale: float, margin: int) -> np.ndarray:
    expected = probabilities @ representatives
    return choices_from_prediction(expected, scale, margin)


def make_classifier(classes: int, seed: int) -> XGBClassifier:
    params: dict[str, Any] = {
        "n_estimators": 350, "max_depth": 2, "learning_rate": 0.05,
        "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85,
        "reg_lambda": 5, "random_state": seed, "n_jobs": 4, "tree_method": "hist",
    }
    if classes == 2:
        params.update(objective="binary:logistic", eval_metric="logloss")
    else:
        params.update(objective="multi:softprob", num_class=classes, eval_metric="mlogloss")
    return XGBClassifier(**params)


def run_classification(output: Path, data: FrozenData, families: Mapping[str, np.ndarray], names: Mapping[str, list[str]], targets: Mapping[str, np.ndarray], sources: np.ndarray, feature_spec: str) -> list[dict[str, Any]]:
    matrix, _ = combine_families(families, names, FEATURE_SPECS[feature_spec]); target = targets["k100"]
    schemes = {
        "bins2": [10],
        "bins3": [10, 70],
        "bins4": [10, 20, 70],
        "bins5": [10, 20, 50, 100],
        "ordinal": [10, 20, 50, 100],
    }
    results = []
    for scheme, thresholds in schemes.items():
        labels = np.digitize(target, thresholds)
        representatives = np.asarray([np.median(target[labels == label]) if np.any(labels == label) else (thresholds + [CAP])[label] for label in range(len(thresholds) + 1)], dtype=float)
        probabilities_oof = np.zeros((len(data.rows), len(thresholds) + 1)); choices = np.zeros(len(data.rows), dtype=np.int16)
        calibrations = []
        for outer_fold, (train_idx, test_idx) in enumerate(fixed_folds(sources)):
            inner_prob = np.zeros((len(train_idx), len(thresholds) + 1))
            splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 100 + outer_fold)
            for inner_fold, (train_pos, valid_pos) in enumerate(splitter.split(matrix[train_idx], sources[train_idx])):
                model = make_classifier(len(thresholds) + 1, SEED + inner_fold)
                model.fit(matrix[train_idx[train_pos]], labels[train_idx[train_pos]])
                inner_prob[valid_pos] = model.predict_proba(matrix[train_idx[valid_pos]])
            inner_expected = inner_prob @ representatives
            scale, margin, metric = choose_calibration(data, train_idx, inner_expected)
            model = make_classifier(len(thresholds) + 1, SEED + outer_fold)
            model.fit(matrix[train_idx], labels[train_idx])
            probabilities_oof[test_idx] = model.predict_proba(matrix[test_idx])
            choices[test_idx] = classification_policy(probabilities_oof[test_idx], representatives, scale, margin)
            calibrations.append({"fold": outer_fold, "scale": scale, "margin": margin, **metric})
        predicted_labels = np.argmax(probabilities_oof, axis=1)
        summary = metric_summary(data.rows, choices)
        payload = {"name": scheme, "summary": {k: v for k, v in summary.items() if k != "recalls"}, "thresholds": thresholds, "representatives": representatives, "confusion_matrix": confusion_matrix(labels, predicted_labels, labels=list(range(len(thresholds) + 1))), "calibrations": calibrations, "raw_prediction": probabilities_oof @ representatives, "choices": choices, "per_query_recall": summary["recalls"], "histories": [], "feature_importance": []}
        save_result(output, "classification", payload, data, target); results.append(payload)
    return results


def run_alternative_models(output: Path, data: FrozenData, families: Mapping[str, np.ndarray], names: Mapping[str, list[str]], targets: Mapping[str, np.ndarray], sources: np.ndarray, feature_spec: str) -> list[dict[str, Any]]:
    matrix, _ = combine_families(families, names, FEATURE_SPECS[feature_spec]); target = targets["k100"]
    factories: dict[str, Callable[[int], Any]] = {
        "hgb": lambda seed: HistGradientBoostingRegressor(max_iter=350, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=5, random_state=seed),
        "extra_trees": lambda seed: ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.85, random_state=seed, n_jobs=4),
        "random_forest": lambda seed: RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.85, random_state=seed, n_jobs=4),
    }
    results = []
    for name, factory in factories.items():
        raw = np.zeros(len(data.rows)); choices = np.zeros(len(data.rows), dtype=np.int16); calibrations = []
        for outer_fold, (train_idx, test_idx) in enumerate(fixed_folds(sources)):
            inner_prediction = np.zeros(len(train_idx)); splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 100 + outer_fold)
            for train_pos, valid_pos in splitter.split(matrix[train_idx], sources[train_idx]):
                model = factory(SEED + outer_fold); model.fit(matrix[train_idx[train_pos]], target[train_idx[train_pos]]); inner_prediction[valid_pos] = model.predict(matrix[train_idx[valid_pos]])
            scale, margin, metric = choose_calibration(data, train_idx, inner_prediction)
            model = factory(SEED + outer_fold); model.fit(matrix[train_idx], target[train_idx]); raw[test_idx] = np.clip(model.predict(matrix[test_idx]), 1, CAP); choices[test_idx] = choices_from_prediction(raw[test_idx], scale, margin); calibrations.append({"fold": outer_fold, "scale": scale, "margin": margin, **metric})
        summary = metric_summary(data.rows, choices)
        payload = {"name": name, "summary": {k: v for k, v in summary.items() if k != "recalls"}, "calibrations": calibrations, "raw_prediction": raw, "choices": choices, "per_query_recall": summary["recalls"], "histories": [], "feature_importance": []}
        save_result(output, "models", payload, data, target); results.append(payload)
    return results


def normalize_component(raw: np.ndarray, mode: str) -> np.ndarray:
    if mode == "minmax_top100":
        output = np.empty_like(raw, dtype=np.float32)
        for index, scores in enumerate(raw):
            top = np.sort(scores)[-100:]
            output[index] = np.clip((scores - top[0]) / (top[-1] - top[0] + 1e-9), 0.0, None)
        return output
    if mode == "percentile":
        output = np.empty_like(raw, dtype=np.float32)
        for index, scores in enumerate(raw):
            order = np.argsort(scores, kind="stable")
            ranks = np.empty(len(scores), dtype=np.float32); ranks[order] = np.arange(len(scores), dtype=np.float32)
            output[index] = ranks / max(len(scores) - 1, 1)
        return output
    if mode == "robust_z":
        output = np.empty_like(raw, dtype=np.float32)
        for index, scores in enumerate(raw):
            top = np.sort(scores)[-100:]
            median = float(np.median(top)); scale = float(np.percentile(top, 75) - np.percentile(top, 25)) + 1e-9
            output[index] = np.clip((scores - median) / scale, -5.0, 5.0)
        return output
    raise ValueError(mode)


def rerank_fusion(data: FrozenData, weight_bm25: float, normalization: str = "minmax_top100") -> FrozenData:
    if normalization == "minmax_top100":
        lexical, dense = data.lexical_full, data.dense_full
    else:
        if data.lexical_raw_full is None or data.dense_raw_full is None:
            raise ValueError("Raw component scores are required for alternative normalization")
        lexical = normalize_component(data.lexical_raw_full, normalization)
        dense = normalize_component(data.dense_raw_full, normalization)
    fused_full = weight_bm25 * lexical + (1.0 - weight_bm25) * dense
    rows = []
    for index, old in enumerate(data.rows):
        order = np.argsort(-fused_full[index], kind="stable")[:CAP]
        rows.append({
            **old,
            "ranked": [data.page_ids[int(position)] for position in order],
            "ranked_indices": order.astype(np.int32),
            "scores": fused_full[index, order].astype(np.float32),
            "lexical_scores": lexical[index, order].astype(np.float32),
            "dense_scores": dense[index, order].astype(np.float32),
        })
    return FrozenData(rows, data.query_vectors, fused_full, lexical, dense, data.page_ids, data.page_texts, {**data.protocol, "ranking": f"global {weight_bm25:.1f} BM25 + {1-weight_bm25:.1f} dense; {normalization}"}, data.lexical_raw_full, data.dense_raw_full)


def fixed_curve(data: FrozenData, indices: np.ndarray) -> dict[str, float]:
    return {f"recall@{k}": float(np.mean([recall_at(data.rows[int(index)], k) for index in indices])) for k in (20, 50, 70, 100, 200)}


def run_ranking(output: Path, data: FrozenData, targets: Mapping[str, np.ndarray], sources: np.ndarray) -> list[dict[str, Any]]:
    weights = [round(value, 1) for value in np.arange(0.0, 1.01, 0.1)]
    variants = {(normalization, weight): rerank_fusion(data, weight, normalization) for normalization in ("minmax_top100", "percentile", "robust_z") for weight in weights}
    screen = [{"normalization": key[0], "bm25_weight": key[1], **fixed_curve(variant, np.arange(len(data.rows)))} for key, variant in variants.items()]
    write_csv(output / "ranking" / "fusion_weight_diagnostic.csv", screen)
    selected_weights = []
    oof_curve = {k: np.zeros(len(data.rows)) for k in (20, 50, 70, 100, 200)}
    for fold, (train_idx, test_idx) in enumerate(fixed_folds(sources)):
        candidates = []
        for (normalization, weight), variant in variants.items():
            curve = fixed_curve(variant, train_idx)
            candidates.append({"normalization": normalization, "weight": weight, **curve})
        winner = max(candidates, key=lambda row: (row["recall@70"], row["recall@50"], row["recall@200"], -abs(row["weight"] - 0.2)))
        selected_weights.append({"normalization": winner["normalization"], "weight": winner["weight"]})
        variant = variants[(winner["normalization"], winner["weight"])]
        for index in test_idx:
            for k in oof_curve: oof_curve[k][index] = recall_at(variant.rows[int(index)], k)
    selection = {"fold_weights": selected_weights, "oof_fixed_curve": {f"recall@{k}": float(np.mean(values)) for k, values in oof_curve.items()}, "selection": "max train-fold recall@70; ties recall@50, candidate recall@200, closeness to frozen 0.2", "warning": "Fusion-weight screen is exploratory; every reported OOF row uses a weight selected without its outer-test qrels."}
    json_dump(output / "ranking" / "nested_weight_selection.json", selection)

    results = []
    for spec_name in ("baseline21", "score_modality"):
        raw = np.zeros(len(data.rows)); choices = np.zeros(len(data.rows), dtype=np.int16); fold_audit = []
        for outer_fold, (train_idx, test_idx) in enumerate(fixed_folds(sources)):
            choice = selected_weights[outer_fold]
            variant = variants[(choice["normalization"], choice["weight"])]
            variant_families, variant_names = build_score_feature_families(variant)
            matrix, feature_names = combine_families(variant_families, variant_names, FEATURE_SPECS[spec_name])
            target = target_arrays(variant.rows)["k100"]
            inner_prediction, rounds = inner_oof(matrix, target, sources, train_idx, {}, "squared", outer_fold)
            scale, margin, inner_metric = choose_calibration(variant, train_idx, inner_prediction)
            params = {"n_estimators": int(np.clip(np.median(rounds), 10, 1000))}
            fit_idx, stop_idx = train_test_split(train_idx, test_size=.2, random_state=SEED + outer_fold, stratify=sources[train_idx])
            stopped = fit_xgb(matrix[fit_idx], target[fit_idx], matrix[stop_idx], target[stop_idx], params, "squared", SEED + outer_fold)
            params["n_estimators"] = stopped.best_iteration + 1
            model, _ = make_regressor(params, "squared", SEED + outer_fold); model.fit(matrix[train_idx], target[train_idx], verbose=False)
            prediction = np.clip(model.predict(matrix[test_idx]), 1, CAP)
            raw[test_idx] = prediction; choices[test_idx] = choices_from_prediction(prediction, scale, margin)
            fold_audit.append({"fold": outer_fold, "normalization": choice["normalization"], "bm25_weight": choice["weight"], "scale": scale, "margin": margin, **inner_metric})
        # Each row's recall must be evaluated against its fold-specific ranking.
        recalls = np.zeros(len(data.rows))
        for outer_fold, (_, test_idx) in enumerate(fixed_folds(sources)):
            choice = selected_weights[outer_fold]
            variant = variants[(choice["normalization"], choice["weight"])]
            recalls[test_idx] = [recall_at(variant.rows[int(index)], int(choices[index])) for index in test_idx]
        summary = {"queries": len(data.rows), "mean_k": float(np.mean(choices)), "median_k": float(np.median(choices)), "p90_k": float(np.percentile(choices, 90)), "macro_recall": float(np.mean(recalls))}
        payload = {"name": f"upgraded_ranking_{spec_name}", "summary": summary, "folds": fold_audit, "selected_weights": selected_weights, "raw_prediction": raw, "choices": choices, "per_query_recall": recalls, "histories": [], "feature_importance": []}
        save_result(output, "ranking", payload, data, targets["k100"]); results.append(payload)
    return results


def build_dashboard(output: Path) -> None:
    """Build a dependency-free dashboard that reads the generated CSV/JSON files."""
    result_files = sorted(path for stage in ("features", "tuning", "loss", "classification", "neighbor", "models", "ranking", "policy", "censoring", "augmentation") for path in (output / stage).glob("*.json") if path.name not in ("summary.json", "inner_search.json", "nested_weight_selection.json"))
    models = []
    for path in result_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "summary" in payload: models.append({"stage": path.parent.name, **payload})
        except json.JSONDecodeError:
            continue
    feasible = [model for model in models if model["summary"]["mean_k"] < SUCCESS_MEAN_K]
    champion = max(feasible or models, key=lambda model: (model["summary"]["macro_recall"], -model["summary"]["mean_k"])) if models else None
    oof_rows: list[dict[str, Any]] = []
    if champion is not None:
        oof_path = output / champion["stage"] / f"{champion['name']}_oof.csv"
        if oof_path.exists():
            with oof_path.open(encoding="utf-8-sig") as handle:
                oof_rows = list(csv.DictReader(handle))
    data = {"protocol": json.loads((output / "protocol.json").read_text(encoding="utf-8")), "models": models, "champion": champion, "oof": oof_rows}
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = """<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Adaptive-K OOF Dashboard</title><style>
body{font:14px system-ui;margin:0;background:#0d1117;color:#dce3ea}header,main{padding:22px}header{background:#151b23;position:sticky;top:0;z-index:2}h1{margin:0 0 6px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.card,section{background:#151b23;border:1px solid #30363d;border-radius:10px;padding:15px;margin:14px 0;overflow:auto}table{width:100%;border-collapse:collapse}th,td{padding:8px;border-bottom:1px solid #30363d;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}.ok{color:#3fb950}.bad{color:#f85149}select,input{background:#21262d;color:white;padding:7px}.bar{height:8px;background:#30363d}.bar i{display:block;height:100%;background:#58a6ff}pre{white-space:pre-wrap}</style></head><body><header><h1>XGBoost Adaptive-K â€” nested OOF</h1><div id='protocol'></div></header><main><section><h2>Pareto / ablation table</h2><table id='models'></table></section><section><h2>Fold calibration and loss</h2><select id='pick'></select><div id='detail'></div></section><section><h2>Feature importance</h2><div id='importance'></div></section><section><h2>Champion OOF predictions</h2><input id='filter' placeholder='filter qid/source/query'><table id='oof'></table></section></main><script>const D=""" + embedded + """;const p=D.protocol;document.querySelector('#protocol').textContent=`${p.benchmark}; ${p.outer_folds} outer / ${p.inner_folds} inner folds; frozen ${p.ranking}`;const ms=D.models.sort((a,b)=>b.summary.macro_recall-a.summary.macro_recall||a.summary.mean_k-b.summary.mean_k);document.querySelector('#models').innerHTML='<tr><th>Stage / model</th><th>Mean K</th><th>Median</th><th>P90</th><th>Recall</th><th>Status</th></tr>'+ms.map(m=>`<tr><td>${m.stage} / ${m.name}</td><td>${m.summary.mean_k.toFixed(2)}</td><td>${m.summary.median_k.toFixed(1)}</td><td>${m.summary.p90_k.toFixed(1)}</td><td>${(100*m.summary.macro_recall).toFixed(2)}%</td><td class='${m.summary.mean_k<60&&m.summary.macro_recall>=.87?'ok':'bad'}'>${m.summary.mean_k<60&&m.summary.macro_recall>=.87?'PASS':'â€”'}</td></tr>`).join('');const pick=document.querySelector('#pick');pick.innerHTML=ms.map((m,i)=>`<option value='${i}'>${m.stage} / ${m.name}</option>`).join('');function render(){const m=ms[+pick.value];document.querySelector('#detail').innerHTML='<pre>'+JSON.stringify({summary:m.summary,bootstrap_ci:m.bootstrap_ci,per_source:m.per_source,safe_easy_precision:m.safe_easy_precision,easy_coverage:m.easy_coverage,folds:m.folds,confusion_matrix:m.confusion_matrix,histories:m.histories},null,2)+'</pre>';document.querySelector('#importance').innerHTML=(m.feature_importance||[]).slice(0,25).map(x=>`<div>${x.feature} ${(100*x.gain_importance_mean).toFixed(2)}%<div class='bar'><i style='width:${Math.min(100,500*x.gain_importance_mean)}%'></i></div></div>`).join('')||'No importance data for this model.'}pick.onchange=render;render();function renderOOF(){const q=document.querySelector('#filter').value.toLowerCase();const rows=D.oof.filter(r=>JSON.stringify(r).toLowerCase().includes(q));document.querySelector('#oof').innerHTML='<tr><th>QID</th><th>Source</th><th>Target K</th><th>Raw prediction</th><th>Chosen K</th><th>Recall</th><th>Query</th></tr>'+rows.map(r=>`<tr><td>${r.qid}</td><td>${r.source}</td><td>${(+r.target_k).toFixed(0)}</td><td>${(+r.raw_prediction).toFixed(2)}</td><td>${r.chosen_k}</td><td>${(100*+r.recall).toFixed(1)}%</td><td title='${r.query.replaceAll("'","&#39;")}'>${r.query.slice(0,90)}</td></tr>`).join('')}document.querySelector('#filter').oninput=renderOOF;renderOOF();</script></body></html>"""
    (output / "dashboard").mkdir(parents=True, exist_ok=True)
    (output / "dashboard" / "index.html").write_text(html, encoding="utf-8")


def build_final_report(output: Path) -> dict[str, Any]:
    models = []
    for stage in ("features", "tuning", "loss", "classification", "neighbor", "models", "ranking", "policy", "censoring", "augmentation"):
        for path in (output / stage).glob("*.json"):
            if path.name in ("summary.json", "inner_search.json", "nested_weight_selection.json"):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "summary" in payload:
                models.append({"stage": stage, "name": payload["name"], **payload["summary"], "bootstrap_ci": payload.get("bootstrap_ci"), "per_source": payload.get("per_source"), "safe_easy_precision": payload.get("safe_easy_precision"), "easy_coverage": payload.get("easy_coverage")})
    models.sort(key=lambda row: (row["mean_k"], -row["macro_recall"]))
    frontier = []
    best_recall = -1.0
    for row in models:
        if row["macro_recall"] > best_recall:
            frontier.append(row); best_recall = row["macro_recall"]
    feasible = [row for row in models if row["mean_k"] < SUCCESS_MEAN_K]
    champion = max(feasible, key=lambda row: (row["macro_recall"], -row["mean_k"])) if feasible else None
    success = bool(champion and champion["macro_recall"] >= SUCCESS_RECALL)
    write_csv(output / "final" / "all_results.csv", [{key: value for key, value in row.items() if key not in ("bootstrap_ci", "per_source")} for row in models])
    write_csv(output / "final" / "pareto_frontier.csv", [{key: value for key, value in row.items() if key not in ("bootstrap_ci", "per_source")} for row in frontier])
    report = ["# Adaptive-K XGBoost nested-OOF report", "", f"**Acceptance: {'PASS' if success else 'FAIL'}** â€” required Mean K < 60 and macro recall â‰¥ 87.00%.", ""]
    if champion:
        report += [f"Best evaluated model under the strict budget is `{champion['stage']}/{champion['name']}`: Mean K **{champion['mean_k']:.2f}**, median **{champion['median_k']:.1f}**, P90 **{champion['p90_k']:.1f}**, macro recall **{100*champion['macro_recall']:.2f}%**.", ""]
        if champion.get("bootstrap_ci"):
            ci = champion["bootstrap_ci"]
            report += [f"Bootstrap 95% CI: Mean K [{ci['mean_k'][0]:.2f}, {ci['mean_k'][1]:.2f}], recall [{100*ci['macro_recall'][0]:.2f}%, {100*ci['macro_recall'][1]:.2f}%].", ""]
    report += ["## Pareto frontier", "", "| Stage/model | Mean K | Recall |", "|---|---:|---:|"]
    report += [f"| {row['stage']}/{row['name']} | {row['mean_k']:.2f} | {100*row['macro_recall']:.2f}% |" for row in frontier]
    report += ["", "## Interpretation", "", "- Feature redesign improved the fixed score-only XGBoost, but did not close the full constrained-recall gap.", "- Hyperparameter tuning, easy weighting, classification bins, neighbour target summaries, and tree-model swaps did not reach the acceptance threshold under nested OOF.", "- Robust-z component calibration materially improved candidate recall@200, but most of that gain lies beyond K=70; the Adaptive-K selector could not convert it into recall below Mean K 60.", "- Results are exploratory method development on the same 220-query benchmark. No configuration selected after inspecting outer OOF is called validated.", ""]
    (output / "final").mkdir(parents=True, exist_ok=True)
    (output / "final" / "report.md").write_text("\n".join(report), encoding="utf-8")
    payload = {"success": success, "criterion": {"mean_k_lt": 60, "macro_recall_gte": .87}, "champion": champion, "pareto_frontier": frontier}
    json_dump(output / "final" / "report.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("features", "tuning", "loss", "classification", "neighbor", "models", "ranking", "policy", "censoring", "augmentation", "final", "all"), default="features")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-spec", choices=tuple(FEATURE_SPECS), default="score_modality")
    parser.add_argument("--tuning-samples", type=int, default=64)
    parser.add_argument("--loss-filter", help="Optional substring for a partial loss screen, e.g. asymmetric")
    args = parser.parse_args(argv)
    data, families, names, targets, sources = prepare(args.output_dir)
    stages = ("features", "tuning", "loss", "classification", "neighbor", "models", "ranking", "policy", "censoring", "augmentation") if args.stage == "all" else (() if args.stage == "final" else (args.stage,))
    for stage in stages:
        if stage == "features": run_features(args.output_dir, data, families, names, targets, sources)
        elif stage == "tuning": run_tuning(args.output_dir, data, families, names, targets, sources, args.feature_spec, args.tuning_samples)
        elif stage == "loss": run_loss(args.output_dir, data, families, names, targets, sources, args.feature_spec, args.loss_filter)
        elif stage == "classification": run_classification(args.output_dir, data, families, names, targets, sources, args.feature_spec)
        elif stage == "neighbor": run_neighbor(args.output_dir, data, families, names, targets, sources, args.feature_spec)
        elif stage == "models": run_alternative_models(args.output_dir, data, families, names, targets, sources, args.feature_spec)
        elif stage == "ranking": run_ranking(args.output_dir, data, targets, sources)
        elif stage == "policy": run_safe_policy(args.output_dir, data, families, names, targets, sources, args.feature_spec)
        elif stage == "censoring": run_censoring(args.output_dir, data, families, names, targets, sources, args.feature_spec)
        elif stage == "augmentation": run_augmentation(args.output_dir, data, families, names, targets, sources)
    build_dashboard(args.output_dir)
    build_final_report(args.output_dir)
    print(f"Artifacts: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
