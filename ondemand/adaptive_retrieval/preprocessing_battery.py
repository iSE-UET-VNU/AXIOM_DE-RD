"""Leakage-safe preprocessing/feature/data ablations across five adaptive-K regressors."""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

from .features import FEATURE_NAMES
from .full_factorial_ablation import MODEL_LABELS, MODELS
from .model import XGB_PARAMETERS, _bootstrap_ci, _new_model, _row_metrics
from .model_family_ablation import MODEL_SPECS, _load_rows, _new_estimator
from .scaling_ablation import DEFAULT_RUN_DIR, _read_jsonl


DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "preprocessing_battery"

METHODS: dict[str, dict[str, Any]] = {
    "standard_xy_baseline": {
        "description": "Baseline 21 features + fold-local StandardScaler(X,y)",
        "feature_set": "baseline",
        "transform": "standard",
        "balance": "none",
    },
    "robust_xy": {
        "description": "RobustScaler(X) + StandardScaler(y)",
        "feature_set": "baseline",
        "transform": "robust",
        "balance": "none",
    },
    "power_xy": {
        "description": "Yeo-Johnson PowerTransformer(X) + StandardScaler(y)",
        "feature_set": "baseline",
        "transform": "power",
        "balance": "none",
    },
    "quantile_normal_xy": {
        "description": "QuantileTransformer(X→normal) + StandardScaler(y)",
        "feature_set": "baseline",
        "transform": "quantile_normal",
        "balance": "none",
    },
    "winsor_standard_xy": {
        "description": "Train-fold 1–99% winsorization + StandardScaler(X,y)",
        "feature_set": "baseline",
        "transform": "winsor_standard",
        "balance": "none",
    },
    "corr98_standard_xy": {
        "description": "Drop train-fold |correlation|>0.98 redundancies + StandardScaler(X,y)",
        "feature_set": "baseline",
        "transform": "corr98_standard",
        "balance": "none",
    },
    "mutual_info_top10_xy": {
        "description": "Train-fold mutual-information top-10 features + StandardScaler(X,y)",
        "feature_set": "baseline",
        "transform": "mutual_info_top10_standard",
        "balance": "none",
    },
    "pca95_xy": {
        "description": "StandardScaler(X) + train-fold PCA 95% variance + StandardScaler(y)",
        "feature_set": "baseline",
        "transform": "pca95",
        "balance": "none",
    },
    "expanded_standard_xy": {
        "description": "Baseline + expanded point/local/band/drop/entropy curve features + StandardScaler(X,y)",
        "feature_set": "expanded",
        "transform": "standard",
        "balance": "none",
    },
    "expanded_corr98_xy": {
        "description": "Expanded curve features + correlation pruning + StandardScaler(X,y)",
        "feature_set": "expanded",
        "transform": "corr98_standard",
        "balance": "none",
    },
    "balanced_bins_xy": {
        "description": "StandardScaler(X,y) + train-fold ordinal-bin balanced bootstrap",
        "feature_set": "baseline",
        "transform": "standard",
        "balance": "ordinal_bootstrap",
    },
    "balanced_jitter_xy": {
        "description": "Balanced bootstrap + Gaussian jitter on duplicated standardized X",
        "feature_set": "baseline",
        "transform": "standard",
        "balance": "ordinal_bootstrap_jitter",
    },
}


POINT_RANKS = (1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100)
LOCAL_RANKS = (1, 2, 5, 10, 20, 50, 75, 100)
BANDS = ((1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 75), (76, 100))


def _entropy(values: np.ndarray) -> float:
    shifted = values - values.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return float(-(probabilities * np.log(probabilities + 1e-15)).sum())


def _expanded_score_features(scores: list[float]) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (100,) or not np.isfinite(values).all():
        raise ValueError("expanded feature extraction requires exactly 100 finite scores")
    result: dict[str, float] = {}
    for rank in POINT_RANKS:
        result[f"point_r{rank}"] = float(values[rank - 1])
    for rank in LOCAL_RANKS:
        center = rank - 1
        left, right = max(0, center - 2), min(len(values), center + 3)
        result[f"local5_r{rank}"] = float(values[left:right].mean())
    for start, end in BANDS:
        result[f"band_mean_{start}_{end}"] = float(values[start - 1:end].mean())
        result[f"band_std_{start}_{end}"] = float(values[start - 1:end].std())
    for end in (5, 10, 20, 50, 100):
        result[f"drop_1_{end}"] = float(values[0] - values[end - 1])
        result[f"entropy_top_{end}"] = _entropy(values[:end])
    gaps = values[:-1] - values[1:]
    for end in (5, 10, 20, 50, 99):
        subset = gaps[:end]
        result[f"gap_mean_top_{end}"] = float(subset.mean())
        result[f"gap_std_top_{end}"] = float(subset.std())
    result["top1_minus_top5mean"] = float(values[0] - values[:5].mean())
    result["top1_minus_top10mean"] = float(values[0] - values[:10].mean())
    result["top5mean_minus_6_10mean"] = float(values[:5].mean() - values[5:10].mean())
    result["early_late_drop_ratio"] = float((values[0] - values[19]) / (values[19] - values[-1] + 1e-6))
    return result


def _load_feature_matrices(
    run_dir: Path, rows: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    baseline_names = list(FEATURE_NAMES)
    baseline = np.asarray(
        [[row["features"][name] for name in baseline_names] for row in rows], dtype=np.float64
    )
    score_by_qid = {row["qid"]: row["scores"] for row in _read_jsonl(run_dir / "retrieval_scores.jsonl")}
    expanded_dicts = [_expanded_score_features(score_by_qid[row["qid"]]) for row in rows]
    extra_names = list(expanded_dicts[0])
    if any(list(item) != extra_names for item in expanded_dicts):
        raise ValueError("expanded feature ordering is inconsistent")
    extras = np.asarray([[item[name] for name in extra_names] for item in expanded_dicts], dtype=np.float64)
    expanded = np.column_stack([baseline, extras])
    if not np.isfinite(expanded).all():
        raise ValueError("expanded matrix contains NaN/Inf")
    return {"baseline": baseline, "expanded": expanded}, {
        "baseline": baseline_names,
        "expanded": [*baseline_names, *extra_names],
    }


def _correlation_keep_indices(train_x: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    std = train_x.std(axis=0)
    keep: list[int] = []
    for index in range(train_x.shape[1]):
        if std[index] <= 1e-15:
            continue
        redundant = False
        for kept in keep:
            corr = np.corrcoef(train_x[:, index], train_x[:, kept])[0, 1]
            if np.isfinite(corr) and abs(corr) > threshold:
                redundant = True
                break
        if not redundant:
            keep.append(index)
    if not keep:
        raise RuntimeError("correlation pruning removed every feature")
    return np.asarray(keep, dtype=np.int32)


def _fit_transform_x(
    method: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    feature_names: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    diagnostic: dict[str, Any] = {
        "input_features": train_x.shape[1],
        "transform": method,
    }
    if method == "standard":
        transformer: Any = StandardScaler()
        train_z, eval_z = transformer.fit_transform(train_x), transformer.transform(eval_x)
    elif method == "robust":
        transformer = RobustScaler(quantile_range=(25.0, 75.0))
        train_z, eval_z = transformer.fit_transform(train_x), transformer.transform(eval_x)
    elif method == "power":
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        train_z, eval_z = transformer.fit_transform(train_x), transformer.transform(eval_x)
    elif method == "quantile_normal":
        transformer = QuantileTransformer(
            n_quantiles=min(100, len(train_x)), output_distribution="normal", random_state=seed
        )
        train_z, eval_z = transformer.fit_transform(train_x), transformer.transform(eval_x)
        diagnostic["n_quantiles"] = min(100, len(train_x))
    elif method == "winsor_standard":
        lower = np.quantile(train_x, 0.01, axis=0)
        upper = np.quantile(train_x, 0.99, axis=0)
        scaler = StandardScaler()
        train_z = scaler.fit_transform(np.clip(train_x, lower, upper))
        eval_z = scaler.transform(np.clip(eval_x, lower, upper))
        diagnostic["clip_quantiles"] = [0.01, 0.99]
    elif method == "corr98_standard":
        keep = _correlation_keep_indices(train_x, 0.98)
        scaler = StandardScaler()
        train_z = scaler.fit_transform(train_x[:, keep])
        eval_z = scaler.transform(eval_x[:, keep])
        diagnostic["kept_feature_indices"] = keep.tolist()
        diagnostic["kept_feature_names"] = [feature_names[index] for index in keep]
    elif method == "mutual_info_top10_standard":
        k = min(10, train_x.shape[1])
        selector = SelectKBest(
            score_func=lambda x, y: mutual_info_regression(x, y, random_state=seed), k=k
        )
        selected_train = selector.fit_transform(train_x, train_y)
        selected_eval = selector.transform(eval_x)
        scaler = StandardScaler()
        train_z, eval_z = scaler.fit_transform(selected_train), scaler.transform(selected_eval)
        keep = selector.get_support(indices=True)
        diagnostic["kept_feature_indices"] = keep.tolist()
        diagnostic["kept_feature_names"] = [feature_names[index] for index in keep]
        diagnostic["mutual_information_scores"] = [
            None if not np.isfinite(value) else float(value) for value in selector.scores_
        ]
    elif method == "pca95":
        scaler = StandardScaler()
        train_scaled, eval_scaled = scaler.fit_transform(train_x), scaler.transform(eval_x)
        pca = PCA(n_components=0.95, svd_solver="full")
        train_z, eval_z = pca.fit_transform(train_scaled), pca.transform(eval_scaled)
        diagnostic["pca_components"] = int(pca.n_components_)
        diagnostic["explained_variance_ratio_sum"] = float(pca.explained_variance_ratio_.sum())
    else:
        raise ValueError(f"unknown transform: {method}")
    diagnostic["output_features"] = int(train_z.shape[1])
    return np.asarray(train_z), np.asarray(eval_z), diagnostic


def _ordinal_bin(y: np.ndarray) -> np.ndarray:
    return np.select(
        [y == 1, y <= 10, y <= 30, y <= 60, y < 100],
        [0, 1, 2, 3, 4],
        default=5,
    ).astype(np.int32)


def _choose_calibration_fast(
    predictions: np.ndarray,
    gold_ranks: list[dict[str, int | None]],
    cap: int = 100,
    max_mean_k: float = 60.0,
) -> dict[str, Any]:
    """Equivalent calibration search using precomputed per-query metric lookup tables."""
    n_rows = len(predictions)
    recall_lookup = np.zeros((n_rows, cap + 1), dtype=np.float64)
    ndcg_lookup = np.zeros((n_rows, cap + 1), dtype=np.float64)
    for row_index, rank_map in enumerate(gold_ranks):
        present = np.asarray([int(rank) for rank in rank_map.values() if rank is not None], dtype=np.int32)
        gold_count = max(len(rank_map), 1)
        for k in range(1, cap + 1):
            recall_lookup[row_index, k] = float(np.sum(present <= k)) / gold_count
            dcg = float(np.sum(1.0 / np.log2(present[present <= k] + 1))) if present.size else 0.0
            ideal_n = min(len(rank_map), k)
            idcg = float(np.sum(1.0 / np.log2(np.arange(1, ideal_n + 1) + 1)))
            ndcg_lookup[row_index, k] = dcg / idcg if idcg else 0.0
    row_indices = np.arange(n_rows)
    candidates: list[dict[str, Any]] = []
    for scale in np.round(np.arange(0.60, 1.5001, 0.05), 2):
        for margin in np.arange(-20, 60.01, 5):
            chosen = np.clip(np.rint(predictions * float(scale) + float(margin)), 1, cap).astype(np.int32)
            candidates.append({
                "scale": float(scale),
                "margin": float(margin),
                "mean_k": float(np.mean(chosen)),
                "macro_recall": float(np.mean(recall_lookup[row_indices, chosen])),
                "macro_ndcg_at_chosen_k": float(np.mean(ndcg_lookup[row_indices, chosen])),
            })
    under_budget = [row for row in candidates if row["mean_k"] < max_mean_k]
    if under_budget:
        selected = max(
            under_budget,
            key=lambda row: (
                row["macro_recall"], -row["mean_k"], row["macro_ndcg_at_chosen_k"], -row["scale"]
            ),
        )
        constraint_met = True
    else:
        selected = min(candidates, key=lambda row: (row["mean_k"], -row["macro_recall"]))
        constraint_met = False
    chosen = np.clip(
        np.rint(predictions * selected["scale"] + selected["margin"]), 1, cap
    ).astype(np.int32)
    result = {
        "scale": selected["scale"],
        "margin": selected["margin"],
        **_row_metrics(chosen, gold_ranks),
        "constraint_met": constraint_met,
        "calibration_objective": f"max macro recall subject to mean K < {max_mean_k:g}",
    }
    return result


def _balance_training(
    train_x: np.ndarray,
    train_y: np.ndarray,
    method: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if method == "none":
        return train_x, train_y, {"method": "none", "rows_before": len(train_y), "rows_after": len(train_y)}
    bins = _ordinal_bin(train_y)
    counts_before = np.bincount(bins, minlength=6)
    target_count = int(counts_before.max())
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    duplicated: list[bool] = []
    for bin_id in range(6):
        indices = np.flatnonzero(bins == bin_id)
        if not len(indices):
            continue
        selected.extend(indices.tolist())
        duplicated.extend([False] * len(indices))
        extra_count = target_count - len(indices)
        if extra_count > 0:
            extras = rng.choice(indices, size=extra_count, replace=True)
            selected.extend(extras.tolist())
            duplicated.extend([True] * extra_count)
    order = rng.permutation(len(selected))
    selected_array = np.asarray(selected, dtype=np.int32)[order]
    duplicated_array = np.asarray(duplicated, dtype=bool)[order]
    output_x = np.asarray(train_x[selected_array], dtype=np.float64).copy()
    output_y = train_y[selected_array].copy()
    if method == "ordinal_bootstrap_jitter":
        scale = np.std(train_x, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        noise = rng.normal(0.0, 0.02, size=output_x.shape) * scale
        output_x[duplicated_array] += noise[duplicated_array]
    elif method != "ordinal_bootstrap":
        raise ValueError(f"unknown balancing method: {method}")
    counts_after = np.bincount(_ordinal_bin(output_y), minlength=6)
    return output_x, output_y, {
        "method": method,
        "rows_before": len(train_y),
        "rows_after": len(output_y),
        "bin_counts_before": counts_before.tolist(),
        "bin_counts_after": counts_after.tolist(),
        "jitter_sigma_fraction_of_train_feature_std": 0.02 if method.endswith("jitter") else 0.0,
    }


def _fit_predict(
    model_name: str,
    method_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    feature_names: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    method = METHODS[method_name]
    transformed_train, transformed_eval, transform_diag = _fit_transform_x(
        method["transform"], train_x, train_y, eval_x, feature_names, seed
    )
    balanced_x, balanced_y, balance_diag = _balance_training(
        transformed_train, train_y, method["balance"], seed + 5000
    )
    y_scaler = StandardScaler()
    target_z = y_scaler.fit_transform(balanced_y.reshape(-1, 1)).ravel()
    estimator = _new_model(seed) if model_name == "xgboost" else _new_estimator(model_name, seed)
    dtype = np.float32 if model_name == "xgboost" else np.float64
    balanced_x = np.asarray(balanced_x, dtype=dtype)
    transformed_train = np.asarray(transformed_train, dtype=dtype)
    transformed_eval = np.asarray(transformed_eval, dtype=dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(balanced_x, target_z)
        convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    train_prediction_z = np.asarray(estimator.predict(transformed_train), dtype=np.float64)
    eval_prediction_z = np.asarray(estimator.predict(transformed_eval), dtype=np.float64)
    train_prediction = y_scaler.inverse_transform(train_prediction_z.reshape(-1, 1)).ravel()
    eval_prediction = y_scaler.inverse_transform(eval_prediction_z.reshape(-1, 1)).ravel()
    fit_diag: dict[str, Any] = {
        "transform": transform_diag,
        "balance": balance_diag,
        "target_scaler_mean": float(y_scaler.mean_[0]),
        "target_scaler_scale": float(y_scaler.scale_[0]),
        "convergence_warning_count": len(convergence),
        "convergence_warnings": convergence,
    }
    iteration_value = getattr(estimator, "n_iter_", None)
    if iteration_value is not None:
        fit_diag["n_iter"] = int(np.max(iteration_value))
    if hasattr(estimator, "best_validation_score_"):
        fit_diag["best_internal_validation_score"] = float(estimator.best_validation_score_)
    return train_prediction, eval_prediction, fit_diag


def _inner_oof(
    model_name: str,
    method_name: str,
    x: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    feature_names: list[str],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    diagnostics = []
    for fold, (train_indices, valid_indices) in enumerate(splitter.split(x, sources)):
        _, fold_prediction, diagnostic = _fit_predict(
            model_name, method_name, x[train_indices], y[train_indices], x[valid_indices], feature_names, seed + fold
        )
        predictions[valid_indices] = fold_prediction
        diagnostics.append({"fold": fold + 1, **diagnostic})
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"incomplete inner OOF: {method_name}/{model_name}")
    return predictions, diagnostics


def _evaluate_cell(
    model_name: str,
    method_name: str,
    x: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    gold_ranks: list[dict[str, int | None]],
    rows: list[dict[str, Any]],
    feature_names: list[str],
    ranking: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    chosen_k = np.zeros(len(y), dtype=np.int32)
    fold_ids = np.zeros(len(y), dtype=np.int32)
    fold_reports = []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    started = time.perf_counter()
    for fold, (train_indices, test_indices) in enumerate(outer.split(x, sources), start=1):
        inner_prediction, inner_diagnostics = _inner_oof(
            model_name, method_name, x[train_indices], y[train_indices], sources[train_indices], feature_names, 100 + fold
        )
        calibration = _choose_calibration_fast(
            inner_prediction, [gold_ranks[index] for index in train_indices], cap=100, max_mean_k=60.0
        )
        train_prediction, test_prediction, outer_diagnostic = _fit_predict(
            model_name, method_name, x[train_indices], y[train_indices], x[test_indices], feature_names, 1000 + fold
        )
        fold_k = np.clip(np.rint(test_prediction * calibration["scale"] + calibration["margin"]), 1, 100).astype(np.int32)
        predictions[test_indices] = test_prediction
        chosen_k[test_indices] = fold_k
        fold_ids[test_indices] = fold
        test_metrics = _row_metrics(fold_k, [gold_ranks[index] for index in test_indices])
        fold_reports.append({
            "fold": fold,
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "train_rmse": float(np.sqrt(np.mean((train_prediction - y[train_indices]) ** 2))),
            "train_mae": float(np.mean(np.abs(train_prediction - y[train_indices]))),
            "test_rmse": float(np.sqrt(np.mean((test_prediction - y[test_indices]) ** 2))),
            "test_mae": float(np.mean(np.abs(test_prediction - y[test_indices]))),
            "calibration_fit_on_inner_oof": calibration,
            "test_metrics": test_metrics,
            "outer_fit": outer_diagnostic,
            "inner_fit_diagnostics": inner_diagnostics,
        })
        print(
            f"[preprocessing] {method_name}/{model_name} fold {fold}/5 "
            f"MeanK={test_metrics['mean_k']:.2f} R={test_metrics['macro_recall']*100:.2f}% "
            f"P={test_metrics['macro_precision']*100:.2f}%",
            flush=True,
        )
    if not np.isfinite(predictions).all() or np.any(chosen_k < 1):
        raise RuntimeError(f"incomplete outer OOF: {method_name}/{model_name}")
    metrics = _row_metrics(chosen_k, gold_ranks)
    recall_rows = np.asarray([
        _row_metrics(chosen_k[[index]], [gold_ranks[index]])["macro_recall"] for index in range(len(y))
    ])
    precision_rows = np.asarray([
        _row_metrics(chosen_k[[index]], [gold_ranks[index]])["macro_precision"] for index in range(len(y))
    ])
    report = {
        "method": method_name,
        "method_spec": METHODS[method_name],
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "model_spec": XGB_PARAMETERS if model_name == "xgboost" else MODEL_SPECS[model_name],
        "status": "exploratory_nested_oof_preprocessing_screen",
        "elapsed_seconds": time.perf_counter() - started,
        "protocol": {
            "rows": len(y),
            "input_feature_count": len(feature_names),
            "input_feature_names": feature_names,
            "outer_folds": 5,
            "inner_calibration_folds": 4,
            "folds": "source-stratified; outer random_state=42",
            "leakage_control": "All learned transforms, selection, PCA, balancing, augmentation, and y scaling fit/use only the current training partition.",
            "calibration": "inner OOF only; maximize macro recall subject to mean K < 60",
            "ranking": ranking,
            "hyperparameter_tuning": "none; fixed model-family screening settings",
        },
        "metrics": metrics,
        "regression_diagnostics": {
            "oof_rmse": float(np.sqrt(np.mean((predictions - y) ** 2))),
            "oof_mae": float(np.mean(np.abs(predictions - y))),
        },
        "confidence_intervals_95": {
            "mean_k": _bootstrap_ci(chosen_k.astype(np.float64), 71),
            "macro_recall": _bootstrap_ci(recall_rows, 72),
            "macro_precision": _bootstrap_ci(precision_rows, 73),
        },
        "folds": fold_reports,
        "per_source": {},
    }
    for source in sorted(set(sources)):
        indices = np.flatnonzero(sources == source)
        report["per_source"][str(source)] = _row_metrics(chosen_k[indices], [gold_ranks[index] for index in indices])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "oof_predictions.jsonl").write_text(
        "".join(json.dumps({
            "qid": row["qid"], "source": row["source"], "target_k": int(y[index]),
            "prediction_raw_k": float(predictions[index]), "prediction_k_oof": int(chosen_k[index]),
            "outer_fold": int(fold_ids[index]),
        }, ensure_ascii=False) + "\n" for index, row in enumerate(rows)), encoding="utf-8"
    )
    return report


def _cell(metrics: dict[str, Any]) -> str:
    return f"{metrics['macro_recall']*100:.2f}% / {metrics['macro_precision']*100:.2f}% / K={metrics['mean_k']:.2f}"


def _format_report(comparison: dict[str, Any]) -> str:
    lines = [
        "# Adaptive-K preprocessing, feature, and data battery",
        "",
        "All cells use the same frozen ranking, 220 qids, source-stratified nested 5-fold OOF protocol, and fixed model hyperparameters. Cell format is `macro recall / macro precision / Mean K`.",
        "",
        "| Data method | " + " | ".join(MODEL_LABELS[model] for model in MODELS) + " |",
        "|---|" + "---:|" * len(MODELS),
    ]
    for method_name, method in METHODS.items():
        lines.append(
            "| " + method["description"] + " | " + " | ".join(
                _cell(comparison["cells"][f"{method_name}__{model}"]["metrics"]) for model in MODELS
            ) + " |"
        )
    lines.extend([
        "",
        "## Best cells under Mean K < 60",
        "",
        "| Rank | Method | Model | Mean K | Recall | Precision | F1 | nDCG |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for rank, row in enumerate(comparison["ranked_under_budget"], start=1):
        metrics = row["metrics"]
        lines.append(
            f"| {rank} | {row['method']} | {MODEL_LABELS[row['model']]} | {metrics['mean_k']:.2f} | "
            f"{metrics['macro_recall']*100:.2f}% | {metrics['macro_precision']*100:.2f}% | "
            f"{metrics['macro_f1']*100:.2f}% | {metrics['macro_ndcg_at_chosen_k']*100:.2f} |"
        )
    lines.extend([
        "",
        "Selection is exploratory: rows are ranked by recall, then lower Mean K, then precision. Because many methods were screened on the same OOF benchmark, the winner must be confirmed in a fresh validation protocol before being called an improvement.",
    ])
    return "\n".join(lines) + "\n"


def run_battery(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    run_dir, output_dir = run_dir.resolve(), output_dir.resolve()
    rows, manifest = _load_rows(run_dir)
    matrices, feature_names = _load_feature_matrices(run_dir, rows)
    y = np.asarray([row["target_k"] for row in rows], dtype=np.float64)
    sources = np.asarray([row["source"] for row in rows])
    gold_ranks = [row["gold_ranks"] for row in rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps({
        "status": "running", "methods": METHODS, "models": MODEL_LABELS, "queries": len(rows),
        "feature_sets": {name: names for name, names in feature_names.items()},
        "ordinal_balance_bins": ["K=1", "K=2-10", "K=11-30", "K=31-60", "K=61-99", "K=100"],
        "ranking": manifest["settings"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    cells: dict[str, Any] = {}
    started = time.perf_counter()
    for method_name, method in METHODS.items():
        x = matrices[method["feature_set"]]
        names = feature_names[method["feature_set"]]
        for model_name in MODELS:
            key = f"{method_name}__{model_name}"
            completed_report_path = output_dir / key / "report.json"
            if completed_report_path.exists():
                report = json.loads(completed_report_path.read_text(encoding="utf-8"))
                cells[key] = {
                    "method": method_name, "model": model_name,
                    "metrics": report["metrics"], "regression_diagnostics": report["regression_diagnostics"],
                    "confidence_intervals_95": report["confidence_intervals_95"],
                    "per_source": report["per_source"], "elapsed_seconds": report["elapsed_seconds"],
                }
                print(f"[preprocessing] resumed completed {key}: {_cell(report['metrics'])}", flush=True)
                continue
            print(f"[preprocessing] starting {key}", flush=True)
            report = _evaluate_cell(
                model_name, method_name, x, y, sources, gold_ranks, rows, names,
                manifest["settings"], output_dir / key,
            )
            cells[key] = {
                "method": method_name, "model": model_name,
                "metrics": report["metrics"], "regression_diagnostics": report["regression_diagnostics"],
                "confidence_intervals_95": report["confidence_intervals_95"],
                "per_source": report["per_source"], "elapsed_seconds": report["elapsed_seconds"],
            }
            print(f"[preprocessing] completed {key}: {_cell(report['metrics'])}", flush=True)
    ranked = sorted(
        [cell for cell in cells.values() if cell["metrics"]["mean_k"] < 60.0],
        key=lambda cell: (
            -cell["metrics"]["macro_recall"], cell["metrics"]["mean_k"],
            -cell["metrics"]["macro_precision"],
        ),
    )
    comparison = {
        "title": "Leakage-safe adaptive-K preprocessing/feature/data battery",
        "status": "completed_exploratory_nested_oof_screen",
        "queries": len(rows), "method_count": len(METHODS), "model_count": len(MODELS),
        "cell_count": len(cells), "elapsed_seconds": time.perf_counter() - started,
        "protocol": {
            "same_frozen_ranking_labels_and_folds": True,
            "outer_folds": 5, "inner_calibration_folds": 4,
            "all_learned_preprocessing_train_fold_only": True,
            "model_hyperparameter_tuning": False,
            "multiple_comparison_warning": "Methods screened on the same OOF benchmark require fresh confirmation.",
        },
        "methods": METHODS, "models": MODEL_LABELS, "cells": cells,
        "ranked_under_budget": ranked,
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(_format_report(comparison), encoding="utf-8")
    with (output_dir / "matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["method", "description", *[MODEL_LABELS[model] for model in MODELS]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method_name, method in METHODS.items():
            writer.writerow({
                "method": method_name, "description": method["description"],
                **{MODEL_LABELS[model]: _cell(cells[f"{method_name}__{model}"]["metrics"]) for model in MODELS},
            })
    (output_dir / "manifest.json").write_text(json.dumps({
        "status": "completed", "methods": METHODS, "models": MODEL_LABELS,
        "queries": len(rows), "completed_cells": list(cells), "elapsed_seconds": comparison["elapsed_seconds"],
        "feature_sets": {name: names for name, names in feature_names.items()},
        "ordinal_balance_bins": ["K=1", "K=2-10", "K=11-30", "K=31-60", "K=61-99", "K=100"],
        "ranking": manifest["settings"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_battery(args.run_dir, args.output_dir)
    print(
        f"[preprocessing] completed {result['cell_count']} cells in {result['elapsed_seconds']:.1f}s; "
        f"report={args.output_dir.resolve() / 'report.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
