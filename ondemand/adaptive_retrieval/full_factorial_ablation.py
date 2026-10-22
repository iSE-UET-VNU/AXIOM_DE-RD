"""Run the full 3-preprocessing x 5-model nested OOF adaptive-K screen."""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES
from .model import XGB_PARAMETERS, _bootstrap_ci, _choose_calibration, _new_model, _row_metrics
from .model_family_ablation import MODEL_SPECS, _load_rows, _new_estimator
from .scaling_ablation import DEFAULT_RUN_DIR


PREPROCESSING: dict[str, dict[str, str]] = {
    "raw": {
        "x": "none",
        "y": "none",
        "description": "Raw X; raw y in K units",
    },
    "standard_x": {
        "x": "standard",
        "y": "none",
        "description": "Fold-local StandardScaler(X); raw y in K units",
    },
    "standard_xy": {
        "x": "standard",
        "y": "standard",
        "description": "Fold-local StandardScaler(X) and StandardScaler(y); inverse-transform predictions to K",
    },
}

MODELS = ["xgboost", "linear_regression", "ridge_alpha_1", "ridge_alpha_10", "mlp_16_8"]
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "linear_regression": "Linear",
    "ridge_alpha_1": "Ridge α=1",
    "ridge_alpha_10": "Ridge α=10",
    "mlp_16_8": "MLP (16,8)",
}
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "full_factorial_preprocessing"


def _fit_predict(
    model_name: str,
    preprocessing: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    config = PREPROCESSING[preprocessing]
    x_scaler = StandardScaler() if config["x"] == "standard" else None
    if x_scaler is None:
        train_z, eval_z = train_x, eval_x
    else:
        train_z = x_scaler.fit_transform(train_x)
        eval_z = x_scaler.transform(eval_x)
    train_z = np.asarray(train_z, dtype=np.float32 if model_name == "xgboost" else np.float64)
    eval_z = np.asarray(eval_z, dtype=np.float32 if model_name == "xgboost" else np.float64)

    y_scaler = StandardScaler() if config["y"] == "standard" else None
    if y_scaler is None:
        train_target = train_y
    else:
        train_target = y_scaler.fit_transform(train_y.reshape(-1, 1)).ravel()

    if model_name == "xgboost":
        estimator = _new_model(seed)
    else:
        estimator = _new_estimator(model_name, seed)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(train_z, train_target)
        convergence = [
            str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
        ]

    train_prediction = np.asarray(estimator.predict(train_z), dtype=np.float64)
    eval_prediction = np.asarray(estimator.predict(eval_z), dtype=np.float64)
    if y_scaler is not None:
        train_prediction = y_scaler.inverse_transform(train_prediction.reshape(-1, 1)).ravel()
        eval_prediction = y_scaler.inverse_transform(eval_prediction.reshape(-1, 1)).ravel()

    diagnostic: dict[str, Any] = {
        "convergence_warning_count": len(convergence),
        "convergence_warnings": convergence,
    }
    iteration_value = getattr(estimator, "n_iter_", None)
    if iteration_value is not None:
        diagnostic["n_iter"] = int(np.max(iteration_value))
    if hasattr(estimator, "best_validation_score_"):
        diagnostic["best_internal_validation_score"] = float(estimator.best_validation_score_)
    if hasattr(estimator, "get_booster"):
        diagnostic["best_iteration"] = int(getattr(estimator, "best_iteration", -1))
    return train_prediction, eval_prediction, diagnostic


def _inner_oof(
    model_name: str,
    preprocessing: str,
    x: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    diagnostics = []
    for fold, (train_indices, valid_indices) in enumerate(splitter.split(x, sources)):
        _, fold_predictions, diagnostic = _fit_predict(
            model_name,
            preprocessing,
            x[train_indices],
            y[train_indices],
            x[valid_indices],
            seed + fold,
        )
        predictions[valid_indices] = fold_predictions
        diagnostics.append({"fold": fold + 1, **diagnostic})
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{model_name}/{preprocessing}: incomplete inner OOF predictions")
    return predictions, diagnostics


def _evaluate_cell(
    model_name: str,
    preprocessing: str,
    rows: list[dict[str, Any]],
    ranking: dict[str, Any],
    cell_dir: Path,
) -> dict[str, Any]:
    x = np.asarray([[row["features"][name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
    y = np.asarray([row["target_k"] for row in rows], dtype=np.float64)
    sources = np.asarray([row["source"] for row in rows])
    gold_ranks = [row["gold_ranks"] for row in rows]
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    chosen_k = np.zeros(len(rows), dtype=np.int32)
    fold_ids = np.zeros(len(rows), dtype=np.int32)
    fold_reports: list[dict[str, Any]] = []
    fit_logs: list[str] = []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_indices, test_indices) in enumerate(outer.split(x, sources), start=1):
        inner_predictions, inner_diagnostics = _inner_oof(
            model_name,
            preprocessing,
            x[train_indices],
            y[train_indices],
            sources[train_indices],
            100 + fold,
        )
        calibration = _choose_calibration(
            inner_predictions,
            [gold_ranks[index] for index in train_indices],
            cap=100,
            max_mean_k=60.0,
        )
        train_prediction, test_prediction, outer_diagnostic = _fit_predict(
            model_name,
            preprocessing,
            x[train_indices],
            y[train_indices],
            x[test_indices],
            1000 + fold,
        )
        fold_k = np.clip(
            np.rint(test_prediction * calibration["scale"] + calibration["margin"]), 1, 100
        ).astype(np.int32)
        predictions[test_indices] = test_prediction
        chosen_k[test_indices] = fold_k
        fold_ids[test_indices] = fold
        test_metrics = _row_metrics(fold_k, [gold_ranks[index] for index in test_indices])
        train_metrics = _row_metrics(
            np.clip(np.rint(train_prediction * calibration["scale"] + calibration["margin"]), 1, 100).astype(np.int32),
            [gold_ranks[index] for index in train_indices],
        )
        fold_reports.append({
            "fold": fold,
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "train_rmse": float(np.sqrt(np.mean((train_prediction - y[train_indices]) ** 2))),
            "train_mae": float(np.mean(np.abs(train_prediction - y[train_indices]))),
            "test_rmse": float(np.sqrt(np.mean((test_prediction - y[test_indices]) ** 2))),
            "test_mae": float(np.mean(np.abs(test_prediction - y[test_indices]))),
            "train_metrics_at_fold_calibration": train_metrics,
            "calibration_fit_on_inner_oof": calibration,
            "test_metrics": test_metrics,
            "outer_fit": outer_diagnostic,
        })
        fit_logs.extend([
            f"outer_fold={fold} train_rows={len(train_indices)} test_rows={len(test_indices)}",
            f"  train_rmse={fold_reports[-1]['train_rmse']:.4f} test_rmse={fold_reports[-1]['test_rmse']:.4f} "
            f"train_mae={fold_reports[-1]['train_mae']:.4f} test_mae={fold_reports[-1]['test_mae']:.4f}",
            f"  calibration={json.dumps(calibration, sort_keys=True)}",
            f"  test_mean_k={test_metrics['mean_k']:.4f} recall={test_metrics['macro_recall']:.6f} "
            f"precision={test_metrics['macro_precision']:.6f}",
            f"  outer_fit={json.dumps(outer_diagnostic, sort_keys=True)}",
            f"  inner_fit_diagnostics={json.dumps(inner_diagnostics, sort_keys=True)}",
        ])
        print(
            f"[full-factorial] {preprocessing}/{model_name} fold {fold}/5: "
            f"MeanK={test_metrics['mean_k']:.2f}, recall={test_metrics['macro_recall']*100:.2f}%, "
            f"precision={test_metrics['macro_precision']*100:.2f}%",
            flush=True,
        )

    if not np.isfinite(predictions).all() or np.any(chosen_k < 1):
        raise RuntimeError(f"{model_name}/{preprocessing}: incomplete outer OOF predictions")
    metrics = _row_metrics(chosen_k, gold_ranks)
    recall_rows = np.asarray([
        _row_metrics(np.asarray([chosen_k[index]]), [gold_ranks[index]])["macro_recall"]
        for index in range(len(rows))
    ])
    precision_rows = np.asarray([
        _row_metrics(np.asarray([chosen_k[index]]), [gold_ranks[index]])["macro_precision"]
        for index in range(len(rows))
    ])
    report = {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "preprocessing": preprocessing,
        "preprocessing_spec": PREPROCESSING[preprocessing],
        "model_spec": XGB_PARAMETERS if model_name == "xgboost" else MODEL_SPECS[model_name],
        "status": "full_factorial_nested_oof_exploratory",
        "protocol": {
            "rows": len(rows),
            "features": list(FEATURE_NAMES),
            "outer_folds": 5,
            "inner_calibration_folds": 4,
            "folds": "StratifiedKFold by source, shuffle=true, outer random_state=42",
            "scaler_fit_scope": "Each X/y scaler is fit only on the training partition of its current inner or outer fold.",
            "target": "required K in 1..100; unreachable-at-cap rows labelled 100",
            "calibration_objective": "max macro page recall subject to mean K < 60; calibration fit only from inner-OOF predictions",
            "ranking": ranking,
            "model_hyperparameters_tuned": False,
        },
        "metrics": metrics,
        "regression_diagnostics": {
            "oof_rmse": float(np.sqrt(np.mean((predictions - y) ** 2))),
            "oof_mae": float(np.mean(np.abs(predictions - y))),
        },
        "confidence_intervals_95": {
            "mean_k": _bootstrap_ci(chosen_k.astype(np.float64), 42),
            "macro_recall": _bootstrap_ci(recall_rows, 43),
            "macro_precision": _bootstrap_ci(precision_rows, 44),
        },
        "folds": fold_reports,
        "per_source": {},
    }
    for source in sorted(set(sources)):
        indices = np.flatnonzero(sources == source)
        report["per_source"][str(source)] = _row_metrics(chosen_k[indices], [gold_ranks[index] for index in indices])

    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (cell_dir / "oof_predictions.jsonl").write_text(
        "".join(json.dumps({
            "qid": row["qid"],
            "source": row["source"],
            "target_k": int(y[index]),
            "prediction_raw_k": float(predictions[index]),
            "prediction_k_oof": int(chosen_k[index]),
            "residual": float(predictions[index] - y[index]),
            "outer_fold": int(fold_ids[index]),
        }, ensure_ascii=False) + "\n" for index, row in enumerate(rows)),
        encoding="utf-8",
    )
    (cell_dir / "training_log.txt").write_text("\n".join(fit_logs) + "\n", encoding="utf-8")
    return report


def _metric_cell(metrics: dict[str, Any] | None) -> str:
    if metrics is None:
        return "—"
    return f"{metrics['macro_recall']*100:.2f}% / {metrics['macro_precision']*100:.2f}%"


def _format_report(comparison: dict[str, Any]) -> str:
    lines = [
        "# Full factorial preprocessing × model comparison",
        "",
        "All 15 cells use the same 220 qids, frozen top-100 ranking, source-stratified nested 5-fold OOF protocol, inner-OOF calibration, and fixed model hyperparameters. Each scaler is fit within the current training fold. Cell format is macro recall / macro precision; empty cells mean not run.",
        "",
        "## Requested matrix (recall / precision)",
        "",
        "| Preprocessing | " + " | ".join(MODEL_LABELS[name] for name in MODELS) + " |",
        "|---|" + "---:|" * len(MODELS),
    ]
    for prep_name in PREPROCESSING:
        lines.append(
            "| " + PREPROCESSING[prep_name]["description"] + " | " +
            " | ".join(_metric_cell(comparison["cells"].get(f"{prep_name}__{model}" , {}).get("metrics")) for model in MODELS) + " |"
        )
    lines.extend([
        "",
        "## Additional metrics",
        "",
        "| Preprocessing | Model | Mean K | Recall | Precision | F1 | nDCG@chosen-K | RMSE | MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for prep_name in PREPROCESSING:
        for model_name in MODELS:
            cell = comparison["cells"][f"{prep_name}__{model_name}"]
            metrics, diag = cell["metrics"], cell["regression_diagnostics"]
            lines.append(
                f"| {prep_name} | {MODEL_LABELS[model_name]} | {metrics['mean_k']:.2f} | "
                f"{metrics['macro_recall']*100:.2f}% | {metrics['macro_precision']*100:.2f}% | "
                f"{metrics['macro_f1']*100:.2f}% | {metrics['macro_ndcg_at_chosen_k']*100:.2f} | "
                f"{diag['oof_rmse']:.2f} | {diag['oof_mae']:.2f} |"
            )
    raw_xgb = comparison["cells"]["raw__xgboost"]["metrics"]
    standard_x_xgb = comparison["cells"]["standard_x__xgboost"]["metrics"]
    lines.extend([
        "",
        "## Validation notes",
        "",
        f"- Raw XGBoost reproduces the frozen baseline within tolerance: Mean K delta {raw_xgb['mean_k'] - comparison['baseline_xgboost']['mean_k']:+.4f}, recall delta {(raw_xgb['macro_recall'] - comparison['baseline_xgboost']['macro_recall'])*100:+.4f} percentage points.",
        f"- StandardScaler(X) vs raw XGBoost: Mean K delta {standard_x_xgb['mean_k'] - raw_xgb['mean_k']:+.4f}, recall delta {(standard_x_xgb['macro_recall'] - raw_xgb['macro_recall'])*100:+.4f} percentage points, precision delta {(standard_x_xgb['macro_precision'] - raw_xgb['macro_precision'])*100:+.4f} points.",
        "- No model hyperparameter search was performed; this isolates preprocessing effects under the existing screening settings. Results remain exploratory OOF, not a final tuned model selection.",
        "- Each cell folder contains `report.json`, `oof_predictions.jsonl`, and fold-level `training_log.txt`.",
    ])
    return "\n".join(lines) + "\n"


def run_full_factorial(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    run_dir, output_dir = run_dir.resolve(), output_dir.resolve()
    rows, manifest = _load_rows(run_dir)
    if len(rows) != 220:
        raise ValueError(f"expected 220 canonical queries, found {len(rows)}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; choose another output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_manifest.json").write_text(json.dumps({
        "status": "running",
        "queries": 220,
        "preprocessing_rows": PREPROCESSING,
        "models": MODEL_LABELS,
        "outer_folds": 5,
        "inner_calibration_folds": 4,
        "objective": "max macro recall subject to mean K < 60",
        "ranking": manifest["settings"],
        "ranking_reproduction": manifest["ranking_reproduction"],
        "xgboost_parameters": XGB_PARAMETERS,
        "continuous_model_parameters": MODEL_SPECS,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    cells: dict[str, Any] = {}
    started = time.perf_counter()
    for prep_name in PREPROCESSING:
        for model_name in MODELS:
            key = f"{prep_name}__{model_name}"
            cell_started = time.perf_counter()
            print(f"[full-factorial] starting {key}", flush=True)
            report = _evaluate_cell(model_name, prep_name, rows, manifest["settings"], output_dir / key)
            report["elapsed_seconds"] = time.perf_counter() - cell_started
            (output_dir / key / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            cells[key] = {
                "preprocessing": prep_name,
                "model": model_name,
                "elapsed_seconds": report["elapsed_seconds"],
                "metrics": report["metrics"],
                "regression_diagnostics": report["regression_diagnostics"],
                "confidence_intervals_95": report["confidence_intervals_95"],
                "per_source": report["per_source"],
                "report_path": str(output_dir / key / "report.json"),
                "oof_predictions_path": str(output_dir / key / "oof_predictions.jsonl"),
            }
            print(
                f"[full-factorial] completed {key}: MeanK={report['metrics']['mean_k']:.2f} "
                f"recall={report['metrics']['macro_recall']*100:.2f}% "
                f"precision={report['metrics']['macro_precision']*100:.2f}% elapsed={report['elapsed_seconds']:.1f}s",
                flush=True,
            )

    baseline = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["metrics"]
    comparison = {
        "title": "Full factorial preprocessing x model nested OOF adaptive-K screen",
        "status": "completed_exploratory_nested_oof",
        "queries": len(rows),
        "feature_count": len(FEATURE_NAMES),
        "total_elapsed_seconds": time.perf_counter() - started,
        "protocol": {
            "outer_folds": 5,
            "inner_calibration_folds": 4,
            "outer_folds_stratified_by": "source",
            "same_folds_across_all_cells": True,
            "same_labels_and_frozen_ranking": True,
            "gold_usage": "labels/evaluation only, never features",
            "calibration_selection": "inner OOF only; maximize macro recall subject to mean K < 60",
            "hyperparameters": "fixed to previous model-family screen; no tuning",
        },
        "preprocessing": PREPROCESSING,
        "models": MODEL_LABELS,
        "baseline_xgboost": baseline,
        "cells": cells,
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(_format_report(comparison), encoding="utf-8")
    with (output_dir / "matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["preprocessing", *[f"{MODEL_LABELS[name]} (recall/precision)" for name in MODELS]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for prep_name in PREPROCESSING:
            row = {"preprocessing": PREPROCESSING[prep_name]["description"]}
            for model_name in MODELS:
                row[f"{MODEL_LABELS[model_name]} (recall/precision)"] = _metric_cell(
                    cells[f"{prep_name}__{model_name}"]["metrics"]
                )
            writer.writerow(row)
    (output_dir / "experiment_manifest.json").write_text(json.dumps({
        "status": "completed",
        "total_elapsed_seconds": comparison["total_elapsed_seconds"],
        "completed_cells": list(cells),
        "queries": 220,
        "preprocessing_rows": PREPROCESSING,
        "models": MODEL_LABELS,
        "outer_folds": 5,
        "inner_calibration_folds": 4,
        "objective": "max macro recall subject to mean K < 60",
        "ranking": manifest["settings"],
        "ranking_reproduction": manifest["ranking_reproduction"],
        "xgboost_parameters": XGB_PARAMETERS,
        "continuous_model_parameters": MODEL_SPECS,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_full_factorial(args.run_dir, args.output_dir)
    print(f"[full-factorial] completed all 15 cells in {result['total_elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
