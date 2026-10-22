"""Nested OOF screen of linear, ridge, and small neural regressors."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .dataset import _target_from_gold_ranks
from .features import FEATURE_NAMES
from .model import _bootstrap_ci, _choose_calibration, _row_metrics
from .scaling_ablation import DEFAULT_RUN_DIR, _read_jsonl


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "linear_regression": {
        "kind": "linear",
        "parameters": "LinearRegression(fit_intercept=True)",
    },
    "ridge_alpha_1": {
        "kind": "ridge",
        "alpha": 1.0,
        "parameters": "Ridge(alpha=1.0)",
    },
    "ridge_alpha_10": {
        "kind": "ridge",
        "alpha": 10.0,
        "parameters": "Ridge(alpha=10.0)",
    },
    "mlp_16_8": {
        "kind": "mlp",
        "parameters": "MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', alpha=1.0, learning_rate_init=0.003, early_stopping=True)",
    },
}


def _load_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    gate_path = Path(manifest["artifact_root"]) / "light_prep" / "gate_ppocrv5_ungated.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    rows = []
    for row in _read_jsonl(run_dir / "feature_labels.jsonl"):
        qid = row["qid"]
        gold_ranks = gate["gold_rank"].get(qid)
        if not gold_ranks:
            raise ValueError(f"frozen gold-rank labels are missing for {qid}")
        target_k, reachable = _target_from_gold_ranks(gold_ranks, cap=100)
        if int(row["target_k"]) != target_k or bool(row["reachable_at_cap"]) != reachable:
            raise ValueError(f"stored target differs from frozen label for {qid}")
        if set(row["features"]) != set(FEATURE_NAMES):
            raise ValueError(f"feature schema mismatch for {qid}")
        rows.append({**row, "gold_ranks": gold_ranks})
    if len(rows) != 220 or len({row["qid"] for row in rows}) != 220:
        raise ValueError(f"expected 220 unique rows, found {len(rows)}")
    return rows, manifest


def _new_estimator(name: str, seed: int) -> Any:
    spec = MODEL_SPECS[name]
    if spec["kind"] == "linear":
        return LinearRegression()
    if spec["kind"] == "ridge":
        return Ridge(alpha=spec["alpha"])
    if spec["kind"] == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=1.0,
            batch_size=32,
            learning_rate_init=0.003,
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            tol=1e-4,
            random_state=seed,
        )
    raise ValueError(f"unknown model family: {name}")


def _fit_predict(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit fold-local X/y scalers and one estimator, returning original-K predictions."""
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    train_z = x_scaler.fit_transform(train_x)
    eval_z = x_scaler.transform(eval_x)
    train_y_z = y_scaler.fit_transform(train_y.reshape(-1, 1)).ravel()
    estimator = _new_estimator(name, seed)
    captured: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(train_z, train_y_z)
        captured = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    train_predictions = y_scaler.inverse_transform(estimator.predict(train_z).reshape(-1, 1)).ravel()
    eval_predictions = y_scaler.inverse_transform(estimator.predict(eval_z).reshape(-1, 1)).ravel()
    diagnostic = {
        "convergence_warning_count": len(captured),
        "convergence_warnings": captured,
    }
    iteration_value = getattr(estimator, "n_iter_", None)
    if iteration_value is not None:
        diagnostic["n_iter"] = int(np.max(iteration_value))
    if hasattr(estimator, "best_validation_score_"):
        diagnostic["best_internal_validation_score"] = float(estimator.best_validation_score_)
    return train_predictions, eval_predictions, diagnostic


def _inner_oof(
    name: str,
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
            name,
            x[train_indices],
            y[train_indices],
            x[valid_indices],
            seed + fold,
        )
        predictions[valid_indices] = fold_predictions
        diagnostics.append({"fold": fold + 1, **diagnostic})
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{name}: incomplete inner OOF predictions")
    return predictions, diagnostics


def _evaluate_model(
    name: str,
    rows: list[dict[str, Any]],
    ranking: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    x = np.asarray([[row["features"][feature] for feature in FEATURE_NAMES] for row in rows], dtype=np.float64)
    y = np.asarray([row["target_k"] for row in rows], dtype=np.float64)
    sources = np.asarray([row["source"] for row in rows])
    gold_ranks = [row["gold_ranks"] for row in rows]
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    chosen_k = np.zeros(len(rows), dtype=np.int32)
    fold_ids = np.zeros(len(rows), dtype=np.int32)
    fold_reports: list[dict[str, Any]] = []
    oof_diagnostics: list[dict[str, Any]] = []

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_indices, test_indices) in enumerate(outer.split(x, sources), start=1):
        inner_predictions, inner_diagnostics = _inner_oof(
            name, x[train_indices], y[train_indices], sources[train_indices], 100 + fold
        )
        calibration = _choose_calibration(
            inner_predictions,
            [gold_ranks[index] for index in train_indices],
            cap=100,
            max_mean_k=60.0,
        )
        train_predictions, test_predictions, outer_diagnostic = _fit_predict(
            name,
            x[train_indices],
            y[train_indices],
            x[test_indices],
            1000 + fold,
        )
        fold_k = np.clip(
            np.rint(test_predictions * calibration["scale"] + calibration["margin"]), 1, 100
        ).astype(np.int32)
        predictions[test_indices] = test_predictions
        chosen_k[test_indices] = fold_k
        fold_ids[test_indices] = fold
        test_metrics = _row_metrics(fold_k, [gold_ranks[index] for index in test_indices])
        fold_reports.append({
            "fold": fold,
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "train_rmse": float(np.sqrt(np.mean((train_predictions - y[train_indices]) ** 2))),
            "train_mae": float(np.mean(np.abs(train_predictions - y[train_indices]))),
            "test_rmse": float(np.sqrt(np.mean((test_predictions - y[test_indices]) ** 2))),
            "test_mae": float(np.mean(np.abs(test_predictions - y[test_indices]))),
            "calibration": calibration,
            "test_metrics": test_metrics,
            "outer_fit": outer_diagnostic,
        })
        oof_diagnostics.append({"outer_fold": fold, "inner_fits": inner_diagnostics})
        print(
            f"[model-family] {name} fold {fold}/5: Mean K={test_metrics['mean_k']:.2f}, "
            f"recall={test_metrics['macro_recall'] * 100:.2f}%"
        )

    if not np.isfinite(predictions).all() or np.any(chosen_k < 1):
        raise RuntimeError(f"{name}: incomplete outer OOF predictions")
    metrics = _row_metrics(chosen_k, gold_ranks)
    recall_rows = np.asarray([
        _row_metrics(np.asarray([chosen_k[index]]), [gold_ranks[index]])["macro_recall"]
        for index in range(len(rows))
    ])
    report = {
        "model": name,
        "model_spec": MODEL_SPECS[name],
        "status": "exploratory_nested_oof_screen",
        "protocol": {
            "rows": len(rows),
            "features": list(FEATURE_NAMES),
            "outer_folds": 5,
            "inner_calibration_folds": 4,
            "folds": "StratifiedKFold by source, shuffle=true, outer random_state=42",
            "input_scaling": "StandardScaler fitted only on each training partition",
            "target_scaling": "StandardScaler fitted only on each training partition; predictions inverse-transformed to K",
            "target": "required K in 1..100; unreachable-at-cap rows labelled 100",
            "calibration_objective": "max macro page recall subject to mean K < 60",
            "ranking": ranking,
            "hyperparameter_tuning": "none; fixed screening configurations",
        },
        "metrics": metrics,
        "regression_diagnostics": {
            "oof_rmse": float(np.sqrt(np.mean((predictions - y) ** 2))),
            "oof_mae": float(np.mean(np.abs(predictions - y))),
        },
        "confidence_intervals_95": {
            "mean_k": _bootstrap_ci(chosen_k.astype(np.float64), 42),
            "macro_recall": _bootstrap_ci(recall_rows, 43),
        },
        "folds": fold_reports,
        "fit_diagnostics": oof_diagnostics,
        "per_source": {},
    }
    for source in sorted(set(sources)):
        indices = np.flatnonzero(sources == source)
        report["per_source"][str(source)] = _row_metrics(
            chosen_k[indices], [gold_ranks[index] for index in indices]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "oof_predictions.jsonl").write_text(
        "".join(json.dumps({
            "qid": row["qid"],
            "source": row["source"],
            "target_k": int(y[index]),
            "prediction_raw": float(predictions[index]),
            "prediction_k_oof": int(chosen_k[index]),
            "residual": float(predictions[index] - y[index]),
            "outer_fold": int(fold_ids[index]),
        }, ensure_ascii=False) + "\n" for index, row in enumerate(rows)),
        encoding="utf-8",
    )
    return report


def _comparison_report(comparison: dict[str, Any]) -> str:
    lines = [
        "# Linear and neural regression screen for adaptive K",
        "",
        "All model families use the same 220 rows, 21 features, source-stratified nested 5-fold protocol, fold-local input/target scaling, and inner-OOF calibration objective. Model hyperparameters are fixed screening settings, not tuned champions.",
        "",
        "| Model | Mean K | Macro recall | Precision | F1 | nDCG@chosen-K | OOF RMSE | OOF MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in comparison["models"].items():
        metrics = row["metrics"]
        diag = row["regression_diagnostics"]
        lines.append(
            f"| {name} | {metrics['mean_k']:.2f} | {metrics['macro_recall']*100:.2f}% | "
            f"{metrics['macro_precision']*100:.2f}% | {metrics['macro_f1']*100:.2f}% | "
            f"{metrics['macro_ndcg_at_chosen_k']*100:.2f} | {diag['oof_rmse']:.2f} | {diag['oof_mae']:.2f} |"
        )
    xgb = comparison["xgboost_reference"]
    lines.extend([
        f"| XGBoost reference (unscaled) | {xgb['metrics']['mean_k']:.2f} | {xgb['metrics']['macro_recall']*100:.2f}% | "
        f"{xgb['metrics']['macro_precision']*100:.2f}% | {xgb['metrics']['macro_f1']*100:.2f}% | "
        f"{xgb['metrics']['macro_ndcg_at_chosen_k']*100:.2f} | {xgb['regression_diagnostics']['oof_rmse']:.2f} | "
        f"{xgb['regression_diagnostics']['oof_mae']:.2f} |",
        "",
        "The OOF metrics are the decision metrics. Lower K-target RMSE/MAE alone does not imply better recall at the chosen retrieval budget. K=100 remains a capped label for unreachable queries. The OHR / Physics / Industrial source-specific results are in `comparison.json` and each model's `report.json`.",
    ])
    return "\n".join(lines) + "\n"


def run_ablation(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    run_dir, output_dir = run_dir.resolve(), output_dir.resolve()
    comparison_path = output_dir / "comparison.json"
    if comparison_path.exists():
        raise FileExistsError(f"{comparison_path} already exists; choose another output directory")
    rows, manifest = _load_rows(run_dir)
    reports = {}
    for name in MODEL_SPECS:
        reports[name] = _evaluate_model(name, rows, manifest.get("settings", {}), output_dir / name)

    xgb_report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    comparison = {
        "title": "Linear, Ridge, and neural regression comparison for adaptive K",
        "status": "exploratory_nested_oof_screen",
        "rows": len(rows),
        "feature_count": len(FEATURE_NAMES),
        "models": {
            name: {
                "metrics": report["metrics"],
                "regression_diagnostics": report["regression_diagnostics"],
                "confidence_intervals_95": report["confidence_intervals_95"],
                "per_source": report["per_source"],
                "spec": report["model_spec"],
            }
            for name, report in reports.items()
        },
        "xgboost_reference": {
            "metrics": xgb_report["metrics"],
            "regression_diagnostics": xgb_report["regression_diagnostics"],
            "per_source": xgb_report["per_source"],
            "note": "Existing router-matched XGBRegressor nested-OOF result on identical rows/folds and calibration objective.",
        },
        "protocol": reports[next(iter(reports))]["protocol"],
        "target_distribution": xgb_report["label_distribution"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(_comparison_report(comparison), encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN_DIR / "model_family_ablation",
    )
    args = parser.parse_args()
    comparison = run_ablation(args.run_dir, args.output_dir)
    print(f"[model-family] completed: {args.output_dir.resolve()}")
    for name, row in comparison["models"].items():
        print(
            f"[model-family] {name}: Mean K={row['metrics']['mean_k']:.2f}, "
            f"recall={row['metrics']['macro_recall']*100:.2f}%"
        )


if __name__ == "__main__":
    main()
