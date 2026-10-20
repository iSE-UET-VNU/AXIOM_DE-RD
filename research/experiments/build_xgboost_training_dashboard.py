"""Train one auditable XGBoost regression split and build a static dashboard.

The run is diagnostic rather than a benchmark claim.  Validation controls
early stopping; test labels are used only after fitting for held-out metrics
and per-iteration diagnostic loss.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

import numpy as np
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_DIR = ROOT / "experiments" / "adaptive-k-retrieval"
ASSET_DIR = Path(__file__).resolve().parent / "xgboost_training_dashboard"
DEFAULT_OUTPUT = ROOT / "data" / "work" / "xgboost_training_dashboard_20261016"
if str(ADAPTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTIVE_DIR))

from run_canonical_adaptive_methods import load_rows  # noqa: E402
from run_risk_label_statistics import features as score_features  # noqa: E402


CAP = 200
SEED = 42
CLASS_NAMES = ["easy_1_10", "medium_11_70", "hard_71_199", "unreachable_200"]


def recall_at(row: dict[str, Any], cutoff: int) -> float:
    gold = set(row["gold"])
    return len(set(row["ranked"][:cutoff]) & gold) / max(len(gold), 1)


def required_k(row: dict[str, Any]) -> int:
    gold = set(row["gold"])
    ranks = sorted(rank for rank, page in enumerate(row["ranked"][:CAP], 1) if page in gold)
    return int(ranks[-1]) if len(ranks) == len(gold) and ranks else CAP


def class_index(k: int) -> int:
    if k <= 10:
        return 0
    if k <= 70:
        return 1
    if k < CAP:
        return 2
    return 3


def split_indices(sources: np.ndarray) -> dict[str, np.ndarray]:
    all_indices = np.arange(len(sources))
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_val_pos, test_pos = next(outer.split(all_indices, sources))
    train_val = all_indices[train_val_pos]
    test = all_indices[test_pos]
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    train_pos, val_pos = next(inner.split(train_val, sources[train_val]))
    return {"train": train_val[train_pos], "validation": train_val[val_pos], "test": test}


def regression_metrics(
    rows: Sequence[dict[str, Any]], indices: np.ndarray, y: np.ndarray, pred: np.ndarray
) -> dict[str, Any]:
    rounded = np.clip(np.ceil(pred), 1, CAP).astype(int)
    recalls = np.asarray([recall_at(rows[int(i)], int(k)) for i, k in zip(indices, rounded)])
    rank_y = np.argsort(np.argsort(y, kind="stable"), kind="stable")
    rank_p = np.argsort(np.argsort(pred, kind="stable"), kind="stable")
    return {
        "rows": int(len(indices)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "spearman_rank_proxy": float(np.corrcoef(rank_y, rank_p)[0, 1]),
        "mean_predicted_k": float(np.mean(rounded)),
        "macro_recall_at_predicted_k": float(np.mean(recalls)),
    }


def confusion(y: np.ndarray, pred: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((4, 4), dtype=int)
    rounded = np.clip(np.ceil(pred), 1, CAP).astype(int)
    for truth, estimate in zip(y.astype(int), rounded):
        matrix[class_index(int(truth)), class_index(int(estimate))] += 1
    return matrix.tolist()


def distribution(y: np.ndarray) -> list[int]:
    result = [0, 0, 0, 0]
    for value in y.astype(int):
        result[class_index(int(value))] += 1
    return result


def feature_dictionary() -> list[dict[str, str]]:
    output = []
    for k in (1, 5, 10, 20, 50, 100):
        output.append({"name": f"score_mean_top_{k}", "group": "level", "formula": f"mean(score ranks 1..{k})"})
    output.extend(
        [
            {"name": "score_mean_tail", "group": "level", "formula": "mean(score ranks 101..200)"},
            {"name": "score_std", "group": "dispersion", "formula": "population std of ranks 1..200"},
            {"name": "score_range", "group": "dispersion", "formula": "score rank 1 - score rank 200"},
        ]
    )
    for k in (1, 5, 10, 20, 50):
        output.append({"name": f"gap_at_{k}", "group": "local gap", "formula": f"score rank {k} - score rank {k + 1}"})
    output.extend(
        [
            {"name": "max_gap", "group": "local gap", "formula": "maximum adjacent gap"},
            {"name": "max_gap_rank", "group": "local gap", "formula": "argmax adjacent gap / 200"},
        ]
    )
    for k in (10, 20, 50, 100):
        output.append({"name": f"slope_top_{k}", "group": "slope", "formula": f"(score rank {k} - score rank 1) / {k - 1}"})
    output.append({"name": "softmax_entropy", "group": "shape", "formula": "normalized entropy, temperature=0.1"})
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    dist = args.output_dir / "site" / "dist"
    artifacts = args.output_dir / "training_run"
    dist.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    rows, _ = load_rows("hybrid", CAP)
    feature_names, _ = score_features(rows[0]["scores"])
    x = np.vstack([score_features(row["scores"])[1] for row in rows])
    y = np.asarray([required_k(row) for row in rows], dtype=float)
    sources = np.asarray([str(row["source"]) for row in rows])
    splits = split_indices(sources)

    params: dict[str, Any] = {
        "n_estimators": 1000,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "early_stopping_rounds": 50,
        "random_state": SEED,
        "n_jobs": 4,
        "tree_method": "hist",
        "verbosity": 0,
    }
    model = XGBRegressor(**params)
    started = time.perf_counter()
    model.fit(
        x[splits["train"]],
        y[splits["train"]],
        eval_set=[
            (x[splits["train"]], y[splits["train"]]),
            (x[splits["validation"]], y[splits["validation"]]),
        ],
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    best_iteration = int(model.best_iteration)
    iteration_end = best_iteration + 1
    predictions = model.predict(x, iteration_range=(0, iteration_end))
    predictions = np.clip(np.asarray(predictions, dtype=float), 1.0, float(CAP))

    evals = model.evals_result()
    rounds = len(evals["validation_0"]["rmse"])
    test_rmse: list[float] = []
    test_mae: list[float] = []
    for iteration in range(1, rounds + 1):
        estimate = model.predict(x[splits["test"]], iteration_range=(0, iteration))
        test_rmse.append(float(np.sqrt(mean_squared_error(y[splits["test"]], estimate))))
        test_mae.append(float(mean_absolute_error(y[splits["test"]], estimate)))

    split_name = np.full(len(rows), "", dtype=object)
    for name, indices in splits.items():
        split_name[indices] = name

    table_rows = []
    for i, row in enumerate(rows):
        pred_k = int(np.clip(math.ceil(predictions[i]), 1, CAP))
        gold = set(row["gold"])
        gold_ranks = [rank for rank, page in enumerate(row["ranked"], 1) if page in gold]
        item: dict[str, Any] = {
            "row_index": i,
            "qid": row["qid"],
            "source": row["source"],
            "split": str(split_name[i]),
            "query": row["query"],
            "gold_pages": len(gold),
            "gold_ranks": gold_ranks,
            "target_k": int(y[i]),
            "target_class": CLASS_NAMES[class_index(int(y[i]))],
            "prediction_raw": float(predictions[i]),
            "prediction_k": pred_k,
            "prediction_class": CLASS_NAMES[class_index(pred_k)],
            "residual": float(predictions[i] - y[i]),
            "absolute_error": float(abs(predictions[i] - y[i])),
            "recall_at_prediction": recall_at(row, pred_k),
            "candidate_recall_at_200": recall_at(row, CAP),
            "scores": [float(value) for value in row["scores"]],
            "ranked_pages": list(row["ranked"]),
        }
        item.update({name: float(x[i, j]) for j, name in enumerate(feature_names)})
        table_rows.append(item)

    feature_stats = []
    importance = model.get_booster().get_score(importance_type="gain")
    corr = np.corrcoef(x, rowvar=False)
    for j, name in enumerate(feature_names):
        values = x[:, j]
        feature_stats.append(
            {
                **feature_dictionary()[j],
                "min": float(np.min(values)),
                "p10": float(np.percentile(values, 10)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "p90": float(np.percentile(values, 90)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
                "target_correlation": float(np.corrcoef(values, y)[0, 1]),
                "gain_importance": float(importance.get(f"f{j}", 0.0)),
            }
        )

    metrics = {}
    matrices = {}
    distributions = {"all": distribution(y)}
    for name, indices in splits.items():
        metrics[name] = regression_metrics(rows, indices, y[indices], predictions[indices])
        matrices[name] = confusion(y[indices], predictions[indices])
        distributions[name] = distribution(y[indices])

    source_distribution = {}
    for source in sorted(set(sources.tolist())):
        indices = np.flatnonzero(sources == source)
        source_distribution[source] = {
            "rows": int(len(indices)),
            "classes": distribution(y[indices]),
            "mean_target": float(np.mean(y[indices])),
            "mean_gold_pages": float(np.mean([len(rows[int(i)]["gold"]) for i in indices])),
        }

    issues = [
        {
            "severity": "critical",
            "title": "K=200 is a censoring sentinel",
            "detail": f"{int(np.sum(y == CAP))} rows are labeled 200 because at least one gold page is absent from top-200; none is a clean reachable rank-200 case.",
        },
        {
            "severity": "critical",
            "title": "The score source is not the official merged routed run",
            "detail": "The loader rebuilds global 0.2 BM25 + 0.8 Qwen dense fusion over 5,091 pages after top-100 min-max clipping.",
        },
        {
            "severity": "warning",
            "title": "Exact duplicate features",
            "detail": "score_mean_top_1 equals score_range because rank-200 score is zero for every query.",
        },
        {
            "severity": "warning",
            "title": "Strong domain and qrel-cardinality shift",
            "detail": "OHR averages one gold page, while Physics/Industrial average several; random source-stratified splits do not test unseen-domain transfer.",
        },
        {
            "severity": "info",
            "title": "This is one diagnostic split",
            "detail": "Train/validation/test results are transparent but higher variance than repeated or nested cross-validation.",
        },
    ]

    payload = {
        "meta": {
            "title": "XGBoost Continuous-K Training Audit",
            "date": "2026-10-16",
            "queries": len(rows),
            "features": len(feature_names),
            "cap": CAP,
            "class_names": CLASS_NAMES,
            "split_protocol": "60% train / 20% validation / 20% test, fixed seed 42, source-stratified",
            "validation_role": "early stopping and best iteration selection",
            "test_role": "held-out diagnostics only; never used to select the model",
            "score_provenance": "global top-200 of 0.2 * top100-minmax BM25 + 0.8 * top100-minmax Qwen3-Embedding-8B dense",
            "label": "smallest rank containing every gold page; forced to 200 when full gold is unreachable",
            "elapsed_seconds": elapsed,
            "best_iteration": best_iteration,
            "rounds_logged": rounds,
            "versions": {"xgboost": xgboost.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
        },
        "parameters": params,
        "metrics": metrics,
        "loss_curves": {
            "round": list(range(1, rounds + 1)),
            "train_rmse": [float(v) for v in evals["validation_0"]["rmse"]],
            "validation_rmse": [float(v) for v in evals["validation_1"]["rmse"]],
            "test_rmse": test_rmse,
            "train_mae": [float(v) for v in evals["validation_0"]["mae"]],
            "validation_mae": [float(v) for v in evals["validation_1"]["mae"]],
            "test_mae": test_mae,
        },
        "confusion_matrices": matrices,
        "class_distribution": distributions,
        "source_distribution": source_distribution,
        "feature_names": feature_names,
        "feature_stats": feature_stats,
        "feature_correlation": corr.tolist(),
        "issues": issues,
        "rows": table_rows,
    }

    (artifacts / "training_run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (dist / "data.js").write_text(
        "window.DASHBOARD_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    for filename in ("index.html", "styles.css", "app.js"):
        shutil.copyfile(ASSET_DIR / filename, dist / filename)

    summary = {
        "output": str(args.output_dir),
        "best_iteration": best_iteration,
        "rounds_logged": rounds,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
    }
    (artifacts / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
