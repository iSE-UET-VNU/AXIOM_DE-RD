"""Optimize a score-only XGBoost continuous-K predictor under a cap of 200.

The target is the smallest rank that retrieves every gold page when possible;
queries whose full gold set is not reachable in the top 200 receive target K=200.
Only retrieval-score features are used.  Calibration is selected inside each
outer training fold, never from the outer-test qrels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_DIR = ROOT / "experiments" / "adaptive-k-retrieval"
if str(ADAPTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTIVE_DIR))

from run_canonical_adaptive_methods import load_rows  # noqa: E402
from run_risk_label_statistics import features as score_features  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "work" / "xgboost_continuous_k_20261016"
CAP = 200
SEED = 42
TARGET_RECALL = 0.87
MEAN_K_BUDGET = 70.0


CONFIGS = (
    {"name": "raw_squared", "features": "raw", "objective": "squared"},
    {"name": "raw_log_squared", "features": "raw", "objective": "log_squared"},
    {"name": "raw_quantile_070", "features": "raw", "objective": "quantile_070"},
    {"name": "raw_shape_squared", "features": "raw+shape", "objective": "squared"},
    {"name": "raw_shape_log_squared", "features": "raw+shape", "objective": "log_squared"},
    {"name": "raw_shape_quantile_070", "features": "raw+shape", "objective": "quantile_070"},
)


def recall_at(row: dict[str, Any], cutoff: int) -> float:
    gold = set(row["gold"])
    return len(set(row["ranked"][:cutoff]) & gold) / max(len(gold), 1)


def required_k(row: dict[str, Any], target: float = 1.0) -> int:
    gold = set(row["gold"])
    ranks = sorted(
        rank for rank, page_id in enumerate(row["ranked"][:CAP], 1) if page_id in gold
    )
    need = int(math.ceil(target * len(gold) - 1e-12))
    if need > 0 and len(ranks) >= need:
        return int(ranks[need - 1])
    return CAP


def normalized_scores(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    high = float(values[0])
    low = float(values[-1])
    return (values - low) / (high - low + 1e-12)


def build_matrices(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    raw_names, _ = score_features(rows[0]["scores"])
    raw = np.vstack([score_features(row["scores"])[1] for row in rows])
    shape = np.vstack(
        [score_features(normalized_scores(row["scores"]))[1] for row in rows]
    )
    return (
        {"raw": raw, "raw+shape": np.hstack([raw, shape])},
        {
            "raw": raw_names,
            "raw+shape": raw_names + [f"shape_{name}" for name in raw_names],
        },
    )


def make_model(config: dict[str, str], seed: int) -> XGBRegressor:
    common: dict[str, Any] = {
        "n_estimators": 320,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "random_state": seed,
        "n_jobs": 4,
        "tree_method": "hist",
        "verbosity": 0,
    }
    if config["objective"] == "quantile_070":
        common.update(objective="reg:quantileerror", quantile_alpha=0.70)
    else:
        common.update(objective="reg:squarederror")
    return XGBRegressor(**common)


def fit_predict(
    config: dict[str, str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = make_model(config, seed)
    target = np.log1p(y_train) if config["objective"] == "log_squared" else y_train
    model.fit(x_train, target)
    prediction = np.asarray(model.predict(x_test), dtype=float)
    if config["objective"] == "log_squared":
        prediction = np.expm1(prediction)
    return np.clip(prediction, 1.0, float(CAP))


def choices_from_prediction(prediction: np.ndarray, scale: float, margin: int) -> np.ndarray:
    return np.clip(np.ceil(scale * prediction + margin), 1, CAP).astype(int)


def metrics(rows: Sequence[dict[str, Any]], indices: np.ndarray, choices: np.ndarray) -> dict[str, Any]:
    recalls = np.asarray(
        [recall_at(rows[int(index)], int(k)) for index, k in zip(indices, choices)], dtype=float
    )
    return {
        "mean_k": float(np.mean(choices)),
        "macro_recall": float(np.mean(recalls)),
        "recalls": recalls,
    }


def calibration_grid() -> list[tuple[float, int]]:
    return [
        (scale, margin)
        for scale in (0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50)
        for margin in (-10, 0, 5, 10, 20, 30, 40, 60)
    ]


def select_calibration(
    rows: Sequence[dict[str, Any]],
    indices: np.ndarray,
    prediction: np.ndarray,
    mode: str,
) -> tuple[float, int, dict[str, Any]]:
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for scale, margin in calibration_grid():
        choice = choices_from_prediction(prediction, scale, margin)
        item = metrics(rows, indices, choice)
        candidates.append((scale, margin, item))

    if mode == "recall_87":
        feasible = [item for item in candidates if item[2]["macro_recall"] >= TARGET_RECALL]
        pool = feasible or candidates
        return min(pool, key=lambda item: (item[2]["mean_k"], -item[2]["macro_recall"]))
    if mode == "mean_k_70":
        feasible = [item for item in candidates if item[2]["mean_k"] <= MEAN_K_BUDGET]
        pool = feasible or candidates
        return max(pool, key=lambda item: (item[2]["macro_recall"], -item[2]["mean_k"]))
    raise ValueError(mode)


def inner_oof_prediction(
    config: dict[str, str],
    matrix: np.ndarray,
    target: np.ndarray,
    sources: np.ndarray,
    outer_train: np.ndarray,
    outer_fold: int,
) -> np.ndarray:
    local_sources = sources[outer_train]
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 100 + outer_fold)
    prediction = np.zeros(len(outer_train), dtype=float)
    for inner_fold, (inner_train_pos, inner_valid_pos) in enumerate(
        splitter.split(matrix[outer_train], local_sources)
    ):
        inner_train = outer_train[inner_train_pos]
        inner_valid = outer_train[inner_valid_pos]
        prediction[inner_valid_pos] = fit_predict(
            config,
            matrix[inner_train],
            target[inner_train],
            matrix[inner_valid],
            SEED + 1000 * outer_fold + inner_fold,
        )
    return prediction


def summarize_config(
    config: dict[str, str],
    matrix: np.ndarray,
    rows: Sequence[dict[str, Any]],
    target: np.ndarray,
    sources: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    raw_prediction = np.zeros(len(rows), dtype=float)
    choices_by_mode = {
        "recall_87": np.zeros(len(rows), dtype=int),
        "mean_k_70": np.zeros(len(rows), dtype=int),
    }
    calibrations: dict[str, list[dict[str, Any]]] = {mode: [] for mode in choices_by_mode}

    for outer_fold, (train, test) in enumerate(folds):
        inner_prediction = inner_oof_prediction(
            config, matrix, target, sources, train, outer_fold
        )
        test_prediction = fit_predict(
            config,
            matrix[train],
            target[train],
            matrix[test],
            SEED + outer_fold,
        )
        raw_prediction[test] = test_prediction
        for mode in choices_by_mode:
            scale, margin, inner_metric = select_calibration(
                rows, train, inner_prediction, mode
            )
            choices_by_mode[mode][test] = choices_from_prediction(
                test_prediction, scale, margin
            )
            calibrations[mode].append(
                {
                    "fold": outer_fold,
                    "scale": scale,
                    "margin": margin,
                    "inner_mean_k": inner_metric["mean_k"],
                    "inner_recall": inner_metric["macro_recall"],
                }
            )

    indices = np.arange(len(rows), dtype=int)
    raw_choices = choices_from_prediction(raw_prediction, 1.0, 0)
    output: dict[str, Any] = {
        "name": config["name"],
        "features": config["features"],
        "objective": config["objective"],
        "target_mae": float(np.mean(np.abs(raw_prediction - target))),
        "target_spearman": float(
            np.corrcoef(
                np.argsort(np.argsort(raw_prediction)),
                np.argsort(np.argsort(target)),
            )[0, 1]
        ),
        "raw": {
            key: value
            for key, value in metrics(rows, indices, raw_choices).items()
            if key != "recalls"
        },
        "policies": {},
        "calibrations": calibrations,
        "raw_prediction": raw_prediction.tolist(),
    }
    for mode, choice in choices_by_mode.items():
        item = metrics(rows, indices, choice)
        output["policies"][mode] = {
            "mean_k": item["mean_k"],
            "macro_recall": item["macro_recall"],
            "choices": choice.tolist(),
            "per_query_recall": item["recalls"].tolist(),
        }
    return output


def fixed_metrics(rows: Sequence[dict[str, Any]], k: int) -> dict[str, float]:
    values = [recall_at(row, k) for row in rows]
    return {"mean_k": float(k), "macro_recall": float(np.mean(values))}


def oracle_metrics(rows: Sequence[dict[str, Any]], target_recall: float) -> dict[str, Any]:
    choices = np.asarray([required_k(row, target_recall) for row in rows], dtype=int)
    item = metrics(rows, np.arange(len(rows)), choices)
    return {
        "target_per_query": target_recall,
        "mean_k": item["mean_k"],
        "macro_recall": item["macro_recall"],
        "unreachable_queries": int(
            sum(recall_at(row, CAP) + 1e-12 < target_recall for row in rows)
        ),
    }


def by_source(
    rows: Sequence[dict[str, Any]], sources: np.ndarray, choices: np.ndarray
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for source in sorted(set(sources.tolist())):
        indices = np.flatnonzero(sources == source)
        item = metrics(rows, indices, choices[indices])
        output[source] = {
            "queries": len(indices),
            "mean_k": item["mean_k"],
            "macro_recall": item["macro_recall"],
        }
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = load_rows("hybrid", CAP)
    qids = [str(row["qid"]) for row in rows]
    sources = np.asarray([str(row["source"]) for row in rows])
    matrices, feature_names = build_matrices(rows)
    target = np.asarray([required_k(row, 1.0) for row in rows], dtype=float)
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            matrices["raw"], sources
        )
    )

    results = []
    for config in CONFIGS:
        print(f"running {config['name']}", flush=True)
        results.append(
            summarize_config(
                config,
                matrices[config["features"]],
                rows,
                target,
                sources,
                folds,
            )
        )

    def result_key(item: dict[str, Any]) -> tuple[int, float, float]:
        metric = item["policies"]["mean_k_70"]
        achieves = int(
            metric["mean_k"] < MEAN_K_BUDGET and metric["macro_recall"] > TARGET_RECALL
        )
        return achieves, metric["macro_recall"], -metric["mean_k"]

    best = max(results, key=result_key)
    best_choices = np.asarray(best["policies"]["mean_k_70"]["choices"], dtype=int)
    payload = {
        "protocol": {
            "benchmark": "Physics 80 + Industrial 75 + OHR-Bench 65",
            "queries": len(rows),
            "ranking": "hybrid cached top-200",
            "cap": CAP,
            "output_k": "every integer from 1 through 200",
            "target": "rank of final gold page when all gold is reachable; otherwise 200",
            "features": "retrieval score only",
            "outer_folds": 5,
            "inner_folds": 4,
            "calibration": "scale and additive margin selected using inner OOF qrels",
            "selection_warning": "Six configurations are compared on the same outer OOF benchmark; exploratory method selection.",
        },
        "feature_names": feature_names,
        "fixed": {str(k): fixed_metrics(rows, k) for k in (10, 20, 50, 70, 100, 150, 200)},
        "oracles": {
            str(target_recall): oracle_metrics(rows, target_recall)
            for target_recall in (0.80, 0.90, 1.00)
        },
        "target_distribution": {
            "mean": float(np.mean(target)),
            "median": float(np.median(target)),
            "p90": float(np.percentile(target, 90)),
            "at_cap": int(np.sum(target == CAP)),
        },
        "results": results,
        "best_under_mean_k_70": {
            "name": best["name"],
            "features": best["features"],
            "objective": best["objective"],
            "mean_k": best["policies"]["mean_k_70"]["mean_k"],
            "macro_recall": best["policies"]["mean_k_70"]["macro_recall"],
            "by_source": by_source(rows, sources, best_choices),
        },
    }

    (args.output_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for index, qid in enumerate(qids):
            handle.write(
                json.dumps(
                    {
                        "qid": qid,
                        "source": sources[index],
                        "target_k": int(target[index]),
                        "best_predicted_k": int(best_choices[index]),
                        "best_recall": recall_at(rows[index], int(best_choices[index])),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    lines = [
        "# XGBoost continuous-K score-only screen",
        "",
        "Cap is 200 and every integer K from 1 through 200 is allowed. Calibration is selected inside each outer training fold.",
        "",
        "## Oracle and fixed baselines",
        "",
        "| Method | Mean K | Recall |",
        "|---|---:|---:|",
    ]
    for target_name in ("0.8", "0.9", "1.0"):
        item = payload["oracles"][target_name]
        lines.append(
            f"| Oracle target {float(target_name):.0%} | {item['mean_k']:.2f} | {item['macro_recall']:.2%} |"
        )
    lines.extend(
        [
            f"| Fixed K=70 | 70.00 | {payload['fixed']['70']['macro_recall']:.2%} |",
            f"| Fixed K=100 | 100.00 | {payload['fixed']['100']['macro_recall']:.2%} |",
            f"| Fixed K=200 | 200.00 | {payload['fixed']['200']['macro_recall']:.2%} |",
            "",
            "## Nested-calibrated XGBoost results",
            "",
            "| Configuration | Features | Objective | Policy | Mean K | Recall | Target MAE |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for result in results:
        for mode in ("mean_k_70", "recall_87"):
            item = result["policies"][mode]
            lines.append(
                f"| `{result['name']}` | {result['features']} | {result['objective']} | {mode} | "
                f"{item['mean_k']:.2f} | {item['macro_recall']:.2%} | {result['target_mae']:.2f} |"
            )
    best_metric = payload["best_under_mean_k_70"]
    lines.extend(
        [
            "",
            "## Best screened configuration under the K=70 policy",
            "",
            f"`{best_metric['name']}`: mean K={best_metric['mean_k']:.2f}, recall={best_metric['macro_recall']:.2%}.",
            "",
            "This is exploratory model selection, not a final held-out result. No query or neighbour features are used.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload["best_under_mean_k_70"], ensure_ascii=False, indent=2))
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
