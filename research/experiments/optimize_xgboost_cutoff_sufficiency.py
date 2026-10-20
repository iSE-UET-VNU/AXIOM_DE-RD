"""Train one score-only XGBoost model to judge cutoff sufficiency for K=1..200.

Each query is expanded into (query, cutoff) rows.  The label is one when the
cutoff is at least the cap-200 full-evidence target K; unreachable queries have
target K=200.  A probability threshold is calibrated using inner OOF queries,
then the smallest sufficient cutoff is selected on each outer-test query.
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
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_DIR = ROOT / "experiments" / "adaptive-k-retrieval"
if str(ADAPTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTIVE_DIR))

from run_canonical_adaptive_methods import load_rows  # noqa: E402
from run_risk_label_statistics import features as score_features  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "work" / "xgboost_cutoff_sufficiency_20261016"
CAP = 200
SEED = 42
TARGET_RECALL = 0.87
MEAN_K_BUDGET = 70.0
CUTOFFS = np.arange(1, CAP + 1, dtype=int)

CONFIGS = (
    {"name": "raw_depth2", "features": "raw", "max_depth": 2},
    {"name": "raw_depth3", "features": "raw", "max_depth": 3},
    {"name": "raw_shape_depth2", "features": "raw+shape", "max_depth": 2},
    {"name": "raw_shape_depth3", "features": "raw+shape", "max_depth": 3},
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
    return (values - values[-1]) / (values[0] - values[-1] + 1e-12)


def query_feature_matrices(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    raw = np.vstack([score_features(row["scores"])[1] for row in rows])
    shapes = [normalized_scores(row["scores"]) for row in rows]
    shape_global = np.vstack([score_features(values)[1] for values in shapes])
    return {"raw": raw, "raw+shape": np.hstack([raw, shape_global])}, shapes


def dynamic_features(scores: np.ndarray, shape: np.ndarray) -> np.ndarray:
    raw_prefix = np.cumsum(scores) / CUTOFFS
    shape_prefix = np.cumsum(shape) / CUTOFFS
    raw_tail_sum = np.cumsum(scores[::-1])[::-1]
    shape_tail_sum = np.cumsum(shape[::-1])[::-1]
    remaining = CAP - CUTOFFS
    raw_tail = np.where(
        remaining > 0,
        np.r_[raw_tail_sum[1:], 0.0] / np.maximum(remaining, 1),
        0.0,
    )
    shape_tail = np.where(
        remaining > 0,
        np.r_[shape_tail_sum[1:], 0.0] / np.maximum(remaining, 1),
        0.0,
    )
    raw_gap = np.r_[scores[:-1] - scores[1:], 0.0]
    shape_gap = np.r_[shape[:-1] - shape[1:], 0.0]
    return np.column_stack(
        [
            CUTOFFS / CAP,
            np.log1p(CUTOFFS) / math.log1p(CAP),
            scores,
            shape,
            raw_gap,
            shape_gap,
            raw_prefix,
            shape_prefix,
            raw_tail,
            shape_tail,
        ]
    )


def expanded_matrix(
    query_matrix: np.ndarray,
    rows: Sequence[dict[str, Any]],
    shapes: Sequence[np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    blocks = []
    for index in indices:
        scores = np.asarray(rows[int(index)]["scores"], dtype=float)
        base = np.repeat(query_matrix[int(index)][None, :], CAP, axis=0)
        blocks.append(np.hstack([base, dynamic_features(scores, shapes[int(index)])]))
    return np.vstack(blocks)


def expanded_labels(target: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.concatenate([(CUTOFFS >= int(target[index])).astype(int) for index in indices])


def make_model(config: dict[str, Any], seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=int(config["max_depth"]),
        learning_rate=0.035,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
    )


def fit_predict(
    config: dict[str, Any],
    query_matrix: np.ndarray,
    rows: Sequence[dict[str, Any]],
    shapes: Sequence[np.ndarray],
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> np.ndarray:
    x_train = expanded_matrix(query_matrix, rows, shapes, train)
    y_train = expanded_labels(target, train)
    x_test = expanded_matrix(query_matrix, rows, shapes, test)
    model = make_model(config, seed)
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1].reshape(len(test), CAP)
    return np.maximum.accumulate(probability, axis=1)


def choices(probability: np.ndarray, threshold: float) -> np.ndarray:
    reached = probability >= threshold
    first = np.argmax(reached, axis=1) + 1
    return np.where(np.any(reached, axis=1), first, CAP).astype(int)


def evaluate(
    rows: Sequence[dict[str, Any]], indices: np.ndarray, selected_k: np.ndarray
) -> dict[str, Any]:
    recalls = np.asarray(
        [recall_at(rows[int(index)], int(k)) for index, k in zip(indices, selected_k)],
        dtype=float,
    )
    return {
        "mean_k": float(np.mean(selected_k)),
        "macro_recall": float(np.mean(recalls)),
        "recalls": recalls,
    }


def select_threshold(
    rows: Sequence[dict[str, Any]],
    indices: np.ndarray,
    probability: np.ndarray,
    mode: str,
) -> tuple[float, dict[str, Any]]:
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 37):
        item = evaluate(rows, indices, choices(probability, float(threshold)))
        candidates.append((float(threshold), item))
    if mode == "recall_87":
        feasible = [item for item in candidates if item[1]["macro_recall"] >= TARGET_RECALL]
        if feasible:
            return min(feasible, key=lambda item: (item[1]["mean_k"], -item[1]["macro_recall"]))
        return max(candidates, key=lambda item: (item[1]["macro_recall"], -item[1]["mean_k"]))
    if mode == "mean_k_70":
        feasible = [item for item in candidates if item[1]["mean_k"] <= MEAN_K_BUDGET]
        pool = feasible or candidates
        return max(pool, key=lambda item: (item[1]["macro_recall"], -item[1]["mean_k"]))
    raise ValueError(mode)


def inner_oof(
    config: dict[str, Any],
    query_matrix: np.ndarray,
    rows: Sequence[dict[str, Any]],
    shapes: Sequence[np.ndarray],
    target: np.ndarray,
    sources: np.ndarray,
    outer_train: np.ndarray,
    outer_fold: int,
) -> np.ndarray:
    output = np.zeros((len(outer_train), CAP), dtype=float)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 100 + outer_fold)
    for inner_fold, (train_pos, valid_pos) in enumerate(
        splitter.split(query_matrix[outer_train], sources[outer_train])
    ):
        train = outer_train[train_pos]
        valid = outer_train[valid_pos]
        output[valid_pos] = fit_predict(
            config,
            query_matrix,
            rows,
            shapes,
            target,
            train,
            valid,
            SEED + 1000 * outer_fold + inner_fold,
        )
    return output


def run_config(
    config: dict[str, Any],
    query_matrix: np.ndarray,
    rows: Sequence[dict[str, Any]],
    shapes: Sequence[np.ndarray],
    target: np.ndarray,
    sources: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    probabilities = np.zeros((len(rows), CAP), dtype=float)
    selected = {
        "recall_87": np.zeros(len(rows), dtype=int),
        "mean_k_70": np.zeros(len(rows), dtype=int),
    }
    calibration: dict[str, list[dict[str, Any]]] = {key: [] for key in selected}
    for outer_fold, (train, test) in enumerate(folds):
        inner_probability = inner_oof(
            config,
            query_matrix,
            rows,
            shapes,
            target,
            sources,
            train,
            outer_fold,
        )
        test_probability = fit_predict(
            config,
            query_matrix,
            rows,
            shapes,
            target,
            train,
            test,
            SEED + outer_fold,
        )
        probabilities[test] = test_probability
        for mode in selected:
            threshold, inner_metric = select_threshold(
                rows, train, inner_probability, mode
            )
            selected[mode][test] = choices(test_probability, threshold)
            calibration[mode].append(
                {
                    "fold": outer_fold,
                    "threshold": threshold,
                    "inner_mean_k": inner_metric["mean_k"],
                    "inner_recall": inner_metric["macro_recall"],
                }
            )

    indices = np.arange(len(rows), dtype=int)
    result = {
        "name": config["name"],
        "features": config["features"],
        "max_depth": config["max_depth"],
        "policies": {},
        "calibration": calibration,
    }
    for mode, selected_k in selected.items():
        item = evaluate(rows, indices, selected_k)
        result["policies"][mode] = {
            "mean_k": item["mean_k"],
            "macro_recall": item["macro_recall"],
            "choices": selected_k.tolist(),
            "per_query_recall": item["recalls"].tolist(),
        }
    return result


def fixed_recall(rows: Sequence[dict[str, Any]], k: int) -> float:
    return float(np.mean([recall_at(row, k) for row in rows]))


def oracle(rows: Sequence[dict[str, Any]], target_recall: float) -> dict[str, Any]:
    selected = np.asarray([required_k(row, target_recall) for row in rows], dtype=int)
    item = evaluate(rows, np.arange(len(rows)), selected)
    return {"mean_k": item["mean_k"], "macro_recall": item["macro_recall"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = load_rows("hybrid", CAP)
    qids = [str(row["qid"]) for row in rows]
    sources = np.asarray([str(row["source"]) for row in rows])
    matrices, shapes = query_feature_matrices(rows)
    target = np.asarray([required_k(row, 1.0) for row in rows], dtype=int)
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            matrices["raw"], sources
        )
    )

    results = []
    for config in CONFIGS:
        print(f"running {config['name']}", flush=True)
        results.append(
            run_config(
                config,
                matrices[config["features"]],
                rows,
                shapes,
                target,
                sources,
                folds,
            )
        )

    def best_key(item: dict[str, Any]) -> tuple[int, int, float, float]:
        metric = item["policies"]["mean_k_70"]
        within_budget = int(metric["mean_k"] < MEAN_K_BUDGET)
        achieved = int(
            metric["mean_k"] < MEAN_K_BUDGET and metric["macro_recall"] > TARGET_RECALL
        )
        return within_budget, achieved, metric["macro_recall"], -metric["mean_k"]

    best = max(results, key=best_key)
    best_choice = np.asarray(best["policies"]["mean_k_70"]["choices"], dtype=int)
    payload = {
        "protocol": {
            "benchmark": "Physics 80 + Industrial 75 + OHR-Bench 65",
            "queries": len(rows),
            "cap": CAP,
            "output_k": "continuous integer 1..200",
            "target": "final gold rank if fully reachable, otherwise 200",
            "features": "retrieval scores only",
            "model": "one XGBoost cutoff-sufficiency classifier",
            "outer_folds": 5,
            "inner_folds": 4,
            "monotonicity": "cumulative maximum applied to P(sufficient) over K",
            "warning": "Four configurations compared on the same outer OOF benchmark; exploratory selection.",
        },
        "oracles": {str(t): oracle(rows, t) for t in (0.80, 0.90, 1.00)},
        "fixed": {str(k): {"mean_k": k, "macro_recall": fixed_recall(rows, k)} for k in (50, 70, 100, 200)},
        "results": results,
        "best_under_mean_k_70": {
            "name": best["name"],
            "mean_k": best["policies"]["mean_k_70"]["mean_k"],
            "macro_recall": best["policies"]["mean_k_70"]["macro_recall"],
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
                        "predicted_k": int(best_choice[index]),
                        "recall": recall_at(rows[index], int(best_choice[index])),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    lines = [
        "# XGBoost continuous cutoff-sufficiency screen",
        "",
        "One XGBoost model predicts whether each candidate cutoff K is sufficient. Thresholds are calibrated in inner folds.",
        "",
        "| Configuration | Features | Depth | Policy | Mean K | Recall |",
        "|---|---|---:|---|---:|---:|",
    ]
    for result in results:
        for mode in ("mean_k_70", "recall_87"):
            item = result["policies"][mode]
            lines.append(
                f"| `{result['name']}` | {result['features']} | {result['max_depth']} | {mode} | {item['mean_k']:.2f} | {item['macro_recall']:.2%} |"
            )
    lines.extend(
        [
            "",
            f"Best screened under the K=70 policy: `{best['name']}` at mean K={best['policies']['mean_k_70']['mean_k']:.2f}, recall={best['policies']['mean_k_70']['macro_recall']:.2%}.",
            "",
            "This is exploratory selection over four configurations. No query or neighbour features are used.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["best_under_mean_k_70"], indent=2))
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
