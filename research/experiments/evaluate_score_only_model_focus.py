"""Focused score-only adaptive-K comparison.

This experiment intentionally excludes query-text and neighbour features.
It answers two narrow questions:

1. How does XGBoost compare with HistGradientBoosting when folds, score
   features, labels, training rows, and the 30/100 budget policy are fixed?
2. What happens when ExtraTrees directly predicts the action label
   K in {10, 40, 100} that preserves recall reachable at K=100?

The experiment is exploratory OOF evaluation on the merged 220-query set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_DIR = ROOT / "experiments" / "adaptive-k-retrieval"
if str(ADAPTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTIVE_DIR))

from run_canonical_adaptive_methods import load_rows  # noqa: E402
from run_risk_label_statistics import features as score_features  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "work" / "score_only_model_focus_20261016"
MAX_K = 200
EASY_K = 30
HARD_K = 100
HARD_LABEL_K = 50
HISTORICAL_THRESHOLD = 0.2
HISTORICAL_HARD_COUNT = 134
ACTION_LEVELS = np.asarray([10, 40, 100], dtype=int)
SEED = 42


def recall_at(row: dict[str, Any], cutoff: int) -> float:
    gold = set(row["gold"])
    return len(set(row["ranked"][:cutoff]) & gold) / max(len(gold), 1)


def k95(row: dict[str, Any]) -> int:
    gold = set(row["gold"])
    for rank, page_id in enumerate(row["ranked"][:MAX_K], 1):
        if page_id not in gold:
            continue
        found = len(set(row["ranked"][:rank]) & gold)
        if found / max(len(gold), 1) >= 0.95:
            return rank
    return MAX_K + 1


def action_label(row: dict[str, Any]) -> int:
    cap_recall = recall_at(row, HARD_K)
    for cutoff in ACTION_LEVELS:
        if recall_at(row, int(cutoff)) >= cap_recall - 1e-12:
            return int(cutoff)
    return HARD_K


def policy_metrics(rows: Sequence[dict[str, Any]], choices: np.ndarray) -> dict[str, Any]:
    recalls = np.asarray(
        [recall_at(row, int(cutoff)) for row, cutoff in zip(rows, choices)], dtype=float
    )
    return {
        "mean_k": float(np.mean(choices)),
        "macro_recall": float(np.mean(recalls)),
        "choice_counts": {
            str(int(level)): int(np.sum(choices == level)) for level in np.unique(choices)
        },
        "per_query_recall": recalls.tolist(),
    }


def binary_summary(
    rows: Sequence[dict[str, Any]], probabilities: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    threshold_prediction = probabilities >= HISTORICAL_THRESHOLD
    threshold_choices = np.where(threshold_prediction, HARD_K, EASY_K)

    descending = np.argsort(-probabilities, kind="stable")
    matched_prediction = np.zeros(len(rows), dtype=bool)
    matched_prediction[descending[:HISTORICAL_HARD_COUNT]] = True
    matched_choices = np.where(matched_prediction, HARD_K, EASY_K)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "threshold_0.2": policy_metrics(rows, threshold_choices),
        "matched_134_hard_slots": policy_metrics(rows, matched_choices),
    }


def make_xgboost(labels: np.ndarray) -> Any:
    from xgboost import XGBClassifier

    positive = max(int(np.sum(labels == 1)), 1)
    negative = max(int(np.sum(labels == 0)), 1)
    return XGBClassifier(
        n_estimators=220,
        max_depth=3,
        learning_rate=0.035,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=negative / positive,
        random_state=SEED,
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = load_rows("hybrid", MAX_K)
    qids = [str(row["qid"]) for row in rows]
    sources = np.asarray([str(row["source"]) for row in rows])
    feature_names, _ = score_features(rows[0]["scores"])
    matrix = np.vstack([score_features(row["scores"])[1] for row in rows])
    k95_values = np.asarray([k95(row) for row in rows], dtype=int)
    hard_labels = (k95_values > HARD_LABEL_K).astype(int)
    action_labels = np.asarray([action_label(row) for row in rows], dtype=int)

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(matrix, sources))
    fold_ids = np.zeros(len(rows), dtype=int)
    hgb_probability = np.zeros(len(rows), dtype=float)
    xgb_probability = np.zeros(len(rows), dtype=float)
    extra_probability = np.zeros((len(rows), len(ACTION_LEVELS)), dtype=float)

    for fold, (train, test) in enumerate(folds):
        fold_ids[test] = fold

        hgb = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=7,
            random_state=SEED,
        )
        hgb.fit(matrix[train], hard_labels[train])
        hgb_probability[test] = hgb.predict_proba(matrix[test])[:, 1]

        xgb = make_xgboost(hard_labels[train])
        xgb.fit(matrix[train], hard_labels[train])
        xgb_probability[test] = xgb.predict_proba(matrix[test])[:, 1]

        extra = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=4,
        )
        extra.fit(matrix[train], action_labels[train])
        local_probability = extra.predict_proba(matrix[test])
        for local_column, label in enumerate(extra.classes_):
            global_column = int(np.flatnonzero(ACTION_LEVELS == int(label))[0])
            extra_probability[test, global_column] = local_probability[:, local_column]

    extra_prediction = ACTION_LEVELS[np.argmax(extra_probability, axis=1)]
    extra_metrics = policy_metrics(rows, extra_prediction)
    extra_metrics.update(
        {
            "accuracy": float(np.mean(extra_prediction == action_labels)),
            "confusion_matrix": confusion_matrix(
                action_labels, extra_prediction, labels=ACTION_LEVELS
            ).astype(int).tolist(),
        }
    )

    oracle_metrics = policy_metrics(rows, action_labels)
    fixed_100 = policy_metrics(rows, np.full(len(rows), HARD_K, dtype=int))

    historical_path = (
        ROOT / "data" / "work" / "adaptive_retrieval_20261009" / "score_confidence_gate.json"
    )
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical_probability = np.asarray(historical["oof_probability"], dtype=float)

    payload = {
        "protocol": {
            "benchmark": "Physics 80 + Industrial 75 + OHR-Bench 65",
            "queries": len(rows),
            "base_ranking": "hybrid top-200 artifact; adaptive cap is 100",
            "features": feature_names,
            "feature_scope": "retrieval scores only; no query or neighbour features",
            "outer_folds": 5,
            "fold_stratification": "source",
            "binary_label": "hard iff k95 > 50",
            "binary_actions": [30, 100],
            "three_class_label": "smallest K in {10,40,100} preserving recall@100",
            "warning": "Exploratory OOF evaluation; no nested model or policy selection.",
        },
        "fixed_k100": {key: value for key, value in fixed_100.items() if key != "per_query_recall"},
        "historical_hgb_frozen": binary_summary(rows, historical_probability, hard_labels),
        "refit_hgb_score_only_all_rows": binary_summary(rows, hgb_probability, hard_labels),
        "xgboost_score_only_all_rows": binary_summary(rows, xgb_probability, hard_labels),
        "extra_trees_direct_three_class": {
            **{key: value for key, value in extra_metrics.items() if key != "per_query_recall"},
            "true_label_counts": {
                str(int(level)): int(np.sum(action_labels == level)) for level in ACTION_LEVELS
            },
        },
        "three_class_oracle": {
            key: value for key, value in oracle_metrics.items() if key != "per_query_recall"
        },
    }

    per_query = []
    for index, row in enumerate(rows):
        per_query.append(
            {
                "qid": qids[index],
                "source": sources[index],
                "fold": int(fold_ids[index]),
                "k95": int(k95_values[index]),
                "hard_label": int(hard_labels[index]),
                "action_label": int(action_labels[index]),
                "hgb_hard_probability": float(hgb_probability[index]),
                "xgb_hard_probability": float(xgb_probability[index]),
                "extra_trees_probability_k10": float(extra_probability[index, 0]),
                "extra_trees_probability_k40": float(extra_probability[index, 1]),
                "extra_trees_probability_k100": float(extra_probability[index, 2]),
                "extra_trees_action": int(extra_prediction[index]),
            }
        )

    (args.output_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def matched(name: str) -> dict[str, Any]:
        return payload[name]["matched_134_hard_slots"]

    historical_row = matched("historical_hgb_frozen")
    refit_row = matched("refit_hgb_score_only_all_rows")
    xgb_row = matched("xgboost_score_only_all_rows")
    lines = [
        "# Focused score-only model comparison",
        "",
        "No query or neighbour features are used. HGB and XGBoost share the same score features, hard label, folds, training rows, and matched 134-hard-slot policy.",
        "",
        "## Binary easy/hard comparison at matched budget",
        "",
        "| Model | Training rows | Mean K | Recall | ROC-AUC | AP |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Historical HGB artifact | frozen | {historical_row['mean_k']:.2f} | {historical_row['macro_recall']:.2%} | {payload['historical_hgb_frozen']['roc_auc']:.3f} | {payload['historical_hgb_frozen']['average_precision']:.3f} |",
        f"| HGB refit | all | {refit_row['mean_k']:.2f} | {refit_row['macro_recall']:.2%} | {payload['refit_hgb_score_only_all_rows']['roc_auc']:.3f} | {payload['refit_hgb_score_only_all_rows']['average_precision']:.3f} |",
        f"| XGBoost | all | {xgb_row['mean_k']:.2f} | {xgb_row['macro_recall']:.2%} | {payload['xgboost_score_only_all_rows']['roc_auc']:.3f} | {payload['xgboost_score_only_all_rows']['average_precision']:.3f} |",
        "",
        "The matched-budget row is the primary model comparison. Threshold 0.2 diagnostics are retained in report.json but are not budget matched.",
        "",
        "## Direct ExtraTrees three-class probe",
        "",
        "| Method | Mean K | Recall | Accuracy | Predicted K counts |",
        "|---|---:|---:|---:|---|",
        f"| ExtraTrees argmax | {extra_metrics['mean_k']:.2f} | {extra_metrics['macro_recall']:.2%} | {extra_metrics['accuracy']:.2%} | `{json.dumps(extra_metrics['choice_counts'], sort_keys=True)}` |",
        f"| Oracle discrete {{10,40,100}}, preserve Recall@100 | {oracle_metrics['mean_k']:.2f} | {oracle_metrics['macro_recall']:.2%} | 100.00% | `{json.dumps(oracle_metrics['choice_counts'], sort_keys=True)}` |",
        f"| Fixed K=100 | {fixed_100['mean_k']:.2f} | {fixed_100['macro_recall']:.2%} | n/a | `{{\"100\": {len(rows)}}}` |",
        "",
        "The direct three-class result is only a first probe: argmax is not recall-constrained, probabilities are not calibrated, and queries with recall@100=0 receive K=10 because every cutoff preserves the same zero recall.",
        "This discrete cap-100 oracle is not comparable to the earlier cap-200 target-recall oracle (62.49 mean K / 90.47% recall).",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
