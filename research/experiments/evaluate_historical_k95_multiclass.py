"""Extend the historical k95 easy/hard target to 3/5/10 HGB classes.

This is an exploratory five-fold OOF ablation.  It deliberately reuses the
historical score-only features and HGB settings.  The old binary row remains
an anchor because its 30/100 action policy is not an equal-width discretisation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_DIR = ROOT / "experiments" / "adaptive-k-retrieval"
if str(ADAPTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTIVE_DIR))

from run_canonical_adaptive_methods import load_rows  # noqa: E402
from run_risk_label_statistics import features as score_features  # noqa: E402


OUTPUT = ROOT / "data" / "work" / "historical_k95_multiclass_20261016"
SEED = 42
CAP = 100


def recall_at(row: dict, cutoff: int) -> float:
    gold = set(row["gold"])
    return len(set(row["ranked"][:cutoff]) & gold) / max(len(gold), 1)


def required_k95(row: dict) -> int:
    gold = set(row["gold"])
    for rank, page_id in enumerate(row["ranked"][:200], 1):
        if page_id in gold:
            found = len(set(row["ranked"][:rank]) & gold)
            if found / max(len(gold), 1) >= 0.95:
                return rank
    return 201


def retrieval_metrics(rows: list[dict], choices: np.ndarray) -> dict[str, float]:
    recalls, precisions, f1s, ndcgs = [], [], [], []
    for row, raw_k in zip(rows, choices):
        k = int(raw_k)
        gold = set(row["gold"])
        ranked = row["ranked"][:k]
        hits = [1.0 if page_id in gold else 0.0 for page_id in ranked]
        tp = float(sum(hits))
        recall = tp / max(len(gold), 1)
        precision = tp / max(k, 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal = min(len(gold), k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal))
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        ndcgs.append(dcg / idcg if idcg else 0.0)
    return {
        "mean_k": float(np.mean(choices)),
        "macro_recall": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
        "macro_f1": float(np.mean(f1s)),
        "macro_ndcg": float(np.mean(ndcgs)),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, _ = load_rows("hybrid", 200)
    sources = np.asarray([str(row["source"]) for row in rows])
    feature_names, _ = score_features(rows[0]["scores"])
    matrix = np.vstack([score_features(row["scores"])[1] for row in rows])
    target = np.asarray([required_k95(row) for row in rows])
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(matrix, sources))

    results = []
    per_query = []
    for class_count in (3, 5, 10):
        # Equal-width k95 intervals over the historical cap of 100. Values
        # above 100 (including unreachable rows) belong to the final class.
        upper_bounds = np.rint(np.linspace(CAP / class_count, CAP, class_count)).astype(int)
        thresholds = upper_bounds[:-1]
        labels = np.digitize(target, thresholds, right=True)
        prediction = np.zeros(len(rows), dtype=int)
        fold_ids = np.zeros(len(rows), dtype=int)
        for fold, (train, test) in enumerate(folds):
            model = HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=7,
                random_state=SEED,
            )
            model.fit(matrix[train], labels[train])
            prediction[test] = model.predict(matrix[test])
            fold_ids[test] = fold
        choices = upper_bounds[prediction]
        matrix_confusion = confusion_matrix(
            labels, prediction, labels=list(range(class_count))
        ).astype(int)
        result = {
            "classes": class_count,
            "thresholds": thresholds.tolist(),
            "actions": upper_bounds.tolist(),
            "label_counts": np.bincount(labels, minlength=class_count).astype(int).tolist(),
            "predicted_counts": np.bincount(prediction, minlength=class_count).astype(int).tolist(),
            "accuracy": float(np.mean(labels == prediction)),
            "confusion_matrix": matrix_confusion.tolist(),
            **retrieval_metrics(rows, choices),
        }
        results.append(result)
        for index, row in enumerate(rows):
            per_query.append({
                "qid": str(row["qid"]),
                "source": str(row["source"]),
                "fold": int(fold_ids[index]),
                "classes": class_count,
                "k95": int(target[index]),
                "true_class": int(labels[index]),
                "predicted_class": int(prediction[index]),
                "chosen_k": int(choices[index]),
                "recall": recall_at(row, int(choices[index])),
            })

    payload = {
        "protocol": {
            "queries": len(rows),
            "features": feature_names,
            "model": "HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=7)",
            "folds": "5-fold source-stratified OOF",
            "target": "historical required-k95; >100/unreachable maps to last class",
            "binning": "equal-width over K=1..100; predicted action is bin upper bound",
            "warning": "Exploratory OOF ablation; the historical binary 30/100 policy is a separate anchor.",
        },
        "historical_binary_anchor": {
            "actions": [30, 100],
            "mean_k": 72.63636363636364,
            "macro_recall": 0.8642037119553516,
            "macro_precision": 0.05698484848484849,
            "macro_f1": 0.09851529933559905,
            "macro_ndcg": 0.6683527746169428,
        },
        "results": results,
    }
    (OUTPUT / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in per_query:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    lines = [
        "# Historical k95 multiclass ablation",
        "",
        "| Method | Actions | Mean K | Recall | Precision | F1 | nDCG | Accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        "| Historical binary anchor | 30/100 | 72.64 | 86.42% | 5.70% | 9.85% | 66.84 | n/a |",
    ]
    for item in results:
        lines.append(
            f"| HGB {item['classes']} class | {'/'.join(map(str, item['actions']))} | "
            f"{item['mean_k']:.2f} | {item['macro_recall']:.2%} | "
            f"{item['macro_precision']:.2%} | {item['macro_f1']:.2%} | "
            f"{100 * item['macro_ndcg']:.2f} | {item['accuracy']:.2%} |"
        )
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
