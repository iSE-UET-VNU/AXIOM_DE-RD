"""Cross-validated query-cluster budgets for adaptive retrieval.

This is a retrieval-only proxy for query-centric adaptive-k methods: cheap
query/score-shape features are clustered without qrels, then the best K for
each cluster is learned on the training queries of each fold.  Test-fold qrels
are never used for clustering, budget selection, or ranking.
"""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.evaluate_benchmark_adaptive_transfer import (  # noqa: E402
    K_VALUES,
    _features,
    _load,
    _recall_curve,
)


OUTPUT = ROOT / "data/work/benchmark_hierarchical_local/query_cluster_adaptive"


def _choose_cluster_budgets(cluster_ids: np.ndarray, curves: np.ndarray, n_clusters: int, target_mean: float = 12.0) -> dict[int, int]:
    # Exhaustive over five K choices for four clusters is small and makes the
    # budget constraint explicit.  The objective uses train-fold qrels only.
    options = np.asarray(K_VALUES, dtype=int)
    best: tuple[tuple[float, float], dict[int, int]] | None = None
    for flat in np.ndindex(*(len(options),) * n_clusters):
        assignment = {cluster: int(options[flat[cluster]]) for cluster in range(n_clusters)}
        chosen = np.asarray([assignment[int(cluster)] for cluster in cluster_ids])
        mean_k = float(np.mean(chosen))
        utility = float(np.mean(curves[np.arange(len(curves)), [K_VALUES.index(int(k)) for k in chosen]]))
        key = (abs(mean_k - target_mean), -utility)
        if best is None or key < best[0]:
            best = (key, assignment)
    assert best is not None
    return best[1]


def _metrics(qids: Sequence[str], choices: Sequence[int], rankings: dict[str, Sequence[tuple[str, float]]], qrels: dict[str, set[str]]) -> dict[str, float]:
    values, r10, r20 = [], [], []
    for qid, k in zip(qids, choices):
        gold = qrels[qid]
        ranking = rankings[qid]
        denom = max(len(gold), 1)
        values.append(len({pid for pid, _score in ranking[:k]} & gold) / denom)
        r10.append(len({pid for pid, _score in ranking[:10]} & gold) / denom)
        r20.append(len({pid for pid, _score in ranking[:20]} & gold) / denom)
    return {"recall_at_returned_k": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean(choices))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    queries, qrels, rankings_all = _load()
    qids = [str(query["query_id"]) for query in queries]
    features = np.asarray([_features(str(query["query"]), rankings_all[qid]) for query, qid in zip(queries, qids)], dtype=float)
    rankings = {qid: rankings_all[qid]["hybrid"] for qid in qids}
    curves = np.asarray([_recall_curve(qid, rankings[qid], qrels) for qid in qids], dtype=float)
    from sklearn.cluster import KMeans
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=2026)
    fold_rows = []
    oof_choices = np.zeros(len(qids), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(features), start=1):
        scaler = StandardScaler().fit(features[train_idx])
        train_scaled = scaler.transform(features[train_idx])
        test_scaled = scaler.transform(features[test_idx])
        model = KMeans(n_clusters=args.clusters, random_state=2026, n_init=20).fit(train_scaled)
        train_clusters = model.labels_
        test_clusters = model.predict(test_scaled)
        budgets = _choose_cluster_budgets(train_clusters, curves[train_idx], args.clusters)
        choices = np.asarray([budgets[int(cluster)] for cluster in test_clusters], dtype=int)
        oof_choices[test_idx] = choices
        test_qids = [qids[i] for i in test_idx]
        fold_rows.append({"fold": fold, "metrics": _metrics(test_qids, choices, rankings, qrels), "budgets": budgets, "train_cluster_counts": dict(Counter(int(value) for value in train_clusters)), "test_cluster_counts": dict(Counter(int(value) for value in test_clusters))})
    oof = _metrics(qids, oof_choices, rankings, qrels)
    fixed = _metrics(qids, [12] * len(qids), rankings, qrels)
    report = {"queries": len(qids), "features": int(features.shape[1]), "clusters": args.clusters, "folds": args.folds, "method": "standardized score-shape/query features -> KMeans -> train-fold cluster budget", "qrels_used_for_ranking": False, "fixed_k12": fixed, "oof": oof, "folds_detail": fold_rows, "notes": ["The cluster assignment is qrel-free; only training-fold qrels select cluster budgets.", "This is an exploratory cross-validation result, not a final generalization claim."]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Query-cluster adaptive budget", "", "| Method | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    for name, metric in (("fixed K=12", fixed), ("OOF cluster budget", oof)):
        lines.append(f"| {name} | {metric['recall_at_returned_k']:.2%} | {metric['recall@10']:.2%} | {metric['recall@20']:.2%} | {metric['mean_k']:.2f} |")
    lines += ["", "| Fold | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---:|---:|---:|---:|---:|"]
    for row in fold_rows:
        metric = row["metrics"]
        lines.append(f"| {row['fold']} | {metric['recall_at_returned_k']:.2%} | {metric['recall@10']:.2%} | {metric['recall@20']:.2%} | {metric['mean_k']:.2f} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
