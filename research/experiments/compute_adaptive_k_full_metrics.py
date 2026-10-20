"""Compute comparable macro retrieval metrics for saved Adaptive-K policies."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "research" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from adaptive_k_optimization_core import load_frozen_data  # noqa: E402


def policy_metrics(rows: Sequence[Mapping[str, Any]], choices: np.ndarray) -> dict[str, float]:
    recalls, precisions, f1s, ndcgs = [], [], [], []
    for row, cutoff_value in zip(rows, choices):
        cutoff = int(cutoff_value)
        gold = set(row["gold"])
        ranked = list(row["ranked"][:cutoff])
        hits = [1.0 if page_id in gold else 0.0 for page_id in ranked]
        true_positive = float(sum(hits))
        recall = true_positive / max(len(gold), 1)
        precision = true_positive / max(cutoff, 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(hits))
        ideal_hits = min(len(gold), cutoff)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        recalls.append(recall); precisions.append(precision); f1s.append(f1); ndcgs.append(dcg / idcg if idcg else 0.0)
    return {
        "mean_k": float(np.mean(choices)),
        "macro_recall": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
        "macro_f1": float(np.mean(f1s)),
        "macro_ndcg": float(np.mean(ndcgs)),
    }


def main() -> int:
    data = load_frozen_data()
    qid_to_position = {str(row["qid"]): index for index, row in enumerate(data.rows)}

    qcar_path = ROOT / "data/work/qcar_faithful_20261007_source_stratified/per_query.jsonl"
    qcar_records = [json.loads(line) for line in qcar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    qcar_method = "qcar_te3s_umap_3c_kmax100_semantic_mean2std"
    qcar_qwen_method = "qcar_qwen_umap_3c_kmax100_semantic_mean2std"
    qcar_choices = np.zeros(len(data.rows), dtype=int)
    qcar_qwen_choices = np.zeros(len(data.rows), dtype=int)
    for record in qcar_records:
        if record["method"] == qcar_method:
            qcar_choices[qid_to_position[str(record["qid"])]] = int(record["adaptive_k"])
        if record["method"] == qcar_qwen_method:
            qcar_qwen_choices[qid_to_position[str(record["qid"])]] = int(record["adaptive_k"])
    if np.any(qcar_choices == 0) or np.any(qcar_qwen_choices == 0):
        raise ValueError("Incomplete QCAR choices")

    historical_path = ROOT / "data/work/adaptive_retrieval_20261009/score_confidence_gate.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    probability = np.asarray(historical["oof_probability"], dtype=float)
    hard = np.zeros(len(data.rows), dtype=bool)
    hard[np.argsort(-probability, kind="stable")[:134]] = True
    easy_hard_100 = np.where(hard, 100, 30)
    easy_hard_120 = np.where(hard, 120, 30)

    topk_path = ROOT / "data/work/xgboost_adaptive_k_optimization_20261016/neighbor/neighbor_score_modality_oof.csv"
    with topk_path.open(encoding="utf-8-sig") as handle:
        topk_records = list(csv.DictReader(handle))
    topk_choices = np.zeros(len(data.rows), dtype=int)
    for record in topk_records:
        topk_choices[qid_to_position[str(record["qid"])]] = int(record["chosen_k"])

    score_topk_path = ROOT / "data/work/xgboost_adaptive_k_optimization_20261016/features/score_modality_oof.csv"
    with score_topk_path.open(encoding="utf-8-sig") as handle:
        score_topk_records = list(csv.DictReader(handle))
    score_topk_choices = np.zeros(len(data.rows), dtype=int)
    for record in score_topk_records:
        score_topk_choices[qid_to_position[str(record["qid"])]] = int(record["chosen_k"])

    policies = [
        ("QCAR 3-cluster (TE3S, faithful)", qcar_choices),
        ("QCAR 3-cluster (Qwen, faithful)", qcar_qwen_choices),
        ("Historical Easy/Hard HGB, K=30/120", easy_hard_120),
        ("Historical Easy/Hard HGB, K=30/100", easy_hard_100),
        ("Top-K regression, score+modality XGBoost", score_topk_choices),
        ("Top-K regression, neighbor-aware XGBoost (corrected)", topk_choices),
    ]
    output = [{"method": name, **policy_metrics(data.rows, choices)} for name, choices in policies]
    target = ROOT / "data/work/xgboost_adaptive_k_optimization_20261016/final/full_metric_comparison.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
