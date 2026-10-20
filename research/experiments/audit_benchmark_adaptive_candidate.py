"""Fold audit for the selected adaptive candidate and its controls."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data/work/benchmark_hierarchical_local"
SCREEN = CACHE / "adaptive_method_screen"
OUTPUT = CACHE / "adaptive_candidate_audit"


def metrics(rows: list[dict[str, Any]], qrels: dict[str, set[str]]) -> dict[str, float]:
    values, r10, r20 = [], [], []
    for row in rows:
        gold = qrels[row["query_id"]]
        pages = row["returned_pages"]
        denom = max(len(gold), 1)
        values.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
    return {"recall_at_returned_k": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean([len(row["returned_pages"]) for row in rows]))}


def main() -> int:
    rows = [json.loads(line) for line in (SCREEN / "all_policy_rows.jsonl").open(encoding="utf-8")]
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in [json.loads(line) for line in (CACHE / "qrels_page_level.jsonl").open(encoding="utf-8")]:
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    names = ["bm25_fixed_k12", "hybrid_file_sum2_beta0.30_fixed_k12", "hybrid_file_sum2_beta0.30_coverage_aware"]
    by_policy = {name: {row["query_id"]: row for row in rows if row["policy"] == name} for name in names}
    qids = sorted(by_policy[names[0]])
    sources = {qid: qid.split("::", 1)[0] for qid in qids}
    # Deterministic stratified folds: each source is distributed round-robin.
    fold_by_qid = {}
    for source in sorted(set(sources.values())):
        source_qids = sorted(qid for qid in qids if sources[qid] == source)
        for position, qid in enumerate(source_qids):
            fold_by_qid[qid] = position % 5
    folds = []
    for fold in range(5):
        fold_qids = [qid for qid in qids if fold_by_qid[qid] == fold]
        fold_metrics = {name: metrics([by_policy[name][qid] for qid in fold_qids], qrels) for name in names}
        folds.append({"fold": fold + 1, "queries": len(fold_qids), "metrics": fold_metrics})
    report = {"policies": names, "folds": folds, "full": {name: metrics([by_policy[name][qid] for qid in qids], qrels) for name in names}, "notes": ["This is a post-selection stability audit; the file-fusion and budget parameters were selected in the exploratory screen before this audit.", "Folds are source-stratified and use no new ranking or API call."]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Adaptive candidate fold audit", "", "| Fold | Policy | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---:|---|---:|---:|---:|---:|"]
    for fold in folds:
        for name in names:
            value = fold["metrics"][name]
            lines.append(f"| {fold['fold']} | {name} | {value['recall_at_returned_k']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} | {value['mean_k']:.2f} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
