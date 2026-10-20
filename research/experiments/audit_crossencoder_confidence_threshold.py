"""Source-stratified stability audit for the cached cross-encoder confidence cutoff."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/work/benchmark_hierarchical_local"
RANKINGS = CACHE / "mixed_cached_crossencoder/rankings.jsonl"
OUTPUT = CACHE / "crossencoder_confidence_cutoffs"


def metric(qids, rankings, qrels, choices):
    values, r10, r20 = [], [], []
    for qid in qids:
        gold = qrels[qid]
        pages = [pid for pid, _s in rankings[qid][:choices[qid]]]
        denom = max(len(gold), 1)
        values.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
    return {"recall_at_returned_k": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean(list(choices.values())))}


def calibrate(qids, rankings, choices, target=12):
    choices = dict(choices)
    total = sum(choices.values())
    while total > target * len(qids):
        available = [q for q in qids if choices[q] > 1]
        qid = min(available, key=lambda q: (rankings[q][choices[q] - 1][1], q))
        choices[qid] -= 1
        total -= 1
    while total < target * len(qids):
        available = [q for q in qids if choices[q] < len(rankings[q])]
        qid = max(available, key=lambda q: (rankings[q][choices[q]][1], q))
        choices[qid] += 1
        total += 1
    return choices


def main() -> None:
    rows = [json.loads(line) for line in RANKINGS.open(encoding="utf-8")]
    qids = [row["query_id"] for row in rows]
    rankings = {row["query_id"]: [(str(pid), float(score)) for pid, score in row["ranking"]] for row in rows}
    qrels = defaultdict(set)
    for line in (CACHE / "qrels_page_level.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    source = {qid: qid.split("::", 1)[0] for qid in qids}
    fold_by_qid = {}
    for group in sorted(set(source.values())):
        group_qids = sorted(qid for qid in qids if source[qid] == group)
        for i, qid in enumerate(group_qids):
            fold_by_qid[qid] = i % 5
    fold_rows = []
    for fold in range(5):
        test_qids = [qid for qid in qids if fold_by_qid[qid] == fold]
        raw = {qid: max(1, min(20, sum(score >= -8.0 for _pid, score in rankings[qid][:20]))) for qid in test_qids}
        choices = calibrate(test_qids, rankings, raw)
        fixed = {qid: 12 for qid in test_qids}
        fold_rows.append({"fold": fold + 1, "queries": len(test_qids), "confidence": metric(test_qids, rankings, qrels, choices), "fixed": metric(test_qids, rankings, qrels, fixed)})
    all_raw = {qid: max(1, min(20, sum(score >= -8.0 for _pid, score in rankings[qid][:20]))) for qid in qids}
    all_conf = calibrate(qids, rankings, all_raw)
    all_fixed = {qid: 12 for qid in qids}
    report = {"threshold": -8.0, "full_confidence": metric(qids, rankings, qrels, all_conf), "full_fixed": metric(qids, rankings, qrels, all_fixed), "folds": fold_rows, "notes": ["Threshold was fixed before this fold audit; calibration uses score only."]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Cross-encoder confidence threshold audit", "", "| Fold | Policy | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---:|---|---:|---:|---:|---:|"]
    for row in fold_rows:
        for name in ("fixed", "confidence"):
            value = row[name]
            lines.append(f"| {row['fold']} | {name} | {value['recall_at_returned_k']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} | {value['mean_k']:.2f} |")
    (OUTPUT / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
