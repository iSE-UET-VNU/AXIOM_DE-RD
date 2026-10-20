"""Make mixed-text score-gap Adaptive-k use exactly the fixed total page budget."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/work/benchmark_hierarchical_local"
INPUT = CACHE / "adaptive_mixed_text/all_policy_rows.jsonl"
OUTPUT = CACHE / "mixed_gap_budget_calibrated"


def metric(rows, qrels):
    values, r10, r20 = [], [], []
    for row in rows:
        gold = qrels[row["query_id"]]
        pages = row["pages"]
        denom = max(len(gold), 1)
        values.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
    return {"recall_at_returned_k": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean([len(row["pages"]) for row in rows]))}


def main() -> int:
    rows = [json.loads(line) for line in INPUT.open(encoding="utf-8")]
    qrels: dict[str, set[str]] = defaultdict(set)
    for line in (CACHE / "qrels_page_level.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    policy = "file_largest_gap_plus6"
    selected = [dict(row) for row in rows if row["policy"] == policy]
    if len(selected) != 220:
        raise RuntimeError(f"Expected 220 rows for {policy}, got {len(selected)}")
    target_total = 12 * len(selected)
    current_total = sum(len(row["pages"]) for row in selected)
    # Remove low-score tail pages first. This uses only the retriever score,
    # never qrels, and preserves the Adaptive-k order for every query.
    while current_total > target_total:
        candidates = [row for row in selected if len(row["pages"]) > 1]
        row = min(candidates, key=lambda item: (float(item["scores"][-1]), item["query_id"]))
        row["pages"] = row["pages"][:-1]
        row["scores"] = row["scores"][:-1]
        row["k"] = len(row["pages"])
        current_total -= 1
    for row in selected:
        row["policy"] = "file_largest_gap_plus6_exact_mean12"
    fixed_bm25 = [row for row in rows if row["policy"] == "bm25_fixed_k12"]
    fixed_file = [row for row in rows if row["policy"] == "file_fixed_k12"]
    policies = {"bm25_fixed_k12": metric(fixed_bm25, qrels), "file_fixed_k12": metric(fixed_file, qrels), "file_gap_plus6_raw": metric([row for row in rows if row["policy"] == policy], qrels), "file_gap_plus6_exact_mean12": metric(selected, qrels)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"target_total_pages": target_total, "actual_total_pages": current_total, "qrels_used_for_calibration": False, "policies": policies}
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "rows.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in selected) + "\n", encoding="utf-8")
    lines = ["# Calibrated mixed-text Adaptive-k", "", "| Policy | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    for name, value in policies.items():
        lines.append(f"| {name} | {value['recall_at_returned_k']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} | {value['mean_k']:.2f} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
