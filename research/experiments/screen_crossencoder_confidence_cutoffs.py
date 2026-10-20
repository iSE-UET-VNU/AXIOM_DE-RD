"""Offline confidence-threshold cutoffs over cached cross-encoder scores."""

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
    vals, r10, r20 = [], [], []
    for qid in qids:
        gold = qrels[qid]
        pages = [pid for pid, _s in rankings[qid][:choices[qid]]]
        denom = max(len(gold), 1)
        vals.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
    return {"recall_at_returned_k": float(np.mean(vals)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean(list(choices.values())))}


def calibrate(qids, rankings, choices, target=12):
    choices = dict(choices)
    total = sum(choices.values())
    while total > target * len(qids):
        avail = [q for q in qids if choices[q] > 1]
        qid = min(avail, key=lambda q: (rankings[q][choices[q] - 1][1], q))
        choices[qid] -= 1
        total -= 1
    while total < target * len(qids):
        avail = [q for q in qids if choices[q] < len(rankings[q])]
        qid = max(avail, key=lambda q: (rankings[q][choices[q]][1], q))
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
    policies = {}
    details = []
    for threshold in (-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4):
        raw = {qid: max(1, min(20, sum(score >= threshold for _pid, score in rankings[qid][:20]))) for qid in qids}
        choices = calibrate(qids, rankings, raw)
        name = f"absolute_score_{threshold:+.1f}_exact_mean12"
        policies[name] = metric(qids, rankings, qrels, choices)
        details.append({"policy": name, "raw_mean_k": float(np.mean(list(raw.values()))), "metrics": policies[name]})
    for margin in (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        raw = {qid: max(1, min(20, sum(score >= rankings[qid][0][1] - margin for _pid, score in rankings[qid][:20]))) for qid in qids}
        choices = calibrate(qids, rankings, raw)
        name = f"top1_margin_{margin:.2f}_exact_mean12"
        policies[name] = metric(qids, rankings, qrels, choices)
        details.append({"policy": name, "raw_mean_k": float(np.mean(list(raw.values()))), "metrics": policies[name]})
    fixed = {qid: 12 for qid in qids}
    policies["fixed_k12"] = metric(qids, rankings, qrels, fixed)
    report = {"queries": len(qids), "policies": policies, "details": details, "notes": ["All cutoffs use only cached cross-encoder scores and are calibrated to exact mean K=12.", "Qrels are used only for final evaluation."]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best = sorted(((name, value) for name, value in policies.items()), key=lambda item: -item[1]["recall_at_returned_k"])
    lines = ["# Cross-encoder confidence cutoffs", "", "| Policy | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    for name, value in best:
        lines.append(f"| {name} | {value['recall_at_returned_k']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} | {value['mean_k']:.2f} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"fixed": policies["fixed_k12"], "best": best[:10]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
