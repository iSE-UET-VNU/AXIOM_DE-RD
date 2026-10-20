"""Summarise the embedded-image OCR size grid with 5-fold OOF selection."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from math import log2
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ndcg(pages: Sequence[str], gold: Mapping[str, int], k: int) -> float:
    dcg = sum((2.0 ** gold[page] - 1.0) / log2(rank + 1) for rank, page in enumerate(pages[:k], 1) if page in gold)
    ideal = sum((2.0 ** value - 1.0) / log2(rank + 1) for rank, value in enumerate(sorted(gold.values(), reverse=True)[:k], 1))
    return dcg / ideal if ideal else 0.0


def query_metrics(row: Mapping[str, Any], gold: Mapping[str, int]) -> dict[str, float]:
    pages = [str(item["page_id"]) for item in row["chunks"]]
    gold_pages = set(gold)
    gold_files = {page.split("#page=", 1)[0] for page in gold_pages}
    selected = {str(item["file_id"]) for item in row["selected_files"]}
    return {
        "recall10": len(set(pages[:10]) & gold_pages) / len(gold_pages),
        "recall20": len(set(pages[:20]) & gold_pages) / len(gold_pages),
        "ndcg10": ndcg(pages, gold, 10),
        "ndcg20": ndcg(pages, gold, 20),
        "file_recall3": len(selected & gold_files) / len(gold_files),
    }


def avg(per_query: Mapping[str, Mapping[str, float]], qids: Sequence[str]) -> dict[str, float]:
    return {key: mean(per_query[qid][key] for qid in qids) for key in ("recall10", "recall20", "ndcg10", "ndcg20", "file_recall3")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.grid_root / "manifest.json").read_text(encoding="utf-8"))
    metadata = {arm["name"]: arm for arm in manifest["arms"]}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_jsonl(args.qrels):
        qrels[str(row["query_id"])][str(row["page_id"])] = int(row.get("relevance", 1))
    qids = sorted(qrels)
    by_arm: dict[str, dict[str, dict[str, float]]] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for name, arm in metadata.items():
        run = read_jsonl(args.grid_root / "results" / name / "runs" / "hierarchical_bm25_kf3_top20.jsonl")
        rows = {str(row["query_id"]): query_metrics(row, qrels[str(row["query_id"])]) for row in run}
        if set(rows) != set(qids):
            raise ValueError(f"query mismatch for {name}")
        by_arm[name] = rows
        metrics = avg(rows, qids)
        aggregate_rows.append({
            "name": name, "min_area_ratio": arm["min_area_ratio"], "min_area_percent": arm["min_area_percent"],
            "min_pixel_area": arm["min_pixel_area"], "accepted_images": arm.get("accepted_images", 0),
            "rejected_images": arm.get("rejected_images", 0), **metrics,
        })

    def selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
        # Accuracy first. Exact ties choose the less aggressive filter.
        return (row["recall10"], row["ndcg10"], row["recall20"], -row["min_area_ratio"], -row["min_pixel_area"])

    full_best = max(aggregate_rows, key=selection_key)
    folds = {qid: int(hashlib.sha1(qid.encode("utf-8")).hexdigest(), 16) % 5 for qid in qids}
    fold_rows: list[dict[str, Any]] = []
    oof_per_query: dict[str, dict[str, float]] = {}
    for fold in range(5):
        train = [qid for qid in qids if folds[qid] != fold]
        test = [qid for qid in qids if folds[qid] == fold]
        candidates = []
        for name, rows in by_arm.items():
            metric = avg(rows, train)
            arm = metadata[name]
            candidates.append({"name": name, "min_area_ratio": arm["min_area_ratio"], "min_pixel_area": arm["min_pixel_area"], **metric})
        selected = max(candidates, key=selection_key)
        test_metric = avg(by_arm[selected["name"]], test)
        for qid in test:
            oof_per_query[qid] = by_arm[selected["name"]][qid]
        fold_rows.append({"fold": fold, "train_queries": len(train), "test_queries": len(test), "selected": selected["name"], "min_area_ratio": selected["min_area_ratio"], "min_pixel_area": selected["min_pixel_area"], "train": {k: selected[k] for k in ("recall10", "recall20", "ndcg10", "ndcg20", "file_recall3")}, "test": test_metric})
    oof = avg(oof_per_query, qids)
    raw = next(row for row in aggregate_rows if row["min_area_ratio"] == 0 and row["min_pixel_area"] == 0)
    top = sorted(aggregate_rows, key=selection_key, reverse=True)[:10]

    args.grid_root.mkdir(parents=True, exist_ok=True)
    with (args.grid_root / "grid_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(aggregate_rows, key=lambda row: (row["min_area_ratio"], row["min_pixel_area"])))
    report = {"selection_protocol": "5 folds by stable SHA1(query_id); choose on other 4 folds by recall@10, then nDCG@10, recall@20; conservative threshold tie-break", "raw_control": raw, "full_set_best_exploratory": full_best, "oof_metrics": oof, "folds": fold_rows, "selected_arm_counts": dict(Counter(row["selected"] for row in fold_rows)), "top10_full_set": top}
    (args.grid_root / "grid_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Embedded-image OCR size grid", "", f"Grid: {len(aggregate_rows)} arms; 220 queries; hierarchical BM25 Kf=3; top-20 saved.",
        "", "## Full-set ranking (exploratory)", "", "| Area % | Min pixels | Kept images | Recall@10 | Recall@20 | nDCG@10 | nDCG@20 | File recall@3 |", "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(f"| {row['min_area_percent']:.2f}% | {row['min_pixel_area']:,} | {row['accepted_images']:,} | {row['recall10']:.2%} | {row['recall20']:.2%} | {100*row['ndcg10']:.2f} | {100*row['ndcg20']:.2f} | {row['file_recall3']:.2%} |")
    lines += ["", "## Five-fold out-of-fold selection", "", "| Fold | Selected | Train q | Test q | Test Recall@10 | Test nDCG@10 |", "|---:|---|---:|---:|---:|---:|"]
    for row in fold_rows:
        lines.append(f"| {row['fold']} | `{row['selected']}` | {row['train_queries']} | {row['test_queries']} | {row['test']['recall10']:.2%} | {100*row['test']['ndcg10']:.2f} |")
    lines += ["", f"OOF aggregate: Recall@10 **{oof['recall10']:.2%}**, Recall@20 **{oof['recall20']:.2%}**, nDCG@10 **{100*oof['ndcg10']:.2f}**, nDCG@20 **{100*oof['ndcg20']:.2f}**, file recall@3 **{oof['file_recall3']:.2%}**.", ""]
    (args.grid_root / "grid_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
