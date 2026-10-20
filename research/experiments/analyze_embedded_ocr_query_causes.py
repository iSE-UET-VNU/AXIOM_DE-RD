"""Explain per-query ranking changes caused by embedded-image OCR."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunking_embedding.lexical import analyze  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--raw-run", type=Path, required=True)
    parser.add_argument("--filtered-run", type=Path, required=True)
    parser.add_argument("--ocr-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    queries = {str(row["query_id"]): row for row in read_jsonl(args.queries)}
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(args.qrels):
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    runs = {
        "baseline": {str(row["query_id"]): row for row in read_jsonl(args.baseline_run)},
        "raw": {str(row["query_id"]): row for row in read_jsonl(args.raw_run)},
        "filtered": {str(row["query_id"]): row for row in read_jsonl(args.filtered_run)},
    }
    ocr_text: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(args.ocr_results):
        if row.get("status") == "ok" and str(row.get("text") or "").strip():
            ocr_text[str(row["page_id"])].append(str(row["text"]))
    ocr_pages = set(ocr_text)
    ocr_files = {page_id.split("#page=", 1)[0] for page_id in ocr_pages}
    records = []
    category_counts: Counter[str] = Counter()
    qids = sorted(set(queries) & set(qrels) & set(runs["baseline"]) & set(runs["raw"]))
    for qid in qids:
        gold = qrels[qid]
        base_pages = [str(item["page_id"]) for item in runs["baseline"][qid]["chunks"]]
        raw_pages = [str(item["page_id"]) for item in runs["raw"][qid]["chunks"]]
        filtered_pages = [str(item["page_id"]) for item in runs["filtered"][qid]["chunks"]]
        base_top10, raw_top10 = set(base_pages[:10]), set(raw_pages[:10])
        base_top20, raw_top20 = set(base_pages[:20]), set(raw_pages[:20])
        base_gold10, raw_gold10 = base_top10 & gold, raw_top10 & gold
        base_gold20, raw_gold20 = base_top20 & gold, raw_top20 & gold
        added_gold10 = sorted(raw_gold10 - base_gold10)
        removed_gold10 = sorted(base_gold10 - raw_gold10)
        added_pages10 = sorted(raw_top10 - base_top10)
        removed_pages10 = sorted(base_top10 - raw_top10)
        query_terms = set(analyze(str(queries[qid]["query"])))
        direct_added = []
        inherited_added = []
        global_shift_added = []
        for page_id in added_gold10:
            page_terms = set(analyze(" ".join(ocr_text.get(page_id, []))))
            overlap = sorted(query_terms & page_terms)
            if page_id in ocr_pages and overlap:
                direct_added.append({"page_id": page_id, "overlap": overlap, "ocr_text": " ".join(ocr_text[page_id])[:500]})
            elif page_id in ocr_pages:
                direct_added.append({"page_id": page_id, "overlap": [], "ocr_text": " ".join(ocr_text[page_id])[:500]})
            elif page_id.split("#page=", 1)[0] in ocr_files:
                inherited_added.append(page_id)
            else:
                global_shift_added.append(page_id)
        added_ocr_competitors = [page_id for page_id in added_pages10 if page_id in ocr_pages]
        removed_ocr_pages = [page_id for page_id in removed_pages10 if page_id in ocr_pages]
        base_recall10 = len(base_gold10) / max(len(gold), 1)
        raw_recall10 = len(raw_gold10) / max(len(gold), 1)
        base_recall20 = len(base_gold20) / max(len(gold), 1)
        raw_recall20 = len(raw_gold20) / max(len(gold), 1)
        # A compact category is assigned only for Recall@10 changes; rank-only
        # changes are kept as a separate family below.
        if raw_recall10 > base_recall10:
            if direct_added:
                category = "gold_promoted_by_direct_embedded_ocr"
            elif inherited_added:
                category = "gold_promoted_by_same_file_ocr_signal"
            else:
                category = "gold_promoted_by_global_bm25_statistics"
        elif raw_recall10 < base_recall10:
            category = "gold_displaced_by_ocr_induced_reranking"
        else:
            base_ndcg = float(runs["baseline"][qid].get("chunks", [{}])[0].get("score", 0.0)) if False else None
            # The detailed delta JSON contains the metric-level classification.
            category = "no_recall10_change"
        record = {
            "query_id": qid, "source": queries[qid].get("source", qid.split("::", 1)[0]), "query": queries[qid]["query"],
            "gold_count": len(gold), "base_recall10": base_recall10, "raw_recall10": raw_recall10, "base_recall20": base_recall20, "raw_recall20": raw_recall20,
            "added_gold_top10": added_gold10, "removed_gold_top10": removed_gold10, "added_pages_top10": added_pages10, "removed_pages_top10": removed_pages10,
            "direct_ocr_gold": direct_added, "same_file_ocr_gold": inherited_added, "global_shift_gold": global_shift_added,
            "ocr_competitors_added": added_ocr_competitors, "ocr_pages_removed": removed_ocr_pages, "category": category,
        }
        records.append(record)
        if raw_recall10 != base_recall10:
            category_counts[category] += 1

    # Add nDCG-only classification from the already evaluated per-query JSONL
    # if available beside the visualization output.
    detail_path = args.output_dir.parent / "query_delta_visualization" / "per_query_ocr_delta.jsonl"
    detail = {str(row["query_id"]): row for row in read_jsonl(detail_path)} if detail_path.is_file() else {}
    for row in records:
        metrics = detail.get(row["query_id"], {}).get("delta_raw", {})
        row["delta_ndcg10"] = metrics.get("ndcg10", 0.0)
        row["delta_ndcg20"] = metrics.get("ndcg20", 0.0)
        if row["category"] == "no_recall10_change" and (row["delta_ndcg10"] != 0 or row["delta_ndcg20"] != 0):
            row["category"] = "rank_only_change"
            category_counts["rank_only_change"] += 1
    (args.output_dir / "query_cause_analysis.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records if row["category"] != "no_recall10_change"), encoding="utf-8")
    summary = {"changed_query_records": len([row for row in records if row["category"] != "no_recall10_change"]), "categories": dict(category_counts), "direct_ocr_gold_count": sum(bool(row["direct_ocr_gold"]) for row in records), "same_file_inherited_count": sum(bool(row["same_file_ocr_gold"]) for row in records), "global_shift_count": sum(bool(row["global_shift_gold"]) for row in records), "ocr_competitor_count": sum(bool(row["ocr_competitors_added"]) for row in records)}
    (args.output_dir / "cause_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Embedded OCR query-cause analysis", "", "| Category | Queries |", "|---|---:|"]
    for key, value in category_counts.most_common():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## Changed queries", "", "| QID | Source | Category | Added gold top10 | Removed gold top10 | OCR competitors added | Query |", "|---|---|---|---|---|---|---|"]
    for row in records:
        if row["category"] == "no_recall10_change":
            continue
        lines.append(f"| `{row['query_id']}` | {row['source']} | {row['category']} | {', '.join(row['added_gold_top10']) or '-'} | {', '.join(row['removed_gold_top10']) or '-'} | {len(row['ocr_competitors_added'])} | {row['query']} |")
    (args.output_dir / "cause_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
