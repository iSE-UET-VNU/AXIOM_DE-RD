"""Evaluate global and Kf-limited hierarchical BM25 on an OCR-augmented page cache.

The intended comparison is controlled: input page text differs from the
default light run only by embedded-image OCR inserted in PDF reading order.
No qrels are used until both rankings are complete.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import log2
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.sparse import BM25Index  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalise(scores: dict[str, float]) -> dict[str, float]:
    maximum = max((score for score in scores.values() if score > 0), default=0.0)
    return {key: max(0.0, score) / maximum if maximum else 0.0 for key, score in scores.items()}


def ranked(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def ndcg(pages: list[str], gold: dict[str, int], k: int) -> float:
    dcg = sum(
        (2.0 ** gold[page] - 1.0) / log2(rank + 1)
        for rank, page in enumerate(pages[:k], 1)
        if page in gold
    )
    ideal = sum(
        (2.0 ** relevance - 1.0) / log2(rank + 1)
        for rank, relevance in enumerate(sorted(gold.values(), reverse=True)[:k], 1)
    )
    return dcg / ideal if ideal else 0.0


def evaluate(rows: list[dict[str, Any]], qrels: dict[str, dict[str, int]], depth: int, saved_depth: int) -> dict[str, Any]:
    recalls10: list[float] = []
    recalls20: list[float] = []
    ndcgs10: list[float] = []
    ndcgs20: list[float] = []
    hits10: list[float] = []
    candidate_file_recalls: list[float] = []
    derived_file_recalls: list[float] = []
    candidates: list[int] = []
    for row in rows:
        gold = qrels[str(row["query_id"])]
        gold_pages = set(gold)
        pages = [item["page_id"] for item in row["chunks"]]
        top10, top20 = pages[:depth], pages[:saved_depth]
        denom = len(gold_pages)
        recalls10.append(len(set(top10) & gold_pages) / denom)
        recalls20.append(len(set(top20) & gold_pages) / denom)
        ndcgs10.append(ndcg(top10, gold, depth))
        ndcgs20.append(ndcg(top20, gold, saved_depth))
        hits10.append(float(bool(set(top10) & gold_pages)))
        gold_files = {page.split("#page=", 1)[0] for page in gold_pages}
        selected = {item["file_id"] for item in row["selected_files"]}
        candidate_file_recalls.append(len(selected & gold_files) / len(gold_files))
        first_three: list[str] = []
        for page in top20:
            file_id = page.split("#page=", 1)[0]
            if file_id not in first_three:
                first_three.append(file_id)
        derived_file_recalls.append(len(set(first_three) & gold_files) / len(gold_files))
        candidates.append(int(row["candidate_page_count"]))
    average = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "queries": len(rows),
        "page_recall@10": average(recalls10),
        "page_recall@20": average(recalls20),
        "page_hit@10": average(hits10),
        "ndcg@10": 100.0 * average(ndcgs10),
        "ndcg@20": 100.0 * average(ndcgs20),
        "file_candidate_recall@3": average(candidate_file_recalls),
        "derived_file_recall@3": average(derived_file_recalls),
        "mean_candidate_pages": average(candidates),
    }


def make_row(
    query: dict[str, Any], ranking: list[tuple[str, float]], pages_by_id: dict[str, dict[str, Any]],
    selected_files: list[tuple[str, float]], candidate_count: int, policy: str, page_components: dict[str, dict[str, float]],
) -> dict[str, Any]:
    chunks = []
    for rank, (page_id, score) in enumerate(ranking, 1):
        page = pages_by_id[page_id]
        chunks.append({
            "page_id": page_id, "doc_id": page["doc_id"], "source": page["source"],
            "relative_path": page["relative_path"], "page_index": page["page_index"],
            "page_number": page["page_number"], "rank": rank, "score": round(score, 8),
            "components": {key: round(value, 8) for key, value in page_components[page_id].items()},
        })
    return {
        "query_id": query["query_id"], "query": query["query"], "policy": policy,
        "file_k": len(selected_files), "metric_page_k": 10, "saved_page_k": len(chunks),
        "selected_files": [{"file_id": file_id, "score": round(score, 8)} for file_id, score in selected_files],
        "candidate_page_count": candidate_count, "chunks": chunks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/BENCHMARK")
    parser.add_argument("--qrels", type=Path, default=ROOT / "data/work/benchmark_hierarchical_local/qrels_page_level.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file-k", type=int, default=3)
    parser.add_argument("--saved-page-k", type=int, default=20)
    args = parser.parse_args()

    started = perf_counter()
    pages = read_jsonl(args.pages)
    queries = read_jsonl(args.dataset_root / "queries.jsonl")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_jsonl(args.qrels):
        qrels[str(row["query_id"])][str(row["page_id"])] = int(row.get("relevance", 1))
    if len(pages) != len({row["page_id"] for row in pages}):
        raise ValueError("duplicate page ID")
    if set(query["query_id"] for query in queries) != set(qrels):
        raise ValueError("query/qrels mismatch")
    pages_by_id = {row["page_id"]: row for row in pages}
    page_to_file = {row["page_id"]: row["doc_id"] for row in pages}
    pages_by_file: dict[str, list[str]] = defaultdict(list)
    for page in pages:
        pages_by_file[page["doc_id"]].append(page["page_id"])
    file_ids = sorted(pages_by_file)

    setup_started = perf_counter()
    page_index = BM25Index(analyzer_name="auto").build(
        {"chunk_id": page["page_id"], "doc_id": page["doc_id"], "text": page["text"]} for page in pages
    )
    file_index = BM25Index(analyzer_name="auto").build(
        {"chunk_id": file_id, "doc_id": file_id, "text": "\n".join(pages_by_id[page_id]["text"] for page_id in page_ids)}
        for file_id, page_ids in pages_by_file.items()
    )
    setup_seconds = perf_counter() - setup_started

    global_rows: list[dict[str, Any]] = []
    hierarchical_rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    retrieval_started = perf_counter()
    for query in queries:
        query_started = perf_counter()
        raw_pages = {page_index.chunk_ids[pos]: float(score) for pos, score in page_index.search(query["query"], len(pages))}
        bm25_partial = normalise(raw_pages)
        # BM25Index only returns positive postings.  Retain every page/file as
        # a zero-score candidate, matching the original cascade contract.
        bm25 = {page_id: bm25_partial.get(page_id, 0.0) for page_id in pages_by_id}
        global_rank = ranked(bm25)[:args.saved_page_k]
        global_components = {page_id: {"page_bm25": bm25[page_id], "page_base": 0.70 * bm25[page_id]} for page_id in bm25}
        # Global has no file-selection stage.  Its selected_files field is
        # diagnostic only: unique first three files in the global page order.
        global_files: list[tuple[str, float]] = []
        for page_id, score in global_rank:
            file_id = page_to_file[page_id]
            if file_id not in {item[0] for item in global_files}:
                global_files.append((file_id, score))
            if len(global_files) == 3:
                break
        global_rows.append(make_row(query, global_rank, pages_by_id, global_files, len(pages), "global_page_bm25", global_components))

        page_base = {page_id: 0.70 * score for page_id, score in bm25.items()}
        raw_files = {file_index.chunk_ids[pos]: float(score) for pos, score in file_index.search(query["query"], len(file_ids))}
        direct_partial = normalise(raw_files)
        direct = {file_id: direct_partial.get(file_id, 0.0) for file_id in file_ids}
        pooled = {file_id: max((page_base[page_id] for page_id in page_ids), default=0.0) for file_id, page_ids in pages_by_file.items()}
        pooled = normalise(pooled)
        file_scores = {file_id: 0.50 * direct[file_id] + 0.50 * pooled[file_id] for file_id in file_ids}
        selected = ranked(file_scores)[:args.file_k]
        selected_ids = {file_id for file_id, _score in selected}
        parent = normalise(file_scores)
        final = {page_id: 0.85 * page_base[page_id] + 0.15 * parent[page_to_file[page_id]] for page_id in page_base if page_to_file[page_id] in selected_ids}
        hierarchy_rank = ranked(final)[:args.saved_page_k]
        hierarchy_components = {page_id: {"page_bm25": bm25[page_id], "page_base": page_base[page_id], "parent_file": parent[page_to_file[page_id]]} for page_id in final}
        hierarchical_rows.append(make_row(query, hierarchy_rank, pages_by_id, selected, len(final), f"hierarchical_bm25_kf{args.file_k}", hierarchy_components))
        timings.append({"query_id": query["query_id"], "seconds": perf_counter() - query_started, "hierarchical_candidate_pages": len(final)})

    retrieval_seconds = perf_counter() - retrieval_started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "runs/global_page_bm25_top20.jsonl", global_rows)
    write_jsonl(args.output_dir / f"runs/hierarchical_bm25_kf{args.file_k}_top20.jsonl", hierarchical_rows)
    report = {
        "input_pages": str(args.pages.resolve()), "queries": len(queries), "files": len(file_ids), "pages": len(pages),
        "config": {"visual_mode": "disabled", "page_base": "0.70 * normalized_BM25", "file_score": "0.50 * normalized_file_BM25 + 0.50 * normalized_max_page_base", "page_final": "0.85 * page_base + 0.15 * normalized_file_score", "file_k": args.file_k, "saved_page_k": args.saved_page_k, "qrels_used_for_ranking": False},
        "metrics": {"global_page_bm25": evaluate(global_rows, qrels, 10, args.saved_page_k), f"hierarchical_bm25_kf{args.file_k}": evaluate(hierarchical_rows, qrels, 10, args.saved_page_k)},
        "timing_seconds": {"index_build": setup_seconds, "retrieval": retrieval_seconds, "mean_query": sum(row["seconds"] for row in timings) / len(timings), "total": perf_counter() - started},
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "timing_per_query.jsonl").write_text("".join(json.dumps(row) + "\n" for row in timings), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
