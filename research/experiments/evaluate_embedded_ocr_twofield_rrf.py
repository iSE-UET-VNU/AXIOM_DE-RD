"""Two-field BM25/RRF experiment for native PDF text and embedded-image OCR.

The two fields are indexed independently to avoid BM25 length-normalisation
of long native pages diluting a short, exact text hit recovered from an
embedded image.  Fusion is standard rank-only RRF (k=60); no relevance data
or fitted component weight is used in ranking.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.sparse import BM25Index  # noqa: E402
from research.experiments.evaluate_embedded_ocr_bm25 import evaluate, read_jsonl, write_jsonl  # noqa: E402


def ranked_positive(scores: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(((key, float(value)) for key, value in scores.items() if value > 0), key=lambda item: (-item[1], item[0]))


def rrf(
    score_fields: Sequence[Mapping[str, float]],
    *,
    k: int,
    allowed: set[str] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, int | None]]]:
    """Fuse only positive BM25 matches; zero-score ties never get an RRF vote."""
    fused: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int | None]] = defaultdict(dict)
    for field_index, scores in enumerate(score_fields):
        entries = ranked_positive({key: value for key, value in scores.items() if allowed is None or key in allowed})
        for rank, (key, _score) in enumerate(entries, 1):
            fused[key] += 1.0 / (k + rank)
            ranks[key][f"field_{field_index}_rank"] = rank
    return dict(fused), dict(ranks)


def full_scores(index: BM25Index, query: str) -> dict[str, float]:
    return {index.chunk_ids[pos]: float(score) for pos, score in index.search(query, len(index.chunk_ids))}


def make_row(
    query: Mapping[str, Any], ranking: Sequence[tuple[str, float]], pages: Mapping[str, Mapping[str, Any]],
    selected_files: Sequence[tuple[str, float]], candidate_count: int, policy: str,
    rank_components: Mapping[str, Mapping[str, int | None]],
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    for rank, (page_id, score) in enumerate(ranking, 1):
        page = pages[page_id]
        chunks.append({
            "page_id": page_id, "doc_id": page["doc_id"], "source": page["source"],
            "relative_path": page["relative_path"], "page_index": page["page_index"],
            "page_number": page["page_number"], "rank": rank, "score": round(float(score), 10),
            "components": dict(rank_components.get(page_id, {})),
        })
    return {
        "query_id": query["query_id"], "query": query["query"], "policy": policy,
        "file_k": len(selected_files), "metric_page_k": 10, "saved_page_k": len(chunks),
        "selected_files": [{"file_id": file_id, "score": round(float(score), 10)} for file_id, score in selected_files],
        "candidate_page_count": candidate_count, "chunks": chunks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-pages", type=Path, required=True)
    parser.add_argument("--ocr-results", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/BENCHMARK")
    parser.add_argument("--qrels", type=Path, default=ROOT / "data/work/benchmark_hierarchical_local/qrels_page_level.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file-k", type=int, default=3)
    parser.add_argument("--saved-page-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()
    if args.file_k < 1 or args.saved_page_k < 10 or args.rrf_k < 1:
        raise ValueError("file-k and RRF/page depths must be positive; saved depth must be >=10")

    started = perf_counter()
    pages = read_jsonl(args.native_pages)
    queries = read_jsonl(args.dataset_root / "queries.jsonl")
    page_by_id = {str(page["page_id"]): page for page in pages}
    if len(page_by_id) != len(pages):
        raise ValueError("native page cache has duplicate page IDs")
    image_text: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(args.ocr_results):
        if row.get("status") == "ok" and str(row.get("text") or "").strip():
            image_text[str(row["page_id"])].append(str(row["text"]).strip())
    page_to_file = {page_id: str(page["doc_id"]) for page_id, page in page_by_id.items()}
    page_ids_by_file: dict[str, list[str]] = defaultdict(list)
    for page_id, file_id in page_to_file.items():
        page_ids_by_file[file_id].append(page_id)
    file_ids = sorted(page_ids_by_file)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_jsonl(args.qrels):
        qrels[str(row["query_id"])][str(row["page_id"])] = int(row.get("relevance", 1))
    if {str(query["query_id"]) for query in queries} != set(qrels):
        raise ValueError("queries/qrels mismatch")

    index_started = perf_counter()
    native_page_index = BM25Index(analyzer_name="auto").build(
        {"chunk_id": page_id, "doc_id": page_to_file[page_id], "text": str(page_by_id[page_id]["text"])}
        for page_id in page_by_id
    )
    ocr_page_index = BM25Index(analyzer_name="auto").build(
        {"chunk_id": page_id, "doc_id": page_to_file[page_id], "text": "\n".join(image_text.get(page_id, []))}
        for page_id in page_by_id
    )
    native_file_index = BM25Index(analyzer_name="auto").build(
        {"chunk_id": file_id, "doc_id": file_id, "text": "\n".join(str(page_by_id[page_id]["text"]) for page_id in page_ids)}
        for file_id, page_ids in page_ids_by_file.items()
    )
    ocr_file_index = BM25Index(analyzer_name="auto").build(
        {"chunk_id": file_id, "doc_id": file_id, "text": "\n".join("\n".join(image_text.get(page_id, [])) for page_id in page_ids)}
        for file_id, page_ids in page_ids_by_file.items()
    )
    index_seconds = perf_counter() - index_started

    global_rows: list[dict[str, Any]] = []
    hierarchical_rows: list[dict[str, Any]] = []
    query_timing: list[dict[str, Any]] = []
    retrieve_started = perf_counter()
    for query in queries:
        t0 = perf_counter()
        native_pages = full_scores(native_page_index, str(query["query"]))
        ocr_pages = full_scores(ocr_page_index, str(query["query"]))
        global_fused, global_ranks = rrf((native_pages, ocr_pages), k=args.rrf_k)
        global_ranked = sorted(global_fused.items(), key=lambda item: (-item[1], item[0]))[:args.saved_page_k]
        global_files: list[tuple[str, float]] = []
        seen_files: set[str] = set()
        for page_id, score in global_ranked:
            file_id = page_to_file[page_id]
            if file_id not in seen_files:
                seen_files.add(file_id)
                global_files.append((file_id, score))
            if len(global_files) == args.file_k:
                break
        global_rows.append(make_row(query, global_ranked, page_by_id, global_files, len(page_by_id), "global_twofield_bm25_rrf", global_ranks))

        # File score has two independent views.  In each view, direct file
        # BM25 and its strongest page hit receive equal rank-based support.
        native_direct = full_scores(native_file_index, str(query["query"]))
        ocr_direct = full_scores(ocr_file_index, str(query["query"]))
        native_pool = {file_id: max((native_pages.get(page_id, 0.0) for page_id in page_ids), default=0.0) for file_id, page_ids in page_ids_by_file.items()}
        ocr_pool = {file_id: max((ocr_pages.get(page_id, 0.0) for page_id in page_ids), default=0.0) for file_id, page_ids in page_ids_by_file.items()}
        native_file, _ = rrf((native_direct, native_pool), k=args.rrf_k)
        ocr_file, _ = rrf((ocr_direct, ocr_pool), k=args.rrf_k)
        file_fused, _ = rrf((native_file, ocr_file), k=args.rrf_k)
        selected = sorted(file_fused.items(), key=lambda item: (-item[1], item[0]))[:args.file_k]
        selected_ids = {file_id for file_id, _score in selected}
        candidates = {page_id for file_id in selected_ids for page_id in page_ids_by_file[file_id]}
        page_fused, page_ranks = rrf((native_pages, ocr_pages), k=args.rrf_k, allowed=candidates)
        hierarchical_ranked = sorted(page_fused.items(), key=lambda item: (-item[1], item[0]))[:args.saved_page_k]
        hierarchical_rows.append(make_row(query, hierarchical_ranked, page_by_id, selected, len(candidates), f"hierarchical_twofield_bm25_rrf_kf{args.file_k}", page_ranks))
        query_timing.append({"query_id": query["query_id"], "seconds": perf_counter() - t0, "candidate_pages": len(candidates)})

    retrieval_seconds = perf_counter() - retrieve_started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "runs/global_twofield_bm25_rrf_top20.jsonl", global_rows)
    write_jsonl(args.output_dir / f"runs/hierarchical_twofield_bm25_rrf_kf{args.file_k}_top20.jsonl", hierarchical_rows)
    report = {
        "input": {"native_pages": str(args.native_pages.resolve()), "ocr_results": str(args.ocr_results.resolve()), "queries": len(queries), "files": len(file_ids), "pages": len(pages), "pages_with_image_ocr": len(image_text), "image_ocr_records": sum(len(items) for items in image_text.values())},
        "config": {"native_field": "original PDF-inspector/Tesseract page text", "image_ocr_field": "embedded image OCR only", "fusion": f"RRF(k={args.rrf_k})", "file_score": "RRF(RRF(native direct-file, native max-page), RRF(ocr direct-file, ocr max-page))", "page_score": "RRF(native page, image-OCR page) within selected files", "file_k": args.file_k, "saved_page_k": args.saved_page_k, "qrels_used_for_ranking": False},
        "metrics": {"global_twofield_bm25_rrf": evaluate(global_rows, qrels, 10, args.saved_page_k), f"hierarchical_twofield_bm25_rrf_kf{args.file_k}": evaluate(hierarchical_rows, qrels, 10, args.saved_page_k)},
        "timing_seconds": {"index_build": index_seconds, "retrieval": retrieval_seconds, "mean_query": sum(item["seconds"] for item in query_timing) / len(query_timing), "total": perf_counter() - started},
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(args.output_dir / "timing_per_query.jsonl", query_timing)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
