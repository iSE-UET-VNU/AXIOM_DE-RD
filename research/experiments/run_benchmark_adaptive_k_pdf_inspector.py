"""Run and evaluate Adaptive-k using PDF-inspector text only.

The benchmark cache contains a later Tesseract fallback for some pages.  This
runner reconstructs BM25 indexes in memory while retaining only pages whose
cached ``text_source`` is ``pdf_inspector``.  It then applies the upstream
Adaptive-k largest-gap rule with the same settings as the earlier adapter.

No OCR, rendering, embedding, or API call is performed.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.raw_hierarchical import (  # noqa: E402
    RawPage,
    read_jsonl,
)
from research.experiments.evaluate_benchmark_adaptive_metrics import (  # noqa: E402
    _evaluate_rows,
    _write_json,
    _write_jsonl,
)
from research.experiments.run_benchmark_adaptive_k import adaptive_k  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402

DEFAULT_ROOT = ROOT / "data/work/benchmark_hierarchical_local"
DEFAULT_DATASET_ROOT = ROOT / "data/raw/BENCHMARK"
DEFAULT_OUTPUT = DEFAULT_ROOT / "adaptive_k_pdf_inspector_only"


def _build_indexes(
    pages: Sequence[RawPage],
    inspector_text: Mapping[str, str],
) -> BM25Index:
    page_index = BM25Index(analyzer_name="auto").build(
        {
            "chunk_id": page.page_id,
            "doc_id": page.doc_id,
            "text": inspector_text.get(page.page_id, ""),
        }
        for page in pages
        if page.page_id in inspector_text
    )
    return page_index


def _make_row(
    query_id: str,
    query: str,
    ranking: Sequence[tuple[str, float]],
    pages_by_id: Mapping[str, RawPage],
    k: int,
    adaptive_info: Mapping[str, Any] | None = None,
    policy: str = "adaptive_k_pdf_inspector_bm25",
) -> dict[str, Any]:
    returned = [page_id for page_id, _score in ranking[:k]]
    chunks = []
    for rank, (page_id, score) in enumerate(ranking[:k], start=1):
        page = pages_by_id[page_id]
        chunks.append(
            {
                "page_id": page_id,
                "doc_id": page.doc_id,
                "source": page.source,
                "relative_path": page.relative_path,
                "page_index": page.page_index,
                "page_number": page.page_number,
                "rank": rank,
                "score": round(float(score), 8),
            }
        )
    return {
        "query_id": query_id,
        "query": query,
        "policy": policy,
        "kp": len(returned),
        "selected_files": [],
        "candidate_page_count": len(ranking),
        "returned_pages": returned,
        "chunks": chunks,
        "adaptive_k": dict(adaptive_info) if adaptive_info is not None else None,
        "score_source": "BM25 over PDF-inspector page text; global page retrieval",
        "page_scores_returned": [round(float(score), 8) for _page_id, score in ranking[:k]],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ignore-extreme", type=int, default=0)
    parser.add_argument("--ignore-extreme-tail", type=float, default=0.1)
    parser.add_argument("--retrieve-more", type=int, default=5)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    cache = args.cache_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()

    page_rows = read_jsonl(cache / "parsing/pages.jsonl")
    pages = [RawPage(**row) for row in page_rows]
    queries = read_jsonl(dataset_root / "queries.jsonl")
    file_ids = sorted({page.doc_id for page in pages})
    pages_by_id = {page.page_id: page for page in pages}
    inspector_text = {
        str(row["page_id"]): str(row.get("text") or "")
        for row in page_rows
        if str(row.get("text_source") or "") == "pdf_inspector"
    }
    source_counts = defaultdict(int)
    for row in page_rows:
        source_counts[str(row.get("text_source") or "empty")] += 1
    page_index = _build_indexes(pages, inspector_text)

    normalized_qrels = read_jsonl(cache / "qrels_page_level.jsonl")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in normalized_qrels:
        qrels[str(row["query_id"])][str(row["page_id"])] = int(row.get("relevance", 1))

    fixed10_rows: list[dict[str, Any]] = []
    fixed20_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query["query_id"])
        query_text = str(query["query"])
        page_hits = page_index.search(query_text, len(pages))
        ranking = [(page_index.chunk_ids[position], float(score)) for position, score in page_hits]
        adaptive_info = adaptive_k(
            [score for _page_id, score in ranking],
            ignore_extreme=args.ignore_extreme,
            ignore_extreme_tail=args.ignore_extreme_tail,
            retrieve_more=args.retrieve_more,
        )
        fixed10_rows.append(
            _make_row(
                query_id,
                query_text,
                ranking,
                pages_by_id,
                10,
                policy="fixed_global_kp10_pdf_inspector_bm25",
            )
        )
        fixed20_rows.append(
            _make_row(
                query_id,
                query_text,
                ranking,
                pages_by_id,
                20,
                policy="fixed_global_kp20_pdf_inspector_bm25",
            )
        )
        adaptive_rows.append(
            _make_row(
                query_id,
                query_text,
                ranking,
                pages_by_id,
                int(adaptive_info["k"]),
                adaptive_info,
            )
        )

    fixed10_metrics, _fixed10_per_query = _evaluate_rows(fixed10_rows, qrels)
    fixed20_metrics, _fixed20_per_query = _evaluate_rows(fixed20_rows, qrels)
    metrics, per_query = _evaluate_rows(adaptive_rows, qrels)
    report = {
        "contract_version": "benchmark-adaptive-k-pdf-inspector-only-v1",
        "cache_dir": str(cache),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "queries": len(queries),
        "files": len(file_ids),
        "pages": len(pages),
        "text_source": {
            "indexed_only": "pdf_inspector",
            "page_counts": dict(sorted(source_counts.items())),
            "indexed_page_count": len(inspector_text),
            "excluded_page_count": len(pages) - len(inspector_text),
        },
        "retrieval": {
            "page_index": "BM25 over PDF-inspector page text",
            "retrieval_scope": "global all-page index; no hierarchical file stage",
            "page_score": "raw BM25 page score",
            "visual_mode": "disabled",
            "qrels_used_for_ranking": False,
        },
        "adaptive_k": {
            "implementation": "upstream largest descending-score gap adapter",
            "ignore_extreme": args.ignore_extreme,
            "ignore_extreme_tail": args.ignore_extreme_tail,
            "retrieve_more": args.retrieve_more,
        },
        "metrics": {
            "fixed_global_kp10": fixed10_metrics,
            "fixed_global_kp20": fixed20_metrics,
            "adaptive": metrics,
        },
        "timing_seconds": {"offline_index_and_retrieval": round(time.perf_counter() - started, 3)},
        "no_new_inference": True,
    }
    _write_json(output_dir / "report.json", report)
    _write_jsonl(output_dir / "fixed_global_kp10_per_query.jsonl", fixed10_rows)
    _write_jsonl(output_dir / "fixed_global_kp20_per_query.jsonl", fixed20_rows)
    _write_jsonl(output_dir / "adaptive_per_query.jsonl", adaptive_rows)
    _write_jsonl(output_dir / "adaptive_metrics_per_query.jsonl", per_query)

    lines = [
        "# Adaptive-k with PDF-inspector text + BM25",
        "",
        f"- Benchmark: **{len(file_ids)} files**, **{len(pages)} pages**, **{len(queries)} queries**.",
        "- Text/indexing: only cached `text_source=pdf_inspector`; Tesseract and empty-page text were excluded.",
        f"- Indexed pages: **{len(inspector_text)}**; excluded pages: **{len(pages) - len(inspector_text)}**.",
        "- Retrieval: one global BM25 page index over all pages; no hierarchical file stage.",
        f"- Adaptive-k: upstream largest-score-gap rule; tail ignored={args.ignore_extreme_tail}; retrieve_more={args.retrieve_more}.",
        "- No OCR, rendering, embeddings, or API call was performed; qrels were used only for evaluation.",
        "",
        "## Results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Fixed global Kp=10 page recall@10 | {fixed10_metrics['page_recall@10']:.2%} |",
        f"| Fixed global Kp=20 page recall@10 | {fixed20_metrics['page_recall@10']:.2%} |",
        f"| Fixed global Kp=20 page recall@20 | {fixed20_metrics['page_recall@20']:.2%} |",
        f"| Adaptive page recall at returned K | {metrics['page_recall_at_returned_k']:.2%} |",
        f"| Adaptive page recall@10 | {metrics['page_recall@10']:.2%} |",
        f"| Adaptive page recall@20 | {metrics['page_recall@20']:.2%} |",
        f"| Adaptive nDCG@10 | {metrics['ndcg@10']:.2f} |",
        f"| Adaptive nDCG@20 | {metrics['ndcg@20']:.2f} |",
        f"| Adaptive file recall from union of all returned pages | {metrics['file_recall_from_returned_file_union']:.2%} |",
        f"| Adaptive file recall@3 from returned page union | {metrics['file_recall@3_from_returned_file_union']:.2%} |",
        f"| Mean Adaptive Kp | {metrics['mean_adaptive_k']:.2f} |",
        f"| Mean unique files in returned page union | {metrics['mean_unique_returned_files']:.2f} |",
        "",
        "## Dataset breakdown",
        "",
        "| Source | Queries | Recall at Adaptive K | Recall@20 | nDCG@10 | nDCG@20 | File recall from returned union |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_query:
        grouped[str(row["source"])].append(row)
    for source, rows in sorted(grouped.items()):
        lines.append(
            f"| {source} | {len(rows)} | {_mean([float(row['page_recall_at_returned_k']) for row in rows]):.2%} | {_mean([float(row['page_recall@20']) for row in rows]):.2%} | {_mean([float(row['ndcg@10']) for row in rows]):.2f} | {_mean([float(row['ndcg@20']) for row in rows]):.2f} | {_mean([float(row['file_recall_from_returned_file_union']) for row in rows]):.2%} |"
        )
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
