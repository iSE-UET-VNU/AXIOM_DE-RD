"""Evaluate the stronger c014 light hierarchy followed by legacy retrieval.

Protocol:

    c014 hybrid light retrieval -> select top-3 files -> expose all pages in
    those files -> cached legacy BM25+dense second retrieval -> top-10 pages.

This is deliberately separate from the existing c014/page100 experiment:
the latter only lets legacy reorder the first 100 light pages.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.evaluate_physics_hierarchical_legacy_kf import (  # noqa: E402
    _legacy_over_selected_files,
    _load_questions_and_qrels,
)
from research.experiments.evaluate_physics_legacy_second_retrieval import (  # noqa: E402
    _load_cached_query_vectors,
    _load_legacy_chunks,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    HierarchicalRetriever,
    _build_corpus_and_indexes,
    _derived_metrics,
    _load_visual_scores,
    _page_vector_units,
    _write_run,
)
from research.data_discovery.hierarchical import CascadeConfig  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
LEGACY_CACHE = ROOT / "data/work/vidore_physics_emb"
OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_c014_legacy_file_scope"

K_FILES = 3
LEGACY_WEIGHT = 0.25


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _c014_config() -> CascadeConfig:
    return CascadeConfig(
        name="c014-kf3-all-pages",
        file_pool="sum_top2",
        k_files=K_FILES,
        page_depth=1674,
        final_depth=100,
        bm25_weight=0.60,
        parent_weight=0.25,
        file_direct_weight=0.50,
        file_pool_source="page_base",
        fine_unit="none",
        fine_weight=0.0,
    )


def main() -> None:
    started = time.perf_counter()
    questions, qids, qrels = _load_questions_and_qrels(BENCHMARK_ROOT)
    page_ids = _page_vector_units(PAGE_VECTOR_DIR)
    corpus, page_index, file_indexes, _, index_counts = _build_corpus_and_indexes(
        PARSED_RUN, page_ids, fine_units=[]
    )
    visual_scores, visual_meta = _load_visual_scores(
        PAGE_VECTOR_DIR, QUERY_VECTOR_DIR, page_ids, qids
    )
    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes={},
        visual_scores=visual_scores,
    )

    config = _c014_config()
    base_runs: dict[str, list[dict[str, Any]]] = {}
    base_traces: dict[str, dict[str, Any]] = {}
    for question in questions:
        run, trace = retriever.retrieve(question.qid, question.query, config)
        base_runs[question.qid] = run
        base_traces[question.qid] = trace

    chunks, legacy_meta = _load_legacy_chunks(PARSED_RUN, corpus)
    query_vectors, query_meta = _load_cached_query_vectors(questions, LEGACY_CACHE)
    chunk_matrix = np.asarray([chunk.vector for chunk in chunks], dtype=np.float32)
    chunk_matrix /= np.clip(np.linalg.norm(chunk_matrix, axis=1, keepdims=True), 1e-12, None)
    query_matrix = np.asarray([query_vectors[qid] for qid in qids], dtype=np.float32)
    dense_score_matrix = chunk_matrix @ query_matrix.T
    chunk_index = BM25Index(analyzer_name="auto").build(
        [{"chunk_id": chunk.chunk_id, "doc_id": chunk.file_id, "text": chunk.text} for chunk in chunks]
    )
    by_chunk_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunk_positions_by_page: dict[str, list[int]] = {}
    for position, chunk in enumerate(chunks):
        for page_id in chunk.page_ids:
            chunk_positions_by_page.setdefault(page_id, []).append(position)

    second_runs, rows = _legacy_over_selected_files(
        base_traces,
        questions,
        qids,
        qrels,
        corpus,
        chunks,
        by_chunk_id,
        chunk_index,
        dense_score_matrix,
        {qid: index for index, qid in enumerate(qids)},
        chunk_positions_by_page,
        K_FILES,
    )

    base_metrics = _derived_metrics(base_runs, qids, qrels)
    second_metrics = _derived_metrics(second_runs, qids, qrels)
    scope_metrics = {
        "avg_selected_files": float(np.mean([len(row["selected_files"]) for row in rows])),
        "avg_candidate_pages": float(np.mean([row["candidate_page_count"] for row in rows])),
        "file_scope_recall": float(np.mean([row["file_scope_recall"] for row in rows])),
        "page_pool_recall_ceiling": float(np.mean([row["page_pool_recall_ceiling"] for row in rows])),
        "avg_legacy_chunk_pages": float(np.mean([row["legacy_chunk_pages"] for row in rows])),
        "avg_legacy_chunk_hits": float(np.mean([row["legacy_chunk_hits"] for row in rows])),
    }

    output_dir = OUTPUT_DIR
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    _write_run(
        runs_dir / "c014_kf3_all_pages_before_legacy.jsonl",
        base_runs,
        qids,
        queries={question.qid: question.query for question in questions},
    )
    _write_run(
        runs_dir / "c014_kf3_all_pages_legacy_second.jsonl",
        second_runs,
        qids,
        queries={question.qid: question.query for question in questions},
    )
    _write_jsonl(output_dir / "per_query.jsonl", rows)
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "config": {
            "light": "c014: file_pool=sum_top2, BM25=0.60, V-SPLADE=0.40, parent=0.25, Kf=3",
            "scope": "all pages in selected top-3 files",
            "legacy": "BM25+dense alpha=0.70, depth=100, max page pooling, gamma=0.25",
            "qrels_used_for_ranking": False,
        },
        "cache": {
            "legacy": legacy_meta,
            "query_embeddings": query_meta,
            "visual": visual_meta,
        },
        "index_counts": index_counts,
        "light_metrics": base_metrics,
        "scope_metrics": scope_metrics,
        "second_metrics": second_metrics,
        "timing_seconds": {
            "total": time.perf_counter() - started,
            "note": "Cached artifacts only; excludes parsing and model encoding.",
        },
        "reference": {
            "c014_page100_light_page_recall@10": 0.4512,
            "c014_page100_plus_legacy_gamma0.20_page_recall@10": 0.4537,
        },
    }
    _write_json(output_dir / "report.json", report)
    lines = [
        "# Physics c014 light retrieval + legacy second over selected-file pages",
        "",
        "c014 selects the top 3 files; every page in those files is then passed to the cached legacy second retriever.",
        "",
        "| Method | Avg pages/query | File scope recall | Page ceiling | nDCG@10 | Page recall@10 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| c014 light | {scope_metrics['avg_candidate_pages']:.2f} | {scope_metrics['file_scope_recall']:.2%} | {scope_metrics['page_pool_recall_ceiling']:.2%} | {base_metrics['ndcg@10']:.2f} | {base_metrics['page_recall@10']:.2%} |",
        f"| c014 + legacy second | {scope_metrics['avg_candidate_pages']:.2f} | {scope_metrics['file_scope_recall']:.2%} | {scope_metrics['page_pool_recall_ceiling']:.2%} | {second_metrics['ndcg@10']:.2f} | {second_metrics['page_recall@10']:.2%} |",
        "",
        "The legacy second stage cannot recover a page outside the three selected files.",
        "The earlier c014 page100 + legacy gamma=0.20 reference is reported separately; it is not this all-pages scope.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_dir),
        "elapsed_seconds": round(report["timing_seconds"]["total"], 3),
        "light_page_recall@10": round(100 * base_metrics["page_recall@10"], 2),
        "second_page_recall@10": round(100 * second_metrics["page_recall@10"], 2),
        "avg_candidate_pages": round(scope_metrics["avg_candidate_pages"], 2),
        "page_ceiling": round(100 * scope_metrics["page_pool_recall_ceiling"], 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
