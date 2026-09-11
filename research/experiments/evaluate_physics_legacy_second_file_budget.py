"""Screen larger file budgets followed by the cached legacy second retriever.

This is a retrieval-only extension of the existing Physics legacy second-stage
experiment.  It evaluates the fixed cascade configurations Kf=5 and Kf=10,
then re-ranks all pages belonging to those files with the cached legacy
text-chunk BM25 + dense retriever.  No parser, embedding API, LLM or network
call is used.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import CascadeConfig
from research.experiments.evaluate_physics_legacy_second_retrieval import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_LEGACY_CACHE,
    DEFAULT_PAGE_VECTOR_DIR,
    DEFAULT_PARSED_RUN,
    DEFAULT_QUERY_VECTOR_DIR,
    _load_cached_query_vectors,
    _load_legacy_chunks,
    _aggregate_hits_to_pages,
    _base_page_scores,
    _retrieve_legacy_chunks_for_query,
    _second_stage_metrics,
)
from research.experiments.physics_hierarchical_retrieval import (
    HierarchicalRetriever,
    _build_corpus_and_indexes,
    _derived_metrics,
    _file_id,
    _load_visual_scores,
    _page_vector_units,
    _paired_comparison,
    _safe_name,
    _stage_metrics,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3
from src.retrieval.sparse import BM25Index
from research.data_discovery.hierarchical import normalise_scores, sort_scores
import numpy as np


OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_legacy_second_file_budget"
LEGACY_GAMMA = 0.25
LEGACY_ALPHA = 0.70
LEGACY_DEPTH = 100
FILE_BUDGETS = (3, 5, 10)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_config(k_files: int, page_depth: int) -> CascadeConfig:
    return CascadeConfig(
        name=f"full-cascade-kf{k_files}",
        file_representation="all_text",
        file_pool="max",
        k_files=k_files,
        # Keep all pages from the selected files available to the second stage.
        # final_depth remains 100, so the base arm's evaluated output is still
        # top-100 pages.
        page_depth=page_depth,
        final_depth=100,
        bm25_weight=0.70,
        parent_weight=0.15,
        file_direct_weight=0.50,
        file_pool_source="page_base",
        fine_unit="none",
        fine_pool="max",
        fine_weight=0.0,
    )


def _summary_row(name: str, method: dict[str, Any]) -> dict[str, Any]:
    metrics = method["retrieval_metrics"]
    stage = method.get("stage_metrics") or {}
    comparison = method.get("comparison_to_baseline") or {}
    return {
        "method": name,
        "ndcg@10": round(float(metrics["ndcg@10"]), 4),
        "page_recall@10": round(100.0 * float(metrics["page_recall@10"]), 4),
        "page_hit@10": round(100.0 * float(metrics["page_hit@10"]), 4),
        "file_recall@3": round(100.0 * float(metrics["file_metrics_by_k"]["3"]["file_recall"]), 4),
        "file_hit@3": round(100.0 * float(metrics["file_metrics_by_k"]["3"]["file_hit"]), 4),
        "file_recall@10": round(100.0 * float(metrics["file_metrics_by_k"]["10"]["file_recall"]), 4),
        "file_hit@10": round(100.0 * float(metrics["file_metrics_by_k"]["10"]["file_hit"]), 4),
        "file_candidate_recall@5": round(100.0 * float(stage.get("file_candidate_recall", {}).get("@5", 0.0)), 4),
        "file_candidate_recall@10": round(100.0 * float(stage.get("file_candidate_recall", {}).get("@10", 0.0)), 4),
        "page_pool_recall": round(100.0 * float(stage.get("page_pool_recall", 0.0)), 4),
        "legacy_chunk_page_recall_proxy": round(100.0 * float(stage.get("legacy_chunk_page_recall_proxy", 0.0)), 4),
        "page_delta_pp": round(float(comparison.get("page_recall_delta_pp", 0.0)), 4),
        "file_delta_pp": round(float(comparison.get("file_recall_delta_pp", 0.0)), 4),
    }


def _legacy_over_selected_files(
    base_traces: dict[str, dict[str, Any]],
    questions: dict[str, Any],
    chunks: list[Any],
    chunk_index: BM25Index,
    chunk_matrix: np.ndarray,
    query_vectors: dict[str, np.ndarray],
    corpus: Any,
    *,
    k_files: int,
    gamma: float,
    alpha: float,
    depth: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Run legacy retrieval over every page in the selected files.

    The original second-stage experiment scopes legacy retrieval to the first
    100 pages emitted by the first cascade.  This function deliberately uses
    all pages belonging to the selected Kf files, which is the file-budget
    hypothesis under test here.
    """

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    page_to_file = {page_id: _file_id(page_id) for page_id in corpus.page_order}
    runs: dict[str, list[dict[str, Any]]] = {}
    traces: dict[str, dict[str, Any]] = {}
    for qid, question in questions.items():
        base_trace = base_traces[qid]
        selected_files = {
            str(row["node_id"])
            for row in base_trace.get("file_candidates", [])[:k_files]
        }
        allowed_pages = {
            page_id
            for page_id in corpus.page_order
            if page_to_file[page_id] in selected_files
        }
        output_pages = [page_id for page_id in corpus.page_order if page_id in allowed_pages]

        base_scores = {
            page_id: score
            for page_id, score in _base_page_scores(base_trace).items()
            if page_id in allowed_pages
        }
        base_norm = normalise_scores(base_scores)
        hits = _retrieve_legacy_chunks_for_query(
            chunks,
            chunk_index,
            chunk_matrix,
            question.query,
            query_vectors[qid],
            allowed_pages,
            alpha=alpha,
            depth=depth,
        )
        legacy_scores = _aggregate_hits_to_pages(hits, by_id, allowed_pages, "max")
        legacy_norm = normalise_scores(legacy_scores)
        final_scores = {
            page_id: (1.0 - gamma) * base_norm.get(page_id, 0.0)
            + gamma * legacy_norm.get(page_id, 0.0)
            for page_id in output_pages
        }
        ranked = sort_scores(final_scores)[:100]
        runs[qid] = [
            {
                "chunk_id": page_id,
                "doc_id": page_id,
                "text": corpus.pages[page_id].text,
                "score": round(float(score), 8),
                "rank": rank,
                "scores": {
                    "page_base": round(float(base_norm.get(page_id, 0.0)), 8),
                    "legacy_chunk_score": round(float(legacy_norm.get(page_id, 0.0)), 8),
                    "final_score": round(float(score), 8),
                },
            }
            for rank, (page_id, score) in enumerate(ranked, 1)
        ]
        chunk_pages = {
            page_id
            for chunk_id, _score in hits
            for page_id in by_id[chunk_id].page_ids
            if page_id in allowed_pages
        }
        traces[qid] = {
            "qid": qid,
            "scope": "selected_files_all_pages",
            "k_files": k_files,
            "selected_files": sorted(selected_files),
            "page_pool": output_pages,
            "legacy_chunk_hits": [
                {
                    "chunk_id": chunk_id,
                    "page_ids": list(by_id[chunk_id].page_ids),
                    "score": round(float(score), 8),
                }
                for chunk_id, score in hits
            ],
            "legacy_chunk_pages": sorted(chunk_pages),
            "counts": {
                "selected_files": len(selected_files),
                "allowed_pages": len(allowed_pages),
                "legacy_chunks_scoped": sum(
                    1 for chunk in chunks if allowed_pages.intersection(chunk.page_ids)
                ),
                "legacy_chunk_hits": len(hits),
                "legacy_chunk_pages": len(chunk_pages),
                "final_pages": len(ranked),
            },
        }
    return runs, traces


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Physics file-budget + legacy second retrieval",
        "",
        "Offline screening of larger hard file budgets followed by cached legacy text-chunk retrieval.",
        "All page BM25 inputs are PDF-inspector extracted text; no raw PDF is indexed.",
        "",
        f"Legacy parameters: alpha={LEGACY_ALPHA}, gamma={LEGACY_GAMMA}, depth={LEGACY_DEPTH}, scope=selected-files-all-pages, pool=max.",
        "",
        "| Method | nDCG@10 | Page recall@10 | File recall@3 | File recall@10 | File candidate recall@10 | Page-pool recall | Legacy chunk-page proxy | Δ page pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['method']} | {row['ndcg@10']:.2f} | {row['page_recall@10']:.2f}% | "
            f"{row['file_recall@3']:.2f}% | {row['file_recall@10']:.2f}% | "
            f"{row['file_candidate_recall@10']:.2f}% | {row['page_pool_recall']:.2f}% | "
            f"{row['legacy_chunk_page_recall_proxy']:.2f}% | {row['page_delta_pp']:+.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `file_candidate_recall` measures whether the hard file stage preserved gold files before page ranking.",
        "- `file_recall@3` remains the repository's fixed page-derived metric: first 3 unique files from the first 100 final pages.",
        "- The legacy stage searches every page in the selected Kf files; it cannot recover a page discarded by the file stage.",
        "- Gamma=0.25 is fixed from the prior Kf=3 screening; this run is a sensitivity study, not a fresh hyperparameter search.",
        "",
        "Runs and per-query traces are stored under `runs/` and `traces/`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.perf_counter()
    benchmark = ViDoreV3(root=DEFAULT_BENCHMARK_ROOT, subset="physics", language="french")
    questions_list = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    questions = {question.qid: question for question in questions_list}
    qids = [question.qid for question in questions_list]
    qrels = benchmark.qrels()
    page_ids = _page_vector_units(DEFAULT_PAGE_VECTOR_DIR)
    if len(qids) != 302 or len(page_ids) != 1674:
        raise RuntimeError(f"Expected 302 queries and 1,674 pages, got {len(qids)} and {len(page_ids)}")

    corpus, page_index, file_indexes, fine_indexes, index_counts = _build_corpus_and_indexes(
        DEFAULT_PARSED_RUN, page_ids, fine_units=[]
    )
    visual_scores, visual_meta = _load_visual_scores(
        DEFAULT_PAGE_VECTOR_DIR, DEFAULT_QUERY_VECTOR_DIR, page_ids, qids
    )
    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes=fine_indexes,
        visual_scores=visual_scores,
    )

    chunks, legacy_meta = _load_legacy_chunks(DEFAULT_PARSED_RUN, corpus)
    query_vectors, query_meta = _load_cached_query_vectors(questions_list, DEFAULT_LEGACY_CACHE)
    chunk_matrix = np.asarray([chunk.vector for chunk in chunks], dtype=np.float32)
    chunk_matrix /= np.clip(np.linalg.norm(chunk_matrix, axis=1, keepdims=True), 1e-12, None)
    chunk_index = BM25Index(analyzer_name="auto").build(
        [{"chunk_id": chunk.chunk_id, "doc_id": chunk.file_id, "text": chunk.text} for chunk in chunks]
    )

    baseline_path = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl"
    baseline_run = {}
    for line in baseline_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            baseline_run[str(row["qid"])] = list(row["chunks"])
    baseline_metrics = _derived_metrics(baseline_run, qids, qrels)

    output_dir = OUTPUT_DIR
    runs_dir = output_dir / "runs"
    traces_dir = output_dir / "traces"
    runs_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    methods: dict[str, dict[str, Any]] = {
        "PDF-inspector + BM25 baseline": {
            "config": None,
            "retrieval_metrics": baseline_metrics,
            "stage_metrics": {},
        }
    }

    for k_files in FILE_BUDGETS:
        config = _base_config(k_files, page_depth=len(page_ids))
        base_name = f"full-cascade-kf{k_files}"
        base_runs: dict[str, list[dict[str, Any]]] = {}
        base_traces: dict[str, dict[str, Any]] = {}
        for qid, question in questions.items():
            run, trace = retriever.retrieve(qid, question.query, config)
            base_runs[qid] = run
            base_traces[qid] = trace

        base_metrics = _derived_metrics(base_runs, qids, qrels)
        base_stage = _stage_metrics(base_traces, qids, qrels)
        methods[base_name] = {
            "config": asdict(config),
            "retrieval_metrics": base_metrics,
            "stage_metrics": base_stage,
        }
        _write_run(runs_dir / f"{_safe_name(base_name)}.jsonl", base_runs, qids, queries={qid: questions[qid].query for qid in qids})

        second_name = f"{base_name}-legacy-second-selected-files-max-gamma{LEGACY_GAMMA:g}"
        second_runs, second_traces = _legacy_over_selected_files(
            base_traces,
            questions,
            chunks,
            chunk_index,
            chunk_matrix,
            query_vectors,
            corpus,
            k_files=k_files,
            gamma=LEGACY_GAMMA,
            alpha=LEGACY_ALPHA,
            depth=LEGACY_DEPTH,
        )
        methods[second_name] = {
            "config": {
                "base_config": asdict(config),
                "scope": "selected_files_all_pages",
                "pool": "max",
                "gamma": LEGACY_GAMMA,
                "alpha": LEGACY_ALPHA,
                "depth": LEGACY_DEPTH,
            },
            "retrieval_metrics": _derived_metrics(second_runs, qids, qrels),
            "stage_metrics": {
                **base_stage,
                **_second_stage_metrics(second_traces, qrels),
            },
        }
        _write_run(runs_dir / f"{_safe_name(second_name)}.jsonl", second_runs, qids, queries={qid: questions[qid].query for qid in qids})
        with (traces_dir / f"{_safe_name(second_name)}.jsonl").open("w", encoding="utf-8") as handle:
            for qid in qids:
                handle.write(json.dumps(second_traces[qid], ensure_ascii=False) + "\n")

    for name, method in methods.items():
        if name != "PDF-inspector + BM25 baseline":
            method["comparison_to_baseline"] = _paired_comparison(baseline_metrics, method["retrieval_metrics"])

    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "sources": {
            "parsed_run": str(DEFAULT_PARSED_RUN),
            "page_vectors": visual_meta,
            "legacy_cache": query_meta,
        },
        "legacy_cache": legacy_meta,
        "index_counts": index_counts,
        "methods": methods,
        "summary": [_summary_row(name, method) for name, method in methods.items()],
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "The Kf=5/10 file budgets are hard filters before page ranking.",
            "Legacy second retrieval uses persisted text chunks and cached query vectors.",
            "No network or model inference is used.",
            "V-SPLADE query vectors are the existing English-query cache evaluated against French qrels.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "output": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
