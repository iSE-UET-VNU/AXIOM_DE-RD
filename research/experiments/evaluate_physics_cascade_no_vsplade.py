"""Ablate V-SPLADE from the Physics full cascade.

The experiment keeps the same file -> page -> legacy-second architecture and
the same cached legacy chunks/query vectors, but sets page retrieval to
PDF-inspector BM25 only.  It is retrieval-only: no rendering, model inference
or network call is performed.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import CascadeConfig
from research.experiments.evaluate_physics_legacy_second_file_budget import (
    _legacy_over_selected_files,
)
from research.experiments.evaluate_physics_legacy_second_retrieval import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_LEGACY_CACHE,
    DEFAULT_PAGE_VECTOR_DIR,
    DEFAULT_PARSED_RUN,
    DEFAULT_QUERY_VECTOR_DIR,
    _load_cached_query_vectors,
    _load_legacy_chunks,
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


OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_cascade_no_vsplade"
K_FILES = 3
LEGACY_GAMMA = 0.25
LEGACY_ALPHA = 0.70
LEGACY_DEPTH = 100


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _no_visual_config() -> CascadeConfig:
    return CascadeConfig(
        name="full-cascade-no-vsplade-kf3",
        file_representation="all_text",
        file_pool="max",
        k_files=K_FILES,
        page_depth=1674,
        final_depth=100,
        bm25_weight=1.0,
        parent_weight=0.15,
        file_direct_weight=0.50,
        file_pool_source="page_base",
        fine_unit="none",
        fine_pool="max",
        fine_weight=0.0,
    )


def _metrics_row(name: str, method: dict[str, Any]) -> dict[str, Any]:
    metrics = method["retrieval_metrics"]
    return {
        "method": name,
        "ndcg@10": round(float(metrics["ndcg@10"]), 4),
        "page_recall@10": round(100.0 * float(metrics["page_recall@10"]), 4),
        "page_hit@10": round(100.0 * float(metrics["page_hit@10"]), 4),
        "file_recall@3": round(100.0 * float(metrics["file_metrics_by_k"]["3"]["file_recall"]), 4),
        "file_recall@10": round(100.0 * float(metrics["file_metrics_by_k"]["10"]["file_recall"]), 4),
        "page_delta_pp": round(float((method.get("comparison_to_baseline") or {}).get("page_recall_delta_pp", 0.0)), 4),
        "file_delta_pp": round(float((method.get("comparison_to_baseline") or {}).get("file_recall_delta_pp", 0.0)), 4),
        "legacy_chunk_page_recall_proxy": round(100.0 * float((method.get("stage_metrics") or {}).get("legacy_chunk_page_recall_proxy", 0.0)), 4),
    }


def _load_reference(path: Path, method_name: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    method = report.get("methods", {}).get(method_name)
    if not method:
        return None
    metrics = method["retrieval_metrics"]
    return {
        "source": str(path),
        "method": method_name,
        "ndcg@10": float(metrics["ndcg@10"]),
        "page_recall@10": 100.0 * float(metrics["page_recall@10"]),
        "file_recall@3": 100.0 * float(metrics["file_metrics_by_k"]["3"]["file_recall"]),
        "file_recall@10": 100.0 * float(metrics["file_metrics_by_k"]["10"]["file_recall"]),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Physics full cascade without V-SPLADE",
        "",
        "Ablation: PDF-inspector BM25 only at page/file stages, followed by cached legacy second retrieval.",
        "No PDF rendering, V-SPLADE encoding, model inference or network call was used.",
        "",
        "| Method | nDCG@10 | Page recall@10 | File recall@3 | File recall@10 | Δ page pp vs BM25 | Legacy chunk-page proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['method']} | {row['ndcg@10']:.2f} | {row['page_recall@10']:.2f}% | "
            f"{row['file_recall@3']:.2f}% | {row['file_recall@10']:.2f}% | "
            f"{row['page_delta_pp']:+.2f} | {row['legacy_chunk_page_recall_proxy']:.2f}% |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The no-V-SPLADE arm uses normalized PDF-inspector BM25 page scores only; `bm25_weight=1.0`.",
        "- The legacy second stage searches all pages belonging to the selected Kf=3 files and then reranks the top 100 pages.",
        "- Existing V-SPLADE full-cascade references are included separately for impact comparison.",
        "- This is retrieval-only; it does not claim an end-to-end QA improvement.",
        "",
        "Runs are under `runs/`; traces are under `traces/`.",
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
    # Load the visual cache only to verify the corpus/query alignment.  It is
    # deliberately not passed to HierarchicalRetriever for this ablation.
    _load_visual_scores(DEFAULT_PAGE_VECTOR_DIR, DEFAULT_QUERY_VECTOR_DIR, page_ids, qids)
    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes=fine_indexes,
        visual_scores={},
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

    config = _no_visual_config()
    base_runs: dict[str, list[dict[str, Any]]] = {}
    base_traces: dict[str, dict[str, Any]] = {}
    for qid, question in questions.items():
        run, trace = retriever.retrieve(qid, question.query, config)
        base_runs[qid] = run
        base_traces[qid] = trace

    second_runs, second_traces = _legacy_over_selected_files(
        base_traces,
        questions,
        chunks,
        chunk_index,
        chunk_matrix,
        query_vectors,
        corpus,
        k_files=K_FILES,
        gamma=LEGACY_GAMMA,
        alpha=LEGACY_ALPHA,
        depth=LEGACY_DEPTH,
    )

    methods = {
        "PDF-inspector + BM25 baseline": {
            "config": None,
            "retrieval_metrics": baseline_metrics,
            "stage_metrics": {},
        },
        "full cascade Kf=3 without V-SPLADE": {
            "config": asdict(config),
            "retrieval_metrics": _derived_metrics(base_runs, qids, qrels),
            "stage_metrics": _stage_metrics(base_traces, qids, qrels),
        },
        "full cascade Kf=3 without V-SPLADE + legacy second": {
            "config": {
                "base_config": asdict(config),
                "scope": "selected_files_all_pages",
                "gamma": LEGACY_GAMMA,
                "alpha": LEGACY_ALPHA,
                "depth": LEGACY_DEPTH,
            },
            "retrieval_metrics": _derived_metrics(second_runs, qids, qrels),
            "stage_metrics": {
                **_stage_metrics(base_traces, qids, qrels),
                **_second_stage_metrics(second_traces, qrels),
            },
        },
    }
    for name, method in methods.items():
        if name != "PDF-inspector + BM25 baseline":
            method["comparison_to_baseline"] = _paired_comparison(baseline_metrics, method["retrieval_metrics"])

    output_dir = OUTPUT_DIR
    runs_dir = output_dir / "runs"
    traces_dir = output_dir / "traces"
    runs_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    _write_run(runs_dir / "full-cascade-kf3-no-vsplade.jsonl", base_runs, qids, queries={qid: questions[qid].query for qid in qids})
    second_path = runs_dir / "full-cascade-kf3-no-vsplade-legacy-second.jsonl"
    _write_run(second_path, second_runs, qids, queries={qid: questions[qid].query for qid in qids})
    with (traces_dir / "full-cascade-kf3-no-vsplade-legacy-second.jsonl").open("w", encoding="utf-8") as handle:
        for qid in qids:
            handle.write(json.dumps(second_traces[qid], ensure_ascii=False) + "\n")

    references = [
        _load_reference(
            ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_retrieval/report.json",
            "cascade-all_text-max-kf3-fusion",
        ),
        _load_reference(
            ROOT / "data/benchmark/vidore_v3/results/physics_legacy_second_file_budget/report.json",
            "full-cascade-kf3-legacy-second-selected-files-max-gamma0.25",
        ),
    ]
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "config": asdict(config),
        "visual_signal": "disabled; visual cache alignment checked but scores not passed to retriever",
        "sources": {
            "parsed_run": str(DEFAULT_PARSED_RUN),
            "legacy_cache": query_meta,
        },
        "legacy_cache": legacy_meta,
        "index_counts": index_counts,
        "methods": methods,
        "summary": [_metrics_row(name, method) for name, method in methods.items()],
        "existing_vsplade_references": [reference for reference in references if reference],
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "No V-SPLADE score is used in file or page ranking.",
            "Legacy second retrieval uses cached text chunks and cached French query embeddings.",
            "All page BM25 input is PDF-inspector extracted text.",
        ],
    }
    _write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "output": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
