"""Test a cache-only Rocchio feedback loop for Industrial retrieval.

The first legacy chunk retrieval supplies pseudo-relevant chunk embeddings. A
small normalized centroid of those chunks is interpolated into the cached
query vector, and the legacy chunk index is searched again. The second-pass
page scores are then combined with the page/file cascade score.

No qrels, answer labels, rendering, network call or new embedding is used by
the retrieval loop. This is a research adapter over the existing Industrial
KDL + PDF-inspector and legacy embedding caches.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import (  # noqa: E402
    CascadeConfig,
    HierarchyCorpus,
    normalise_scores,
    sort_scores,
)
from research.experiments.industrial_hierarchical_retrieval import (  # noqa: E402
    DEFAULT_BASELINE_RUN,
    _build_indexes,
)
from research.experiments.industrial_legacy_second_retrieval import (  # noqa: E402
    DEFAULT_LEGACY_CACHE,
    DEFAULT_PARSED_RUN,
    _load_legacy_chunks,
    _load_query_vectors,
    _pool,
    _retrieve_chunks,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    HierarchicalRetriever,
    _derived_metrics,
    _file_id,
    _paired_comparison,
    _safe_name,
    _stratified_folds,
    _load_run,
)
from research.experiments.industrial_soft_cascade_feedback import (  # noqa: E402
    _write_compact_run,
)
from src.chunking_embedding.embedders import sanitize_text  # noqa: F401, E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.fusion import alpha_fuse  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/industrial_rocchio_legacy_feedback"
)


def _normalised_centroid(
    chunks: Sequence[Any],
    chunk_positions: Mapping[str, int],
    hits: Sequence[tuple[str, float]],
    feedback_k: int,
) -> np.ndarray | None:
    selected = [(chunk_id, max(float(score), 0.0)) for chunk_id, score in hits[:feedback_k]]
    if not selected:
        return None
    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for chunk_id, score in selected:
        position = chunk_positions.get(chunk_id)
        if position is None:
            continue
        vectors.append(chunks[position].vector)
        # A small floor prevents a zero-scored feedback tail from becoming an
        # accidental uniform centroid.
        weights.append(max(score, 1e-3))
    if not vectors:
        return None
    matrix = np.asarray(vectors, dtype=np.float32)
    centroid = np.average(matrix, axis=0, weights=np.asarray(weights, dtype=np.float32))
    norm = float(np.linalg.norm(centroid))
    return centroid / max(norm, 1e-12)


def _refined_chunk_hits(
    chunks: Sequence[Any],
    index: BM25Index,
    matrix: np.ndarray,
    query: str,
    query_vector: np.ndarray,
    allowed_pages: set[str],
    *,
    feedback_k: int,
    rocchio_lambda: float,
    alpha: float,
    depth: int,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], np.ndarray | None]:
    """Run original chunk retrieval, update the query, and retrieve again."""

    first_hits = _retrieve_chunks(
        chunks,
        index,
        matrix,
        query,
        query_vector,
        allowed_pages,
        alpha,
        depth,
    )
    positions = {chunk.chunk_id: position for position, chunk in enumerate(chunks)}
    centroid = _normalised_centroid(chunks, positions, first_hits, feedback_k)
    if centroid is None or rocchio_lambda <= 0:
        return first_hits, first_hits, None
    refined = (1.0 - rocchio_lambda) * query_vector + rocchio_lambda * centroid
    refined /= max(float(np.linalg.norm(refined)), 1e-12)

    allowed_positions = {
        position
        for position, chunk in enumerate(chunks)
        if allowed_pages.intersection(chunk.page_ids)
    }
    dense_scores = matrix @ refined
    dense_positions = sorted(
        allowed_positions,
        key=lambda position: (-float(dense_scores[position]), position),
    )[:depth]
    dense_hits = [
        (index.chunk_ids[position], float(dense_scores[position]))
        for position in dense_positions
    ]
    sparse_hits = [
        (index.chunk_ids[position], float(score))
        for position, score in index.search(query, depth, allowed_positions)
    ]
    refined_hits = alpha_fuse(dense_hits, sparse_hits, alpha, depth)
    return first_hits, refined_hits, refined


def _retrieve_one(
    corpus: HierarchyCorpus,
    chunks: Sequence[Any],
    chunk_index: BM25Index,
    chunk_matrix: np.ndarray,
    query_vector: np.ndarray,
    query: str,
    base_pages: Sequence[str],
    base_scores_raw: Mapping[str, float],
    *,
    feedback_k: int,
    rocchio_lambda: float,
    gamma: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed_pages = set(base_pages)
    first_hits, refined_hits, refined_vector = _refined_chunk_hits(
        chunks,
        chunk_index,
        chunk_matrix,
        query,
        query_vector,
        allowed_pages,
        feedback_k=feedback_k,
        rocchio_lambda=rocchio_lambda,
        alpha=0.70,
        depth=100,
    )
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for chunk_id, score in refined_hits:
        for page_id in by_id[chunk_id].page_ids:
            if page_id in allowed_pages:
                grouped[page_id].append(score)
    fine_scores = _pool(grouped, "max")
    base_norm = normalise_scores(base_scores_raw)
    fine_norm = normalise_scores(fine_scores)
    final_scores = {
        page_id: (1.0 - gamma) * base_norm.get(page_id, 0.0)
        + gamma * fine_norm.get(page_id, 0.0)
        for page_id in base_pages
    }
    ranked = sort_scores(final_scores)[:100]
    run = [
        {
            "chunk_id": page_id,
            "doc_id": page_id,
            "text": corpus.pages[page_id].text,
            "score": round(float(score), 8),
            "rank": rank,
            "scores": {
                "cascade_page": round(base_norm.get(page_id, 0.0), 8),
                "legacy_feedback_page": round(fine_norm.get(page_id, 0.0), 8),
                "final": round(float(score), 8),
            },
        }
        for rank, (page_id, score) in enumerate(ranked, 1)
    ]
    trace = {
        "query": query,
        "feedback_k": feedback_k,
        "rocchio_lambda": rocchio_lambda,
        "gamma": gamma,
        "base_page_count": len(base_pages),
        "first_feedback_chunks": [
            {"chunk_id": chunk_id, "score": round(float(score), 8)}
            for chunk_id, score in first_hits[:feedback_k]
        ],
        "refined_chunks": [
            {"chunk_id": chunk_id, "score": round(float(score), 8)}
            for chunk_id, score in refined_hits[:20]
        ],
        "refined_query_used": refined_vector is not None,
    }
    return run, trace


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> tuple[float, float]:
    rows = [row for row in metrics["per_query"] if str(row["qid"]) in qids]
    if not rows:
        return 0.0, 0.0
    page = sum(float(row["page_recall@10"]) for row in rows) / len(rows)
    files: list[float] = []
    for row in rows:
        candidates = row["top10_files_from_top100_pages"][:3]
        gold = set(row["gold_files"])
        files.append(len(set(candidates) & gold) / len(gold) if gold else 0.0)
    return page, sum(files) / len(files)


def _oof_selection(
    qids: Sequence[str],
    questions: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int]],
    methods: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    folds = _stratified_folds(qids, questions, qrels)
    all_qids = set(qids)
    oof_run: dict[str, list[dict[str, Any]]] = {}
    selected: list[dict[str, Any]] = []
    for fold, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train = all_qids - heldout_set
        ranking = [
            (*_metric_for_qids(method["retrieval_metrics"], train), name)
            for name, method in methods.items()
        ]
        _, _, winner = max(ranking, key=lambda row: (row[0], row[1], row[2]))
        for qid in heldout:
            oof_run[qid] = runs[winner][qid]
        selected.append(
            {
                "fold": fold,
                "heldout_qids": heldout,
                "selected_method": winner,
                "train_page_recall@10": _metric_for_qids(
                    methods[winner]["retrieval_metrics"], train
                )[0],
                "train_file_recall@3": _metric_for_qids(
                    methods[winner]["retrieval_metrics"], train
                )[1],
            }
        )
    return {
        "folds": selected,
        "selected_method_counts": {
            name: sum(row["selected_method"] == name for row in selected)
            for name in methods
        },
        "run": oof_run,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--legacy-cache", type=Path, default=DEFAULT_LEGACY_CACHE)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    benchmark = ViDoreV3(
        root=ROOT / "data/raw/benchmarks/vidore_v3",
        subset="industrial",
        language="english",
    )
    questions_list = sorted(
        list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1])
    )
    qids = [q.qid for q in questions_list]
    questions = {q.qid: q for q in questions_list}
    qrels = benchmark.qrels()
    page_ids = sorted(document.doc_id for document in benchmark.corpus())
    corpus, page_index, file_indexes, index_counts = _build_indexes(
        args.parsed_run, page_ids
    )
    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes={},
        visual_scores={},
    )
    chunks, legacy_meta = _load_legacy_chunks(args.parsed_run, corpus)
    query_vectors, query_meta = _load_query_vectors(questions_list, args.legacy_cache)
    chunk_matrix = np.asarray([chunk.vector for chunk in chunks], dtype=np.float32)
    chunk_matrix /= np.clip(np.linalg.norm(chunk_matrix, axis=1, keepdims=True), 1e-12, None)
    chunk_index = BM25Index(analyzer_name="auto").build(
        [
            {"chunk_id": chunk.chunk_id, "doc_id": chunk.file_id, "text": chunk.text}
            for chunk in chunks
        ]
    )

    # Materialise the Kf=10 page scope once. Each feedback arm shares this
    # first-pass context, so the comparison isolates the iterative update.
    base_config = CascadeConfig(
        name="industrial-kf10",
        file_representation="all_text",
        file_pool="max",
        k_files=10,
        page_depth=len(corpus.page_order),
        final_depth=len(corpus.page_order),
        bm25_weight=1.0,
        parent_weight=0.15,
        file_direct_weight=0.5,
    )
    base_context: dict[str, tuple[list[str], dict[str, float]]] = {}
    for question in questions_list:
        _, trace = retriever.retrieve(question.qid, question.query, base_config)
        rows = trace["page_candidates"]
        base_context[question.qid] = (
            [str(row["node_id"]) for row in rows],
            {str(row["node_id"]): float(row["score"]) for row in rows},
        )

    baseline = _load_run(args.baseline_run)
    methods: dict[str, dict[str, Any]] = {
        "Cached page BM25 baseline": {
            "config": None,
            "retrieval_metrics": _derived_metrics(baseline, qids, qrels),
            "timing_seconds": {"retrieval": 0.0},
        }
    }
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {
        "Cached page BM25 baseline": baseline
    }
    traces: dict[str, dict[str, Any]] = {}
    specs = [
        (1, 0.03),
        (1, 0.05),
        (3, 0.03),
        (3, 0.05),
        (3, 0.10),
        (5, 0.03),
        (5, 0.05),
        (5, 0.10),
        (5, 0.20),
        (10, 0.03),
        (10, 0.05),
        (10, 0.10),
    ]
    for feedback_k, rocchio_lambda in specs:
        name = f"rocchio-file10-k{feedback_k}-lambda{rocchio_lambda:g}-gamma0.1"
        arm_started = time.perf_counter()
        arm_runs: dict[str, list[dict[str, Any]]] = {}
        arm_traces: dict[str, Any] = {}
        for question in questions_list:
            base_pages, base_scores = base_context[question.qid]
            run, trace = _retrieve_one(
                corpus,
                chunks,
                chunk_index,
                chunk_matrix,
                query_vectors[question.qid],
                question.query,
                base_pages,
                base_scores,
                feedback_k=feedback_k,
                rocchio_lambda=rocchio_lambda,
                gamma=0.1,
            )
            arm_runs[question.qid] = run
            arm_traces[question.qid] = trace
        metrics = _derived_metrics(arm_runs, qids, qrels)
        methods[name] = {
            "config": {
                "scope": "file10",
                "feedback_k": feedback_k,
                "rocchio_lambda": rocchio_lambda,
                "gamma": 0.1,
                "legacy_alpha": 0.70,
                "depth": 100,
            },
            "retrieval_metrics": metrics,
            "comparison_to_baseline": _paired_comparison(
                methods["Cached page BM25 baseline"]["retrieval_metrics"], metrics
            ),
            "timing_seconds": {"retrieval": round(time.perf_counter() - arm_started, 6)},
        }
        runs[name] = arm_runs
        traces[name] = arm_traces

    oof = _oof_selection(qids, questions, qrels, methods, runs)
    oof_metrics = _derived_metrics(oof["run"], qids, qrels)
    baseline_metrics = methods["Cached page BM25 baseline"]["retrieval_metrics"]
    report = {
        "dataset": "vidore_v3/industrial",
        "evaluation_language": "english",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "index_counts": index_counts,
        "legacy_cache": legacy_meta,
        "query_cache": query_meta,
        "methods": methods,
        "cv": {
            "folds": oof["folds"],
            "selected_method_counts": oof["selected_method_counts"],
            "retrieval_metrics": oof_metrics,
            "comparison_to_baseline": _paired_comparison(baseline_metrics, oof_metrics),
        },
        "sources": {
            "parsed_run": str(args.parsed_run),
            "baseline_run": str(args.baseline_run),
            "legacy_cache": str(args.legacy_cache),
        },
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "Rocchio feedback uses only cached legacy chunk embeddings and the first-pass top-100 chunk hits.",
            "The first-pass file10 page scope is shared by all arms; no feedback arm can recover a page outside that scope.",
            "No qrels are used in retrieval. Full-set arm screening and OOF selection are reported separately.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)
    (args.output_dir / "traces").mkdir(exist_ok=True)
    query_texts = {q.qid: q.query for q in questions_list}
    best_name = max(
        (name for name in methods if name != "Cached page BM25 baseline"),
        key=lambda name: (
            methods[name]["retrieval_metrics"]["page_recall@10"],
            methods[name]["retrieval_metrics"]["ndcg@10"],
        ),
    )
    _write_compact_run(args.output_dir / "runs" / f"{_safe_name(best_name)}.jsonl", runs[best_name], qids, query_texts)
    _write_compact_run(args.output_dir / "oof_run.jsonl", oof["run"], qids, query_texts)
    with (args.output_dir / "traces" / f"{_safe_name(best_name)}.jsonl").open("w", encoding="utf-8") as handle:
        for qid in qids:
            handle.write(json.dumps(traces[best_name][qid], ensure_ascii=False) + "\n")
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Industrial Rocchio legacy feedback",
        "",
        "Cache-only iterative query-vector refinement over legacy chunk embeddings inside the Kf=10 file scope.",
        "",
        "| Method | nDCG@10 | Page hit@10 | Page recall@10 | File recall@3 | Δ page pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, method in methods.items():
        metrics = method["retrieval_metrics"]
        delta = method.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0)
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.2f} | {metrics['page_hit@10']:.2%} | "
            f"{metrics['page_recall@10']:.2%} | {metrics['file_metrics_by_k']['3']['file_recall']:.2%} | {delta:+.2f} |"
        )
    cv_metrics = report["cv"]["retrieval_metrics"]
    cv_compare = report["cv"]["comparison_to_baseline"]
    lines += [
        "",
        "## Out-of-fold selection",
        "",
        f"- Selected methods: `{report['cv']['selected_method_counts']}`.",
        f"- OOF nDCG@10: **{cv_metrics['ndcg@10']:.2f}**; page recall@10: **{cv_metrics['page_recall@10']:.2%}** ({cv_compare['page_recall_delta_pp']:+.2f}pp vs baseline).",
        f"- OOF file recall@3: **{cv_metrics['file_metrics_by_k']['3']['file_recall']:.2%}**.",
        f"- OOF paired bootstrap page-recall CI: **{cv_compare['page_recall_ci95_pp']} pp**, p={cv_compare['page_recall_p_two_sided']:.4f}.",
        "",
        "Full-set rows are sensitivity results; OOF is the primary generalisation check.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: {"page_recall@10": round(method["retrieval_metrics"]["page_recall@10"] * 100, 2), "file_recall@3": round(method["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"] * 100, 2), "delta_pp": round(method.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0), 2)} for name, method in methods.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
