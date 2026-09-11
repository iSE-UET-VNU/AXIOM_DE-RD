"""Evaluate the old hierarchical light stage followed by legacy second retrieval.

Protocol under test:

    old hierarchical BM25 + cached V-SPLADE
        -> select top-K files
        -> expose every page in those files
        -> cached legacy chunk BM25+dense second retrieval
        -> final top-10 pages

This intentionally does not use the later proposal-union/file-synopsis export.
It revisits the original ``all_text`` hierarchical arm associated with the
44.28% page-recall result and varies only Kf.  All ranking inputs are cached;
qrels are read only for evaluation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
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
from research.experiments.evaluate_physics_legacy_second_retrieval import (  # noqa: E402
    _aggregate_hits_to_pages,
    _load_cached_query_vectors,
    _load_legacy_chunks,
    _top_dense,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    HierarchicalRetriever,
    _build_corpus_and_indexes,
    _derived_metrics,
    _file_id,
    _load_visual_scores,
    _paired_comparison,
    _page_vector_units,
    _stage_metrics,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.fusion import alpha_fuse  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_LEGACY_CACHE = ROOT / "data/work/vidore_physics_emb"
DEFAULT_BASELINE_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_legacy_kf"

K_FILES = (3, 5, 10, 15, 20)
FINAL_K = 10
LEGACY_ALPHA = 0.70
LEGACY_DEPTH = 100
LEGACY_WEIGHT = 0.25


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _load_questions_and_qrels(root: Path) -> tuple[list[Any], list[str], dict[str, dict[str, int]]]:
    benchmark = ViDoreV3(root=root, subset="physics", language="french")
    questions = sorted(
        list(benchmark.questions()),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    qids = [question.qid for question in questions]
    qrels = benchmark.qrels()
    if len(qids) != 302 or len(qrels) != 302:
        raise RuntimeError(f"Expected 302 queries/qrels, got {len(qids)}/{len(qrels)}")
    return questions, qids, qrels


def _load_run(path: Path, qids: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                output[str(row["qid"])] = list(row["chunks"])
    if set(output) != set(qids):
        raise ValueError(f"Baseline run qid mismatch: {path}")
    return output


def _base_config(k_files: int) -> CascadeConfig:
    return CascadeConfig(
        name=f"old-hierarchical-all-text-max-kf{k_files}",
        file_representation="all_text",
        file_pool="max",
        k_files=k_files,
        # Every page in the selected files remains available to the second
        # stage. The final evaluation output is still top-100 pages.
        page_depth=1674,
        final_depth=100,
        bm25_weight=0.70,
        parent_weight=0.15,
        file_direct_weight=0.50,
        file_pool_source="page_base",
        fine_unit="none",
        fine_pool="max",
        fine_weight=0.0,
    )


def _retrieve_legacy_hits(
    chunk_index: BM25Index,
    dense_scores: np.ndarray,
    query: str,
    allowed_pages: set[str],
    chunk_positions_by_page: Mapping[str, Sequence[int]],
) -> list[tuple[str, float]]:
    allowed_positions = sorted({
        position
        for page_id in allowed_pages
        for position in chunk_positions_by_page.get(page_id, [])
    })
    dense = [
        (chunk_index.chunk_ids[position], score)
        for position, score in _top_dense(dense_scores, allowed_positions, LEGACY_DEPTH)
    ]
    sparse = [
        (chunk_index.chunk_ids[position], float(score))
        for position, score in chunk_index.search(
            query, LEGACY_DEPTH, set(allowed_positions)
        )
    ]
    return alpha_fuse(dense, sparse, LEGACY_ALPHA, LEGACY_DEPTH)


def _legacy_over_selected_files(
    base_traces: Mapping[str, Mapping[str, Any]],
    questions: Sequence[Any],
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    corpus: HierarchyCorpus,
    chunks: Sequence[Any],
    by_chunk_id: Mapping[str, Any],
    chunk_index: BM25Index,
    dense_score_matrix: np.ndarray,
    qid_to_index: Mapping[str, int],
    chunk_positions_by_page: Mapping[str, Sequence[int]],
    k_files: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for question in questions:
        qid = question.qid
        base_trace = base_traces[qid]
        selected_files = [
            str(item["node_id"])
            for item in base_trace["file_candidates"][:k_files]
        ]
        selected_file_set = set(selected_files)
        allowed_pages = {
            page_id
            for page_id in corpus.page_order
            if _file_id(page_id) in selected_file_set
        }
        output_pages = [page_id for page_id in corpus.page_order if page_id in allowed_pages]
        base_scores = {
            str(item["node_id"]): float(item["score"])
            for item in base_trace["page_candidates"]
            if str(item["node_id"]) in allowed_pages
        }
        base_norm = normalise_scores(base_scores)
        hits = _retrieve_legacy_hits(
            chunk_index,
            dense_score_matrix[:, qid_to_index[qid]],
            question.query,
            allowed_pages,
            chunk_positions_by_page,
        )
        legacy_scores = _aggregate_hits_to_pages(hits, by_chunk_id, allowed_pages, "max")
        legacy_norm = normalise_scores(legacy_scores)
        final_scores = {
            page_id: (1.0 - LEGACY_WEIGHT) * base_norm.get(page_id, 0.0)
            + LEGACY_WEIGHT * legacy_norm.get(page_id, 0.0)
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
                    "legacy_chunk": round(float(legacy_norm.get(page_id, 0.0)), 8),
                    "final": round(float(score), 8),
                },
            }
            for rank, (page_id, score) in enumerate(ranked, 1)
        ]
        gold_pages = set(qrels[qid])
        gold_files = {_file_id(page_id) for page_id in gold_pages}
        top10 = [page_id for page_id, _score in ranked[:FINAL_K]]
        found = set(top10) & gold_pages
        touched_pages = {
            page_id
            for chunk_id, _score in hits
            for page_id in by_chunk_id[chunk_id].page_ids
            if page_id in allowed_pages
        }
        rows.append({
            "qid": qid,
            "k_files": k_files,
            "selected_files": selected_files,
            "candidate_page_count": len(allowed_pages),
            "page_pool_recall_ceiling": len(allowed_pages & gold_pages) / len(gold_pages),
            "file_scope_recall": len(selected_file_set & gold_files) / len(gold_files),
            "final_top10_pages": top10,
            "page_hit@10": bool(found),
            "page_recall@10": len(found) / len(gold_pages),
            "nDCG@10": _ndcg(top10, qrels[qid], FINAL_K),
            "legacy_chunk_pages": len(touched_pages),
            "legacy_chunk_hits": len(hits),
            "latency_seconds": base_trace["timing_seconds"].get("total", 0.0),
        })
    return runs, rows


def _ndcg(ranked: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    def gain(value: int) -> float:
        return float((2 ** int(value)) - 1)

    dcg = sum(
        gain(qrel.get(page_id, 0)) / math.log2(rank + 1)
        for rank, page_id in enumerate(ranked[:k], 1)
        if qrel.get(page_id, 0) > 0
    )
    ideal = sorted((int(value) for value in qrel.values()), reverse=True)[:k]
    idcg = sum(gain(value) / math.log2(rank + 1) for rank, value in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def _legacy_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "queries": len(rows),
        "avg_selected_files": float(np.mean([len(row["selected_files"]) for row in rows])),
        "avg_candidate_pages": float(np.mean([row["candidate_page_count"] for row in rows])),
        "file_scope_recall": float(np.mean([row["file_scope_recall"] for row in rows])),
        "page_pool_recall_ceiling": float(np.mean([row["page_pool_recall_ceiling"] for row in rows])),
        "nDCG@10": float(np.mean([row["nDCG@10"] for row in rows])),
        "page_hit@10": float(np.mean([row["page_hit@10"] for row in rows])),
        "page_recall@10": float(np.mean([row["page_recall@10"] for row in rows])),
        "final_file_recall@3": _derived_file_recall(rows),
        "avg_legacy_chunk_pages": float(np.mean([row["legacy_chunk_pages"] for row in rows])),
        "avg_legacy_chunk_hits": float(np.mean([row["legacy_chunk_hits"] for row in rows])),
    }


def _derived_file_recall(rows: Sequence[Mapping[str, Any]]) -> float:
    values: list[float] = []
    for row in rows:
        gold_files = {
            _file_id(page_id)
            for page_id in row["final_top10_pages"]
        }
        selected = set(row["selected_files"])
        # The legacy fixed file metric is most usefully reported from the
        # selected file scope here; the final top-10 has no 100-page tail.
        values.append(len(selected & gold_files) / len(selected) if selected else 0.0)
    return float(np.mean(values)) if values else 0.0


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Physics old hierarchical file budget + legacy second retrieval",
        "",
        "Old hierarchical light stage (all_text + BM25/V-SPLADE) selects top-K files, then legacy retrieval searches every page in those files.",
        "This is separate from the later proposal-union/file-synopsis experiment.",
        "",
        "## Fixed configuration",
        "",
        f"- Hierarchical light: all_text, max file pool, BM25 weight 0.70, V-SPLADE weight 0.30, parent weight 0.15.",
        f"- Legacy second: dense alpha `{LEGACY_ALPHA}`, chunk depth `{LEGACY_DEPTH}`, max chunk/page pool, legacy weight `{LEGACY_WEIGHT}`.",
        "- All pages in selected files are eligible for the second stage; final output is top-10 for the main metric.",
        "",
        "## Results",
        "",
        "| Arm | Kf | Avg pages/query | File scope recall | Page ceiling | nDCG@10 | Page recall@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        light = arm["light_metrics"]
        second = arm["second_metrics"]
        lines.append(
            f"| Light hierarchical | {arm['k_files']} | {light['avg_candidate_pages']:.2f} | "
            f"{light['file_scope_recall']:.2%} | {light['page_pool_recall_ceiling']:.2%} | "
            f"{light['nDCG@10']:.2%} | {light['page_recall@10']:.2%} |"
        )
        lines.append(
            f"| + legacy second | {arm['k_files']} | {second['avg_candidate_pages']:.2f} | "
            f"{second['file_scope_recall']:.2%} | {second['page_pool_recall_ceiling']:.2%} | "
            f"{second['nDCG@10']:.2%} | {second['page_recall@10']:.2%} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- File scope recall is determined before legacy retrieval; legacy cannot recover a page from a file not selected by the light stage.",
        "- Page ceiling is the gold-page coverage after expanding selected files to all their pages.",
        "- The 44.28% reference is the original Kf=3 hierarchical arm; the old Kf=3 + legacy row is reported separately.",
        "- No parser, embedding API, OCR, LLM or qrel-guided ranking was used.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--legacy-cache", type=Path, default=DEFAULT_LEGACY_CACHE)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    questions, qids, qrels = _load_questions_and_qrels(args.benchmark_root)
    page_ids = _page_vector_units(args.page_vector_dir)
    if len(page_ids) != 1674 or len(set(page_ids)) != len(page_ids):
        raise RuntimeError(f"Expected 1,674 unique pages, got {len(page_ids)}")

    corpus, page_index, file_indexes, _fine_indexes, index_counts = _build_corpus_and_indexes(
        args.parsed_run, page_ids, fine_units=[]
    )
    visual_scores, visual_meta = _load_visual_scores(
        args.page_vector_dir, args.query_vector_dir, page_ids, qids
    )
    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes={},
        visual_scores=visual_scores,
    )
    chunks, legacy_meta = _load_legacy_chunks(args.parsed_run, corpus)
    query_vectors, query_meta = _load_cached_query_vectors(questions, args.legacy_cache)
    chunk_matrix = np.asarray([chunk.vector for chunk in chunks], dtype=np.float32)
    chunk_matrix /= np.clip(np.linalg.norm(chunk_matrix, axis=1, keepdims=True), 1e-12, None)
    query_matrix = np.asarray([query_vectors[qid] for qid in qids], dtype=np.float32)
    dense_score_matrix = chunk_matrix @ query_matrix.T
    chunk_index = BM25Index(analyzer_name="auto").build([
        {"chunk_id": chunk.chunk_id, "doc_id": chunk.file_id, "text": chunk.text}
        for chunk in chunks
    ])
    by_chunk_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunk_positions_by_page: dict[str, list[int]] = defaultdict(list)
    for position, chunk in enumerate(chunks):
        for page_id in chunk.page_ids:
            chunk_positions_by_page[page_id].append(position)
    qid_to_index = {qid: index for index, qid in enumerate(qids)}

    baseline_run = _load_run(args.baseline_run, qids)
    baseline_metrics = _derived_metrics(baseline_run, qids, qrels)
    output_dir = args.output_dir
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    arms: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for k_files in K_FILES:
        config = _base_config(k_files)
        base_runs: dict[str, list[dict[str, Any]]] = {}
        base_traces: dict[str, dict[str, Any]] = {}
        for question in questions:
            run, trace = retriever.retrieve(question.qid, question.query, config)
            base_runs[question.qid] = run
            base_traces[question.qid] = trace
        base_metrics = _derived_metrics(base_runs, qids, qrels)
        base_stage = _stage_metrics(base_traces, qids, qrels)
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
            qid_to_index,
            chunk_positions_by_page,
            k_files,
        )
        second_metrics = _derived_metrics(second_runs, qids, qrels)
        second_summary = _legacy_metrics(rows)
        _write_run(
            runs_dir / f"hierarchical_kf{k_files}_before_legacy.jsonl",
            base_runs,
            qids,
            queries={question.qid: question.query for question in questions},
        )
        _write_run(
            runs_dir / f"hierarchical_kf{k_files}_legacy_second.jsonl",
            second_runs,
            qids,
            queries={question.qid: question.query for question in questions},
        )
        for row in rows:
            row["light_page_recall@10"] = next(
                item["page_recall@10"]
                for item in _derived_per_query(base_runs, qids, qrels)
                if item["qid"] == row["qid"]
            )
            row["light_stage"] = base_stage
        arms.append({
            "k_files": k_files,
            "config": config.__dict__ if hasattr(config, "__dict__") else str(config),
            "light_metrics": {
                "nDCG@10": base_metrics["ndcg@10"] / 100.0,
                "page_recall@10": base_metrics["page_recall@10"],
                # `second_summary` is computed from the exact selected-file
                # scope used by the second stage.  Unlike the shared stage
                # helper, it supports arbitrary Kf values such as 15/20 and
                # includes the all-pages page ceiling.
                "file_scope_recall": second_summary["file_scope_recall"],
                "page_pool_recall_ceiling": second_summary["page_pool_recall_ceiling"],
                "avg_candidate_pages": second_summary["avg_candidate_pages"],
            },
            "second_metrics": second_summary,
            "second_retrieval_metrics": second_metrics,
        })
        all_rows.extend(rows)

    report = {
        "dataset": "vidore_v3/physics",
        "language": "french",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "reference": {
            "original_hierarchical_kf3_page_recall@10": 0.4428,
            "baseline_pdf_inspector_bm25_page_recall@10": baseline_metrics["page_recall@10"],
        },
        "config": {
            "k_files": list(K_FILES),
            "page_scope": "all pages in selected files",
            "hierarchical_file_representation": "all_text",
            "hierarchical_file_pool": "max",
            "hierarchical_bm25_weight": 0.70,
            "hierarchical_parent_weight": 0.15,
            "legacy_dense_alpha": LEGACY_ALPHA,
            "legacy_chunk_depth": LEGACY_DEPTH,
            "legacy_weight": LEGACY_WEIGHT,
            "legacy_page_pool": "max",
            "qrels_used_for_ranking": False,
        },
        "cache": {
            "legacy_text_chunks": legacy_meta["text_chunks"],
            "cross_page_chunks": legacy_meta["cross_page_chunks"],
            "query_embedding_hits": query_meta["cache_hits"],
            "query_embedding_misses": query_meta["cache_misses"],
            "embedding_model": legacy_meta["embedding_model"],
            "visual_cache": visual_meta,
        },
        "sources": {
            "parsed_run": str(args.parsed_run),
            "baseline_run": str(args.baseline_run),
            "page_vector_dir": str(args.page_vector_dir),
            "query_vector_dir": str(args.query_vector_dir),
        },
        "index_counts": index_counts,
        "baseline_metrics": baseline_metrics,
        "arms": arms,
        "timing_seconds": {
            "total": time.perf_counter() - started,
            "note": "Includes cached artifact loading, hierarchical index construction, cached dense score multiplication and all Kf retrievals; excludes parsing and model encoding.",
        },
    }
    _write_json(output_dir / "report.json", report)
    _write_jsonl(output_dir / "per_query.jsonl", all_rows)
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "elapsed_seconds": round(report["timing_seconds"]["total"], 3),
        "results": {
            f"Kf={arm['k_files']}": {
                "light_page_recall@10": round(100 * arm["light_metrics"]["page_recall@10"], 2),
                "legacy_page_recall@10": round(100 * arm["second_metrics"]["page_recall@10"], 2),
                "page_ceiling": round(100 * arm["second_metrics"]["page_pool_recall_ceiling"], 2),
            }
            for arm in arms
        },
    }, ensure_ascii=False, indent=2))


def _derived_per_query(
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    output = []
    for qid in qids:
        gold = set(qrels[qid])
        top10 = [str(item["chunk_id"]) for item in runs[qid][:10]]
        output.append({
            "qid": qid,
            "page_recall@10": len(set(top10) & gold) / len(gold),
        })
    return output


if __name__ == "__main__":
    main()
