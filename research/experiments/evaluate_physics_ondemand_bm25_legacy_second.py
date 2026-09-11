"""Evaluate the first on-demand cascade: BM25 page pool -> legacy second pass.

This is deliberately the simplest on-demand baseline:

    PDF-inspector page text + BM25 top-Klight
        -> cached legacy chunk BM25+dense retrieval within those pages
        -> max chunk-to-page pooling and fixed light/legacy fusion
        -> final top-10 pages

There is no file retrieval, union proposal or hierarchical compact ranking in
this experiment.  The existing PDF-inspector+BM25 top-100 run is reused, while
the parser/legacy chunks and both query/chunk embeddings are read from cache.
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
from research.experiments.physics_hierarchical_retrieval import _page_vector_units  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.fusion import alpha_fuse  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_LEGACY_CACHE = ROOT / "data/work/vidore_physics_emb"
DEFAULT_LIGHT_RUN = (
    ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/"
    "bm25_french_bm25-french_vs-english.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_ondemand_bm25_legacy_second"

LIGHT_PAGE_K = (10, 20, 50, 100)
FINAL_PAGE_K = 10
LEGACY_DENSE_ALPHA = 0.70
LEGACY_CHUNK_DEPTH = 100
LEGACY_WEIGHT = 0.25
PAGE_POOL = "max"


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


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _ndcg_at_k(ranked: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
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


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _load_light_run(path: Path, qids: Sequence[str], page_set: set[str]) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing PDF-inspector BM25 light run: {path}")
    output: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            if qid in output:
                raise ValueError(f"Duplicate qid {qid!r} in {path}:{line_number}")
            chunks = list(row.get("chunks") or [])
            if len(chunks) < max(LIGHT_PAGE_K):
                raise ValueError(f"Light BM25 row {qid} has only {len(chunks)} pages")
            ids = [str(chunk["chunk_id"]) for chunk in chunks]
            if len(ids) != len(set(ids)) or any(page_id not in page_set for page_id in ids):
                raise ValueError(f"Invalid or duplicate page IDs in light run for {qid}")
            output[qid] = chunks
    if set(output) != set(qids):
        raise ValueError("Light BM25 run does not contain exactly the benchmark qids")
    return output


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
    dense_hits = [
        (chunk_index.chunk_ids[position], score)
        for position, score in _top_dense(dense_scores, allowed_positions, LEGACY_CHUNK_DEPTH)
    ]
    sparse_hits = [
        (chunk_index.chunk_ids[position], float(score))
        for position, score in chunk_index.search(
            query, LEGACY_CHUNK_DEPTH, set(allowed_positions)
        )
    ]
    return alpha_fuse(dense_hits, sparse_hits, LEGACY_DENSE_ALPHA, LEGACY_CHUNK_DEPTH)


def _run_query(
    qid: str,
    question: str,
    light_chunks: Sequence[Mapping[str, Any]],
    qrel: Mapping[str, int],
    corpus: HierarchyCorpus,
    chunks: Sequence[Any],
    by_chunk_id: Mapping[str, Any],
    chunk_index: BM25Index,
    dense_scores: np.ndarray,
    chunk_positions_by_page: Mapping[str, Sequence[int]],
    light_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    light_page_ids = [str(item["chunk_id"]) for item in light_chunks[:light_k]]
    allowed_pages = set(light_page_ids)
    light_scores = {
        page_id: float(item.get("score", 0.0))
        for page_id, item in zip(light_page_ids, light_chunks[:light_k])
    }
    hits = _retrieve_legacy_hits(
        chunk_index,
        dense_scores,
        question,
        allowed_pages,
        chunk_positions_by_page,
    )
    legacy_scores = _aggregate_hits_to_pages(hits, by_chunk_id, allowed_pages, PAGE_POOL)
    light_norm = normalise_scores(light_scores)
    legacy_norm = normalise_scores(legacy_scores)
    final_scores = {
        page_id: (1.0 - LEGACY_WEIGHT) * light_norm.get(page_id, 0.0)
        + LEGACY_WEIGHT * legacy_norm.get(page_id, 0.0)
        for page_id in allowed_pages
    }
    ranked = sort_scores(final_scores)
    run = [
        {
            "chunk_id": page_id,
            "doc_id": page_id,
            "text": corpus.pages[page_id].text,
            "score": round(float(score), 8),
            "rank": rank,
            "scores": {
                "light_bm25": round(float(light_norm.get(page_id, 0.0)), 8),
                "legacy_chunk": round(float(legacy_norm.get(page_id, 0.0)), 8),
                "final": round(float(score), 8),
            },
        }
        for rank, (page_id, score) in enumerate(ranked, 1)
    ]
    gold_pages = set(qrel)
    gold_files = {_file_id(page_id) for page_id in gold_pages}
    candidate_files: list[str] = []
    for page_id in light_page_ids:
        file_id = _file_id(page_id)
        if file_id not in candidate_files:
            candidate_files.append(file_id)
    final_pages = [str(item["chunk_id"]) for item in run[:FINAL_PAGE_K]]
    found = set(final_pages) & gold_pages
    final_files: list[str] = []
    for page_id in [str(item["chunk_id"]) for item in run[:100]]:
        file_id = _file_id(page_id)
        if file_id not in final_files:
            final_files.append(file_id)
        if len(final_files) == 3:
            break
    touched_pages = {
        page_id
        for chunk_id, _score in hits
        for page_id in by_chunk_id[chunk_id].page_ids
        if page_id in allowed_pages
    }
    row = {
        "qid": qid,
        "light_page_k": light_k,
        "light_candidate_page_ids": light_page_ids,
        "final_top10_page_ids": final_pages,
        "gold_page_count": len(gold_pages),
        "candidate_page_recall_ceiling": len(allowed_pages & gold_pages) / len(gold_pages),
        "candidate_file_recall": len(set(candidate_files) & gold_files) / len(gold_files),
        "final_page_hit@10": bool(found),
        "final_page_recall@10": len(found) / len(gold_pages),
        "final_page_precision@10": len(found) / FINAL_PAGE_K,
        "final_nDCG@10": _ndcg_at_k(final_pages, qrel, FINAL_PAGE_K),
        "final_file_recall@3": len(set(final_files) & gold_files) / len(gold_files),
        "final_file_hit@3": bool(set(final_files) & gold_files),
        "final_top3_files": final_files,
        "legacy_chunk_pages": len(touched_pages),
        "legacy_chunk_hits": len(hits),
        "scoped_chunks": sum(
            1 for chunk in chunks if allowed_pages.intersection(chunk.page_ids)
        ),
        "latency_seconds": time.perf_counter() - started,
    }
    trace = {
        "allowed_pages": light_page_ids,
        "legacy_chunk_hits": [
            {"chunk_id": chunk_id, "score": float(score)} for chunk_id, score in hits
        ],
        "legacy_chunk_touched_pages": sorted(touched_pages),
    }
    return run, {"metrics": row, "trace": trace}


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "queries": len(rows),
        "avg_candidate_pages": _mean([len(row["light_candidate_page_ids"]) for row in rows]),
        "avg_scoped_chunks": _mean([float(row["scoped_chunks"]) for row in rows]),
        "avg_latency_seconds": _mean([float(row["latency_seconds"]) for row in rows]),
        "candidate_page_recall_ceiling": _mean(
            [float(row["candidate_page_recall_ceiling"]) for row in rows]
        ),
        "candidate_file_recall": _mean([float(row["candidate_file_recall"]) for row in rows]),
        "final_file_recall@3": _mean([float(row["final_file_recall@3"]) for row in rows]),
        "final_file_hit@3": _mean([float(row["final_file_hit@3"]) for row in rows]),
        "nDCG@10": _mean([float(row["final_nDCG@10"]) for row in rows]),
        "page_hit@10": _mean([float(row["final_page_hit@10"]) for row in rows]),
        "page_recall@10": _mean([float(row["final_page_recall@10"]) for row in rows]),
        "page_precision@10": _mean([float(row["final_page_precision@10"]) for row in rows]),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Physics on-demand BM25 + legacy second retrieval",
        "",
        "Simple baseline: PDF-inspector + BM25 retrieves top-Klight pages, then cached legacy retrieval returns final top-10 pages.",
        "There is no file retrieval, hierarchical selection or candidate union in this experiment.",
        "",
        "## Fixed configuration",
        "",
        f"- Legacy dense alpha: `{LEGACY_DENSE_ALPHA}`; chunk depth: `{LEGACY_CHUNK_DEPTH}`; page pool: `{PAGE_POOL}`.",
        f"- Final fusion: `{1.0 - LEGACY_WEIGHT:.2f} * light BM25 + {LEGACY_WEIGHT:.2f} * legacy`.",
        f"- Cache: `{report['cache']['text_chunks']}` chunks and `{report['cache']['query_embedding_hits']}/{report['queries']}` query embeddings.",
        "",
        "## Results",
        "",
        "| Klight | Avg pages/query | Candidate page ceiling | Candidate file recall | nDCG@10 | Page hit@10 | Page recall@10 | Final file recall@3 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        m = arm["metrics"]
        lines.append(
            f"| {arm['light_page_k']} | {m['avg_candidate_pages']:.2f} | "
            f"{m['candidate_page_recall_ceiling']:.2%} | {m['candidate_file_recall']:.2%} | "
            f"{m['nDCG@10']:.2%} | {m['page_hit@10']:.2%} | "
            f"{m['page_recall@10']:.2%} | {m['final_file_recall@3']:.2%} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Candidate page ceiling is measured immediately after the BM25 light stage, before legacy ranking.",
        "- Final page metrics are measured after legacy second retrieval and top-10 selection.",
        "- Increasing Klight expands the legacy input scope; it does not guarantee a better top-10 ranking.",
        "- Qrels are used only for evaluation, never for candidate construction or scoring.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--legacy-cache", type=Path, default=DEFAULT_LEGACY_CACHE)
    parser.add_argument("--light-run", type=Path, default=DEFAULT_LIGHT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(
        list(benchmark.questions()),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    qids = [question.qid for question in questions]
    qrels = benchmark.qrels()
    page_ids = _page_vector_units(args.page_vector_dir)
    if len(qids) != 302 or len(page_ids) != 1674:
        raise RuntimeError(f"Expected 302 queries and 1,674 pages, got {len(qids)} and {len(page_ids)}")
    corpus = HierarchyCorpus.from_parsed_run(
        args.parsed_run, subset="physics", page_ids=page_ids
    )
    light_run = _load_light_run(args.light_run, qids, set(page_ids))
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

    output_dir = args.output_dir
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    qid_to_index = {qid: index for index, qid in enumerate(qids)}
    for light_k in LIGHT_PAGE_K:
        rows: list[dict[str, Any]] = []
        run_rows: list[dict[str, Any]] = []
        for question in questions:
            qid = question.qid
            run, payload = _run_query(
                qid,
                question.query,
                light_run[qid],
                qrels[qid],
                corpus,
                chunks,
                by_chunk_id,
                chunk_index,
                dense_score_matrix[:, qid_to_index[qid]],
                chunk_positions_by_page,
                light_k,
            )
            row = payload["metrics"]
            rows.append(row)
            all_rows.append(row)
            all_traces.append({"qid": qid, "light_page_k": light_k, **payload["trace"]})
            run_rows.append({
                "qid": qid,
                "query": question.query,
                "retriever_id": f"physics-ondemand-bm25-legacy-k{light_k}",
                "index_id": f"physics-legacy-chunks-scoped-to-bm25-top{light_k}",
                "chunks": run[:100],
            })
        _write_jsonl(runs_dir / f"ondemand_bm25_legacy_lightk{light_k}_top100.jsonl", run_rows)
        arms.append({"light_page_k": light_k, "metrics": _aggregate(rows)})

    report = {
        "dataset": "vidore_v3/physics",
        "language": "french",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "method": "PDF-inspector+BM25 top-Klight -> cached legacy second retrieval -> top-10",
        "file_retrieval_used": False,
        "config": {
            "light_page_k": list(LIGHT_PAGE_K),
            "final_page_k": FINAL_PAGE_K,
            "legacy_dense_alpha": LEGACY_DENSE_ALPHA,
            "legacy_chunk_depth": LEGACY_CHUNK_DEPTH,
            "legacy_weight": LEGACY_WEIGHT,
            "page_pool": PAGE_POOL,
            "qrels_used_for_ranking": False,
        },
        "sources": {
            "light_run": str(args.light_run),
            "parsed_run": str(args.parsed_run),
            "page_inventory": str(args.page_vector_dir),
        },
        "cache": {
            "text_chunks": legacy_meta["text_chunks"],
            "cross_page_chunks": legacy_meta["cross_page_chunks"],
            "query_embedding_hits": query_meta["cache_hits"],
            "query_embedding_misses": query_meta["cache_misses"],
            "embedding_model": legacy_meta["embedding_model"],
        },
        "arms": arms,
        "timing_seconds": {
            "total": time.perf_counter() - started,
            "note": "Includes cache loading, BM25 index rebuild and retrieval; excludes parsing and model encoding.",
        },
    }
    _write_json(output_dir / "report.json", report)
    _write_jsonl(output_dir / "per_query.jsonl", all_rows)
    _write_jsonl(output_dir / "traces.jsonl", all_traces)
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "elapsed_seconds": round(report["timing_seconds"]["total"], 3),
        "results": {
            f"Klight={arm['light_page_k']}": {
                "candidate_page_ceiling": round(100 * arm["metrics"]["candidate_page_recall_ceiling"], 2),
                "page_recall@10": round(100 * arm["metrics"]["page_recall@10"], 2),
                "nDCG@10": round(100 * arm["metrics"]["nDCG@10"], 2),
            }
            for arm in arms
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
