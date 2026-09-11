"""Hybrid semantic page retrieval followed by a strong page reranker.

This is a controlled two-dataset experiment:

    lexical/hierarchical base -> E5 semantic page signal -> Cohere reranker

The file stage is not rebuilt here.  ``semantic_scoped`` preserves the input
run's candidate pages and therefore isolates page scoring.  ``semantic_union``
adds the global E5 top-100 pages to the input top-100 pages, which is a soft
hierarchical candidate-expansion arm and may introduce pages from files outside
the original hard file scope.

E5 is a bi-encoder semantic model, not a reranker.  Cohere is called through
OpenRouter's ``/api/v1/rerank`` endpoint and is used only after candidate
generation.  No qrels are used for encoding, candidate construction or model
scoring; qrels are used only for evaluation and fold selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import HierarchyCorpus  # noqa: E402
from research.experiments.physics_fielded_kf3_rerank import (  # noqa: E402
    _dotenv_value,
    _openrouter_rerank_scores,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _load_run,
    _safe_name,
    _stratified_folds,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DATASET_DEFAULTS = {
    "physics": {
        "benchmark_root": ROOT / "data/benchmark/vidore_v3",
        "parsed_run": ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb",
        "base_run": ROOT / "data/benchmark/vidore_v3/results/physics_fielded_kf3_openrouter_cohere_rerank/runs/base_fielded_kf3.jsonl",
        "e5_scores": ROOT / "data/benchmark/vidore_v3/results/physics_multilingual_e5_baseline/page_dense_scores.npy",
        "page_metadata": ROOT / "data/output/vsplade/vidore_v3_physics_48q/page_metadata.json",
        "e5_cache": None,
        "language": "french",
        "expected_queries": 302,
        "expected_pages": 1674,
        "expected_files": 42,
        "expected_model": "intfloat/multilingual-e5-small",
        "output": ROOT / "data/benchmark/vidore_v3/results/physics_hybrid_semantic_rerank",
    },
    "industrial": {
        "benchmark_root": ROOT / "data/raw/benchmarks/vidore_v3",
        "parsed_run": ROOT / "data/output/vidore-v3-industrial-kdl-pdf-inspector/fcfa9a665c256e86",
        "base_run": ROOT / "data/benchmark/vidore_v3/results/industrial_discovery_bm25_english.jsonl",
        "e5_scores": None,
        "page_metadata": None,
        "e5_cache": ROOT / "data/output/e5/vidore_v3_industrial_multilingual_e5_small",
        "language": "english",
        "expected_queries": 283,
        "expected_pages": 5244,
        "expected_files": 27,
        "expected_model": "intfloat/multilingual-e5-small",
        "output": ROOT / "data/benchmark/vidore_v3/results/industrial_hybrid_semantic_rerank",
    },
}

SEMANTIC_WEIGHT = 0.30
BASE_WEIGHT = 1.0 - SEMANTIC_WEIGHT
RRF_CONSTANT = 20
RERANK_DEPTH = 50
FINAL_DEPTH = 10
RERANK_WEIGHT = 0.80
E5_WINDOW = 240
E5_OVERLAP = 40


def _normalise(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not array.size:
        return array
    minimum = float(array.min())
    shifted = array - minimum if minimum < 0.0 else array.copy()
    maximum = float(shifted.max())
    return shifted / maximum if maximum > 0.0 else np.zeros_like(shifted)


def _segments(text: str, window: int = E5_WINDOW, overlap: int = E5_OVERLAP) -> list[str]:
    words = str(text).split()
    if not words:
        return [" "]
    if len(words) <= window:
        return [str(text)]
    step = max(1, window - overlap)
    output: list[str] = []
    for start in range(0, len(words), step):
        output.append(" ".join(words[start : start + window]))
        if start + window >= len(words):
            break
    return output


def _save_or_load_industrial_e5(
    *,
    cache_dir: Path,
    page_ids: Sequence[str],
    page_texts: Sequence[str],
    queries: Sequence[str],
    model_name: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    scores_path = cache_dir / "page_dense_scores.npy"
    pages_path = cache_dir / "page_ids.json"
    queries_path = cache_dir / "queries.json"
    meta_path = cache_dir / "metadata.json"
    if scores_path.exists() and pages_path.exists() and queries_path.exists():
        cached_pages = json.loads(pages_path.read_text(encoding="utf-8"))
        cached_queries = json.loads(queries_path.read_text(encoding="utf-8"))
        scores = np.load(scores_path, mmap_mode="r")
        if cached_pages == list(page_ids) and cached_queries == list(queries) and scores.shape == (len(queries), len(page_ids)):
            metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            metadata["cache_hit"] = True
            return scores, metadata

    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    segments: list[str] = []
    segment_page_indices: list[int] = []
    for page_index, text in enumerate(page_texts):
        page_segments = _segments(text)
        segments.extend("passage: " + segment for segment in page_segments)
        segment_page_indices.extend([page_index] * len(page_segments))
    model = SentenceTransformer(model_name)
    passage_vectors = model.encode(
        segments,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    query_vectors = model.encode(
        ["query: " + query for query in queries],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    segment_scores = query_vectors @ passage_vectors.T
    page_scores = np.full((len(queries), len(page_ids)), -np.inf, dtype=np.float32)
    segment_pages_np = np.asarray(segment_page_indices, dtype=np.int64)
    for query_index in range(len(queries)):
        np.maximum.at(page_scores[query_index], segment_pages_np, segment_scores[query_index].astype(np.float32, copy=False))

    np.save(scores_path, page_scores)
    pages_path.write_text(json.dumps(list(page_ids), ensure_ascii=False), encoding="utf-8")
    queries_path.write_text(json.dumps(list(queries), ensure_ascii=False), encoding="utf-8")
    metadata = {
        "model": model_name,
        "window_words": E5_WINDOW,
        "overlap_words": E5_OVERLAP,
        "segments": len(segments),
        "pages": len(page_ids),
        "queries": len(queries),
        "encoding_seconds": round(time.perf_counter() - started, 6),
        "cache_hit": False,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return page_scores, metadata


def _load_page_ids(path: Path | None, benchmark: ViDoreV3) -> list[str]:
    if path is not None:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [str(row["unit_id"]) for row in rows]
    return sorted(str(document.doc_id) for document in benchmark.corpus())


def _load_cached_rerank_scores(path: Path) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    run = _load_run(path)
    for qid, items in run.items():
        scores[qid] = {
            str(item["chunk_id"]): float(item["rerank_score"])
            for item in items
            if item.get("rerank_score") is not None
        }
    if not scores or any(not values for values in scores.values()):
        raise RuntimeError(f"Cached rerank run has no complete rerank scores: {path}")
    return scores


def _base_page_score(item: Mapping[str, Any]) -> float:
    return float(item.get("score", 0.0))


def _build_semantic_runs(
    *,
    base_run: Mapping[str, list[dict[str, Any]]],
    e5_scores: np.ndarray,
    page_ids: Sequence[str],
    qids: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    position = {page_id: index for index, page_id in enumerate(page_ids)}
    runs: dict[str, list[dict[str, Any]]] = {}
    details: dict[str, Any] = {
        "semantic_model": "intfloat/multilingual-e5-small",
        "semantic_weight": SEMANTIC_WEIGHT,
        "base_weight": BASE_WEIGHT,
        "rrf_constant": RRF_CONSTANT,
        "global_e5_depth": 100,
        "base_depth": 100,
    }
    e5_global: dict[str, list[str]] = {}
    for query_index, qid in enumerate(qids):
        if e5_scores.shape[1] != len(page_ids):
            raise RuntimeError(f"E5 score/page mismatch: {e5_scores.shape} versus {len(page_ids)} pages")
        base_items = list(base_run[qid][:100])
        base_page_ids = [str(item["chunk_id"]) for item in base_items]
        base_ranks = {page_id: rank for rank, page_id in enumerate(base_page_ids, 1)}
        base_scores = {str(item["chunk_id"]): _base_page_score(item) for item in base_items}
        base_norm = _normalise([base_scores[page_id] for page_id in base_page_ids])
        e5_values_base = np.asarray([e5_scores[query_index, position[page_id]] for page_id in base_page_ids], dtype=np.float32)
        e5_norm_base = _normalise(e5_values_base)
        scoped_values = BASE_WEIGHT * base_norm + SEMANTIC_WEIGHT * e5_norm_base
        scoped_order = np.argsort(-scoped_values, kind="stable")
        runs.setdefault("semantic_scoped", {})
        runs["semantic_scoped"][qid] = [
            {"chunk_id": base_page_ids[int(index)], "doc_id": base_page_ids[int(index)], "score": float(scoped_values[int(index)]), "rank": rank}
            for rank, index in enumerate(scoped_order[:100], 1)
        ]

        global_order = np.argsort(-e5_scores[query_index], kind="stable")[:100]
        semantic_pages = [page_ids[int(index)] for index in global_order]
        e5_global[qid] = semantic_pages
        candidate_pages = sorted(set(base_page_ids) | set(semantic_pages))
        base_prior = np.asarray(
            [1.0 / (RRF_CONSTANT + base_ranks[page_id]) if page_id in base_ranks else 0.0 for page_id in candidate_pages],
            dtype=np.float32,
        )
        e5_candidate = np.asarray([e5_scores[query_index, position[page_id]] for page_id in candidate_pages], dtype=np.float32)
        union_values = BASE_WEIGHT * _normalise(base_prior) + SEMANTIC_WEIGHT * _normalise(e5_candidate)
        union_order = np.argsort(-union_values, kind="stable")
        runs.setdefault("semantic_union", {})
        runs["semantic_union"][qid] = [
            {"chunk_id": candidate_pages[int(index)], "doc_id": candidate_pages[int(index)], "score": float(union_values[int(index)]), "rank": rank}
            for rank, index in enumerate(union_order[:100], 1)
        ]

        e5_only_values = e5_scores[query_index]
        e5_order = np.argsort(-e5_only_values, kind="stable")[:100]
        runs.setdefault("e5_global", {})
        runs["e5_global"][qid] = [
            {"chunk_id": page_ids[int(index)], "doc_id": page_ids[int(index)], "score": float(e5_only_values[int(index)]), "rank": rank}
            for rank, index in enumerate(e5_order, 1)
        ]
    details["semantic_global_candidates"] = e5_global
    return runs, details


def _rerank_local_scores(
    source_run: Mapping[str, list[dict[str, Any]]],
    rerank_scores: Mapping[str, Mapping[str, float]],
    *,
    mode: str,
    candidate_depth: int = RERANK_DEPTH,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for qid, items in source_run.items():
        candidates = list(items[:100])
        band = candidates[:candidate_depth]
        if not all(str(item["chunk_id"]) in rerank_scores[qid] for item in band):
            missing = [str(item["chunk_id"]) for item in band if str(item["chunk_id"]) not in rerank_scores[qid]][:3]
            raise RuntimeError(f"Missing reranker scores for {qid}: {missing}")
        rr = np.asarray([rerank_scores[qid][str(item["chunk_id"])] for item in band], dtype=np.float32)
        base = np.asarray([_base_page_score(item) for item in band], dtype=np.float32)
        rr_norm = _normalise(rr)
        base_norm = _normalise(base)
        if mode == "rerank_only":
            combined = rr_norm
        elif mode == "rerank_base80":
            combined = RERANK_WEIGHT * rr_norm + (1.0 - RERANK_WEIGHT) * base_norm
        else:
            raise ValueError(mode)
        order = np.argsort(-combined, kind="stable")
        ranked: list[dict[str, Any]] = []
        band_ids = {str(item["chunk_id"]) for item in band}
        for rank, index in enumerate(order, 1):
            item = dict(band[int(index)])
            item.update(
                {
                    "rank": rank,
                    "score": float(combined[int(index)]),
                    "rerank_score": float(rr[int(index)]),
                    "base_score": float(base[int(index)]),
                    "rerank_mode": mode,
                }
            )
            ranked.append(item)
        tail = [dict(item) for item in candidates if str(item["chunk_id"]) not in band_ids]
        for offset, item in enumerate(tail, len(ranked) + 1):
            item["rank"] = offset
            item["rerank_mode"] = mode
        output[qid] = ranked + tail
    return output


def _candidate_recall(
    run: Mapping[str, list[dict[str, Any]]],
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> float:
    values = []
    for qid in qids:
        gold = set(qrels[qid])
        candidates = {str(item["chunk_id"]) for item in run[qid][:100]}
        values.append(len(gold & candidates) / len(gold) if gold else 0.0)
    return float(np.mean(values)) if values else 0.0


def _metric_for_qids(metric: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metric["per_query"] if row["qid"] in qids]
    return float(np.mean([float(row["page_recall@10"]) for row in rows])) if rows else 0.0


def _run_metrics(
    run: Mapping[str, list[dict[str, Any]]],
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    candidate_source: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    metric = _derived_metrics(run, qids, qrels)
    return {
        "page_recall@10": float(metric["page_recall@10"]),
        "ndcg@10": float(metric["ndcg@10"]),
        "page_hit@10": float(metric["page_hit@10"]),
        "file_recall@3": float(metric["file_metrics_by_k"]["3"]["file_recall"]),
        "candidate_page_recall@100": _candidate_recall(candidate_source, qids, qrels),
        "metrics": metric,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# {report['dataset']} hybrid semantic retrieval + reranking",
        "",
        "E5 is used as a semantic page signal; Cohere is used as a post-retrieval reranker through OpenRouter.",
        "",
        "## Full-set screening",
        "",
        "| Method | Page recall@10 | nDCG@10 | Page hit@10 | File recall@3 | Candidate recall@100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    methods = report["methods"]
    for name, value in sorted(methods.items(), key=lambda item: item[1]["page_recall@10"], reverse=True):
        lines.append(
            f"| {name} | {value['page_recall@10']:.2%} | {value['ndcg@10']:.2f} | "
            f"{value['page_hit@10']:.2%} | {value['file_recall@3']:.2%} | "
            f"{value['candidate_page_recall@100']:.2%} |"
        )
    lines += [
        "",
        "## OOF selection",
        "",
        f"Selected method by fold: `{report['oof']['selected_method']}`.",
        f"OOF page recall@10: **{report['oof']['metrics']['page_recall@10']:.2%}**.",
        f"OOF nDCG@10: **{report['oof']['metrics']['ndcg@10']:.2f}**.",
        f"OOF page hit@10: **{report['oof']['metrics']['page_hit@10']:.2%}**.",
        "",
        "| Fold | Selected method | Test page recall@10 |",
        "|---:|---|---:|",
    ]
    for row in report["oof"]["folds"]:
        lines.append(f"| {row['fold']} | `{row['selected_method']}` | {row['test_page_recall@10']:.2%} |")
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- `semantic_scoped` keeps the base top-100 page set and changes only its page score.",
        "- `semantic_union` adds global E5 top-100 pages before ranking; it is a soft hierarchy arm and can escape the base hard file scope.",
        "- Rerank arms use R=50 and final top-10. The reranker cannot recover pages absent from its source top-100.",
        "- Reranking does not change the file proposal itself; file recall is derived from the page run and may remain unchanged when the candidate set is unchanged.",
        "- Physics V-SPLADE remains an English-query visual sparse signal against French qrels; Industrial has no V-SPLADE signal here.",
        f"- OpenRouter reranker model: `{report['reranker']['model']}`; requests: {report['reranker']['requests']}.",
        "",
        "## Timing",
        "",
        f"- E5 encoding/cache: {report['timing_seconds']['e5']:.2f}s.",
        f"- Reranker API: {report['timing_seconds']['reranker_api']:.2f}s.",
        f"- Total: {report['timing_seconds']['total']:.2f}s.",
    ]
    return "\n".join(lines) + "\n"


def _load_base_run(path: Path, qids: Sequence[str], page_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    run = _load_run(path)
    if set(run) != set(qids):
        raise RuntimeError(f"Base run qids mismatch: {len(run)} versus {len(qids)}")
    for qid in qids:
        rows = run[qid]
        if len(rows) < FINAL_DEPTH:
            raise RuntimeError(f"Base run has fewer than {FINAL_DEPTH} pages for {qid}: {len(rows)}")
        ids = [str(item["chunk_id"]) for item in rows[:100]]
        if len(ids) != len(set(ids)) or not set(ids).issubset(page_ids):
            raise RuntimeError(f"Invalid base page candidates for {qid}")
    return {qid: list(run[qid][:100]) for qid in qids}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFAULTS), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--parsed-run", type=Path)
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--e5-scores", type=Path)
    parser.add_argument("--page-metadata", type=Path)
    parser.add_argument("--e5-cache", type=Path)
    parser.add_argument("--e5-model", default=None)
    parser.add_argument("--e5-batch-size", type=int, default=16)
    parser.add_argument("--openrouter-model", default="cohere/rerank-v3.5")
    parser.add_argument("--openrouter-workers", type=int, default=8)
    parser.add_argument("--openrouter-timeout", type=float, default=120.0)
    parser.add_argument("--openrouter-retries", type=int, default=2)
    parser.add_argument("--reuse-cached-physics-rerank", action="store_true")
    args = parser.parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    benchmark_root = args.benchmark_root or defaults["benchmark_root"]
    parsed_run = args.parsed_run or defaults["parsed_run"]
    base_run_path = args.base_run or defaults["base_run"]
    output_dir = args.output_dir or defaults["output"]
    e5_scores_path = args.e5_scores if args.e5_scores is not None else defaults["e5_scores"]
    page_metadata_path = args.page_metadata if args.page_metadata is not None else defaults["page_metadata"]
    e5_cache = args.e5_cache if args.e5_cache is not None else defaults["e5_cache"]
    e5_model = args.e5_model or defaults["expected_model"]
    started = time.perf_counter()

    benchmark = ViDoreV3(root=benchmark_root, subset=args.dataset, language=defaults["language"])
    questions_list = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions_list]
    queries = {q.qid: q.query for q in questions_list}
    qrels = benchmark.qrels()
    if len(qids) != defaults["expected_queries"]:
        raise RuntimeError(f"Expected {defaults['expected_queries']} queries, got {len(qids)}")
    page_ids = _load_page_ids(page_metadata_path, benchmark)
    if len(page_ids) != defaults["expected_pages"] or len(set(page_ids)) != len(page_ids):
        raise RuntimeError(f"Expected {defaults['expected_pages']} unique pages, got {len(set(page_ids))}")
    corpus = HierarchyCorpus.from_parsed_run(parsed_run, subset=args.dataset, page_ids=page_ids)
    if len(corpus.file_order) != defaults["expected_files"] or len(corpus.page_order) != len(page_ids):
        raise RuntimeError("Parsed corpus inventory mismatch")
    page_texts = [corpus.pages[page_id].text for page_id in page_ids]
    base_run = _load_base_run(base_run_path, qids, set(page_ids))

    e5_started = time.perf_counter()
    if e5_scores_path is not None:
        e5_scores = np.load(e5_scores_path, mmap_mode="r")
        e5_meta = {"source": str(e5_scores_path), "cache_hit": True}
    else:
        if e5_cache is None:
            raise RuntimeError("Industrial E5 cache path is missing")
        e5_scores, e5_meta = _save_or_load_industrial_e5(
            cache_dir=e5_cache,
            page_ids=page_ids,
            page_texts=page_texts,
            queries=[queries[qid] for qid in qids],
            model_name=e5_model,
            batch_size=args.e5_batch_size,
        )
        e5_meta["source"] = str(e5_cache / "page_dense_scores.npy")
    if e5_scores.shape != (len(qids), len(page_ids)):
        raise RuntimeError(f"Unexpected E5 score shape: {e5_scores.shape}")
    e5_seconds = time.perf_counter() - e5_started

    source_runs, semantic_details = _build_semantic_runs(
        base_run=base_run, e5_scores=e5_scores, page_ids=page_ids, qids=qids
    )
    source_runs["base"] = base_run

    rerank_scores_by_source: dict[str, dict[str, dict[str, float]]] = {}
    reranker_requests = 0
    reranker_started = time.perf_counter()
    cached_rerank_path = ROOT / "data/benchmark/vidore_v3/results/physics_fielded_kf3_openrouter_cohere_rerank/runs/or_only_r100.jsonl"
    if args.dataset == "physics" and args.reuse_cached_physics_rerank and cached_rerank_path.exists():
        cached = _load_cached_rerank_scores(cached_rerank_path)
        base_ids = {qid: [str(item["chunk_id"]) for item in source_runs["base"][qid][:100]] for qid in qids}
        if set(cached) == set(qids) and all(set(base_ids[qid]).issubset(cached[qid]) for qid in qids):
            rerank_scores_by_source["base"] = cached
            rerank_scores_by_source["semantic_scoped"] = cached
    sources_to_call = [name for name in ("base", "semantic_scoped", "semantic_union") if name not in rerank_scores_by_source]
    # Candidate sets for base and semantic_scoped are identical by construction.
    if "base" in sources_to_call and "semantic_scoped" in sources_to_call:
        sources_to_call.remove("semantic_scoped")
    for source_name in sources_to_call:
        candidate_pages_by_qid = {
            qid: [str(item["chunk_id"]) for item in source_runs[source_name][qid][:100]]
            for qid in qids
        }
        scores, meta = _openrouter_rerank_scores(
            corpus=corpus,
            qids=qids,
            queries=queries,
            candidate_pages_by_qid=candidate_pages_by_qid,
            model_name=args.openrouter_model,
            endpoint="https://openrouter.ai/api/v1/rerank",
            workers=args.openrouter_workers,
            timeout_seconds=args.openrouter_timeout,
            max_retries=args.openrouter_retries,
        )
        rerank_scores_by_source[source_name] = scores
        reranker_requests += int(meta["requests"])
        if source_name == "base" and "semantic_scoped" not in rerank_scores_by_source:
            rerank_scores_by_source["semantic_scoped"] = scores
    reranker_seconds = time.perf_counter() - reranker_started

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    candidate_source: dict[str, Mapping[str, list[dict[str, Any]]]] = {}
    for source_name, source_run in source_runs.items():
        runs[source_name] = source_run
        candidate_source[source_name] = source_run
        if source_name not in rerank_scores_by_source:
            # e5_global is a semantic-only control; it has no reranker scores.
            continue
        for mode in ("rerank_only", "rerank_base80"):
            name = f"{source_name}_{mode}_r{RERANK_DEPTH}"
            runs[name] = _rerank_local_scores(source_run, rerank_scores_by_source[source_name], mode=mode)
            candidate_source[name] = source_run

    metrics = {
        name: _run_metrics(run, qids, qrels, candidate_source=candidate_source[name])
        for name, run in runs.items()
    }
    folds = _stratified_folds(qids, {q.qid: q for q in questions_list}, qrels)
    oof_run: dict[str, list[dict[str, Any]]] = {}
    fold_rows: list[dict[str, Any]] = []
    candidates = list(runs)
    for fold, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train = set(qids) - heldout_set
        winner = max(candidates, key=lambda name: (_metric_for_qids(metrics[name]["metrics"], train), name))
        oof_run.update({qid: runs[winner][qid] for qid in heldout})
        fold_rows.append(
            {
                "fold": fold,
                "selected_method": winner,
                "train_page_recall@10": _metric_for_qids(metrics[winner]["metrics"], train),
                "test_page_recall@10": _metric_for_qids(metrics[winner]["metrics"], heldout_set),
            }
        )
    oof_metric = _derived_metrics(oof_run, qids, qrels)
    selected_method = max(candidates, key=lambda name: (metrics[name]["page_recall@10"], metrics[name]["ndcg@10"], name))
    oof_selected = max(
        set(row["selected_method"] for row in fold_rows),
        key=lambda name: sum(row["selected_method"] == name for row in fold_rows),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name, run in runs.items():
        _write_run(runs_dir / f"{_safe_name(name)}.jsonl", run, qids, queries=queries)
    _write_run(output_dir / "oof_run.jsonl", oof_run, qids, queries=queries)
    per_query_path = output_dir / "per_query.jsonl"
    with per_query_path.open("w", encoding="utf-8") as handle:
        for name, run in runs.items():
            per_query = {row["qid"]: row for row in metrics[name]["metrics"]["per_query"]}
            for qid in qids:
                handle.write(
                    json.dumps(
                        {
                            "method": name,
                            "qid": qid,
                            "candidate_source": name if name in source_runs else name.rsplit("_rerank_", 1)[0],
                            "candidate_pages": [str(item["chunk_id"]) for item in candidate_source[name][qid][:100]],
                            "metrics": per_query[qid],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    report: dict[str, Any] = {
        "dataset": f"vidore_v3/{args.dataset}",
        "language": defaults["language"],
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.file_order),
        "base_run": str(base_run_path),
        "e5": {**e5_meta, "model": e5_model, **semantic_details, "encoding_seconds": e5_seconds},
        "reranker": {
            "model": args.openrouter_model,
            "endpoint": "https://openrouter.ai/api/v1/rerank",
            "candidate_depth": 100,
            "rerank_depth": RERANK_DEPTH,
            "final_depth": FINAL_DEPTH,
            "rerank_weight": RERANK_WEIGHT,
            "requests": reranker_requests,
            "cached_physics_scores_reused": args.dataset == "physics" and args.reuse_cached_physics_rerank,
        },
        "methods": {
            name: {key: value for key, value in data.items() if key != "metrics"}
            for name, data in metrics.items()
        },
        "best_full_set_method": selected_method,
        "oof": {
            "selected_method": oof_selected,
            "metrics": {
                "page_recall@10": float(oof_metric["page_recall@10"]),
                "ndcg@10": float(oof_metric["ndcg@10"]),
                "page_hit@10": float(oof_metric["page_hit@10"]),
                "file_recall@3": float(oof_metric["file_metrics_by_k"]["3"]["file_recall"]),
            },
            "folds": fold_rows,
        },
        "timing_seconds": {
            "e5": e5_seconds,
            "reranker_api": reranker_seconds,
            "total": time.perf_counter() - started,
        },
        "validation": {
            "query_count": len(qids),
            "page_count": len(page_ids),
            "file_count": len(corpus.file_order),
            "base_candidates_up_to_100": all(FINAL_DEPTH <= len(base_run[qid]) <= 100 for qid in qids),
            "qrel_pages_reachable": all(set(qrels[qid]).issubset(set(page_ids)) for qid in qids),
            "semantic_union_is_superset_of_base": all(
                set(item["chunk_id"] for item in source_runs["base"][qid]).issubset(
                    set(item["chunk_id"] for item in source_runs["semantic_union"][qid])
                )
                for qid in qids
            ),
        },
        "notes": [
            "E5 is a semantic bi-encoder score, not a reranker.",
            "semantic_scoped preserves the base candidate page set; semantic_union adds global E5 top-100 pages.",
            "Cohere reranking is performed only after the page source is built, with R=50 and final top-10.",
            "Physics cached V-SPLADE remains part of the fielded base run; Industrial has no V-SPLADE artifact.",
            "All model scores and candidate construction are qrel-free; OOF fold selection is used only for reporting the selected arm.",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "dataset": args.dataset,
                "best_full_set_method": selected_method,
                "best_full_page_recall@10": round(metrics[selected_method]["page_recall@10"] * 100, 2),
                "oof_selected_method": oof_selected,
                "oof_page_recall@10": round(oof_metric["page_recall@10"] * 100, 2),
                "e5_seconds": round(e5_seconds, 3),
                "reranker_seconds": round(reranker_seconds, 3),
                "reranker_requests": reranker_requests,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
