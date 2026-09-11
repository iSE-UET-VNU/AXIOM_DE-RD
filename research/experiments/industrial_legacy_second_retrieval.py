"""Evaluate cached legacy chunk retrieval as an Industrial second pass.

This is the closest existing implementation of a stronger iterative loop for
the Industrial benchmark: the cheap page BM25 pass proposes pages/files, then
the cached legacy text-chunk BM25 + ``text-embedding-3-small`` retriever
refines the evidence inside that scope.  All chunk/page mappings come from
the persisted parser offsets; no qrels, API call, rendering, or re-embedding
is performed.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re
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
    build_bm25,
    normalise_scores,
    sort_scores,
)
from research.experiments.industrial_hierarchical_retrieval import (  # noqa: E402
    DEFAULT_BASELINE_RUN,
    _build_indexes,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    HierarchicalRetriever,
    _derived_metrics,
    _file_id,
    _load_run,
    _paired_comparison,
    _safe_name,
    _stratified_folds,
    _write_run,
)
from src.chunking_embedding.embedders import sanitize_text  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.evaluation.pipeline_pages import canonical_doc  # noqa: E402
from src.retrieval.fusion import alpha_fuse  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_PARSED_RUN = (
    ROOT / "data/output/vidore-v3-industrial-kdl-pdf-inspector/fcfa9a665c256e86"
)
DEFAULT_BENCHMARK_ROOT = ROOT / "data/raw/benchmarks/vidore_v3"
DEFAULT_LEGACY_CACHE = ROOT / "data/work/embedding_cache/text-embedding-3-small"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/industrial_legacy_second_retrieval"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
PAGE_MARKER = re.compile(r"\*\*Page\s+(\d+)\*\*")


@dataclass(frozen=True)
class LegacyChunk:
    chunk_id: str
    file_id: str
    page_ids: tuple[str, ...]
    text: str
    vector: np.ndarray


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _page_intervals(main_text: str, page_ids: Sequence[str]) -> list[tuple[int, int, str]]:
    page_by_number = {index: page_id for index, page_id in enumerate(page_ids)}
    markers = [(match.start(), int(match.group(1)) - 1) for match in PAGE_MARKER.finditer(main_text)]
    intervals: list[tuple[int, int, str]] = []
    if markers:
        first_start, _ = markers[0]
        if first_start > 0 and 0 in page_by_number:
            intervals.append((0, first_start, page_by_number[0]))
        for index, (start, page_number) in enumerate(markers):
            end = markers[index + 1][0] if index + 1 < len(markers) else len(main_text)
            page_id = page_by_number.get(page_number)
            if page_id is not None and end > start:
                intervals.append((start, end, page_id))
    elif page_ids:
        intervals.append((0, len(main_text), page_ids[0]))
    return intervals


def _pages_for_span(start: int, end: int, intervals: Sequence[tuple[int, int, str]], page_ids: Sequence[str]) -> tuple[str, ...]:
    touched = [page_id for left, right, page_id in intervals if max(start, left) < min(end, right)]
    if touched:
        return tuple(dict.fromkeys(touched))
    if not intervals:
        return (page_ids[0],) if page_ids else ()
    nearest = min(intervals, key=lambda item: abs(item[0] - start))
    return (nearest[2],)


def _load_legacy_chunks(parsed_run: Path, corpus: HierarchyCorpus) -> tuple[list[LegacyChunk], dict[str, Any]]:
    chunks: list[LegacyChunk] = []
    documents = 0
    cross_page = 0
    by_file: defaultdict[str, int] = defaultdict(int)
    for path in sorted((parsed_run / "documents").glob("*.json")):
        payload = _load_json(path)
        document = payload.get("document") or {}
        file_id = f"industrial::{canonical_doc(document.get('file_name'))}"
        page_ids = corpus.file_to_pages.get(file_id, [])
        if not page_ids:
            raise RuntimeError(f"Legacy file is absent from corpus: {file_id}")
        documents += 1
        intervals = _page_intervals(str((payload.get("content") or {}).get("main_text") or ""), page_ids)
        for item in (payload.get("retrieval") or {}).get("items", []):
            if item.get("type") != "text":
                continue
            text = str(((item.get("content") or {}).get("text")) or "").strip()
            position = item.get("position") or {}
            embeddings = item.get("embeddings") or []
            if not text or position.get("start_char") is None or position.get("end_char") is None:
                continue
            if not embeddings or not embeddings[0].get("values"):
                raise RuntimeError(f"Missing embedding for {item.get('item_id')} in {path}")
            vector = np.asarray(embeddings[0]["values"], dtype=np.float32)
            if vector.shape != (1536,):
                raise RuntimeError(f"Unexpected embedding shape {vector.shape}")
            pages = _pages_for_span(int(position["start_char"]), int(position["end_char"]), intervals, page_ids)
            chunk_id = f"{file_id}#legacy_chunk={item['item_id']}"
            chunks.append(LegacyChunk(chunk_id, file_id, pages, text, vector))
            by_file[file_id] += 1
            cross_page += int(len(pages) > 1)
    if documents != len(corpus.files) or not chunks:
        raise RuntimeError(f"Expected {len(corpus.files)} documents and nonempty legacy chunks; got {documents}, {len(chunks)}")
    return chunks, {
        "documents": documents,
        "text_chunks": len(chunks),
        "cross_page_chunks": cross_page,
        "cross_page_rate": cross_page / len(chunks),
        "chunks_by_file_min": min(by_file.values()),
        "chunks_by_file_max": max(by_file.values()),
        "embedding_model": EMBEDDING_MODEL,
        "source": str(parsed_run),
    }


def _load_query_vectors(questions: Sequence[Any], cache_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    result: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for question in questions:
        key = hashlib.sha1(f"{EMBEDDING_MODEL}|{sanitize_text(question.query)}".encode("utf-8", "ignore")).hexdigest()
        path = cache_dir / f"emb_{key}.json"
        if not path.is_file():
            missing.append(question.qid)
            continue
        vector = np.asarray(_load_json(path), dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        if vector.shape != (1536,):
            raise RuntimeError(f"Unexpected query vector shape {vector.shape}")
        result[question.qid] = vector
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} cached query vectors: {missing[:5]}")
    return result, {"cache_dir": str(cache_dir), "query_count": len(result), "cache_hits": len(result), "cache_misses": len(missing), "model": EMBEDDING_MODEL}


def _pool(values: Mapping[str, list[float]], mode: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for page_id, raw in values.items():
        scores = sorted((max(0.0, float(value)) for value in raw), reverse=True)
        if mode == "max":
            result[page_id] = scores[0] if scores else 0.0
        elif mode == "sum_top2":
            result[page_id] = sum(scores[:2])
        elif mode == "coverage":
            result[page_id] = sum(1.0 / (10.0 + rank) for rank in range(1, min(10, len(scores)) + 1))
        else:
            raise ValueError(mode)
    return result


def _retrieve_chunks(chunks: Sequence[LegacyChunk], index: BM25Index, matrix: np.ndarray, query: str, query_vector: np.ndarray, allowed_pages: set[str], alpha: float, depth: int) -> list[tuple[str, float]]:
    allowed = {position for position, chunk in enumerate(chunks) if allowed_pages.intersection(chunk.page_ids)}
    if not allowed:
        return []
    dense_scores = matrix @ query_vector
    dense_positions = sorted(allowed, key=lambda position: (-float(dense_scores[position]), position))[:depth]
    dense_hits = [(index.chunk_ids[position], float(dense_scores[position])) for position in dense_positions]
    sparse_hits = [(index.chunk_ids[position], float(score)) for position, score in index.search(query, depth, allowed)]
    return alpha_fuse(dense_hits, sparse_hits, alpha, depth)


def _retrieve_one(
    corpus: HierarchyCorpus,
    page_index: BM25Index,
    retriever: HierarchicalRetriever,
    chunks: Sequence[LegacyChunk],
    chunk_index: BM25Index,
    chunk_matrix: np.ndarray,
    query_vector: np.ndarray,
    qid: str,
    query: str,
    *,
    scope: str,
    gamma: float,
    alpha: float,
    pool: str,
    depth: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if scope == "global":
        base_scores_raw = {page_index.chunk_ids[pos]: float(score) for pos, score in page_index.search(query, len(corpus.page_order))}
        base_pages = list(corpus.page_order)
    elif scope == "file10":
        config = CascadeConfig(name="industrial-kf10", file_representation="all_text", file_pool="max", k_files=10, page_depth=len(corpus.page_order), final_depth=len(corpus.page_order), bm25_weight=1.0, parent_weight=0.15, file_direct_weight=0.5)
        _, trace = retriever.retrieve(qid, query, config)
        base_pages = [str(row["node_id"]) for row in trace["page_candidates"]]
        base_scores_raw = {str(row["node_id"]): float(row["score"]) for row in trace["page_candidates"]}
    else:
        raise ValueError(scope)
    allowed_pages = set(base_pages)
    hits = _retrieve_chunks(chunks, chunk_index, chunk_matrix, query, query_vector, allowed_pages, alpha, depth)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for chunk_id, score in hits:
        for page_id in by_id[chunk_id].page_ids:
            if page_id in allowed_pages:
                grouped[page_id].append(score)
    fine_scores = _pool(grouped, pool)
    base_norm = normalise_scores(base_scores_raw)
    fine_norm = normalise_scores(fine_scores)
    final_scores = {
        page_id: (1.0 - gamma) * base_norm.get(page_id, 0.0) + gamma * fine_norm.get(page_id, 0.0)
        for page_id in base_pages
    }
    ranked = sort_scores(final_scores)[:100]
    run = [
        {"chunk_id": page_id, "doc_id": page_id, "text": corpus.pages[page_id].text, "score": round(score, 8), "rank": rank, "scores": {"page_bm25": round(base_norm.get(page_id, 0.0), 8), "legacy_chunk": round(fine_norm.get(page_id, 0.0), 8), "final": round(score, 8)}}
        for rank, (page_id, score) in enumerate(ranked, 1)
    ]
    chunk_pages = {page_id for chunk_id, _ in hits for page_id in by_id[chunk_id].page_ids if page_id in allowed_pages}
    return run, {"qid": qid, "scope": scope, "gamma": gamma, "alpha": alpha, "pool": pool, "page_pool": base_pages[:100], "legacy_chunk_pages": sorted(chunk_pages), "legacy_hits": [{"chunk_id": chunk_id, "page_ids": list(by_id[chunk_id].page_ids), "score": round(score, 8)} for chunk_id, score in hits], "selected_file_count": len({_file_id(page_id) for page_id in allowed_pages})}


def _metric_for_qids(method_metrics: Mapping[str, Any], qids: set[str]) -> tuple[float, float]:
    rows = [row for row in method_metrics["per_query"] if str(row["qid"]) in qids]
    if not rows:
        return 0.0, 0.0
    page = sum(float(row["page_recall@10"]) for row in rows) / len(rows)
    file_values: list[float] = []
    for row in rows:
        candidates = row["top10_files_from_top100_pages"][:3]
        gold = set(row["gold_files"])
        file_values.append(len(set(candidates) & gold) / len(gold) if gold else 0.0)
    return page, sum(file_values) / len(file_values)


def _oof_selection(
    qids: Sequence[str],
    questions: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int]],
    methods: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Select the feedback arm on four folds and evaluate on the fifth."""
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
        selected.append({
            "fold": fold,
            "heldout_qids": heldout,
            "selected_method": winner,
            "train_page_recall@10": _metric_for_qids(methods[winner]["retrieval_metrics"], train)[0],
            "train_file_recall@3": _metric_for_qids(methods[winner]["retrieval_metrics"], train)[1],
        })
    return {
        "folds": selected,
        "selected_method_counts": {name: sum(row["selected_method"] == name for row in selected) for name in methods},
        "run": oof_run,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--legacy-cache", type=Path, default=DEFAULT_LEGACY_CACHE)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="industrial", language="english")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    qrels = benchmark.qrels()
    page_ids = sorted(document.doc_id for document in benchmark.corpus())
    corpus, page_index, file_indexes, index_counts = _build_indexes(args.parsed_run, page_ids)
    retriever = HierarchicalRetriever(corpus, page_index=page_index, file_indexes=file_indexes, fine_indexes={}, visual_scores={})
    chunks, legacy_meta = _load_legacy_chunks(args.parsed_run, corpus)
    query_vectors, query_meta = _load_query_vectors(questions, args.legacy_cache)
    chunk_matrix = np.asarray([chunk.vector for chunk in chunks], dtype=np.float32)
    chunk_matrix /= np.clip(np.linalg.norm(chunk_matrix, axis=1, keepdims=True), 1e-12, None)
    chunk_index = BM25Index(analyzer_name="auto").build([{"chunk_id": c.chunk_id, "doc_id": c.file_id, "text": c.text} for c in chunks])

    baseline = _load_run(args.baseline_run)
    methods: dict[str, dict[str, Any]] = {"Cached page BM25 baseline": {"config": None, "retrieval_metrics": _derived_metrics(baseline, qids, qrels), "timing_seconds": {"retrieval": 0.0}}}
    specs = [
        ("global", "max", 0.10), ("global", "max", 0.25), ("global", "max", 0.40),
        ("global", "sum_top2", 0.25), ("global", "coverage", 0.25),
        ("file10", "max", 0.03), ("file10", "max", 0.05), ("file10", "max", 0.08),
        ("file10", "max", 0.10), ("file10", "max", 0.12), ("file10", "max", 0.15),
        ("file10", "max", 0.25), ("file10", "max", 0.40),
        ("file10", "sum_top2", 0.05), ("file10", "sum_top2", 0.10),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)
    (args.output_dir / "traces").mkdir(exist_ok=True)
    all_traces: dict[str, dict[str, Any]] = {}
    all_runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for scope, pool, gamma in specs:
        name = f"legacy-second-{scope}-{pool}-gamma{gamma:g}"
        config_started = time.perf_counter()
        runs: dict[str, list[dict[str, Any]]] = {}
        traces: dict[str, Any] = {}
        for question in questions:
            run, trace = _retrieve_one(corpus, page_index, retriever, chunks, chunk_index, chunk_matrix, query_vectors[question.qid], question.qid, question.query, scope=scope, gamma=gamma, alpha=0.70, pool=pool, depth=100)
            runs[question.qid] = run
            traces[question.qid] = trace
        metrics = _derived_metrics(runs, qids, qrels)
        methods[name] = {"config": {"scope": scope, "pool": pool, "gamma": gamma, "alpha": 0.70, "depth": 100}, "retrieval_metrics": metrics, "timing_seconds": {"retrieval": round(time.perf_counter() - config_started, 6)}}
        all_traces[name] = traces
        all_runs[name] = runs
        _write_run(args.output_dir / "runs" / f"{_safe_name(name)}.jsonl", runs, qids, queries={q.qid: q.query for q in questions})
        with (args.output_dir / "traces" / f"{_safe_name(name)}.jsonl").open("w", encoding="utf-8") as handle:
            for qid in qids:
                handle.write(json.dumps(traces[qid], ensure_ascii=False) + "\n")

    baseline_metrics = methods["Cached page BM25 baseline"]["retrieval_metrics"]
    for name, method in methods.items():
        if name != "Cached page BM25 baseline":
            method["comparison_to_baseline"] = _paired_comparison(baseline_metrics, method["retrieval_metrics"])
    oof = _oof_selection(qids, {q.qid: q for q in questions}, qrels, methods, {"Cached page BM25 baseline": baseline, **all_runs})
    oof_metrics = _derived_metrics(oof["run"], qids, qrels)
    report = {"dataset": "vidore_v3/industrial", "evaluation_language": "english", "queries": len(qids), "pages": len(page_ids), "files": len(corpus.files), "index_counts": index_counts, "legacy_cache": legacy_meta, "query_cache": query_meta, "sources": {"parsed_run": str(args.parsed_run), "baseline_run": str(args.baseline_run)}, "methods": methods, "cv": {"folds": oof["folds"], "selected_method_counts": oof["selected_method_counts"], "retrieval_metrics": oof_metrics, "comparison_to_baseline": _paired_comparison(baseline_metrics, oof_metrics)}, "timing_seconds": {"total": round(time.perf_counter() - started, 6)}, "notes": ["Legacy text chunks are mapped to every touched page for cross-page spans.", "The global scope is a candidate-union ceiling; file10 is the hierarchical second-pass scope.", "No qrels are used in retrieval."]}
    _write_run(args.output_dir / "oof_run.jsonl", oof["run"], qids, queries={q.qid: q.query for q in questions})
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Industrial legacy second retrieval", "", "Cached legacy text-chunk BM25 + dense retrieval after the page-level PDF-inspector BM25 pass.", "", "| Method | nDCG@10 | Page hit@10 | Page recall@10 | File recall@3 | Δ page pp | Retrieval s |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, method in methods.items():
        m = method["retrieval_metrics"]
        f = m["file_metrics_by_k"]["3"]["file_recall"]
        d = method.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0)
        lines.append(f"| {name} | {m['ndcg@10']:.2f} | {m['page_hit@10']:.2%} | {m['page_recall@10']:.2%} | {f:.2%} | {d:+.2f} | {method['timing_seconds']['retrieval']:.2f} |")
    cv_metrics = report["cv"]["retrieval_metrics"]
    cv_delta = report["cv"]["comparison_to_baseline"]["page_recall_delta_pp"]
    lines += ["", "## Out-of-fold selection", "", f"- Selected methods: `{report['cv']['selected_method_counts']}`.", f"- OOF nDCG@10: **{cv_metrics['ndcg@10']:.2f}**; page recall@10: **{cv_metrics['page_recall@10']:.2%}** ({cv_delta:+.2f}pp vs cached page BM25).", f"- OOF file recall@3: **{cv_metrics['file_metrics_by_k']['3']['file_recall']:.2%}**."]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: {"page_recall@10": round(m["retrieval_metrics"]["page_recall@10"] * 100, 2), "file_recall@3": round(m["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"] * 100, 2), "delta_pp": round(m.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0), 2)} for name, m in methods.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
