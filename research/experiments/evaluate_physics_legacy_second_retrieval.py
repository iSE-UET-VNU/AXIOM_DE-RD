"""Evaluate the legacy AXIOM second retrieval after the Physics page cascade.

This is a research-only, retrieval-only experiment.  It reuses the persisted
KDL + pdf-inspector output, the persisted legacy text-chunk embeddings, and the
cached text-embedding-3-small Physics query vectors.  No parser, embedding API,
LLM or network call is made.

The first stage is the current explicit file -> page cascade with Kf=3.  The
second stage mirrors the legacy chunks arm: BM25 + dense alpha fusion over
legacy text chunks, scoped to a page candidate pool.  Chunk scores are lifted
back to pages so the result can be evaluated under the frozen page-level
ViDoRe protocol.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
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
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    HierarchicalRetriever,
    _build_corpus_and_indexes,
    _derived_metrics,
    _file_id,
    _load_visual_scores,
    _page_vector_units,
    _paired_comparison,
    _stage_metrics,
    _stratified_folds,
    _write_run,
)
from src.chunking_embedding.embedders import sanitize_text  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.evaluation.pipeline_pages import canonical_doc  # noqa: E402
from src.retrieval.fusion import alpha_fuse  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_LEGACY_CACHE = ROOT / "data/work/vidore_physics_emb"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_legacy_second_retrieval"
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
    """Map the legacy document-level main_text offsets to canonical pages.

    Legacy fixed-overlap chunks store character offsets in the document's
    concatenated ``main_text``.  The persisted representation contains explicit
    ``**Page N**`` markers; page N is zero-based parser page N-1.  The cover
    content before the first marker is page 0.
    """

    if not page_ids:
        return []
    page_by_number = {index: page_id for index, page_id in enumerate(page_ids)}
    markers = [
        (match.start(), int(match.group(1)) - 1)
        for match in PAGE_MARKER.finditer(main_text)
    ]
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
    else:
        intervals.append((0, len(main_text), page_ids[0]))
    return intervals


def _pages_for_span(
    start: int,
    end: int,
    intervals: Sequence[tuple[int, int, str]],
    page_ids: Sequence[str],
) -> tuple[str, ...]:
    touched = [
        page_id
        for interval_start, interval_end, page_id in intervals
        if max(start, interval_start) < min(end, interval_end)
    ]
    if touched:
        return tuple(dict.fromkeys(touched))
    # Defensive fallback for a malformed/zero-width span.
    if not intervals:
        return (page_ids[0],) if page_ids else ()
    nearest = min(intervals, key=lambda item: abs(item[0] - start))
    return (nearest[2],)


def _load_legacy_chunks(
    parsed_run: Path,
    corpus: HierarchyCorpus,
) -> tuple[list[LegacyChunk], dict[str, Any]]:
    chunks: list[LegacyChunk] = []
    documents = 0
    text_items = 0
    cross_page = 0
    per_file: dict[str, int] = defaultdict(int)
    for path in sorted((parsed_run / "documents").glob("*.json")):
        payload = _load_json(path)
        document = payload.get("document") or {}
        file_id = f"physics::{canonical_doc(document.get('file_name'))}"
        page_ids = corpus.file_to_pages.get(file_id, [])
        if not page_ids:
            raise RuntimeError(f"Legacy artifact file is not in the page corpus: {file_id}")
        documents += 1
        intervals = _page_intervals(str((payload.get("content") or {}).get("main_text") or ""), page_ids)
        for item in (payload.get("retrieval") or {}).get("items", []):
            if item.get("type") != "text":
                continue
            text = str(((item.get("content") or {}).get("text")) or "").strip()
            position = item.get("position") or {}
            if not text or position.get("start_char") is None or position.get("end_char") is None:
                continue
            start = int(position["start_char"])
            end = int(position["end_char"])
            pages = _pages_for_span(start, end, intervals, page_ids)
            embeddings = item.get("embeddings") or []
            if not embeddings or not embeddings[0].get("values"):
                raise RuntimeError(f"Missing legacy embedding for {item.get('item_id')} in {path}")
            vector = np.asarray(embeddings[0]["values"], dtype=np.float32)
            if vector.shape != (1536,):
                raise RuntimeError(f"Unexpected legacy embedding shape {vector.shape} for {item.get('item_id')}")
            chunk_id = f"{file_id}#legacy_chunk={item['item_id']}"
            chunks.append(LegacyChunk(chunk_id, file_id, pages, text, vector))
            text_items += 1
            per_file[file_id] += 1
            cross_page += int(len(pages) > 1)

    if documents != len(corpus.files):
        raise RuntimeError(f"Expected {len(corpus.files)} legacy documents, got {documents}")
    if not chunks:
        raise RuntimeError("No legacy text chunks were loaded")
    return chunks, {
        "documents": documents,
        "text_chunks": text_items,
        "cross_page_chunks": cross_page,
        "cross_page_rate": cross_page / text_items,
        "chunks_by_file_min": min(per_file.values()),
        "chunks_by_file_max": max(per_file.values()),
        "embedding_model": EMBEDDING_MODEL,
        "source": str(parsed_run),
    }


def _load_cached_query_vectors(
    questions: Sequence[Any],
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    vectors: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for question in questions:
        text = sanitize_text(question.query)
        key = hashlib.sha1(f"{EMBEDDING_MODEL}|{text}".encode("utf-8", "ignore")).hexdigest()
        path = cache_dir / f"emb_{key}.json"
        if not path.is_file():
            missing.append(question.qid)
            continue
        vector = np.asarray(_load_json(path), dtype=np.float32)
        if vector.shape != (1536,):
            raise RuntimeError(f"Unexpected cached query vector shape {vector.shape}: {path}")
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        vectors[question.qid] = vector
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} cached French query embeddings; refusing a network/API fallback. "
            f"Examples: {missing[:5]}"
        )
    return vectors, {
        "cache_dir": str(cache_dir),
        "query_count": len(vectors),
        "cache_hits": len(vectors),
        "cache_misses": len(missing),
        "language": "french",
        "model": EMBEDDING_MODEL,
    }


def _pool_page_scores(values: Mapping[str, list[float]], pool: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for page_id, raw in values.items():
        scores = sorted((max(0.0, float(value)) for value in raw), reverse=True)
        if pool == "max":
            output[page_id] = scores[0] if scores else 0.0
        elif pool == "sum_top2":
            output[page_id] = sum(scores[:2])
        elif pool == "coverage":
            output[page_id] = sum(1.0 / (10.0 + rank) for rank in range(1, min(10, len(scores)) + 1))
            output[page_id] += (scores[0] * 1e-3 if scores else 0.0)
        else:
            raise ValueError(f"Unknown legacy page pool: {pool}")
    return output


def _top_dense(scores: np.ndarray, positions: Sequence[int], depth: int) -> list[tuple[int, float]]:
    if not positions:
        return []
    selected = np.asarray(positions, dtype=np.int32)
    count = min(depth, selected.size)
    if count <= 0:
        return []
    if count == selected.size:
        ordered = selected[np.argsort(-scores[selected])]
    else:
        candidate = selected[np.argpartition(-scores[selected], count - 1)[:count]]
        ordered = candidate[np.argsort(-scores[candidate])]
    return [(int(position), float(scores[position])) for position in ordered]


def _retrieve_legacy_chunks_for_query(
    chunks: Sequence[LegacyChunk],
    index: BM25Index,
    matrix: np.ndarray,
    query: str,
    query_vector: np.ndarray,
    allowed_pages: set[str],
    *,
    alpha: float,
    depth: int,
) -> list[tuple[str, float]]:
    allowed_positions = {
        position
        for position, chunk in enumerate(chunks)
        if allowed_pages.intersection(chunk.page_ids)
    }
    dense_hits = [
        (index.chunk_ids[position], score)
        for position, score in _top_dense(matrix @ query_vector, sorted(allowed_positions), depth)
    ]
    sparse_hits = [
        (index.chunk_ids[position], float(score))
        for position, score in index.search(query, depth, allowed_positions)
    ]
    return alpha_fuse(dense_hits, sparse_hits, alpha, depth)


def _aggregate_hits_to_pages(
    hits: Sequence[tuple[str, float]],
    by_id: Mapping[str, LegacyChunk],
    allowed_pages: set[str],
    pool: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for chunk_id, score in hits:
        chunk = by_id[chunk_id]
        for page_id in chunk.page_ids:
            if page_id in allowed_pages:
                grouped[page_id].append(float(score))
    return _pool_page_scores(grouped, pool)


def _base_page_scores(trace: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(row["node_id"]): float(row["score"])
        for row in trace.get("page_candidates", [])
    }


def _second_stage_run(
    base_traces: Mapping[str, Mapping[str, Any]],
    questions: Mapping[str, Any],
    chunks: Sequence[LegacyChunk],
    chunk_index: BM25Index,
    chunk_matrix: np.ndarray,
    query_vectors: Mapping[str, np.ndarray],
    corpus: HierarchyCorpus,
    *,
    scope: str,
    pool: str,
    gamma: float,
    alpha: float = 0.70,
    depth: int = 100,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    runs: dict[str, list[dict[str, Any]]] = {}
    traces: dict[str, dict[str, Any]] = {}
    for qid, question in questions.items():
        trace = base_traces[qid]
        base_ranked = [str(row["node_id"]) for row in trace.get("page_candidates", [])]
        if scope == "page100":
            allowed_pages = set(base_ranked[:100])
            output_pages = base_ranked[:100]
        elif scope == "page10":
            allowed_pages = set(base_ranked[:10])
            # The second retrieval only reorders the first 10 pages, but the
            # frozen file metric still needs the first 100 final pages.  Keep
            # the untouched page-stage tail in the output.
            output_pages = base_ranked[:100]
        elif scope == "all_pages":
            allowed_pages = set(corpus.page_order)
            output_pages = list(corpus.page_order)
        else:
            raise ValueError(f"Unknown second-stage scope: {scope}")

        base_scores = {
            page_id: score
            for page_id, score in _base_page_scores(trace).items()
            if page_id in output_pages
        }
        base_norm = normalise_scores(base_scores)
        started = time.perf_counter()
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
        fine_scores = _aggregate_hits_to_pages(hits, by_id, allowed_pages, pool)
        fine_norm = normalise_scores(fine_scores)
        final_scores = {
            page_id: (1.0 - gamma) * base_norm.get(page_id, 0.0)
            + gamma * fine_norm.get(page_id, 0.0)
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
                    "legacy_chunk_score": round(float(fine_norm.get(page_id, 0.0)), 8),
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
            "scope": scope,
            "pool": pool,
            "gamma": gamma,
            "legacy_alpha_dense": alpha,
            "legacy_depth": depth,
            "page_pool": base_ranked[:100] if scope != "page10" else base_ranked[:10],
            "legacy_chunk_hits": [
                {
                    "chunk_id": chunk_id,
                    "page_ids": list(by_id[chunk_id].page_ids),
                    "score": round(float(score), 8),
                }
                for chunk_id, score in hits
            ],
            "legacy_chunk_pages": sorted(chunk_pages),
            "timing_seconds": {"second_stage": round(time.perf_counter() - started, 6)},
            "counts": {
                "allowed_pages": len(allowed_pages),
                "legacy_chunks_scoped": sum(
                    1 for chunk in chunks if allowed_pages.intersection(chunk.page_ids)
                ),
                "legacy_chunk_hits": len(hits),
                "legacy_chunk_pages": len(chunk_pages),
            },
        }
    return runs, traces


def _second_stage_metrics(
    traces: Mapping[str, Mapping[str, Any]],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    pool_recall: list[float] = []
    chunk_page_recall: list[float] = []
    conditional: list[float] = []
    conditional_count = 0
    for qid, trace in traces.items():
        gold = set(qrels.get(qid, {}))
        page_pool = set(trace["page_pool"])
        chunk_pages = set(trace["legacy_chunk_pages"])
        pool_value = len(page_pool & gold) / len(gold) if gold else 0.0
        chunk_value = len(chunk_pages & gold) / len(gold) if gold else 0.0
        pool_recall.append(pool_value)
        chunk_page_recall.append(chunk_value)
        if page_pool & gold:
            conditional_count += 1
            conditional.append(chunk_value)
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "page_pool_recall": mean(pool_recall),
        "legacy_chunk_page_recall_proxy": mean(chunk_page_recall),
        "legacy_chunk_page_recall_proxy_conditional_on_page_pool": mean(conditional),
        "conditional_page_pool_queries": conditional_count,
        "queries": len(traces),
    }


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> tuple[float, float]:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    if not rows:
        return 0.0, 0.0
    page = sum(float(row["page_recall@10"]) for row in rows) / len(rows)
    file_values = []
    for row in rows:
        candidates = row["top10_files_from_top100_pages"][:3]
        gold = set(row["gold_files"])
        file_values.append(len(set(candidates) & gold) / len(gold) if gold else 0.0)
    return page, sum(file_values) / len(file_values)


def _oof_selection(
    qids: Sequence[str],
    questions: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int]],
    metrics: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    traces: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    folds = _stratified_folds(qids, questions, qrels)
    all_qids = set(qids)
    oof_run: dict[str, list[dict[str, Any]]] = {}
    oof_trace: dict[str, dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    for fold, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train = all_qids - heldout_set
        ranking = []
        for name, method_metrics in metrics.items():
            page, file = _metric_for_qids(method_metrics, train)
            ranking.append((page, file, name))
        _, _, winner = max(ranking, key=lambda row: (row[0], row[1], row[2]))
        for qid in heldout:
            oof_run[qid] = runs[winner][qid]
            oof_trace[qid] = traces[winner][qid]
        selected.append({
            "fold": fold,
            "heldout_qids": heldout,
            "selected_method": winner,
            "train_page_recall@10": _metric_for_qids(metrics[winner], train)[0],
            "train_file_recall@3": _metric_for_qids(metrics[winner], train)[1],
        })
    return {
        "folds": selected,
        "selected_method_counts": {
            name: sum(row["selected_method"] == name for row in selected)
            for name in metrics
        },
        "run": oof_run,
        "trace": oof_trace,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Physics legacy second retrieval",
        "",
        "Research-only experiment: current file → page cascade followed by cached legacy text-chunk BM25 + dense retrieval.",
        "",
        "## Cache and protocol",
        "",
        f"- Physics: **{report['queries']}** French queries, **{report['pages']}** pages, **{report['files']}** files.",
        f"- Legacy cache: **{report['legacy_cache']['text_chunks']}** text chunks; **{report['legacy_cache']['cross_page_rate']:.2%}** cross-page chunks.",
        f"- Query embedding cache: **{report['query_cache']['cache_hits']}/{report['query_cache']['query_count']}** hits; no API fallback.",
        "- Main page metrics use top-10 page output. File recall@3 is the fixed first-3-unique-files derived metric from the first 100 final pages.",
        "- Legacy chunk results are lifted to page IDs only for evaluation; qrels remain page-level.",
        "",
        "## Results",
        "",
        "| Method | nDCG@10 | Page recall@10 | Δ page pp | File recall@3 | Δ file pp | Page pool recall | Chunk-page proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, method in report["methods"].items():
        metrics = method["retrieval_metrics"]
        comparison = method.get("comparison_to_baseline") or {}
        stage = method.get("stage_metrics") or {}
        file_recall = metrics["file_metrics_by_k"]["3"]["file_recall"]
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.2f} | {metrics['page_recall@10']:.2%} | "
            f"{comparison.get('page_recall_delta_pp', 0.0):+.2f} | {file_recall:.2%} | "
            f"{comparison.get('file_recall_delta_pp', 0.0):+.2f} | "
            f"{stage.get('page_pool_recall', 0.0):.2%} | "
            f"{stage.get('legacy_chunk_page_recall_proxy', 0.0):.2%} |"
        )
    cv = report.get("cv") or {}
    if cv.get("retrieval_metrics"):
        metrics = cv["retrieval_metrics"]
        comparison = cv["comparison_to_baseline"]
        lines += [
            "",
            "## Out-of-fold selection",
            "",
            f"- OOF selected methods: `{cv['selected_method_counts']}`.",
            f"- OOF nDCG@10: **{metrics['ndcg@10']:.2f}**; page recall@10: **{metrics['page_recall@10']:.2%}** (**{comparison['page_recall_delta_pp']:+.2f}pp** vs baseline).",
            f"- OOF file recall@3: **{metrics['file_metrics_by_k']['3']['file_recall']:.2%}** (**{comparison['file_recall_delta_pp']:+.2f}pp** vs baseline).",
        ]
    lines += [
        "",
        "## Interpretation",
        "",
        "- The page100 arms are the deployable cascade test: second retrieval may reorder only pages already retained by the first cascade.",
        "- The all_pages arm is a relaxed candidate-union ceiling, not a valid hard cascade; it estimates how much the legacy retriever could contribute if early filtering were removed.",
        "- Cross-page chunks are assigned to every touched page and are reported explicitly; this is a conservative provenance caveat for page-level lifting.",
        "- This experiment does not make an end-to-end QA claim.",
        "",
        "Runs are in `runs/`; per-query second-stage traces are in `stage_traces.jsonl`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--legacy-cache", type=Path, default=DEFAULT_LEGACY_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions_list = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    questions = {question.qid: question for question in questions_list}
    qids = [question.qid for question in questions_list]
    qrels = benchmark.qrels()
    page_ids = _page_vector_units(args.page_vector_dir)
    if len(qids) != 302 or len(page_ids) != 1674:
        raise RuntimeError(f"Expected 302 queries and 1,674 pages, got {len(qids)} and {len(page_ids)}")

    corpus, page_index, file_indexes, fine_indexes, index_counts = _build_corpus_and_indexes(
        args.parsed_run, page_ids, fine_units=[]
    )
    visual_scores, visual_meta = _load_visual_scores(args.page_vector_dir, args.query_vector_dir, page_ids, qids)
    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes=fine_indexes,
        visual_scores=visual_scores,
    )
    base_config = CascadeConfig(
        name="full-cascade-kf3",
        file_representation="all_text",
        file_pool="max",
        k_files=3,
        page_depth=100,
        final_depth=100,
        bm25_weight=0.70,
        parent_weight=0.15,
        fine_weight=0.0,
    )
    base_runs: dict[str, list[dict[str, Any]]] = {}
    base_traces: dict[str, dict[str, Any]] = {}
    for qid, question in questions.items():
        run, trace = retriever.retrieve(qid, question.query, base_config)
        base_runs[qid] = run
        base_traces[qid] = trace

    chunks, legacy_meta = _load_legacy_chunks(args.parsed_run, corpus)
    query_vectors, query_meta = _load_cached_query_vectors(questions_list, args.legacy_cache)
    chunk_matrix = np.asarray([chunk.vector for chunk in chunks], dtype=np.float32)
    chunk_matrix /= np.clip(np.linalg.norm(chunk_matrix, axis=1, keepdims=True), 1e-12, None)
    chunk_index = BM25Index(analyzer_name="auto").build(
        [{"chunk_id": chunk.chunk_id, "doc_id": chunk.file_id, "text": chunk.text} for chunk in chunks]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl"
    baseline_run = {}
    for line in baseline_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            baseline_run[str(row["qid"])] = list(row["chunks"])
    methods: dict[str, dict[str, Any]] = {
        "PDF-inspector + BM25 baseline": {
            "retrieval_metrics": _derived_metrics(baseline_run, qids, qrels),
            "stage_metrics": {},
        },
        "full cascade (Kf=3), before legacy second retrieval": {
            "retrieval_metrics": _derived_metrics(base_runs, qids, qrels),
            "stage_metrics": _stage_metrics(base_traces, qids, qrels),
        },
    }
    all_runs: dict[str, dict[str, list[dict[str, Any]]]] = {
        "full cascade (Kf=3), before legacy second retrieval": base_runs,
    }
    all_traces: dict[str, dict[str, dict[str, Any]]] = {}

    # The 0.70 dense weight matches the existing legacy Physics hybrid arm.
    variant_specs = [
        ("page100", "max", 0.10),
        ("page100", "max", 0.25),
        ("page100", "max", 0.40),
        ("page100", "max", 0.60),
        ("page100", "sum_top2", 0.25),
        ("page100", "coverage", 0.25),
        ("page10", "max", 0.25),
        ("all_pages", "max", 0.25),
    ]
    for scope, pool, gamma in variant_specs:
        name = f"legacy-second-{scope}-{pool}-gamma{gamma:g}"
        variant_runs, variant_traces = _second_stage_run(
            base_traces,
            questions,
            chunks,
            chunk_index,
            chunk_matrix,
            query_vectors,
            corpus,
            scope=scope,
            pool=pool,
            gamma=gamma,
        )
        metrics = _derived_metrics(variant_runs, qids, qrels)
        stage = _second_stage_metrics(variant_traces, qrels)
        methods[name] = {"retrieval_metrics": metrics, "stage_metrics": stage}
        all_runs[name] = variant_runs
        all_traces[name] = variant_traces
        _write_run(runs_dir / f"{_safe_name(name)}.jsonl", variant_runs, qids, queries={qid: questions[qid].query for qid in qids})

    baseline_metrics = methods["PDF-inspector + BM25 baseline"]["retrieval_metrics"]
    for name, method in methods.items():
        if name != "PDF-inspector + BM25 baseline":
            method["comparison_to_baseline"] = _paired_comparison(baseline_metrics, method["retrieval_metrics"])

    if args.skip_cv:
        cv_payload: dict[str, Any] = {"skipped": True, "retrieval_metrics": {}}
    else:
        cv_candidate_names = [name for name in all_traces if name in all_runs]
        cv = _oof_selection(qids, questions, qrels, metrics={
            name: method["retrieval_metrics"]
            for name, method in methods.items()
            if name in cv_candidate_names
        }, runs={name: all_runs[name] for name in cv_candidate_names}, traces={name: all_traces[name] for name in cv_candidate_names})
        oof_metrics = _derived_metrics(cv["run"], qids, qrels)
        baseline_for_cv = baseline_metrics
        cv_payload = {
            "folds": cv["folds"],
            "selected_method_counts": cv["selected_method_counts"],
            "retrieval_metrics": oof_metrics,
            "stage_metrics": _second_stage_metrics(cv["trace"], qrels),
            "comparison_to_baseline": _paired_comparison(baseline_for_cv, oof_metrics),
        }
        _write_run(args.output_dir / "oof_run.jsonl", cv["run"], qids, queries={qid: questions[qid].query for qid in qids})

    with (args.output_dir / "stage_traces.jsonl").open("w", encoding="utf-8") as handle:
        for name, traces in all_traces.items():
            for qid in qids:
                handle.write(json.dumps({"method": name, **traces[qid]}, ensure_ascii=False) + "\n")

    report: dict[str, Any] = {
        "dataset": "vidore_v3/physics",
        "evaluation_language": "french",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "legacy_cache": legacy_meta,
        "query_cache": query_meta,
        "sources": {
            "parsed_run": str(args.parsed_run),
            "baseline_run": str(baseline_path),
            "page_vector_artifact": visual_meta,
            "legacy_chunk_index": "rebuilt from persisted retrieval.items lexical text",
        },
        "index_counts": index_counts,
        "methods": methods,
        "cv": cv_payload,
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "The first stage is the existing research-only full cascade with Kf=3 and page BM25 + cached V-SPLADE fusion.",
            "Legacy text chunks and their 1536-dimensional embeddings are read from the persisted Physics output artifact; query vectors are read from data/work/vidore_physics_emb.",
            "The legacy hybrid arm uses dense alpha=0.70 and depth=100, matching the existing Physics hybrid experiment.",
            "page100 is the valid hard-cascade scope; all_pages is a relaxed ceiling and must not be presented as a hard cascade.",
            "Cross-page chunks are mapped to every touched page for page-level analysis; this mapping is reported as a provenance caveat.",
            "No qrels are used as routing features and no end-to-end QA claim is made.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "legacy_cache": legacy_meta,
        "query_cache": query_meta,
        "methods": {
            name: {
                "nDCG@10": round(method["retrieval_metrics"]["ndcg@10"], 2),
                "page_recall@10": round(method["retrieval_metrics"]["page_recall@10"] * 100, 2),
                "file_recall@3": round(method["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"] * 100, 2),
                "page_delta_pp": round(method.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0), 2),
                "file_delta_pp": round(method.get("comparison_to_baseline", {}).get("file_recall_delta_pp", 0.0), 2),
            }
            for name, method in methods.items()
        },
        "cv": {
            "page_recall@10": round(cv_payload.get("retrieval_metrics", {}).get("page_recall@10", 0.0) * 100, 2),
            "file_recall@3": round(cv_payload.get("retrieval_metrics", {}).get("file_metrics_by_k", {}).get("3", {}).get("file_recall", 0.0) * 100, 2),
        },
    }, ensure_ascii=False, indent=2))


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


if __name__ == "__main__":
    main()
