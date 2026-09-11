"""Run cached legacy second retrieval over the exported Physics file scopes.

The light stage has already selected files for each Kf.  This experiment sends
*all pages belonging to those files* to the legacy second retriever:

    cached legacy chunks -> BM25 + dense alpha fusion -> max chunk/page pool

The parser output, legacy chunk embeddings and query embeddings are all read
from local caches.  No API, model encoding, OCR or qrels are used to build a
ranking.  This is intentionally separate from the older Kf=3/page100 report,
which restricted the second stage to a page top-K and therefore tested a much
smaller scope.
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
from research.experiments.physics_topk_budget_and_proposal_ablation import (  # noqa: E402
    BRANCHES,
    _active_page_score,
    _build_query_states,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.fusion import alpha_fuse  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_LEGACY_CACHE = ROOT / "data/work/vidore_physics_emb"
DEFAULT_SCOPE_DIR = ROOT / "data/benchmark/vidore_v3/exports/physics_file_retrieval_for_colembed"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_legacy_second_retrieval_file_scopes"

KF_VALUES = (3, 5, 10, 15, 20)
KP_VALUES = (10, 50, 100)
LEGACY_DENSE_ALPHA = 0.70
LEGACY_CHUNK_DEPTH = 100
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


def _load_scopes(
    scope_dir: Path,
    qids: Sequence[str],
    corpus: HierarchyCorpus,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Load and validate file scopes, including exact all-pages membership."""

    qid_set = set(qids)
    corpus_pages = set(corpus.page_order)
    output: dict[int, dict[str, dict[str, Any]]] = {}
    for kf in KF_VALUES:
        path = scope_dir / f"queries_kf{kf}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing scope export: {path}")
        rows: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row["qid"])
                if qid in rows:
                    raise ValueError(f"Duplicate qid {qid!r} in {path}:{line_number}")
                if int(row["k_files"]) != kf:
                    raise ValueError(f"Wrong k_files in {path}:{line_number}")
                file_ids = [str(value) for value in row["selected_file_ids"]]
                page_ids = [str(value) for value in row["candidate_page_ids"]]
                if len(file_ids) != len(set(file_ids)):
                    raise ValueError(f"Duplicate selected file in {path}:{line_number}")
                if len(page_ids) != len(set(page_ids)):
                    raise ValueError(f"Duplicate candidate page in {path}:{line_number}")
                if any(page_id not in corpus_pages for page_id in page_ids):
                    raise ValueError(f"Scope contains an unknown page in {path}:{line_number}")
                expected_pages = {
                    page_id
                    for file_id in file_ids
                    for page_id in corpus.file_to_pages.get(file_id, [])
                }
                if set(page_ids) != expected_pages:
                    raise ValueError(
                        f"Scope is not exactly all pages of selected files for {qid} in {path}"
                    )
                rows[qid] = {
                    "selected_file_ids": file_ids,
                    "candidate_page_ids": page_ids,
                    "candidate_page_count": len(page_ids),
                }
        if set(rows) != qid_set:
            raise ValueError(f"QID set mismatch in {path}")
        output[kf] = rows
    return output


def _load_questions_and_qrels(
    benchmark_root: Path,
) -> tuple[list[Any], list[str], dict[str, dict[str, int]]]:
    benchmark = ViDoreV3(root=benchmark_root, subset="physics", language="french")
    questions = sorted(
        list(benchmark.questions()),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    qids = [question.qid for question in questions]
    qrels = benchmark.qrels()
    if len(qids) != 302 or len(qrels) != 302:
        raise RuntimeError(f"Expected 302 queries/qrels, got {len(qids)}/{len(qrels)}")
    return questions, qids, qrels


def _retrieve_chunk_hits(
    chunk_index: BM25Index,
    dense_scores: np.ndarray,
    query: str,
    allowed_pages: set[str],
    chunk_positions_by_page: Mapping[str, Sequence[int]],
    *,
    alpha: float,
    depth: int,
) -> list[tuple[str, float]]:
    allowed_positions = sorted({
        position
        for page_id in allowed_pages
        for position in chunk_positions_by_page.get(page_id, [])
    })
    dense_hits = [
        (chunk_index.chunk_ids[position], score)
        for position, score in _top_dense(dense_scores, allowed_positions, depth)
    ]
    sparse_hits = [
        (chunk_index.chunk_ids[position], float(score))
        for position, score in chunk_index.search(query, depth, set(allowed_positions))
    ]
    return alpha_fuse(dense_hits, sparse_hits, alpha, depth)


def _run_one_scope(
    qid: str,
    query: str,
    scope_row: Mapping[str, Any],
    corpus: HierarchyCorpus,
    chunks: Sequence[Any],
    by_chunk_id: Mapping[str, Any],
    chunk_index: BM25Index,
    dense_scores: np.ndarray,
    chunk_positions_by_page: Mapping[str, Sequence[int]],
    light_page_scores: Mapping[str, float] | None = None,
    light_prior_weight: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    allowed_pages = set(scope_row["candidate_page_ids"])
    hits = _retrieve_chunk_hits(
        chunk_index,
        dense_scores,
        query,
        allowed_pages,
        chunk_positions_by_page,
        alpha=LEGACY_DENSE_ALPHA,
        depth=LEGACY_CHUNK_DEPTH,
    )
    page_scores = _aggregate_hits_to_pages(hits, by_chunk_id, allowed_pages, PAGE_POOL)
    legacy_norm = normalise_scores(page_scores)
    if light_page_scores is None or light_prior_weight <= 0.0:
        final_scores = {
            page_id: legacy_norm.get(page_id, 0.0) for page_id in allowed_pages
        }
        fusion_mode = "legacy_only"
    else:
        light_norm = normalise_scores({
            page_id: float(light_page_scores.get(page_id, 0.0))
            for page_id in allowed_pages
        })
        final_scores = {
            page_id: (1.0 - light_prior_weight) * light_norm.get(page_id, 0.0)
            + light_prior_weight * legacy_norm.get(page_id, 0.0)
            for page_id in allowed_pages
        }
        fusion_mode = "light_page_prior_plus_legacy"
    # Keep zero-score pages in the candidate pool. They cannot outrank a page
    # with evidence, but this makes the output a complete scoped ranking.
    ranked = sort_scores(final_scores)
    run = [
        {
            "chunk_id": page_id,
            "doc_id": page_id,
            "text": corpus.pages[page_id].text,
            "score": round(float(score), 8),
            "rank": rank,
            "scores": {"legacy_chunk_score": round(float(score), 8)},
        }
        for rank, (page_id, score) in enumerate(ranked[:100], 1)
    ]
    touched_pages = {
        page_id
        for chunk_id, _score in hits
        for page_id in by_chunk_id[chunk_id].page_ids
        if page_id in allowed_pages
    }
    trace = {
        "qid": qid,
        "candidate_page_count": len(allowed_pages),
        "scoped_chunk_count": sum(
            1 for chunk in chunks if allowed_pages.intersection(chunk.page_ids)
        ),
        "legacy_chunk_hits": len(hits),
        "legacy_chunk_pages": len(touched_pages),
        "fusion_mode": fusion_mode,
        "light_prior_weight": light_prior_weight,
        "legacy_chunk_hit_ids": [chunk_id for chunk_id, _score in hits],
        "legacy_chunk_touched_pages": sorted(touched_pages),
        "latency_seconds": time.perf_counter() - started,
    }
    return run, trace


def _metric_row(
    qid: str,
    run: Sequence[Mapping[str, Any]],
    trace: Mapping[str, Any],
    scope_row: Mapping[str, Any],
    qrel: Mapping[str, int],
    kps: Sequence[int],
) -> dict[str, Any]:
    gold_pages = set(qrel)
    gold_files = {_file_id(page_id) for page_id in gold_pages}
    candidate_pages = set(scope_row["candidate_page_ids"])
    selected_files = set(scope_row["selected_file_ids"])
    row: dict[str, Any] = {
        "qid": qid,
        "gold_page_count": len(gold_pages),
        "gold_file_count": len(gold_files),
        "candidate_page_count": len(candidate_pages),
        "page_pool_recall_ceiling": len(candidate_pages & gold_pages) / len(gold_pages),
        "selected_file_count": len(selected_files),
        "file_scope_recall": len(selected_files & gold_files) / len(gold_files),
        "file_scope_hit": bool(selected_files & gold_files),
        "ranked_page_ids": [str(item["chunk_id"]) for item in run],
        "legacy_chunk_pages": trace["legacy_chunk_pages"],
        "scoped_chunk_count": trace["scoped_chunk_count"],
        "legacy_chunk_hits": trace["legacy_chunk_hits"],
        "latency_seconds": trace["latency_seconds"],
    }
    for kp in kps:
        ranked = row["ranked_page_ids"][:kp]
        found = set(ranked) & gold_pages
        row[f"page_hit@{kp}"] = bool(found)
        row[f"page_recall@{kp}"] = len(found) / len(gold_pages)
        row[f"page_precision@{kp}"] = len(found) / kp
        row[f"nDCG@{kp}"] = _ndcg_at_k(ranked, qrel, kp)

    # Keep the project's fixed derived file diagnostic visible as well.
    files: list[str] = []
    for page_id in row["ranked_page_ids"][:100]:
        file_id = _file_id(page_id)
        if file_id not in files:
            files.append(file_id)
        if len(files) == 3:
            break
    row["final_file_recall@3"] = len(set(files) & gold_files) / len(gold_files)
    row["final_file_hit@3"] = bool(set(files) & gold_files)
    row["final_top3_files"] = files
    return row


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]], kps: Sequence[int]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "queries": len(rows),
        "avg_candidate_pages": _mean([float(row["candidate_page_count"]) for row in rows]),
        "avg_selected_files": _mean([float(row["selected_file_count"]) for row in rows]),
        "avg_scoped_chunks": _mean([float(row["scoped_chunk_count"]) for row in rows]),
        "avg_legacy_chunk_hits": _mean([float(row["legacy_chunk_hits"]) for row in rows]),
        "avg_latency_seconds": _mean([float(row["latency_seconds"]) for row in rows]),
        "file_scope_recall": _mean([float(row["file_scope_recall"]) for row in rows]),
        "file_scope_hit": _mean([float(row["file_scope_hit"]) for row in rows]),
        "page_pool_recall_ceiling": _mean(
            [float(row["page_pool_recall_ceiling"]) for row in rows]
        ),
        "final_file_recall@3": _mean([float(row["final_file_recall@3"]) for row in rows]),
        "final_file_hit@3": _mean([float(row["final_file_hit@3"]) for row in rows]),
        "k_values": {},
    }
    for kp in kps:
        output["k_values"][str(kp)] = {
            "nDCG": _mean([float(row[f"nDCG@{kp}"]) for row in rows]),
            "page_hit": _mean([float(row[f"page_hit@{kp}"]) for row in rows]),
            "page_recall": _mean([float(row[f"page_recall@{kp}"]) for row in rows]),
            "page_precision": _mean([float(row[f"page_precision@{kp}"]) for row in rows]),
        }
    return output


def _report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Physics legacy second retrieval over file scopes",
        "",
        "The legacy second retriever receives all pages belonging to the files selected by the light stage.",
        "It uses cached legacy text chunks, BM25 + dense alpha fusion and max chunk-to-page pooling.",
        "",
        "## Fixed configuration",
        "",
        f"- Dense alpha: `{LEGACY_DENSE_ALPHA}`; chunk depth: `{LEGACY_CHUNK_DEPTH}`; page pool: `{PAGE_POOL}`.",
        f"- Fusion mode: `{report['config']['fusion_mode']}`; legacy weight: `{report['config']['legacy_weight']}`.",
        "- Query/page scope is file-based: no top-10 light-page truncation is applied.",
        f"- Cache: `{report['cache']['text_chunks']}` legacy chunks and `{report['cache']['query_embedding_hits']}/{report['queries']}` query embeddings.",
        "",
        "## Results",
        "",
        "| Kf | Avg files | Avg pages/query | File scope recall | Page-pool ceiling | Kp | nDCG | Page hit | Page recall | Final file recall@3 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        metrics = arm["metrics"]
        for kp in report["config"]["k_pages"]:
            value = metrics["k_values"][str(kp)]
            lines.append(
                f"| {arm['k_files']} | {metrics['avg_selected_files']:.2f} | "
                f"{metrics['avg_candidate_pages']:.2f} | {metrics['file_scope_recall']:.2%} | "
                f"{metrics['page_pool_recall_ceiling']:.2%} | {kp} | {value['nDCG']:.2%} | "
                f"{value['page_hit']:.2%} | {value['page_recall']:.2%} | "
                f"{metrics['final_file_recall@3']:.2%} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- File scope recall is inherited from light file retrieval and is not improved by the legacy stage.",
        "- Page-pool ceiling is the maximum possible page recall after Kf pruning.",
        "- Page recall/nDCG measure legacy ranking inside that file scope.",
        "- The experiment uses only cached parser/chunk/embedding artifacts; it makes no end-to-end QA claim.",
        "- The old `physics_legacy_second_retrieval` directory is unchanged; this report uses a separate output directory.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--legacy-cache", type=Path, default=DEFAULT_LEGACY_CACHE)
    parser.add_argument("--scope-dir", type=Path, default=DEFAULT_SCOPE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--with-light-page-prior",
        action="store_true",
        help="Fuse cached light page score with legacy score using fixed legacy weight 0.25.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    questions, qids, qrels = _load_questions_and_qrels(args.benchmark_root)
    page_ids = _page_vector_units(args.page_vector_dir)
    if len(page_ids) != 1674 or len(set(page_ids)) != len(page_ids):
        raise RuntimeError(f"Expected 1,674 unique page IDs, got {len(page_ids)}")
    corpus = HierarchyCorpus.from_parsed_run(
        args.parsed_run, subset="physics", page_ids=page_ids
    )
    scopes = _load_scopes(args.scope_dir, qids, corpus)

    light_page_scores: dict[str, dict[str, float]] | None = None
    legacy_weight = 0.25 if args.with_light_page_prior else 1.0
    if args.with_light_page_prior:
        # Rebuild only the cached light page scores used to select the files:
        # 0.70 normalized BM25 + 0.30 normalized V-SPLADE.  The score build is
        # local and qrel-free; no page top-K is imposed here.
        light_benchmark = ViDoreV3(
            root=args.benchmark_root, subset="physics", language="french"
        )
        light_corpus, _light_questions, light_qids, states, _visual_meta = _build_query_states(
            light_benchmark,
            args.parsed_run,
            args.page_vector_dir,
            args.query_vector_dir,
        )
        if light_qids != qids or light_corpus.page_order != corpus.page_order:
            raise RuntimeError("Light page-score inventory does not match benchmark inventory")
        light_page_scores = {
            qid: _active_page_score(states[qid], set(BRANCHES)) for qid in qids
        }

    chunks, legacy_meta = _load_legacy_chunks(args.parsed_run, corpus)
    query_vectors, query_meta = _load_cached_query_vectors(questions, args.legacy_cache)
    chunk_matrix = np.asarray([chunk.vector for chunk in chunks], dtype=np.float32)
    chunk_matrix /= np.clip(np.linalg.norm(chunk_matrix, axis=1, keepdims=True), 1e-12, None)
    chunk_index = BM25Index(analyzer_name="auto").build([
        {"chunk_id": chunk.chunk_id, "doc_id": chunk.file_id, "text": chunk.text}
        for chunk in chunks
    ])
    by_chunk_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunk_positions_by_page: dict[str, list[int]] = defaultdict(list)
    for position, chunk in enumerate(chunks):
        for page_id in chunk.page_ids:
            chunk_positions_by_page[page_id].append(position)
    query_matrix = np.asarray([query_vectors[qid] for qid in qids], dtype=np.float32)
    # Compute every cached chunk/query dense score once.  The Kf arms only
    # change the allowed page positions, so recomputing this per Kf would be
    # redundant and substantially slower on CPU.
    dense_score_matrix = chunk_matrix @ query_matrix.T
    qid_to_index = {qid: index for index, qid in enumerate(qids)}

    output_dir = args.output_dir
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    all_per_query: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    for kf in KF_VALUES:
        runs: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for question in questions:
            qid = question.qid
            run, trace = _run_one_scope(
                qid,
                question.query,
                scopes[kf][qid],
                corpus,
                chunks,
                by_chunk_id,
                chunk_index,
                dense_score_matrix[:, qid_to_index[qid]],
                chunk_positions_by_page,
                None if light_page_scores is None else light_page_scores[qid],
                legacy_weight,
            )
            row = _metric_row(qid, run, trace, scopes[kf][qid], qrels[qid], KP_VALUES)
            row["k_files"] = kf
            rows.append(row)
            runs.append({
                "qid": qid,
                "query": question.query,
                "retriever_id": f"physics-legacy-second-file-scope-kf{kf}",
                "index_id": f"physics-legacy-chunks-file-scope-kf{kf}",
                "chunks": run,
            })
        _write_jsonl(runs_dir / f"legacy_second_kf{kf}_top100.jsonl", runs)
        metrics = _aggregate_metrics(rows, KP_VALUES)
        arms.append({"k_files": kf, "metrics": metrics})
        all_per_query.extend(rows)

    report: dict[str, Any] = {
        "dataset": "vidore_v3/physics",
        "language": "french",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "method": (
            "cached light page score + legacy chunk BM25+dense second retrieval over all pages of selected files"
            if args.with_light_page_prior
            else "cached legacy chunk BM25+dense second retrieval over all pages of selected files"
        ),
        "config": {
            "k_files": list(KF_VALUES),
            "k_pages": list(KP_VALUES),
            "dense_alpha": LEGACY_DENSE_ALPHA,
            "chunk_depth": LEGACY_CHUNK_DEPTH,
            "page_pool": PAGE_POOL,
            "fusion_mode": (
                "light_page_prior_plus_legacy" if args.with_light_page_prior else "legacy_only"
            ),
            "legacy_weight": legacy_weight,
            "light_page_score": (
                "0.70 normalized BM25 + 0.30 normalized V-SPLADE"
                if args.with_light_page_prior else None
            ),
            "qrels_used_for_ranking": False,
        },
        "cache": {
            "parsed_run": str(args.parsed_run),
            "legacy_cache": str(args.legacy_cache),
            "text_chunks": legacy_meta["text_chunks"],
            "cross_page_chunks": legacy_meta["cross_page_chunks"],
            "query_embedding_hits": query_meta["cache_hits"],
            "query_embedding_misses": query_meta["cache_misses"],
            "embedding_model": legacy_meta["embedding_model"],
        },
        "sources": {
            "scope_dir": str(args.scope_dir),
            "page_vector_dir_for_page_inventory": str(args.page_vector_dir),
        },
        "arms": arms,
        "timing_seconds": {
            "total": time.perf_counter() - started,
            "note": "Includes cached artifact loading, in-memory BM25 construction and all Kf retrievals; excludes parsing and model encoding.",
        },
    }
    _write_json(output_dir / "report.json", report)
    _write_jsonl(output_dir / "per_query.jsonl", all_per_query)
    (output_dir / "report.md").write_text(_report_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "elapsed_seconds": round(report["timing_seconds"]["total"], 3),
        "results": {
            f"Kf={arm['k_files']}": {
                "page_recall@10": round(100 * arm["metrics"]["k_values"]["10"]["page_recall"], 2),
                "nDCG@10": round(100 * arm["metrics"]["k_values"]["10"]["nDCG"], 2),
                "file_scope_recall": round(100 * arm["metrics"]["file_scope_recall"], 2),
                "avg_pages": round(arm["metrics"]["avg_candidate_pages"], 2),
            }
            for arm in arms
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
