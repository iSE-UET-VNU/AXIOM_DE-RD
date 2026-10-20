"""Evaluate a layout-routed BM25/ColVec ingestion baseline on Physics.

This is an offline simulation of adaptive page ingestion.  A page is assigned
to exactly one retrieval branch using only cached PDF-inspector/KDL page
metadata:

* text-rich pages are indexed by BM25;
* pages with visual blocks and relatively little text are indexed by ColVec.

The ColVec score matrix and PDF-inspector output are already cached, so this
script does not parse, render, encode, call an API, or read qrels while
building the route or the indexes.  Qrels are loaded only for evaluation.

The hard route is deliberately simple and fixed.  It is a baseline for the
research question, not a tuned production gate:

    ColVec if (visual_blocks >= 1 and words <= 120)
          or (visual_blocks >= 2 and words <= 300)
    BM25 otherwise

Run::

    python research/experiments/physics_adaptive_ingestion.py
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    HierarchyCorpus,
    build_bm25,
    normalise_scores,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _file_id,
    _index_scores,
    _load_run,
    _path_size_bytes,
)
from src.chunking_embedding.lexical import analyze  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_PARSED_RUN = (
    ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
)
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_COLVEC_DIR = ROOT / "data/work/physics_colvec_export"
DEFAULT_PAIR_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"
DEFAULT_OUTPUT_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/physics_adaptive_ingestion"
)

EXPECTED_PAGES = 1674
EXPECTED_QUERIES = 302
TOP_K = 100
FINAL_K = 10
LOW_TEXT_WORDS = 120
MEDIUM_TEXT_WORDS = 300

VISUAL_TYPES = frozenset(
    {"figure", "picture", "image", "diagram", "flowchart", "chart"}
)
TEXT_TYPES = frozenset(
    {"text", "sectionheader", "title", "list-item", "footnote", "paragraph"}
)
STRUCTURED_TYPES = frozenset(
    {"table", "tablegroup", "equationblock", "formula", "equation", "caption"}
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _load_colvec(
    score_dir: Path, page_ids: Sequence[str], qids: Sequence[str]
) -> tuple[np.ndarray, list[str], list[str], dict[str, Any]]:
    required = (
        "physics_colvec_scores.npy",
        "physics_colvec_keys.json",
        "physics_colvec_qids.json",
        "physics_colvec_meta.json",
    )
    missing = [name for name in required if not (score_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ColVec export files: {missing}")

    scores = np.load(score_dir / "physics_colvec_scores.npy", mmap_mode="r")
    keys = [str(value) for value in _read_json(score_dir / "physics_colvec_keys.json")]
    exported_qids = [
        str(value) for value in _read_json(score_dir / "physics_colvec_qids.json")
    ]
    meta = _read_json(score_dir / "physics_colvec_meta.json")

    if scores.ndim != 2 or scores.shape != (len(exported_qids), len(keys)):
        raise ValueError(
            f"ColVec score shape {scores.shape} disagrees with metadata "
            f"({len(exported_qids)}, {len(keys)})"
        )
    if not np.isfinite(scores).all():
        raise ValueError("ColVec scores contain non-finite values")
    if len(set(keys)) != len(keys) or len(set(exported_qids)) != len(exported_qids):
        raise ValueError("ColVec export contains duplicate page or query IDs")
    if set(keys) != set(page_ids):
        missing_pages = sorted(set(page_ids) - set(keys))[:5]
        extra_pages = sorted(set(keys) - set(page_ids))[:5]
        raise ValueError(
            f"ColVec page inventory mismatch; missing={missing_pages}, extra={extra_pages}"
        )
    if set(exported_qids) != set(qids):
        missing_qids = sorted(set(qids) - set(exported_qids))[:5]
        extra_qids = sorted(set(exported_qids) - set(qids))[:5]
        raise ValueError(
            f"ColVec query inventory mismatch; missing={missing_qids}, extra={extra_qids}"
        )
    if int(meta.get("n_pages", -1)) != len(keys) or int(meta.get("n_queries", -1)) != len(exported_qids):
        raise ValueError("ColVec metadata counts disagree with exported arrays")

    # Put the matrix into the canonical page/query order used by the benchmark.
    page_position = {page_id: position for position, page_id in enumerate(keys)}
    qid_position = {qid: position for position, qid in enumerate(exported_qids)}
    page_columns = np.asarray([page_position[page_id] for page_id in page_ids])
    query_rows = np.asarray([qid_position[qid] for qid in qids])
    ordered = np.asarray(scores[query_rows][:, page_columns], dtype=np.float32)
    return ordered, list(page_ids), list(qids), meta


def _page_layout_stats(
    parsed_run: Path, page_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Extract route features from raw cached KDL block labels and text."""
    output: dict[str, dict[str, Any]] = {}
    for document in documents(parsed_run):
        file_name = str((document.get("document") or {}).get("file_name") or "")
        file_id = f"physics::{canonical_doc(file_name)}"
        for page_number, blocks in page_blocks(document).items():
            page_id = f"{file_id}#page={int(page_number)}"
            raw_text = " ".join(
                str(block.get("text") or "").strip()
                for block in blocks
                if str(block.get("text") or "").strip()
            )
            types = [str(block.get("type") or "").casefold() for block in blocks]
            words = analyze(raw_text)
            visual_count = sum(kind in VISUAL_TYPES for kind in types)
            text_count = sum(kind in TEXT_TYPES for kind in types)
            structured_count = sum(kind in STRUCTURED_TYPES for kind in types)
            output[page_id] = {
                "page_id": page_id,
                "word_count": len(words),
                "char_count": len(raw_text),
                "block_count": len(blocks),
                "text_block_count": text_count,
                "visual_block_count": visual_count,
                "structured_block_count": structured_count,
                "has_visual_block": bool(visual_count),
            }

    # Empty parser pages remain in the retrieval inventory and are routed to
    # ColVec because BM25 has no useful text representation for them.
    for page_id in page_ids:
        output.setdefault(
            page_id,
            {
                "page_id": page_id,
                "word_count": 0,
                "char_count": 0,
                "block_count": 0,
                "text_block_count": 0,
                "visual_block_count": 0,
                "structured_block_count": 0,
                "has_visual_block": False,
            },
        )
    if set(output) != set(page_ids):
        raise ValueError("Page layout stats do not cover exactly the page inventory")
    return output


def _route(stats: Mapping[str, Any]) -> str:
    words = int(stats.get("word_count", 0))
    visual_blocks = int(stats.get("visual_block_count", 0))
    if words == 0:
        return "colvec"
    if visual_blocks >= 1 and words <= LOW_TEXT_WORDS:
        return "colvec"
    if visual_blocks >= 2 and words <= MEDIUM_TEXT_WORDS:
        return "colvec"
    return "bm25"


def _sort_scores(scores: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))


def _rank_colvec(
    scores: np.ndarray,
    qrow: int,
    page_ids: Sequence[str],
    allowed_ids: set[str] | None = None,
) -> list[tuple[str, float]]:
    values = {
        page_id: float(scores[qrow, position])
        for position, page_id in enumerate(page_ids)
        if allowed_ids is None or page_id in allowed_ids
    }
    return _sort_scores(values)


def _compact_run(
    qid: str, ranked: Sequence[tuple[str, float]], top_k: int = TOP_K
) -> dict[str, Any]:
    return {
        "qid": qid,
        "chunks": [
            {
                "chunk_id": page_id,
                "doc_id": page_id,
                "score": round(float(score), 8),
                "rank": rank,
            }
            for rank, (page_id, score) in enumerate(ranked[:top_k], 1)
        ],
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _branch_candidate_metrics(
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    branch_runs: Mapping[str, Sequence[Mapping[str, Any]]],
    visual_runs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    branch_recall: list[float] = []
    visual_recall: list[float] = []
    union_recall: list[float] = []
    for qid in qids:
        gold = set(qrels.get(qid, {}))
        bm25 = {str(row["chunk_id"]) for row in branch_runs[qid][:TOP_K]}
        colvec = {str(row["chunk_id"]) for row in visual_runs[qid][:TOP_K]}
        denominator = len(gold) or 1
        branch_recall.append(len(bm25 & gold) / denominator)
        visual_recall.append(len(colvec & gold) / denominator)
        union_recall.append(len((bm25 | colvec) & gold) / denominator)
    return {
        "bm25_route_candidate_recall": _mean(branch_recall),
        "colvec_route_candidate_recall": _mean(visual_recall),
        "two_route_candidate_recall": _mean(union_recall),
        "two_route_candidate_hit": _mean(
            [float(value > 0.0) for value in union_recall]
        ),
    }


def _route_report(
    page_ids: Sequence[str],
    stats: Mapping[str, Mapping[str, Any]],
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    counts = Counter(_route(stats[page_id]) for page_id in page_ids)
    by_route: dict[str, dict[str, float]] = {}
    for route_name in ("bm25", "colvec"):
        selected = [stats[page_id] for page_id in page_ids if _route(stats[page_id]) == route_name]
        by_route[route_name] = {
            "pages": len(selected),
            "page_share": len(selected) / len(page_ids) if page_ids else 0.0,
            "avg_words": _mean([float(row["word_count"]) for row in selected]),
            "avg_blocks": _mean([float(row["block_count"]) for row in selected]),
            "avg_visual_blocks": _mean(
                [float(row["visual_block_count"]) for row in selected]
            ),
        }

    gold_instances = Counter()
    gold_unique: dict[str, set[str]] = {"bm25": set(), "colvec": set()}
    for qid in qids:
        for page_id in qrels.get(qid, {}):
            route_name = _route(stats[page_id])
            gold_instances[route_name] += 1
            gold_unique[route_name].add(page_id)
    return {
        "page_counts": dict(counts),
        "by_route": by_route,
        "gold_page_instances_by_route": dict(gold_instances),
        "gold_page_instance_share_by_route": {
            route_name: value / sum(gold_instances.values())
            for route_name, value in gold_instances.items()
        },
        "gold_unique_page_counts_by_route": {
            route_name: len(page_ids) for route_name, page_ids in gold_unique.items()
        },
    }


def _metrics_without_per_query(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in metrics.items() if key != "per_query"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--colvec-dir", type=Path, default=DEFAULT_COLVEC_DIR)
    parser.add_argument("--pair-dir", type=Path, default=DEFAULT_PAIR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(
        list(benchmark.questions()),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    qids = [question.qid for question in questions]
    if len(qids) != EXPECTED_QUERIES:
        raise RuntimeError(f"Expected {EXPECTED_QUERIES} queries, got {len(qids)}")

    # The ColVec export defines the canonical page order and is validated before
    # any qrel-dependent calculation.
    page_ids = [
        str(value)
        for value in _read_json(args.colvec_dir / "physics_colvec_keys.json")
    ]
    if len(page_ids) != EXPECTED_PAGES or len(set(page_ids)) != EXPECTED_PAGES:
        raise RuntimeError(f"Expected {EXPECTED_PAGES} unique Physics pages")

    layout_stats = _page_layout_stats(args.parsed_run, page_ids)
    route_by_page = {page_id: _route(layout_stats[page_id]) for page_id in page_ids}
    text_page_ids = [page_id for page_id in page_ids if route_by_page[page_id] == "bm25"]
    visual_page_ids = [page_id for page_id in page_ids if route_by_page[page_id] == "colvec"]
    if not text_page_ids or not visual_page_ids:
        raise RuntimeError("Adaptive route collapsed to one branch")

    colvec_scores, ordered_page_ids, ordered_qids, colvec_meta = _load_colvec(
        args.colvec_dir, page_ids, qids
    )
    if ordered_page_ids != page_ids or ordered_qids != qids:
        raise RuntimeError("Canonical ordering changed while loading ColVec")

    # Only text-route pages are put in the adaptive BM25 index.  The source
    # text is cached KDL/pdf-inspector output; no new parsing occurs here.
    corpus = HierarchyCorpus.from_parsed_run(
        args.parsed_run, subset="physics", page_ids=page_ids
    )
    text_index_started = time.perf_counter()
    adaptive_bm25 = build_bm25(
        (page_id, corpus.pages[page_id].text) for page_id in text_page_ids
    )
    index_build_seconds = time.perf_counter() - text_index_started

    # Reuse the exact cached full-corpus BM25 run used by the earlier Physics
    # comparison.  If absent, construct the same run from the cached corpus.
    full_bm25_path = args.pair_dir / "bm25_french_bm25-french_vs-english.jsonl"
    if full_bm25_path.is_file():
        full_bm25_run = _load_run(full_bm25_path)
    else:
        full_index = build_bm25(
            (page_id, corpus.pages[page_id].text) for page_id in page_ids
        )
        full_bm25_run = {}
        for question in questions:
            scores = _index_scores(full_index, question.query, top_k=len(page_ids))
            full_bm25_run[question.qid] = [
                {
                    "chunk_id": page_id,
                    "doc_id": page_id,
                    "score": score,
                    "rank": rank,
                }
                for rank, (page_id, score) in enumerate(
                    _sort_scores(scores), 1
                )
            ]
    if set(full_bm25_run) != set(qids):
        raise RuntimeError("Full-corpus BM25 run does not contain exactly the Physics qids")

    qid_to_colvec_row = {qid: position for position, qid in enumerate(qids)}
    adaptive_run: dict[str, list[dict[str, Any]]] = {}
    colvec_run: dict[str, list[dict[str, Any]]] = {}
    adaptive_bm25_route_run: dict[str, list[dict[str, Any]]] = {}
    adaptive_colvec_route_run: dict[str, list[dict[str, Any]]] = {}
    retrieval_times: dict[str, float] = {"adaptive": 0.0, "colvec": 0.0}

    for question in questions:
        qid = question.qid
        row = qid_to_colvec_row[qid]

        colvec_started = time.perf_counter()
        colvec_ranked = _rank_colvec(colvec_scores, row, page_ids)
        colvec_run[qid] = [
            {
                "chunk_id": page_id,
                "doc_id": page_id,
                "score": score,
                "rank": rank,
            }
            for rank, (page_id, score) in enumerate(colvec_ranked[:TOP_K], 1)
        ]
        retrieval_times["colvec"] += time.perf_counter() - colvec_started

        adaptive_started = time.perf_counter()
        bm25_scores = _index_scores(
            adaptive_bm25,
            question.query,
            top_k=len(text_page_ids),
        )
        bm25_ranked = _sort_scores(bm25_scores)
        colvec_route_ranked = _rank_colvec(
            colvec_scores, row, page_ids, set(visual_page_ids)
        )
        adaptive_bm25_route_run[qid] = [
            _compact_run(qid, bm25_ranked, TOP_K)["chunks"][i]
            for i in range(min(TOP_K, len(bm25_ranked)))
        ]
        adaptive_colvec_route_run[qid] = [
            _compact_run(qid, colvec_route_ranked, TOP_K)["chunks"][i]
            for i in range(min(TOP_K, len(colvec_route_ranked)))
        ]

        # Each route is normalized only against its own active index, then the
        # two disjoint page populations are merged.  This avoids adding raw
        # BM25 and ColVec scores with incompatible units.
        bm25_norm = normalise_scores(dict(bm25_ranked))
        colvec_norm = normalise_scores(dict(colvec_route_ranked))
        merged_scores = {
            page_id: (
                bm25_norm.get(page_id, 0.0)
                if route_by_page[page_id] == "bm25"
                else colvec_norm.get(page_id, 0.0)
            )
            for page_id in page_ids
        }
        merged_ranked = _sort_scores(merged_scores)
        adaptive_run[qid] = [
            {
                "chunk_id": page_id,
                "doc_id": page_id,
                "score": score,
                "rank": rank,
            }
            for rank, (page_id, score) in enumerate(merged_ranked[:TOP_K], 1)
        ]
        retrieval_times["adaptive"] += time.perf_counter() - adaptive_started

    # The qrels are intentionally loaded only after score, route and inventory
    # validation is complete.
    qrels = benchmark.qrels()
    if set(qrels) != set(qids):
        raise RuntimeError("Qrels do not contain exactly the 302 Physics qids")

    full_bm25_metrics = _derived_metrics(full_bm25_run, qids, qrels)
    colvec_metrics = _derived_metrics(colvec_run, qids, qrels)
    adaptive_metrics = _derived_metrics(adaptive_run, qids, qrels)
    branch_metrics = _branch_candidate_metrics(
        qids, qrels, adaptive_bm25_route_run, adaptive_colvec_route_run
    )
    route_metrics = _route_report(page_ids, layout_stats, qids, qrels)

    per_query: list[dict[str, Any]] = []
    full_rows = {row["qid"]: row for row in full_bm25_metrics["per_query"]}
    colvec_rows = {row["qid"]: row for row in colvec_metrics["per_query"]}
    adaptive_rows = {row["qid"]: row for row in adaptive_metrics["per_query"]}
    for qid in qids:
        gold = set(qrels[qid])
        bm25_route_top = {
            str(row["chunk_id"])
            for row in adaptive_bm25_route_run[qid][:TOP_K]
        }
        colvec_route_top = {
            str(row["chunk_id"])
            for row in adaptive_colvec_route_run[qid][:TOP_K]
        }
        per_query.append(
            {
                "qid": qid,
                "gold_page_count": len(gold),
                "gold_route_counts": dict(
                    Counter(_route(layout_stats[page_id]) for page_id in gold)
                ),
                "adaptive_top10": adaptive_rows[qid]["top10_pages"],
                "adaptive_page_recall@10": adaptive_rows[qid]["page_recall@10"],
                "adaptive_page_hit@10": adaptive_rows[qid]["page_hit@10"],
                "full_bm25_page_recall@10": full_rows[qid]["page_recall@10"],
                "colvec_page_recall@10": colvec_rows[qid]["page_recall@10"],
                "bm25_route_candidate_recall@100": len(bm25_route_top & gold) / len(gold),
                "colvec_route_candidate_recall@100": len(colvec_route_top & gold) / len(gold),
                "two_route_candidate_recall@100": len((bm25_route_top | colvec_route_top) & gold) / len(gold),
                "visual_pages_in_adaptive_top10": sum(
                    route_by_page[page_id] == "colvec"
                    for page_id in adaptive_rows[qid]["top10_pages"]
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        runs_dir / "bm25_full_corpus.jsonl",
        [
            _compact_run(qid, [(str(row["chunk_id"]), float(row.get("score", 0.0))) for row in full_bm25_run[qid]], TOP_K)
            for qid in qids
        ],
    )
    _write_jsonl(
        runs_dir / "colvec_full_corpus.jsonl",
        [
            _compact_run(qid, [(str(row["chunk_id"]), float(row.get("score", 0.0))) for row in colvec_run[qid]], TOP_K)
            for qid in qids
        ],
    )
    _write_jsonl(
        runs_dir / "adaptive_layout_route.jsonl",
        [
            _compact_run(qid, [(str(row["chunk_id"]), float(row.get("score", 0.0))) for row in adaptive_run[qid]], TOP_K)
            for qid in qids
        ],
    )
    _write_jsonl(args.output_dir / "per_query.jsonl", per_query)

    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len({page_id.split("#page=", 1)[0] for page_id in page_ids}),
        "method": "exclusive page routing: cached PDF-inspector/KDL layout -> BM25 or ColVec",
        "config": {
            "route": {
                "colvec_if": [
                    f"word_count == 0",
                    f"visual_block_count >= 1 and word_count <= {LOW_TEXT_WORDS}",
                    f"visual_block_count >= 2 and word_count <= {MEDIUM_TEXT_WORDS}",
                ],
                "otherwise": "BM25",
                "thresholds_fixed_before_qrel_evaluation": True,
            },
            "final_depth": FINAL_K,
            "run_depth": TOP_K,
            "score_merge": "normalize BM25 within text route and ColVec within visual route; merge disjoint pages",
            "qrels_used_for_route_or_index": False,
        },
        "route_stats": route_metrics,
        "sources": {
            "parsed_run": str(args.parsed_run),
            "colvec_dir": str(args.colvec_dir),
            "colvec_model": colvec_meta.get("model"),
            "colvec_score_bytes": int((args.colvec_dir / "physics_colvec_scores.npy").stat().st_size),
            "parsed_run_bytes": _path_size_bytes(args.parsed_run),
        },
        "timing_seconds": {
            "adaptive_bm25_index_build": round(index_build_seconds, 6),
            "adaptive_retrieval_total": round(retrieval_times["adaptive"], 6),
            "adaptive_retrieval_ms_per_query": round(
                1000.0 * retrieval_times["adaptive"] / len(qids), 6
            ),
            "colvec_ranking_total": round(retrieval_times["colvec"], 6),
            "colvec_ranking_ms_per_query": round(
                1000.0 * retrieval_times["colvec"] / len(qids), 6
            ),
            "wall_seconds": round(time.perf_counter() - started, 6),
        },
        "arms": {
            "bm25_full_corpus": _metrics_without_per_query(full_bm25_metrics),
            "colvec_full_corpus": _metrics_without_per_query(colvec_metrics),
            "adaptive_layout_route": {
                **_metrics_without_per_query(adaptive_metrics),
                "candidate_metrics": branch_metrics,
            },
        },
        "validation": {
            "qids": len(qids),
            "pages": len(page_ids),
            "unique_pages": len(set(page_ids)),
            "qrels": len(qrels),
            "finite_colvec_scores": bool(np.isfinite(colvec_scores).all()),
            "text_route_pages": len(text_page_ids),
            "colvec_route_pages": len(visual_page_ids),
        },
    }
    _write_json(args.output_dir / "report.json", report)

    lines = [
        "# Physics adaptive ingestion baseline",
        "",
        "Offline simulation of page-level adaptive ingestion using cached KDL/PDF-inspector layout signals.",
        "",
        "## Fixed route",
        "",
        f"- ColVec route: empty pages, or visual block >= 1 with <= {LOW_TEXT_WORDS} words, or visual block >= 2 with <= {MEDIUM_TEXT_WORDS} words.",
        "- BM25 route: all other pages.",
        "- The route is exclusive: each page is indexed by exactly one branch.",
        "- BM25 and ColVec scores are normalized inside their own route before merging; raw scores are not added.",
        "- Qrels are not used to build the route or either index.",
        "",
        "## Page routing",
        "",
        "| Route | Pages | Share | Avg words | Avg blocks | Gold page instances |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route_name in ("bm25", "colvec"):
        values = route_metrics["by_route"][route_name]
        lines.append(
            f"| {route_name} | {int(values['pages'])} | {values['page_share']:.2%} | "
            f"{values['avg_words']:.1f} | {values['avg_blocks']:.1f} | "
            f"{route_metrics['gold_page_instances_by_route'].get(route_name, 0)} |"
        )
    lines += [
        "",
        "## Retrieval results",
        "",
        "| Arm | nDCG@10 | Page hit@10 | Page recall@10 | File recall@3 | Page population used |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, key in (
        ("BM25 full corpus", "bm25_full_corpus"),
        ("ColVec full corpus", "colvec_full_corpus"),
        ("Adaptive layout route", "adaptive_layout_route"),
    ):
        metrics = report["arms"][key]
        file_metrics = metrics["file_metrics_by_k"]["3"]
        scope = len(page_ids)
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.2f} | {metrics['page_hit@10']:.2%} | "
            f"{metrics['page_recall@10']:.2%} | {file_metrics['file_recall']:.2%} | {scope} |"
        )
    lines += [
        "",
        "## Adaptive candidate diagnostics",
        "",
        f"- BM25-route top-{TOP_K} candidate recall: **{branch_metrics['bm25_route_candidate_recall']:.2%}**.",
        f"- ColVec-route top-{TOP_K} candidate recall: **{branch_metrics['colvec_route_candidate_recall']:.2%}**.",
        f"- Union of the two route-local top-{TOP_K} lists: **{branch_metrics['two_route_candidate_recall']:.2%}** candidate recall and **{branch_metrics['two_route_candidate_hit']:.2%}** query hit rate.",
        "",
        "## Cost interpretation",
        "",
        f"- The simulated BM25 index contains {len(text_page_ids)} pages; the ColVec route contains {len(visual_page_ids)} pages.",
        f"- The route would avoid parsing {len(visual_page_ids) / len(page_ids):.2%} of pages only if the layout gate were available before PDF-inspector parsing. This offline run already uses cached parser output, so it does not measure that preparation saving.",
        f"- Adaptive retrieval wall time (index setup plus query loop) is reported in `report.json`; ColVec encoding/inference time is not included because scores are cached.",
        "",
        "## Caveats",
        "",
        "- This is a first fixed-rule baseline, not an optimized adaptive policy.",
        "- ColVec is the cached `webAI-Official/webAI-ColVec1.1-4b` score matrix; it is visual-only and no new GPU inference is performed.",
        "- The experiment evaluates retrieval quality, not end-to-end answer accuracy.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "route_pages": route_metrics["page_counts"],
                "bm25_page_recall@10": full_bm25_metrics["page_recall@10"],
                "colvec_page_recall@10": colvec_metrics["page_recall@10"],
                "adaptive_page_recall@10": adaptive_metrics["page_recall@10"],
                "adaptive_file_recall@3": adaptive_metrics["file_metrics_by_k"]["3"]["file_recall"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
