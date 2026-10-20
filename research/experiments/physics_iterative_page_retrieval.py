"""Probe iterative page retrieval and score-curve breakpoints on Physics.

This is a deliberately small, score-only experiment.  The file scope is fixed
to the current hierarchical Kf=3 arm.  The first page is selected by the
existing hierarchical score; after each selection, a few high-IDF terms from
the selected page are added as pseudo-relevance feedback for selecting the
next page.  The original hierarchical score remains the anchor, so feedback
cannot completely replace the query signal.

The script also reports a breakpoint diagnostic: the largest adjacent drop in
the initial page-score curve after rank 10.  This is an offline diagnostic for
whether a query-specific stop point is plausible; it is not used to tune the
feedback weight or to inspect qrels.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
    normalise_scores,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    HierarchicalRetriever,
    _build_corpus_and_indexes,
    _derived_metrics,
    _file_id,
    _index_scores,
    _load_visual_scores,
    _page_vector_units,
)
from src.chunking_embedding.lexical import analyze  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_PARSED_RUN = (
    ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
)
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_OUTPUT_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/physics_iterative_page_retrieval"
)

EXPECTED_QUERIES = 302
EXPECTED_PAGES = 1674
K_FILES = 3
MAX_OUTPUT_PAGES = 100
MIN_BREAK_RANK = 10
FEEDBACK_WEIGHT = 0.25
FEEDBACK_TERMS_PER_PAGE = 8
MAX_FEEDBACK_TERMS = 24


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _run_records(ranked: Sequence[tuple[str, float]], corpus: Any) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": page_id,
            "doc_id": page_id,
            "text": corpus.pages[page_id].text,
            "score": float(score),
            "rank": rank,
        }
        for rank, (page_id, score) in enumerate(ranked[:MAX_OUTPUT_PAGES], 1)
    ]


def _page_terms(
    text: str,
    query_terms: set[str],
    document_frequency: Mapping[str, int],
    document_count: int,
) -> list[str]:
    counts = Counter(analyze(text))
    scored: list[tuple[float, str]] = []
    for term, frequency in counts.items():
        if len(term) < 4 or term in query_terms:
            continue
        df = int(document_frequency.get(term, 0))
        if not df:
            continue
        idf = float(np.log((document_count + 1) / (df + 1)) + 1.0)
        scored.append((float(frequency) * idf, term))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [term for _, term in scored[:FEEDBACK_TERMS_PER_PAGE]]


def _document_frequencies(corpus: Any) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for page in corpus.pages.values():
        frequencies.update(set(analyze(page.text)))
    return frequencies


def _curve_break(
    ranked: Sequence[tuple[str, float]],
    *,
    min_rank: int,
) -> tuple[int, float]:
    """Return the prefix ending at the largest adjacent score drop."""
    if len(ranked) <= min_rank:
        return len(ranked), 0.0
    start = min_rank - 1
    end = min(len(ranked) - 1, MAX_OUTPUT_PAGES - 1)
    drops = [float(ranked[index][1] - ranked[index + 1][1]) for index in range(start, end)]
    if not drops:
        return min(len(ranked), MAX_OUTPUT_PAGES), 0.0
    offset = int(np.argmax(drops))
    return start + offset + 1, drops[offset]


def _iterative_rank(
    query: str,
    candidate_pages: Sequence[str],
    base_scores: Mapping[str, float],
    page_index: Any,
    corpus: Any,
    document_frequency: Mapping[str, int],
) -> tuple[list[tuple[str, float]], list[str]]:
    """Select pages sequentially using anchored pseudo-relevance feedback."""
    query_terms = set(analyze(query))
    selected: list[tuple[str, float]] = []
    remaining = set(candidate_pages)
    feedback_terms: list[str] = []
    selected_terms: list[str] = []

    while remaining and len(selected) < MAX_OUTPUT_PAGES:
        feedback_scores: dict[str, float] = {}
        if feedback_terms:
            expanded_query = f"{query} {' '.join(feedback_terms)}"
            raw = _index_scores(
                page_index,
                expanded_query,
                allowed_ids=remaining,
                top_k=len(remaining),
            )
            feedback_scores = normalise_scores(raw)

        scored = {
            page_id: (1.0 - FEEDBACK_WEIGHT) * float(base_scores.get(page_id, 0.0))
            + FEEDBACK_WEIGHT * float(feedback_scores.get(page_id, 0.0))
            for page_id in remaining
        }
        page_id, score = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[0]
        selected.append((page_id, score))
        remaining.remove(page_id)

        new_terms = _page_terms(
            corpus.pages[page_id].text,
            query_terms,
            document_frequency,
            len(corpus.pages),
        )
        selected_terms.extend(new_terms)
        feedback_terms = list(dict.fromkeys(selected_terms))[-MAX_FEEDBACK_TERMS:]

    return selected, feedback_terms


def _prefix_recall(
    ranked: Sequence[tuple[str, float]],
    gold_pages: set[str],
    prefix_length: int,
) -> float:
    if not gold_pages:
        return 0.0
    return len({page_id for page_id, _ in ranked[:prefix_length]} & gold_pages) / len(gold_pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
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
    page_ids = _page_vector_units(args.page_vector_dir)
    if len(page_ids) != EXPECTED_PAGES or len(set(page_ids)) != EXPECTED_PAGES:
        raise RuntimeError("Physics page inventory is not exactly 1,674 unique pages")

    corpus, page_index, file_indexes, fine_indexes, index_counts = _build_corpus_and_indexes(
        args.parsed_run, page_ids, fine_units=()
    )
    visual_scores, visual_meta = _load_visual_scores(
        args.page_vector_dir, args.query_vector_dir, page_ids, qids
    )
    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes=fine_indexes,
        visual_scores=visual_scores,
    )
    document_frequency = _document_frequencies(corpus)
    config = CascadeConfig(
        name="iterative-kf3",
        file_representation="all_text",
        file_pool="max",
        k_files=K_FILES,
        page_depth=len(page_ids),
        final_depth=len(page_ids),
        bm25_weight=0.70,
        parent_weight=0.15,
    )

    qrels = benchmark.qrels()
    base_runs: dict[str, list[dict[str, Any]]] = {}
    iterative_runs: dict[str, list[dict[str, Any]]] = {}
    per_query: list[dict[str, Any]] = []

    for question in questions:
        _, trace = retriever.retrieve(question.qid, question.query, config)
        base_ranked = [
            (str(row["node_id"]), float(row.get("score") or 0.0))
            for row in trace["page_candidates"]
        ]
        if not base_ranked:
            raise RuntimeError(f"No page candidates for {question.qid}")
        selected_files = [
            str(row["node_id"]) for row in trace["file_candidates"][:K_FILES]
        ]
        candidate_pages = [
            page_id
            for file_id in selected_files
            for page_id in corpus.file_to_pages.get(file_id, [])
        ]
        base_scores = dict(base_ranked)
        iterative_ranked, feedback_terms = _iterative_rank(
            question.query,
            candidate_pages,
            base_scores,
            page_index,
            corpus,
            document_frequency,
        )
        base_runs[question.qid] = _run_records(base_ranked, corpus)
        iterative_runs[question.qid] = _run_records(iterative_ranked, corpus)

        gold_pages = set(qrels.get(question.qid, {}))
        break_rank_any, break_drop_any = _curve_break(base_ranked, min_rank=1)
        break_rank_after10, break_drop_after10 = _curve_break(
            base_ranked, min_rank=MIN_BREAK_RANK
        )
        per_query.append(
            {
                "qid": question.qid,
                "candidate_file_count": len(selected_files),
                "candidate_page_count": len(candidate_pages),
                "gold_page_count": len(gold_pages),
                "base_page_recall@10": _prefix_recall(base_ranked, gold_pages, 10),
                "iterative_page_recall@10": _prefix_recall(iterative_ranked, gold_pages, 10),
                "break_rank_any": break_rank_any,
                "break_drop_any": break_drop_any,
                "base_prefix_recall_at_any_break": _prefix_recall(base_ranked, gold_pages, break_rank_any),
                "iterative_prefix_recall_at_any_break": _prefix_recall(iterative_ranked, gold_pages, break_rank_any),
                "break_rank_after10": break_rank_after10,
                "break_drop_after10": break_drop_after10,
                "base_prefix_recall_at_break_after10": _prefix_recall(base_ranked, gold_pages, break_rank_after10),
                "iterative_prefix_recall_at_break_after10": _prefix_recall(iterative_ranked, gold_pages, break_rank_after10),
                "iterative_feedback_term_count": len(feedback_terms),
                "iterative_feedback_terms": feedback_terms,
            }
        )

    base_metrics = _derived_metrics(base_runs, qids, qrels)
    iterative_metrics = _derived_metrics(iterative_runs, qids, qrels)
    break_ranks_any = [float(row["break_rank_any"]) for row in per_query]
    break_ranks_after10 = [float(row["break_rank_after10"]) for row in per_query]
    base_break_recall_any = [float(row["base_prefix_recall_at_any_break"]) for row in per_query]
    iterative_break_recall_any = [float(row["iterative_prefix_recall_at_any_break"]) for row in per_query]
    base_break_recall_after10 = [float(row["base_prefix_recall_at_break_after10"]) for row in per_query]
    iterative_break_recall_after10 = [float(row["iterative_prefix_recall_at_break_after10"]) for row in per_query]
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "method": "Kf=3 anchored iterative pseudo-relevance feedback",
        "config": {
            "k_files": K_FILES,
            "candidate_page_scope": "all pages in the selected Kf files",
            "base_page_score": "hierarchical all_text/max, BM25 0.70, V-SPLADE 0.30, parent 0.15",
            "feedback": "high-IDF terms from selected pages, original query retained",
            "feedback_terms_per_page": FEEDBACK_TERMS_PER_PAGE,
            "max_feedback_terms": MAX_FEEDBACK_TERMS,
            "feedback_weight": FEEDBACK_WEIGHT,
            "output_depth": MAX_OUTPUT_PAGES,
            "breakpoint": "largest adjacent base-score drop after rank 10",
            "qrels_used_for_selection": False,
        },
        "arms": {
            "hierarchical_kf3": base_metrics,
            "iterative_feedback_kf3": iterative_metrics,
        },
        "breakpoint_diagnostic": {
            "any_break_rank_min": min(break_ranks_any),
            "any_break_rank_p25": float(np.quantile(break_ranks_any, 0.25)),
            "any_break_rank_median": float(np.median(break_ranks_any)),
            "any_break_rank_p75": float(np.quantile(break_ranks_any, 0.75)),
            "any_break_rank_max": max(break_ranks_any),
            "mean_base_prefix_recall_at_any_break": _mean(base_break_recall_any),
            "mean_iterative_prefix_recall_at_any_break": _mean(iterative_break_recall_any),
            "mean_any_break_prefix_length": _mean(break_ranks_any),
            "any_break_at_or_before_6": int(sum(rank <= 6 for rank in break_ranks_any)),
            "any_break_at_or_before_10": int(sum(rank <= 10 for rank in break_ranks_any)),
            "post10_break_rank_min": min(break_ranks_after10),
            "post10_break_rank_p25": float(np.quantile(break_ranks_after10, 0.25)),
            "post10_break_rank_median": float(np.median(break_ranks_after10)),
            "post10_break_rank_p75": float(np.quantile(break_ranks_after10, 0.75)),
            "post10_break_rank_max": max(break_ranks_after10),
            "mean_base_prefix_recall_at_post10_break": _mean(base_break_recall_after10),
            "mean_iterative_prefix_recall_at_post10_break": _mean(iterative_break_recall_after10),
            "mean_post10_break_prefix_length": _mean(break_ranks_after10),
            "post10_break_at_or_before_10": int(sum(rank <= 10 for rank in break_ranks_after10)),
        },
        "comparison": {
            "page_recall_delta_pp": 100.0 * (
                iterative_metrics["page_recall@10"] - base_metrics["page_recall@10"]
            ),
            "ndcg_delta": iterative_metrics["ndcg@10"] - base_metrics["ndcg@10"],
            "better_queries": int(
                sum(row["iterative_page_recall@10"] > row["base_page_recall@10"] for row in per_query)
            ),
            "same_queries": int(
                sum(row["iterative_page_recall@10"] == row["base_page_recall@10"] for row in per_query)
            ),
            "worse_queries": int(
                sum(row["iterative_page_recall@10"] < row["base_page_recall@10"] for row in per_query)
            ),
        },
        "sources": {
            "parsed_run": str(args.parsed_run),
            "visual_cache": visual_meta,
            "index_counts": index_counts,
        },
        "validation": {
            "qids": len(qids),
            "pages": len(page_ids),
            "unique_pages": len(set(page_ids)),
            "qrels": len(qrels),
            "base_all_qids": set(base_runs) == set(qids),
            "iterative_all_qids": set(iterative_runs) == set(qids),
            "all_iterative_runs_have_unique_pages": all(
                len({row["chunk_id"] for row in run}) == len(run)
                for run in iterative_runs.values()
            ),
        },
        "wall_seconds": round(time.perf_counter() - started, 6),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name, runs in (("hierarchical_kf3", base_runs), ("iterative_feedback_kf3", iterative_runs)):
        with (runs_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for qid in qids:
                handle.write(json.dumps({"qid": qid, "chunks": runs[qid]}, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Physics iterative page retrieval probe",
        "",
        "## Protocol",
        "",
        "- Fixed hierarchical file scope: Kf=3; all pages in those files are available to the page stage.",
        "- First page follows the existing hierarchical score. Each next page uses the original score plus fixed 25% anchored pseudo-relevance feedback from selected page terms.",
        "- Breakpoint is diagnostic only: largest adjacent initial score drop, reported both from rank 1 and after rank 10; no qrels are used to choose it.",
        "",
        "| Arm | nDCG@10 | Page hit@10 | Page recall@10 | Derived file recall@3 |",
        "|---|---:|---:|---:|---:|",
        f"| Hierarchical Kf=3 | {base_metrics['ndcg@10']:.2f} | {base_metrics['page_hit@10']:.2%} | {base_metrics['page_recall@10']:.2%} | {base_metrics['file_metrics_by_k']['3']['file_recall']:.2%} |",
        f"| Iterative feedback Kf=3 | {iterative_metrics['ndcg@10']:.2f} | {iterative_metrics['page_hit@10']:.2%} | {iterative_metrics['page_recall@10']:.2%} | {iterative_metrics['file_metrics_by_k']['3']['file_recall']:.2%} |",
        "",
        "## Breakpoint diagnostic",
        "",
        f"- Unrestricted break rank (min / p25 / median / p75 / max): `{report['breakpoint_diagnostic']['any_break_rank_min']}` / `{report['breakpoint_diagnostic']['any_break_rank_p25']:.1f}` / `{report['breakpoint_diagnostic']['any_break_rank_median']:.1f}` / `{report['breakpoint_diagnostic']['any_break_rank_p75']:.1f}` / `{report['breakpoint_diagnostic']['any_break_rank_max']}`.",
        f"- Unrestricted break at or before rank 6: `{report['breakpoint_diagnostic']['any_break_at_or_before_6']}/{len(qids)}`; at or before rank 10: `{report['breakpoint_diagnostic']['any_break_at_or_before_10']}/{len(qids)}`.",
        f"- Break after rank 10 (min / p25 / median / p75 / max): `{report['breakpoint_diagnostic']['post10_break_rank_min']}` / `{report['breakpoint_diagnostic']['post10_break_rank_p25']:.1f}` / `{report['breakpoint_diagnostic']['post10_break_rank_median']:.1f}` / `{report['breakpoint_diagnostic']['post10_break_rank_p75']:.1f}` / `{report['breakpoint_diagnostic']['post10_break_rank_max']}`.",
        f"- Mean prefix recall at unrestricted break: base `{report['breakpoint_diagnostic']['mean_base_prefix_recall_at_any_break']:.2%}`, iterative `{report['breakpoint_diagnostic']['mean_iterative_prefix_recall_at_any_break']:.2%}`.",
        "",
        "## Interpretation",
        "",
        "- This is a probe of query-time feedback, not a trained reranker. The page text is only used to form deterministic lexical feedback terms.",
        "- A breakpoint can reduce the page scope sent downstream, but stopping alone cannot improve top-10 ranking; it changes cost/ceiling unless the next stage uses the variable scope.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "base_page_recall@10": base_metrics["page_recall@10"],
        "iterative_page_recall@10": iterative_metrics["page_recall@10"],
        "page_recall_delta_pp": report["comparison"]["page_recall_delta_pp"],
        "any_break_rank_median": report["breakpoint_diagnostic"]["any_break_rank_median"],
        "post10_break_rank_median": report["breakpoint_diagnostic"]["post10_break_rank_median"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
