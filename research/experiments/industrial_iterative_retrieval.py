"""Offline iterative and feedback retrieval experiments for ViDoRe Industrial.

This runner keeps the existing light-preparation contract: the corpus is the
KDL + PDF-inspector page text, and all retrieval is lexical BM25.  It tests
cheap second-pass feedback mechanisms rather than adding a new model:

* pseudo-relevance feedback (PRF) with IDF-weighted expansion terms;
* a residual-query pass that asks BM25 for terms not covered by the first
  retrieved pages;
* soft file-to-page feedback, where the file score is a prior and never a
  hard filter.

The first pass is kept in every final score.  This makes the experiment an
iterative refinement of the original query rather than an opaque replacement
of the baseline.  No qrels are read during retrieval; qrels are only used in
the final evaluation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import (  # noqa: E402
    HierarchyCorpus,
    build_bm25,
    normalise_scores,
    sort_scores,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _file_id,
    _index_scores,
    _load_run,
    _paired_comparison,
    _safe_name,
    _write_run,
)
from src.chunking_embedding.lexical import analyze  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_PARSED_RUN = (
    ROOT / "data/output/vidore-v3-industrial-kdl-pdf-inspector/fcfa9a665c256e86"
)
DEFAULT_BENCHMARK_ROOT = ROOT / "data/raw/benchmarks/vidore_v3"
DEFAULT_BASELINE_RUN = (
    ROOT / "data/benchmark/vidore_v3/results/industrial_discovery_bm25_english.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/industrial_iterative_retrieval"
)


@dataclass(frozen=True)
class IterativeConfig:
    name: str
    feedback_k: int = 3
    expansion_terms: int = 8
    expansion_weight: float = 0.25
    first_pass_weight: float = 0.65
    parent_weight: float = 0.15
    file_k: int = 10
    residual_weight: float = 0.0
    residual_terms: int = 6
    final_depth: int = 100


def _weighted_search(
    index: BM25Index,
    weighted_terms: Mapping[str, float],
    *,
    allowed_ids: set[str] | None = None,
) -> dict[str, float]:
    """Score an index with explicit term weights.

    ``BM25Index.search`` intentionally exposes an unweighted query interface.
    PRF needs a small amount of term weighting, so this mirrors the index's
    BM25 contribution while applying a weight per analyzed term.
    """
    allowed_positions = None
    if allowed_ids is not None:
        allowed_positions = {
            position for position, node_id in enumerate(index.chunk_ids)
            if node_id in allowed_ids
        }
    total = len(index.lengths)
    scores: dict[int, float] = defaultdict(float)
    for term, weight in weighted_terms.items():
        if weight <= 0:
            continue
        posting = index.postings.get(term)
        if not posting:
            continue
        df = len(posting)
        idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
        for position, frequency in posting:
            if allowed_positions is not None and position not in allowed_positions:
                continue
            length = index.lengths[position] or 1
            denominator = frequency + index.k1 * (
                1 - index.b + index.b * length / (index.avgdl or 1.0)
            )
            scores[position] += float(weight) * idf * (
                frequency * (index.k1 + 1)
            ) / denominator
    return {
        index.chunk_ids[position]: float(score)
        for position, score in scores.items()
        if score > 0
    }


def _query_terms(query: str) -> list[str]:
    return list(dict.fromkeys(analyze(query)))


def _idf(index: BM25Index, term: str) -> float:
    posting = index.postings.get(term)
    if not posting:
        return 0.0
    total = len(index.lengths)
    return math.log(1.0 + (total - len(posting) + 0.5) / (len(posting) + 0.5))


def _feedback_terms(
    index: BM25Index,
    pages: Mapping[str, Any],
    feedback_hits: Sequence[tuple[str, float]],
    *,
    original_terms: set[str],
    limit: int,
) -> list[tuple[str, float]]:
    """Return high-IDF terms from top pages, excluding original terms.

    The rank discount prevents a noisy third result from dominating a strong
    first result.  An IDF floor removes page boilerplate and generic terms.
    """
    weights: defaultdict[str, float] = defaultdict(float)
    for rank, (page_id, _) in enumerate(feedback_hits, 1):
        text = str(pages[page_id].text)
        counts = Counter(_query_terms(text))
        for term, frequency in counts.items():
            if term in original_terms or len(term) < 3:
                continue
            idf = _idf(index, term)
            if idf <= 0.15:
                continue
            weights[term] += (
                (1.0 + math.log(float(frequency)))
                * idf
                / math.sqrt(float(rank))
            )
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    if not ranked:
        return []
    maximum = ranked[0][1]
    return [(term, value / maximum) for term, value in ranked[:limit]]


def _residual_terms(query: str, feedback_pages: Sequence[str], pages: Mapping[str, Any]) -> list[str]:
    """Keep query terms absent from the first evidence pages."""
    query_terms = _query_terms(query)
    covered: set[str] = set()
    for page_id in feedback_pages:
        covered.update(_query_terms(str(pages[page_id].text)))
    return [term for term in query_terms if term not in covered]


def _run_rows(
    corpus: HierarchyCorpus,
    scores: Mapping[str, float],
    *,
    depth: int,
    component_maps: Mapping[str, Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:
    ranked = sort_scores(scores)[:depth]
    return [
        {
            "chunk_id": page_id,
            "doc_id": page_id,
            "text": corpus.pages[page_id].text,
            "score": round(float(score), 8),
            "rank": rank,
            "scores": {
                key: round(float(value), 8)
                for key, value in (component_maps or {}).get(page_id, {}).items()
            },
        }
        for rank, (page_id, score) in enumerate(ranked, 1)
    ]


def _retrieve_one(
    corpus: HierarchyCorpus,
    page_index: BM25Index,
    file_index: BM25Index,
    qid: str,
    query: str,
    config: IterativeConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first = _index_scores(page_index, query, top_k=len(corpus.page_order))
    first_norm = normalise_scores(first)
    first_hits = sort_scores(first)[: max(config.feedback_k, 1)]
    original_terms = set(_query_terms(query))
    expansions = _feedback_terms(
        page_index,
        corpus.pages,
        first_hits,
        original_terms=original_terms,
        limit=config.expansion_terms,
    )
    weighted_terms = {term: 1.0 for term in original_terms}
    for term, weight in expansions:
        weighted_terms[term] = config.expansion_weight * weight
    second = _weighted_search(page_index, weighted_terms)
    second_norm = normalise_scores(second)

    residual_page_ids = [page_id for page_id, _ in first_hits]
    residual = _residual_terms(query, residual_page_ids, corpus.pages)
    residual_scores = (
        _weighted_search(
            page_index,
            {term: 1.0 for term in residual[: config.residual_terms]},
        )
        if config.residual_weight > 0 and residual
        else {}
    )
    residual_norm = normalise_scores(residual_scores)

    # File retrieval uses the same first-pass page signal as the page stage,
    # plus a direct BM25 score on concatenated extracted file text.  It is a
    # soft prior: all pages remain eligible in the final ranking.
    page_pool: dict[str, float] = {}
    for page_id, score in first_norm.items():
        file_id = _file_id(page_id)
        page_pool[file_id] = max(page_pool.get(file_id, 0.0), float(score))
    direct_file = _index_scores(file_index, query, top_k=len(corpus.file_order))
    direct_file_norm = normalise_scores(direct_file)
    file_scores = {
        file_id: 0.5 * direct_file_norm.get(file_id, 0.0)
        + 0.5 * page_pool.get(file_id, 0.0)
        for file_id in corpus.file_order
    }
    file_norm = normalise_scores(file_scores)
    file_ranked = sort_scores(file_scores)
    selected_files = {file_id for file_id, _ in file_ranked[: config.file_k]}

    final_scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for page_id in corpus.page_order:
        parent = file_norm.get(_file_id(page_id), 0.0)
        # Parent prior is gated to selected files.  Outside the soft file
        # budget it contributes zero, but the lexical first/second pass stays
        # available, so a file-stage miss can still be recovered.
        parent_signal = parent if _file_id(page_id) in selected_files else 0.0
        score = (
            config.first_pass_weight * first_norm.get(page_id, 0.0)
            + (1.0 - config.first_pass_weight) * second_norm.get(page_id, 0.0)
            + config.parent_weight * parent_signal
            + config.residual_weight * residual_norm.get(page_id, 0.0)
        )
        final_scores[page_id] = score
        components[page_id] = {
            "first_page_bm25": first_norm.get(page_id, 0.0),
            "feedback_page_bm25": second_norm.get(page_id, 0.0),
            "residual_page_bm25": residual_norm.get(page_id, 0.0),
            "parent_file_score": parent_signal,
            "final_score": score,
        }
    run = _run_rows(
        corpus,
        final_scores,
        depth=config.final_depth,
        component_maps=components,
    )
    trace = {
        "qid": qid,
        "config": asdict(config),
        "first_pass": [
            {"page_id": page_id, "score": round(score, 8)}
            for page_id, score in first_hits
        ],
        "feedback_terms": expansions,
        "residual_terms": residual,
        "file_candidates": [
            {"file_id": file_id, "score": round(score, 8), "rank": rank}
            for rank, (file_id, score) in enumerate(file_ranked, 1)
        ],
        "selected_files": sorted(selected_files),
        "final_pages": [row["chunk_id"] for row in run],
    }
    return run, trace


def _validation(run: Mapping[str, Sequence[Mapping[str, Any]]], qids: Sequence[str]) -> None:
    if set(run) != set(qids):
        raise RuntimeError("iterative run qids do not match benchmark")
    for qid in qids:
        ids = [str(row["chunk_id"]) for row in run[qid]]
        if len(ids) != 100 or len(ids) != len(set(ids)):
            raise RuntimeError(f"{qid}: expected 100 unique page candidates")


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Industrial iterative retrieval",
        "",
        "Offline feedback-loop retrieval over KDL + PDF-inspector extracted page text.",
        "",
        "## Protocol",
        "",
        f"- Queries: **{report['queries']}** English Industrial queries.",
        f"- Corpus: **{report['index_counts']['files']} files**, **{report['index_counts']['pages']} pages**.",
        "- First pass: page BM25; file scores combine direct BM25 on concatenated extracted file text and max page pooling.",
        "- Second pass: IDF-weighted PRF expansion, optional residual query, and a soft parent-file prior.",
        "- The file stage never hard-filters pages in this experiment.",
        "",
        "## Results",
        "",
        "| Method | nDCG@10 | Page hit@10 | Page recall@10 | File recall@3 | Δ vs cached page BM25 (pp) | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, method in report["methods"].items():
        metrics = method["retrieval_metrics"]
        delta = method.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0)
        file_recall = metrics["file_metrics_by_k"]["3"]["file_recall"]
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.2f} | {metrics['page_hit@10']:.2%} | "
            f"{metrics['page_recall@10']:.2%} | {file_recall:.2%} | {delta:+.2f} | "
            f"{method.get('timing_seconds', {}).get('retrieval', 0.0):.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "PRF terms are generated from first-pass pages without qrels. The original query terms remain active in every second pass.",
        "A positive page gain with a negative file gain indicates page-level refinement without better discovery; both metrics are reported separately.",
        "",
        "## Caveats",
        "",
        "- This is a screening experiment on the 283-query Industrial benchmark; weights are fixed in the script and should be reselected with held-out folds before a final claim.",
        "- All BM25 inputs are extracted text, never raw PDF bytes.",
        "- File recall is derived from the first 100 final page candidates, not an independent production file index.",
        "",
        "Runs and traces are stored under `runs/` and `traces/`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="industrial", language="english")
    questions_list = sorted(
        list(benchmark.questions()),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    qids = [question.qid for question in questions_list]
    qrels = benchmark.qrels()
    page_ids = sorted(document.doc_id for document in benchmark.corpus())
    if len(qids) != 283 or len(page_ids) != 5244:
        raise RuntimeError(f"Unexpected Industrial inventory: {len(qids)} queries, {len(page_ids)} pages")

    build_started = time.perf_counter()
    corpus = HierarchyCorpus.from_parsed_run(args.parsed_run, subset="industrial", page_ids=page_ids)
    page_index = build_bm25((page_id, corpus.pages[page_id].text) for page_id in corpus.page_order)
    file_index = build_bm25(corpus.file_texts("all_text").items())
    index_build = time.perf_counter() - build_started

    baseline = _load_run(args.baseline_run)
    methods: dict[str, dict[str, Any]] = {
        "Cached page BM25 baseline": {
            "config": None,
            "retrieval_metrics": _derived_metrics(baseline, qids, qrels),
            "timing_seconds": {"retrieval": 0.0},
        }
    }
    configs = [
        IterativeConfig(name="soft-parent-kf10", feedback_k=1, expansion_terms=0, expansion_weight=0.0, first_pass_weight=1.0, parent_weight=0.15),
        IterativeConfig(name="prf-global-k3-m5-b10", feedback_k=3, expansion_terms=5, expansion_weight=0.10, first_pass_weight=0.65, parent_weight=0.0),
        IterativeConfig(name="prf-global-k3-m8-b25", feedback_k=3, expansion_terms=8, expansion_weight=0.25, first_pass_weight=0.65, parent_weight=0.0),
        IterativeConfig(name="prf-global-k5-m8-b25", feedback_k=5, expansion_terms=8, expansion_weight=0.25, first_pass_weight=0.65, parent_weight=0.0),
        IterativeConfig(name="prf-global-k3-m8-b40", feedback_k=3, expansion_terms=8, expansion_weight=0.40, first_pass_weight=0.65, parent_weight=0.0),
        IterativeConfig(name="prf-parent-k3-m8-b25", feedback_k=3, expansion_terms=8, expansion_weight=0.25, first_pass_weight=0.65, parent_weight=0.15),
        IterativeConfig(name="prf-parent-k5-m8-b25", feedback_k=5, expansion_terms=8, expansion_weight=0.25, first_pass_weight=0.65, parent_weight=0.15),
        IterativeConfig(name="prf-parent-k3-m12-b25", feedback_k=3, expansion_terms=12, expansion_weight=0.25, first_pass_weight=0.65, parent_weight=0.15),
        IterativeConfig(name="prf-residual-k3", feedback_k=3, expansion_terms=8, expansion_weight=0.25, first_pass_weight=0.65, parent_weight=0.15, residual_weight=0.15, residual_terms=6),
        IterativeConfig(name="prf-residual-k1", feedback_k=1, expansion_terms=5, expansion_weight=0.20, first_pass_weight=0.65, parent_weight=0.15, residual_weight=0.15, residual_terms=6),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)
    (args.output_dir / "traces").mkdir(exist_ok=True)
    all_traces: dict[str, dict[str, Any]] = {}
    for config in configs:
        config_started = time.perf_counter()
        run_by_qid: dict[str, list[dict[str, Any]]] = {}
        traces: dict[str, dict[str, Any]] = {}
        for question in questions_list:
            run, trace = _retrieve_one(corpus, page_index, file_index, question.qid, question.query, config)
            run_by_qid[question.qid] = run
            traces[question.qid] = trace
        _validation(run_by_qid, qids)
        metrics = _derived_metrics(run_by_qid, qids, qrels)
        methods[config.name] = {
            "config": asdict(config),
            "retrieval_metrics": metrics,
            "timing_seconds": {"retrieval": round(time.perf_counter() - config_started, 6)},
        }
        all_traces[config.name] = traces
        _write_run(
            args.output_dir / "runs" / f"{_safe_name(config.name)}.jsonl",
            run_by_qid,
            qids,
            queries={question.qid: question.query for question in questions_list},
        )

    baseline_metrics = methods["Cached page BM25 baseline"]["retrieval_metrics"]
    for name, method in methods.items():
        if name != "Cached page BM25 baseline":
            method["comparison_to_baseline"] = _paired_comparison(
                baseline_metrics, method["retrieval_metrics"]
            )
    for name, traces in all_traces.items():
        with (args.output_dir / "traces" / f"{_safe_name(name)}.jsonl").open("w", encoding="utf-8") as handle:
            for qid in qids:
                handle.write(json.dumps(traces[qid], ensure_ascii=False) + "\n")

    report: dict[str, Any] = {
        "dataset": "vidore_v3/industrial",
        "evaluation_language": "english",
        "queries": len(qids),
        "pages": len(page_ids),
        "index_counts": {
            "files": len(corpus.files),
            "pages": len(corpus.pages),
            "pages_with_parser_text": sum(bool(node.metadata.get("has_parser_text")) for node in corpus.pages.values()),
            "blocks": len(corpus.blocks),
            "paragraphs": len(corpus.paragraphs),
            "sentences": len(corpus.sentences),
            "evidence_atoms": len(corpus.evidence_atoms),
            "index_build_seconds": round(index_build, 6),
        },
        "sources": {
            "parsed_run": str(args.parsed_run),
            "baseline_run": str(args.baseline_run),
            "file_score": "0.5 * direct BM25 on concatenated PDF-inspector page text + 0.5 * max pooled first-pass page BM25",
        },
        "methods": methods,
        "timing_seconds": {"index_build": round(index_build, 6), "total": round(time.perf_counter() - started, 6)},
        "notes": [
            "No qrels are used to form feedback terms, parent scores or selected files.",
            "The cached page BM25 baseline is retained as the primary comparator.",
            "The 47.47% target corresponds to the existing Industrial Kf=10 cascade artifact; the cached page BM25 baseline itself is 47.71% in this repository.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    summary = {
        name: {
            "ndcg@10": round(method["retrieval_metrics"]["ndcg@10"], 4),
            "page_recall@10": round(method["retrieval_metrics"]["page_recall@10"] * 100, 2),
            "file_recall@3": round(method["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"] * 100, 2),
            "page_delta_pp": round(method.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0), 2),
        }
        for name, method in methods.items()
    }
    print(json.dumps({"summary": summary, "output": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
