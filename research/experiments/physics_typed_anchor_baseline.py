"""Typed-anchor (dictionary/bitmap-like) baseline for Physics retrieval.

Lakehouse engines use dictionaries, bloom filters and column statistics to
protect selective equality/range predicates.  For document discovery, a
cheap analogue is a separate exact field for high-information numeric,
unit and formula-variable tokens.  This experiment asks whether that field
can rescue pages whose general BM25 score is diluted by long natural-language
questions, while preserving the cached c014 hierarchical candidate pool.

The typed field is deliberately fixed and explainable.  It is not a learned
router and does not use qrels or benchmark content-modality labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunking_embedding.lexical import analyze  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _index_scores,
    _load_run,
    _paired_comparison,
    _write_run,
)


DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_PAIR_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"
DEFAULT_BASE_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_cascade_diversity_search/runs/oof_selected.jsonl"
DEFAULT_HIERARCHICAL_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_retrieval/oof_run.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_typed_anchor_baseline"

UNIT_TOKENS = {
    "a", "ampere", "bar", "c", "cal", "cm", "db", "ev", "g", "ghz", "h", "hz",
    "j", "k", "kg", "km", "kv", "l", "m", "ma", "mev", "mg", "mhz", "min", "mm",
    "mol", "ms", "mw", "n", "nm", "pa", "s", "t", "v", "w", "x", "y", "z",
}
VARIABLE_TOKENS = set("cdefghiklmnopqrstuvwxyz") | {"alpha", "beta", "gamma", "delta", "lambda", "mu", "omega", "phi", "psi", "sigma", "theta"}


def typed_tokens(text: str) -> list[str]:
    """Keep deterministic high-selectivity tokens for the auxiliary field."""
    output: list[str] = []
    for token in analyze(text):
        if any(char.isdigit() for char in token) or token in UNIT_TOKENS:
            output.append(token)
        elif len(token) == 1 and token in VARIABLE_TOKENS:
            output.append(token)
        elif token in {"alpha", "beta", "gamma", "delta", "lambda", "mu", "omega", "phi", "psi", "sigma", "theta"}:
            output.append(token)
    return output


def typed_text(text: str) -> str:
    return " ".join(typed_tokens(text))


def _normalise(scores: Mapping[str, float]) -> dict[str, float]:
    positive = [float(value) for value in scores.values() if float(value) > 0]
    maximum = max(positive, default=0.0)
    return {key: (max(0.0, float(value)) / maximum if maximum > 0 else 0.0) for key, value in scores.items()}


def _page_ids_from_metadata(path: Path) -> list[str]:
    rows = json.loads((path / "page_metadata.json").read_text(encoding="utf-8"))
    return [str(row["unit_id"]) for row in rows]


def _load_page_texts(parsed_run: Path, page_ids: list[str]) -> dict[str, str]:
    from research.data_discovery.hierarchical import HierarchyCorpus

    corpus = HierarchyCorpus.from_parsed_run(parsed_run, subset="physics", page_ids=page_ids)
    return {page_id: corpus.pages[page_id].text for page_id in page_ids}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--pair-dir", type=Path, default=DEFAULT_PAIR_DIR)
    parser.add_argument("--base-run", type=Path, default=DEFAULT_BASE_RUN)
    parser.add_argument("--hierarchical-run", type=Path, default=DEFAULT_HIERARCHICAL_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    started = time.perf_counter()

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda question: int(question.qid.rsplit("::", 1)[1]))
    qids = [question.qid for question in questions]
    qrels = benchmark.qrels()
    page_ids = _page_ids_from_metadata(args.page_vector_dir)
    texts = _load_page_texts(args.parsed_run, page_ids)
    page_index = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": page_id, "doc_id": page_id, "text": texts[page_id]} for page_id in page_ids]
    )
    typed_index = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": page_id, "doc_id": page_id, "text": typed_text(texts[page_id])} for page_id in page_ids]
    )
    # Cached weighted/c014 output stores page_base in its score components.
    # We use c014's OOF output as the existing structural candidate pool.
    base_run = _load_run(args.base_run)
    hierarchical_run = _load_run(args.hierarchical_run)
    baseline_run = _load_run(args.pair_dir / "bm25_french_bm25-french_vs-english.jsonl")

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    anchor_counts: dict[str, int] = {}
    for qid, question in zip(qids, questions):
        page_bm25 = _normalise(_index_scores(page_index, question.query, top_k=len(page_ids)))
        # Use the same cached c014 page_base when present, otherwise BM25 is a
        # conservative fallback.  Visual scores are already represented in
        # the candidate run's page_base field.
        base_rows = base_run[qid][:100]
        page_base = {
            str(row["chunk_id"]): float((row.get("scores") or {}).get("page_base", page_bm25.get(str(row["chunk_id"]), 0.0)))
            for row in base_rows
        }
        page_base = _normalise(page_base)
        anchor_scores = _normalise(_index_scores(typed_index, question.query, top_k=len(page_ids)))
        anchor_counts[qid] = sum(1 for token in typed_tokens(question.query))
        candidates = set(page_base)

        # A fixed small dictionary-field candidate expansion.  This is the
        # only arm allowed to add pages beyond the existing cascade.
        anchor_top20 = list(anchor_scores)[:20]
        candidates.update(anchor_top20)

        rerank_scores: dict[str, float] = {}
        for page_id in candidates:
            # Existing page_base is exact hybrid score for c014 candidates;
            # recomputed BM25 is used only for new dictionary candidates.
            base_value = page_base.get(page_id, page_bm25.get(page_id, 0.0))
            rerank_scores[page_id] = 0.85 * base_value + 0.15 * anchor_scores.get(page_id, 0.0)
        ranked = sorted(rerank_scores.items(), key=lambda item: (-item[1], item[0]))[:100]
        rows = [
            {
                "chunk_id": page_id,
                "doc_id": page_id,
                "text": texts[page_id],
                "score": round(float(score), 8),
                "rank": rank,
                "scores": {
                    "base_page": round(float(page_base.get(page_id, page_bm25.get(page_id, 0.0))), 8),
                    "typed_anchor": round(float(anchor_scores.get(page_id, 0.0)), 8),
                    "final": round(float(score), 8),
                },
            }
            for rank, (page_id, score) in enumerate(ranked, 1)
        ]
        runs.setdefault("typed-union20", {})[qid] = rows

        # A no-expansion control: same candidate pages, typed field only
        # adjusts the exact-page ordering.
        control_candidates = set(page_base)
        control_scores = {
            page_id: 0.85 * page_base.get(page_id, 0.0) + 0.15 * anchor_scores.get(page_id, 0.0)
            for page_id in control_candidates
        }
        control_ranked = sorted(control_scores.items(), key=lambda item: (-item[1], item[0]))[:100]
        runs.setdefault("typed-rerank-only", {})[qid] = [
            {"chunk_id": page_id, "doc_id": page_id, "text": texts[page_id], "score": round(float(score), 8), "rank": rank}
            for rank, (page_id, score) in enumerate(control_ranked, 1)
        ]

        # Pure typed field is included as a diagnostic, not a proposed arm.
        typed_ranked = sorted(anchor_scores.items(), key=lambda item: (-item[1], item[0]))[:100]
        runs.setdefault("typed-only", {})[qid] = [
            {"chunk_id": page_id, "doc_id": page_id, "text": texts[page_id], "score": round(float(score), 8), "rank": rank}
            for rank, (page_id, score) in enumerate(typed_ranked, 1)
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods: dict[str, dict[str, Any]] = {}
    for name, run in runs.items():
        methods[name] = {
            "retrieval_metrics": _derived_metrics(run, qids, qrels),
            "typed_query_with_at_least_one_anchor": sum(value > 0 for value in anchor_counts.values()) / len(qids),
        }
        _write_run(args.output_dir / f"{name}.jsonl", run, qids, queries={question.qid: question.query for question in questions})
    references = {
        "cached_page_bm25": _derived_metrics(baseline_run, qids, qrels),
        "current_hierarchical": _derived_metrics(hierarchical_run, qids, qrels),
        "c014_base": _derived_metrics(base_run, qids, qrels),
    }
    for item in methods.values():
        item["comparison_to_c014"] = _paired_comparison(references["c014_base"], item["retrieval_metrics"])
        item["comparison_to_hierarchical"] = _paired_comparison(references["current_hierarchical"], item["retrieval_metrics"])
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "references": references,
        "methods": methods,
        "typed_field": {
            "unit_tokens": sorted(UNIT_TOKENS),
            "variable_tokens": sorted(VARIABLE_TOKENS),
            "fraction_queries_with_anchor": sum(value > 0 for value in anchor_counts.values()) / len(qids),
        },
        "sources": {"parsed_run": str(args.parsed_run), "base_run": str(args.base_run), "page_vector_dir": str(args.page_vector_dir)},
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "Typed field is a deterministic dictionary-like exact signal, not a learned modality router.",
            "typed-union20 can add at most 20 typed-field candidates to the existing c014 top-100 pool.",
            "No qrels or qrel content-modality labels are used during scoring.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Physics typed-anchor baseline",
        "",
        "| Method | nDCG@10 | Page recall@10 | Page hit@10 | File recall@3 | Δ vs c014 pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in methods.items():
        metric = item["retrieval_metrics"]
        lines.append(
            f"| {name} | {metric['ndcg@10']:.2f} | {metric['page_recall@10']:.2%} | {metric['page_hit@10']:.2%} | "
            f"{metric['file_metrics_by_k']['3']['file_recall']:.2%} | {item['comparison_to_c014']['page_recall_delta_pp']:+.2f} |"
        )
    lines += [
        "",
        f"Queries with at least one typed anchor: **{report['typed_field']['fraction_queries_with_anchor']:.2%}**.",
        "The typed-only row is diagnostic; the candidate-expansion row is the intended light-retrieval test.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "methods": {name: round(item["retrieval_metrics"]["page_recall@10"] * 100, 2) for name, item in methods.items()},
        "c014_base_page_recall@10": round(references["c014_base"]["page_recall@10"] * 100, 2),
        "fraction_queries_with_anchor": report["typed_field"]["fraction_queries_with_anchor"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
