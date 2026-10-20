"""Offline screen of light adaptive-retrieval methods on the 220-query benchmark.

The screen keeps the benchmark protocol global and page-level.  BM25 is built
only from cached ``text_source=pdf_inspector`` pages.  A cached text-embedding
index is optionally used as a semantic candidate/ranking signal; no embedding
or API call is made here.  Qrels are loaded only after rankings and K choices
have been produced.

This is a research screen, not a final held-out claim.  It compares fixed
Kp=12 with score-only largest-gap policies at approximately the same budget.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.raw_hierarchical import RawPage, read_jsonl  # noqa: E402
from research.experiments.run_benchmark_adaptive_k import adaptive_k  # noqa: E402
from src.chunking_embedding.lexical import analyze  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DEFAULT_DATASET = ROOT / "data/raw/BENCHMARK"
DEFAULT_EMBEDDING = DEFAULT_CACHE / "embedding_adaptive_openrouter_te3s_fulltext/embeddings"
DEFAULT_OUTPUT = DEFAULT_CACHE / "adaptive_method_screen"

STOPWORDS = set(
    "the a an and or of to in on for with from by is are was were be been being "
    "what when where how why who whom which this that these those do does did "
    "should can could would will may might as at it its into about all any each "
    "per their there while during explain list number page pages".split()
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _query_variants(query: str) -> list[str]:
    variants = [query]
    content = " ".join(token for token in analyze(query) if token not in STOPWORDS)
    if content and content != query:
        variants.append(content)
    for clause in re.split(r"[?;,:()]+|\b(?:and|or|while|during)\b", query, flags=re.I):
        clause = clause.strip()
        if len(analyze(clause)) >= 2 and clause not in variants:
            variants.append(clause)
    return variants


def _bm25_ranking(index: BM25Index, query: str, *, top_k: int = 100) -> list[tuple[str, float]]:
    return [
        (index.chunk_ids[position], float(score))
        for position, score in index.search(query, top_k)
    ]


def _rrf_ranking(index: BM25Index, query: str, *, constant: int = 10) -> list[tuple[str, float]]:
    fused: dict[str, float] = defaultdict(float)
    for variant in _query_variants(query):
        for rank, (page_id, _score) in enumerate(_bm25_ranking(index, variant), start=1):
            fused[page_id] += 1.0 / (constant + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def _normalise(scores: Mapping[str, float]) -> dict[str, float]:
    maximum = max((float(value) for value in scores.values()), default=0.0)
    return {key: float(value) / maximum if maximum > 0 else 0.0 for key, value in scores.items()}


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, set[str]],
) -> dict[str, float]:
    returned_recall: list[float] = []
    recall10: list[float] = []
    recall20: list[float] = []
    page_counts: list[int] = []
    for row in rows:
        gold = qrels.get(str(row["query_id"]), set())
        returned = [str(page_id) for page_id in row["returned_pages"]]
        denominator = len(gold) or 1
        returned_recall.append(len(set(returned) & gold) / denominator)
        recall10.append(len(set(returned[:10]) & gold) / denominator)
        recall20.append(len(set(returned[:20]) & gold) / denominator)
        page_counts.append(len(returned))
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "queries": len(rows),
        "page_recall_at_returned_k": mean(returned_recall),
        "page_recall@10": mean(recall10),
        "page_recall@20": mean(recall20),
        "mean_k": mean(page_counts),
    }


def _rows_for_policy(
    queries: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    qrels: Mapping[str, set[str]],
    *,
    policy: str,
    fixed_k: int | None = None,
    retrieve_more: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query["query_id"])
        ranking = list(rankings[query_id])
        if fixed_k is not None:
            k = min(int(fixed_k), len(ranking))
            adapter = None
        else:
            adapter = adaptive_k(
                [score for _page_id, score in ranking],
                ignore_extreme=0,
                ignore_extreme_tail=0.1,
                retrieve_more=int(retrieve_more or 0),
            )
            k = int(adapter["k"])
        rows.append(
            {
                "query_id": query_id,
                "query": str(query["query"]),
                "policy": policy,
                "k": k,
                "returned_pages": [page_id for page_id, _score in ranking[:k]],
                "scores": [round(float(score), 8) for _page_id, score in ranking[:k]],
                "adaptive_k": adapter,
            }
        )
    return rows, _evaluate(rows, qrels)


def _rows_for_relative_threshold(
    queries: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    qrels: Mapping[str, set[str]],
    *,
    policy: str,
    threshold: float,
    minimum_k: int = 4,
    maximum_k: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query["query_id"])
        ranking = list(rankings[query_id])
        top_score = float(ranking[0][1]) if ranking else 0.0
        k = sum(float(score) >= threshold * top_score for _page_id, score in ranking[:maximum_k])
        k = max(minimum_k, min(maximum_k, int(k)))
        rows.append({"query_id": query_id, "query": str(query["query"]), "policy": policy, "k": k, "returned_pages": [page_id for page_id, _score in ranking[:k]], "scores": [round(float(score), 8) for _page_id, score in ranking[:k]], "adaptive_k": {"rule": "relative_score_threshold", "threshold": threshold}})
    return rows, _evaluate(rows, qrels)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixed-k", type=int, default=12)
    args = parser.parse_args(argv)

    pages_all = [RawPage(**row) for row in read_jsonl(args.cache_dir / "parsing/pages.jsonl")]
    pages = [page for page in pages_all if page.text_source == "pdf_inspector"]
    page_by_id = {page.page_id: page for page in pages}
    queries = read_jsonl(args.dataset_root / "queries.jsonl")
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(args.cache_dir / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))

    bm25 = BM25Index(analyzer_name="auto").build(
        {"chunk_id": page.page_id, "doc_id": page.doc_id, "text": page.text}
        for page in pages
    )
    page_ids = [page.page_id for page in pages]
    page_position = {page_id: position for position, page_id in enumerate(page_ids)}

    embedding_dir = args.embedding_dir.resolve()
    page_vectors = np.load(embedding_dir / "page_embeddings.npy", mmap_mode="r")
    query_vectors = np.load(embedding_dir / "query_embeddings.npy", mmap_mode="r")
    all_page_ids = [page.page_id for page in pages_all]
    if page_vectors.shape[0] != len(all_page_ids) or query_vectors.shape[0] != len(queries):
        raise RuntimeError("Embedding/page/query count mismatch")
    inspector_positions = np.asarray(
        [all_page_ids.index(page_id) for page_id in page_ids], dtype=np.int64
    )
    dense_pages = page_vectors[inspector_positions]
    dense_norms = np.linalg.norm(dense_pages, axis=1)
    query_norms = np.linalg.norm(query_vectors, axis=1)

    bm25_rankings: dict[str, list[tuple[str, float]]] = {}
    rrf_rankings: dict[str, list[tuple[str, float]]] = {}
    multi_max_rankings: dict[str, list[tuple[str, float]]] = {}
    multi_sum_rankings: dict[str, list[tuple[str, float]]] = {}
    dense_rankings: dict[str, list[tuple[str, float]]] = {}
    hybrid_rankings: dict[str, list[tuple[str, float]]] = {}
    hybrid_variants: dict[str, dict[str, list[tuple[str, float]]]] = {
        f"hybrid_bm25_{alpha:.2f}_dense_{1.0-alpha:.2f}": {}
        for alpha in (0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00)
    }
    dynamic_rankings: dict[str, dict[str, list[tuple[str, float]]]] = {
        "hybrid_dynamic_anchor": {},
        "hybrid_dynamic_margin": {},
        "hybrid_dynamic_entropy": {},
    }
    source_filtered_rankings: dict[str, dict[str, list[tuple[str, float]]]] = {
        "hybrid_source_filtered_bm25_0.20_dense_0.80": {},
    }
    file_aggregate_rankings: dict[str, dict[str, list[tuple[str, float]]]] = {
        f"hybrid_a{alpha:.2f}_file_sum2_beta{beta:.2f}": {}
        for alpha in (0.10, 0.20)
        for beta in (0.10, 0.20, 0.30, 0.50)
    }
    rrf_dense_rankings: dict[str, list[tuple[str, float]]] = {}
    query_index = {str(query["query_id"]): index for index, query in enumerate(queries)}
    for query in queries:
        query_id = str(query["query_id"])
        query_text = str(query["query"])
        bm25_rankings[query_id] = _bm25_ranking(bm25, query_text)
        rrf_rankings[query_id] = _rrf_ranking(bm25, query_text)
        variant_scores: dict[str, list[float]] = {}
        for variant in _query_variants(query_text):
            variant_scores[variant] = [0.0] * len(page_ids)
            for position, score in bm25.search(variant, 100):
                variant_scores[variant][position] = float(score)
        max_scores = {page_id: max((values[index] for values in variant_scores.values()), default=0.0) for index, page_id in enumerate(page_ids)}
        sum_scores = {page_id: sum(values[index] / max(max(values), 1e-9) for values in variant_scores.values()) for index, page_id in enumerate(page_ids)}
        multi_max_rankings[query_id] = sorted(max_scores.items(), key=lambda item: (-item[1], item[0]))[:100]
        multi_sum_rankings[query_id] = sorted(sum_scores.items(), key=lambda item: (-item[1], item[0]))[:100]
        dense_scores = (
            dense_pages @ query_vectors[query_index[query_id]]
        ) / (dense_norms * query_norms[query_index[query_id]] + 1e-9)
        dense_scores_by_id = {
            page_id: float(score) for page_id, score in zip(page_ids, dense_scores)
        }
        dense_rankings[query_id] = sorted(
            dense_scores_by_id.items(),
            key=lambda item: (-item[1], item[0]),
        )[:100]
        bm_scores = dict(bm25_rankings[query_id])
        bm_norm = _normalise(bm_scores)
        dense_norm = _normalise(dense_scores_by_id)
        # The screen below keeps several predeclared alphas so we can separate
        # score calibration from the adaptive cutoff question.
        for alpha in (0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00):
            key = f"hybrid_bm25_{alpha:.2f}_dense_{1.0-alpha:.2f}"
            scores = {page_id: alpha * bm_norm.get(page_id, 0.0) + (1.0 - alpha) * dense_norm.get(page_id, 0.0) for page_id in page_ids}
            hybrid_variants[key][query_id] = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:100]
        bm_top = np.asarray([score for _pid, score in bm25_rankings[query_id][:20]], dtype=float)
        de_top = np.asarray([score for _pid, score in dense_rankings[query_id][:20]], dtype=float)
        bm_gap = float(bm_top[0] - bm_top[1]) if len(bm_top) > 1 else 0.0
        bm_concentration = float((bm_top[0] - np.mean(bm_top[1:])) / (np.std(bm_top) + 1e-9)) if len(bm_top) > 2 else 0.0
        de_concentration = float((de_top[0] - np.mean(de_top[1:])) / (np.std(de_top) + 1e-9)) if len(de_top) > 2 else 0.0
        anchor = bool(re.search(r"\d|[%=+\-/]|\b(?:formula|equation|number|quantity|mass|energy|pressure)\b", query_text, flags=re.I))
        dynamic_alphas = {
            "hybrid_dynamic_anchor": 0.30 if anchor else 0.10,
            "hybrid_dynamic_margin": 0.30 if bm_gap > float(np.median(bm_top[1:])) else 0.10,
            "hybrid_dynamic_entropy": 0.30 if bm_concentration > de_concentration else 0.10,
        }
        for dynamic_name, alpha in dynamic_alphas.items():
            scores = {page_id: alpha * bm_norm.get(page_id, 0.0) + (1.0 - alpha) * dense_norm.get(page_id, 0.0) for page_id in page_ids}
            dynamic_rankings[dynamic_name][query_id] = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:100]
        hybrid_scores = {
            page_id: 0.10 * bm_norm.get(page_id, 0.0) + 0.90 * dense_norm.get(page_id, 0.0)
            for page_id in page_ids
        }
        hybrid_rankings[query_id] = sorted(
            hybrid_scores.items(), key=lambda item: (-item[1], item[0])
        )[:100]
        for file_alpha in (0.10, 0.20):
            base_key = f"hybrid_bm25_{file_alpha:.2f}_dense_{1.0-file_alpha:.2f}"
            candidate = hybrid_variants[base_key][query_id]
            by_file: dict[str, list[float]] = defaultdict(list)
            for page_id, score in candidate:
                by_file[page_id.split("#page=", 1)[0]].append(float(score))
            file_scores = {file_id: sum(sorted(scores, reverse=True)[:2]) for file_id, scores in by_file.items()}
            max_file_score = max(file_scores.values(), default=1.0)
            file_scores = {file_id: score / max_file_score for file_id, score in file_scores.items()}
            page_scores = dict(candidate)
            max_page_score = max(page_scores.values(), default=1.0)
            page_scores = {page_id: score / max_page_score for page_id, score in page_scores.items()}
            for beta in (0.10, 0.20, 0.30, 0.50):
                key = f"hybrid_a{file_alpha:.2f}_file_sum2_beta{beta:.2f}"
                scored = {
                    page_id: (1.0 - beta) * page_scores[page_id] + beta * file_scores.get(page_id.split("#page=", 1)[0], 0.0)
                    for page_id in page_scores
                }
                file_aggregate_rankings[key][query_id] = sorted(scored.items(), key=lambda item: (-item[1], item[0]))[:100]
        source_prefix = query_id.split("::", 1)[0]
        source_filtered_rankings["hybrid_source_filtered_bm25_0.20_dense_0.80"][query_id] = [
            item for item in sorted(hybrid_scores.items(), key=lambda item: (-item[1], item[0]))
            if item[0].split("::", 1)[0] == source_prefix
        ][:100]
        bm_rank = {page_id: rank for rank, (page_id, _score) in enumerate(bm25_rankings[query_id], start=1)}
        dense_rank = {page_id: rank for rank, (page_id, _score) in enumerate(dense_rankings[query_id], start=1)}
        rrf_scores = {page_id: 1.0 / (10 + bm_rank.get(page_id, 1000)) + 1.0 / (10 + dense_rank.get(page_id, 1000)) for page_id in page_ids}
        rrf_dense_rankings[query_id] = sorted(rrf_scores.items(), key=lambda item: (-item[1], item[0]))[:100]

    ranking_specs = {
        "bm25": bm25_rankings,
        "bm25_rrf10": rrf_rankings,
        "bm25_multi_max": multi_max_rankings,
        "bm25_multi_sum": multi_sum_rankings,
        "dense_pdf_inspector_pages": dense_rankings,
        "hybrid_bm25_0.10_dense_0.90": hybrid_rankings,
        "dense_bm25_rrf10": rrf_dense_rankings,
    }
    ranking_specs.update(hybrid_variants)
    ranking_specs.update(dynamic_rankings)
    ranking_specs.update(source_filtered_rankings)
    ranking_specs.update(file_aggregate_rankings)
    policies: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for ranking_name, rankings in ranking_specs.items():
        fixed_rows, fixed_metrics = _rows_for_policy(
            queries, rankings, qrels, policy=f"{ranking_name}_fixed_k{args.fixed_k}", fixed_k=args.fixed_k
        )
        policies[f"{ranking_name}_fixed_k{args.fixed_k}"] = fixed_metrics
        all_rows.extend(fixed_rows)
        if ranking_name in {"dense_pdf_inspector_pages", "hybrid_bm25_0.10_dense_0.90", "hybrid_bm25_0.20_dense_0.80", "hybrid_source_filtered_bm25_0.20_dense_0.80"}:
            gap_offsets = (11, 12, 13, 14, 15, 16, 17, 18) if ranking_name == "hybrid_source_filtered_bm25_0.20_dense_0.80" else (8, 9, 10)
            for retrieve_more in gap_offsets:
                adaptive_rows, adaptive_metrics = _rows_for_policy(
                    queries,
                    rankings,
                    qrels,
                    policy=f"{ranking_name}_largest_gap_plus{retrieve_more}",
                    retrieve_more=retrieve_more,
                )
                key = f"{ranking_name}_largest_gap_plus{retrieve_more}"
                policies[key] = adaptive_metrics
                all_rows.extend(adaptive_rows)
        if ranking_name in {"bm25", "hybrid_bm25_0.20_dense_0.80", "hybrid_a0.10_file_sum2_beta0.30", "hybrid_a0.20_file_sum2_beta0.30", "hybrid_source_filtered_bm25_0.20_dense_0.80"}:
            for query_budget in ("coverage_aware",):
                rows: list[dict[str, Any]] = []
                for query in queries:
                    query_id = str(query["query_id"])
                    ranking = list(rankings[query_id])
                    # PDF-inspector has no indexed MPDocVQA pages in this
                    # cache. Return no impossible candidates and spend the
                    # released budget on the two covered sources.
                    k = 0 if query_id.startswith("mpdocvqa::") else 17
                    rows.append({"query_id": query_id, "query": str(query["query"]), "policy": f"{ranking_name}_{query_budget}", "k": k, "returned_pages": [page_id for page_id, _score in ranking[:k]], "scores": [round(float(score), 8) for _page_id, score in ranking[:k]], "adaptive_k": {"rule": "index_coverage_aware", "k": k}})
                key = f"{ranking_name}_{query_budget}"
                policies[key] = _evaluate(rows, qrels)
                all_rows.extend(rows)
        if ranking_name in {"hybrid_bm25_0.10_dense_0.90", "hybrid_bm25_0.20_dense_0.80"}:
            for threshold in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90):
                threshold_rows, threshold_metrics = _rows_for_relative_threshold(
                    queries, rankings, qrels,
                    policy=f"{ranking_name}_relative_threshold_{threshold:.2f}",
                    threshold=threshold,
                )
                key = f"{ranking_name}_relative_threshold_{threshold:.2f}"
                policies[key] = threshold_metrics
                all_rows.extend(threshold_rows)

    report = {
        "contract_version": "benchmark-adaptive-method-screen-v1",
        "cache_dir": str(args.cache_dir.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "embedding_dir": str(embedding_dir),
        "queries": len(queries),
        "pages": len(pages_all),
        "indexed_pdf_inspector_pages": len(pages),
        "ranking_protocol": {
            "scope": "global page retrieval; no hierarchical file stage",
            "bm25_text": "cached text_source=pdf_inspector only",
            "hybrid": "normalized BM25 0.10 + cached dense cosine 0.90",
            "adaptive": "largest descending score gap; tail=0.1",
            "qrels_used_for_ranking": False,
        },
        "policies": policies,
        "notes": [
            "The cached dense vectors are reused offline; no new embedding or API call was made.",
            "This is an exploratory screen. Fixed K=12 is the current trivial budget control.",
            "coverage_aware uses only query source/index coverage metadata: cap MPDocVQA at K=0 because the PDF-inspector index contains zero MPDocVQA pages, and use K=17 for covered sources; it does not inspect qrels.",
            "source_filtered restricts candidates to the query source prefix; this is a corpus-routing ablation, not pure global Adaptive-k.",
        ],
    }
    output_dir = args.output_dir.resolve()
    _write_json(output_dir / "report.json", report)
    for name, rankings in ranking_specs.items():
        fixed_rows, _ = _rows_for_policy(
            queries, rankings, qrels, policy=f"{name}_fixed_k{args.fixed_k}", fixed_k=args.fixed_k
        )
        _write_jsonl(output_dir / f"{name}_fixed_k{args.fixed_k}.jsonl", fixed_rows)
    _write_jsonl(output_dir / "all_policy_rows.jsonl", all_rows)

    lines = [
        "# Adaptive-method screen",
        "",
        f"- Benchmark: {len(queries)} queries, {len(pages_all)} pages; PDF-inspector index: {len(pages)} pages.",
        "- Global page retrieval only; no hierarchical file stage.",
        f"- Fixed trivial control: Kp={args.fixed_k}.",
        "- Dense vectors are cached and reused offline; no new API call.",
        "- The coverage-aware row uses source/index coverage metadata only; it is reported separately from pure score-gap Adaptive-k.",
        "",
        "| Policy | Recall at returned K | Recall@10 | Recall@20 | Mean Kp |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in policies.items():
        lines.append(
            f"| {name} | {metrics['page_recall_at_returned_k']:.2%} | {metrics['page_recall@10']:.2%} | {metrics['page_recall@20']:.2%} | {metrics['mean_k']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The fixed K=12 rows are the same-budget controls. Adaptive rows only change the score-based page cutoff; they do not rerank the page order.",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
