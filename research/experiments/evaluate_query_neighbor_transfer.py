"""Quick OOF demo of query-neighbor transfer for adaptive page retrieval.

For each held-out query, retrieve similar training-fold queries using cached
Qwen query embeddings and fold-fitted TF-IDF keyword vectors. Transfer (a) a
top-k budget from training queries' QCAR-style k* labels and (b) a weak page
prior from those queries' known relevant pages. Test-fold qrels are used only
for evaluation.

This is an exploratory diagnostic on the merged 220-query benchmark, not a
production-ready policy. It makes no network calls and reuses cached rankings
and query embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QCAR_DIR = ROOT / "experiments/adaptive-k-retrieval"
if str(QCAR_DIR) not in sys.path:
    sys.path.insert(0, str(QCAR_DIR))

from run_canonical_adaptive_methods import load_rows  # noqa: E402
from run_qcar_faithful import _kstar, _load_query_vectors  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data/work/query_neighbor_transfer_20261014"
MAX_K = 100
FOLDS = 5
RANDOM_STATE = 2026
TOP_NEIGHBORS = 5


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> int:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights) / max(float(np.sum(ordered_weights)), 1e-12)
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(order) - 1)
    return int(np.clip(math.ceil(float(ordered_values[index])), 1, MAX_K))


def _rank_positions(similarities: np.ndarray) -> np.ndarray:
    """Return zero-based rank positions, preserving deterministic tie order."""
    order = np.argsort(-similarities, kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order))
    return ranks


def _recall(ranking: Sequence[str], gold: set[str], k: int) -> float:
    return len(set(ranking[:k]) & gold) / max(len(gold), 1)


def _evaluate(
    qids: Sequence[str],
    rows_by_qid: dict[str, dict[str, Any]],
    choices: Sequence[int],
    rankings: dict[str, Sequence[str]] | None = None,
) -> dict[str, float]:
    recalls = []
    for qid, k in zip(qids, choices):
        row = rows_by_qid[qid]
        ranking = rankings[qid] if rankings is not None else row["ranked"]
        recalls.append(_recall(ranking, set(row["gold"]), int(k)))
    return {
        "macro_recall_at_returned_k": float(np.mean(recalls)),
        "mean_k": float(np.mean(choices)),
        "median_k": float(np.median(choices)),
        "p90_k": float(np.percentile(choices, 90)),
    }


def _build_neighbor_rankings(
    test_indices: np.ndarray,
    train_indices: np.ndarray,
    rows: Sequence[dict[str, Any]],
    combined_rank: np.ndarray,
    top_neighbors: int,
    page_prior_weight: float,
) -> tuple[dict[int, list[int]], dict[int, list[str]]]:
    """Transfer train-fold relevant pages into a rank-normalized hybrid run."""
    neighbors_by_test: dict[int, list[int]] = {}
    rankings_by_test: dict[int, list[str]] = {}
    for test_pos, test_index in enumerate(test_indices):
        neighbor_order = np.argsort(combined_rank[test_pos], kind="stable")[:top_neighbors]
        neighbor_indices = train_indices[neighbor_order]
        ranks = combined_rank[test_pos, neighbor_order]
        # Reciprocal-rank weights are robust to uncalibrated dense/lexical
        # similarity scales and keep each neighbor's vote bounded.
        weights = 1.0 / (60.0 + ranks.astype(float))
        weights /= max(float(weights.sum()), 1e-12)
        neighbors_by_test[int(test_index)] = [int(i) for i in neighbor_indices]

        row = rows[int(test_index)]
        base_ids = list(row["ranked"])
        base_scores = np.asarray(row["scores"], dtype=np.float64)
        if len(base_scores):
            lo, hi = float(np.min(base_scores)), float(np.max(base_scores))
            base_norm = (base_scores - lo) / max(hi - lo, 1e-12)
        else:
            base_norm = np.zeros(0, dtype=np.float64)
        score_by_page = {page: float(score) for page, score in zip(base_ids, base_norm)}

        page_votes: dict[str, float] = {}
        for neighbor_index, weight in zip(neighbor_indices, weights):
            neighbor_gold = set(rows[int(neighbor_index)]["gold"])
            if not neighbor_gold:
                continue
            per_page_vote = float(weight) / len(neighbor_gold)
            for page in neighbor_gold:
                page_votes[page] = page_votes.get(page, 0.0) + per_page_vote
        max_vote = max(page_votes.values(), default=0.0)
        if max_vote > 0:
            candidate_pages = set(score_by_page) | set(page_votes)
            score_by_page = {
                page: (1.0 - page_prior_weight) * score_by_page.get(page, 0.0)
                + page_prior_weight * page_votes.get(page, 0.0) / max_vote
                for page in candidate_pages
            }
        rankings_by_test[int(test_index)] = sorted(
            score_by_page,
            key=lambda page: (-score_by_page[page], page),
        )
    return neighbors_by_test, rankings_by_test


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--neighbors", type=int, default=TOP_NEIGHBORS)
    parser.add_argument("--page-prior-weight", type=float, default=0.15)
    parser.add_argument("--kstar-depth", type=int, default=MAX_K)
    args = parser.parse_args(argv)
    if not 0.0 <= args.page_prior_weight <= 1.0:
        raise ValueError("--page-prior-weight must be in [0, 1]")

    rows, _ = load_rows("hybrid", 200)
    qids = [str(row["qid"]) for row in rows]
    queries = [str(row["query"]) for row in rows]
    sources = np.asarray([str(row["source"]) for row in rows])
    rows_by_qid = {qid: row for qid, row in zip(qids, rows)}
    qwen = _load_query_vectors("qwen", qids)
    kstars = np.asarray(
        [
            _kstar(row["ranked"], set(row["gold"]), args.kstar_depth, 0.001, "max")
            for row in rows
        ],
        dtype=np.float64,
    )

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    methods: dict[str, dict[str, Any]] = {
        name: {"choices": np.zeros(len(rows), dtype=np.int64), "rankings": {}}
        for name in (
            "semantic_knn_k",
            "keyword_knn_k",
            "semantic_keyword_knn_k",
            "semantic_keyword_knn_k_plus_neighbor_pages",
            "same_source_semantic_keyword_knn_k",
            "same_source_semantic_keyword_knn_k_plus_neighbor_pages",
        )
    }
    fold_details: list[dict[str, Any]] = []

    for fold, (train_indices, test_indices) in enumerate(splitter.split(np.zeros(len(rows)), sources), start=1):
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=20000,
            strip_accents="unicode",
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w][\w-]+\b",
        )
        train_lexical = vectorizer.fit_transform([queries[i] for i in train_indices])
        test_lexical = vectorizer.transform([queries[i] for i in test_indices])
        lexical_similarity = (test_lexical @ train_lexical.T).toarray()
        semantic_similarity = qwen[test_indices] @ qwen[train_indices].T

        semantic_ranks = np.vstack([_rank_positions(row) for row in semantic_similarity])
        lexical_ranks = np.vstack([_rank_positions(row) for row in lexical_similarity])
        combined_ranks = semantic_ranks + lexical_ranks
        same_source_combined_ranks = combined_ranks.copy()
        same_source_combined_ranks[
            sources[test_indices, None] != sources[train_indices][None, :]
        ] = len(train_indices) * 4
        # Smaller combined rank means closer overall; rank reciprocal fusion
        # combines semantic neighborhood and exact/near-keyword overlap.
        for test_pos, test_index in enumerate(test_indices):
            for method_name, rank_matrix in (
                ("semantic_knn_k", semantic_ranks),
                ("keyword_knn_k", lexical_ranks),
                ("semantic_keyword_knn_k", combined_ranks),
                ("same_source_semantic_keyword_knn_k", same_source_combined_ranks),
            ):
                neighbor_order = np.argsort(rank_matrix[test_pos], kind="stable")[: args.neighbors]
                neighbor_k = kstars[train_indices[neighbor_order]]
                # The 90th weighted neighbor quantile is a simple conservative
                # transfer rule; no test-fold labels tune this choice.
                weights = 1.0 / (60.0 + rank_matrix[test_pos, neighbor_order].astype(float))
                weights /= max(float(weights.sum()), 1e-12)
                methods[method_name]["choices"][test_index] = _weighted_quantile(neighbor_k, weights, 0.90)

        neighbors_by_test, transferred_rankings = _build_neighbor_rankings(
            test_indices,
            train_indices,
            rows,
            combined_ranks,
            args.neighbors,
            args.page_prior_weight,
        )
        same_source_neighbors_by_test, same_source_transferred_rankings = _build_neighbor_rankings(
            test_indices,
            train_indices,
            rows,
            same_source_combined_ranks,
            args.neighbors,
            args.page_prior_weight,
        )
        for test_index in test_indices:
            neighbor_indices = neighbors_by_test[int(test_index)]
            # Recompute the same reciprocal-rank weights for k* transfer.
            matching_pos = int(np.where(test_indices == test_index)[0][0])
            neighbor_ranks = combined_ranks[matching_pos]
            neighbor_order = np.argsort(neighbor_ranks, kind="stable")[: args.neighbors]
            neighbor_k = kstars[train_indices[neighbor_order]]
            neighbor_weights = 1.0 / (60.0 + neighbor_ranks[neighbor_order].astype(float))
            neighbor_weights /= max(float(neighbor_weights.sum()), 1e-12)
            methods["semantic_keyword_knn_k_plus_neighbor_pages"]["choices"][test_index] = _weighted_quantile(
                neighbor_k, neighbor_weights, 0.90
            )
            methods["semantic_keyword_knn_k_plus_neighbor_pages"]["rankings"][qids[test_index]] = transferred_rankings[int(test_index)]

            same_source_neighbor_ranks = same_source_combined_ranks[matching_pos]
            same_source_order = np.argsort(same_source_neighbor_ranks, kind="stable")[: args.neighbors]
            same_source_k = kstars[train_indices[same_source_order]]
            same_source_weights = 1.0 / (60.0 + same_source_neighbor_ranks[same_source_order].astype(float))
            same_source_weights /= max(float(same_source_weights.sum()), 1e-12)
            methods["same_source_semantic_keyword_knn_k"]["choices"][test_index] = _weighted_quantile(
                same_source_k, same_source_weights, 0.90
            )
            methods["same_source_semantic_keyword_knn_k_plus_neighbor_pages"]["choices"][test_index] = _weighted_quantile(
                same_source_k, same_source_weights, 0.90
            )
            methods["same_source_semantic_keyword_knn_k_plus_neighbor_pages"]["rankings"][qids[test_index]] = same_source_transferred_rankings[int(test_index)]

        fold_details.append(
            {
                "fold": fold,
                "train_queries": int(len(train_indices)),
                "test_queries": int(len(test_indices)),
                "train_source_counts": {
                    source: int(np.sum(sources[train_indices] == source))
                    for source in sorted(set(sources.tolist()))
                },
                "test_source_counts": {
                    source: int(np.sum(sources[test_indices] == source))
                    for source in sorted(set(sources.tolist()))
                },
            }
        )

    fixed_k = int(round(float(np.mean(methods["semantic_keyword_knn_k"]["choices"]))))
    fixed_choices = np.full(len(rows), fixed_k, dtype=np.int64)
    result_rows: list[dict[str, Any]] = [
        {"method": f"fixed_k{fixed_k}", **_evaluate(qids, rows_by_qid, fixed_choices)}
    ]
    result_rows.append(
        {
            "method": "fixed_k88",
            **_evaluate(qids, rows_by_qid, np.full(len(rows), 88, dtype=np.int64)),
        }
    )
    for name, result in methods.items():
        adaptive_metrics = _evaluate(
            qids,
            rows_by_qid,
            result["choices"],
            result["rankings"] if result["rankings"] else None,
        )
        matched_k = int(round(adaptive_metrics["mean_k"]))
        matched_fixed = _evaluate(
            qids,
            rows_by_qid,
            np.full(len(rows), matched_k, dtype=np.int64),
        )
        result_rows.append(
            {
                "method": name,
                **adaptive_metrics,
                "matched_fixed_k": matched_k,
                "matched_fixed_recall": matched_fixed["macro_recall_at_returned_k"],
                "delta_vs_matched_fixed": adaptive_metrics["macro_recall_at_returned_k"]
                - matched_fixed["macro_recall_at_returned_k"],
            }
        )

    prior_rankings = methods["semantic_keyword_knn_k_plus_neighbor_pages"]["rankings"]
    result_rows.append(
        {
            "method": "semantic_keyword_neighbor_pages_fixed_k88",
            **_evaluate(
                qids,
                rows_by_qid,
                np.full(len(rows), 88, dtype=np.int64),
                prior_rankings,
            ),
        }
    )
    qcar_path = ROOT / "data/work/qcar_faithful_20261007_source_stratified/report.json"
    qcar_report = json.loads(qcar_path.read_text(encoding="utf-8"))
    qcar_method_name = "qcar_qwen_umap_3c_kmax100_semantic_mean2std"
    qcar_method = next(row for row in qcar_report["metrics"] if row["name"] == qcar_method_name)
    result_rows.append(
        {
            "method": "qcar_3cluster_oof_reference",
            "macro_recall_at_returned_k": float(qcar_method["recall_at_adaptive_k"]),
            "mean_k": float(qcar_method["mean_k"]),
            "median_k": float(qcar_method["median_k"]),
            "p90_k": float(qcar_method["p90_k"]),
            "artifact": str(qcar_path.relative_to(ROOT)),
        }
    )

    output: dict[str, Any] = {
        "protocol": {
            "benchmark": "merged Physics 80 + Industrial 75 + OHR-Bench 65",
            "queries": len(rows),
            "split": "5-fold stratified by source, random_state=2026",
            "ranking": "cached PP-OCRv5 + Qwen3-Embedding-8B hybrid top-200",
            "label": "QCAR-style k*: last NDCG gain >= 0.001 through rank 100; no-hit prefix => 100",
            "query_similarity": "cached normalized Qwen embeddings + fold-fitted word unigram/bigram TF-IDF",
            "neighbor_selection": "RRF-style sum of semantic and lexical ranks; top-N train queries only",
            "budget_transfer": "weighted 90th percentile of neighboring train-fold k* labels",
            "page_transfer": f"weighted relevant-page votes from top-{args.neighbors} train-fold queries; base/page-prior weights={1.0-args.page_prior_weight:.2f}/{args.page_prior_weight:.2f}",
            "test_qrels_used_for": "evaluation only",
            "network_calls": 0,
            "warning": "Exploratory benchmark transfer; no nested tuning, historical query log or user feedback is available.",
        },
        "parameters": {
            "neighbors": args.neighbors,
            "page_prior_weight": args.page_prior_weight,
            "kstar_depth": args.kstar_depth,
            "mean_neighbor_budget_k": fixed_k,
        },
        "methods": result_rows,
        "folds": fold_details,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Query-neighbor transfer: quick OOF demo",
        "",
        "All query-neighbor and page-prior signals come from training-fold queries. Test qrels are evaluation-only.",
        "",
        "| Method | Macro recall at returned K | Mean K | Median K | P90 K | vs matched fixed K |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in result_rows:
        matched = (
            f"K={row['matched_fixed_k']}: {row['matched_fixed_recall']:.2%}; "
            f"delta {row['delta_vs_matched_fixed'] * 100:+.2f} pp"
            if "matched_fixed_k" in row
            else "—"
        )
        cells = [
            row["method"],
            f"{row['macro_recall_at_returned_k']:.2%}",
            f"{row['mean_k']:.2f}",
            f"{row['median_k']:.1f}",
            f"{row['p90_k']:.1f}",
            matched,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Caveat",
        "",
        "The neighbor-page arm transfers benchmark qrels from training queries, which is a proxy for a historical relevance/click memory. It is leakage-safe across folds but does not establish that production feedback will have the same quality or coverage.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", args.output_dir / "report.md")
    for row in result_rows:
        print(row["method"], f"recall={row['macro_recall_at_returned_k']:.4f}", f"mean_k={row['mean_k']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
