"""Compare Adaptive-k with a fixed global K equal to its mean K.

This is the budget-matched experiment complementary to the per-query matched
control.  For each Adaptive-k policy, compute its mean returned page count,
round that to an integer global K, and give every query that fixed K.  Both
systems use the same dense cosine ranking and the same top-3 file scope.

No embedding/API call is made. Qrels are used only for evaluation.
"""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.compare_benchmark_adaptive_matched_k import (  # noqa: E402
    _evaluate,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)

DEFAULT_ROOT = ROOT / "data/work/benchmark_hierarchical_local"
DEFAULT_EMBEDDING_ROOT = DEFAULT_ROOT / "embedding_adaptive_openrouter_te3s_fulltext"
DEFAULT_OUTPUT = DEFAULT_EMBEDDING_ROOT / "global_mean_budget_comparison"


def _rounded_mean(values: Sequence[int]) -> int:
    return int(sum(values) / len(values) + 0.5) if values else 0


def _pair_summary(
    adaptive: Mapping[str, Any],
    fixed: Mapping[str, Any],
    *,
    adaptive_k: Sequence[int],
    fixed_k: int,
) -> dict[str, Any]:
    adaptive_recall = [float(row["page_recall_at_returned_k"]) for row in adaptive["per_query"]]
    fixed_recall = [float(row["page_recall_at_returned_k"]) for row in fixed["per_query"]]
    deltas = [fixed_value - adaptive_value for fixed_value, adaptive_value in zip(fixed_recall, adaptive_recall)]
    epsilon = 1e-12
    outcomes = Counter(
        "fixed_wins" if delta > epsilon else "adaptive_wins" if delta < -epsilon else "tie"
        for delta in deltas
    )
    return {
        "adaptive_mean_k": float(sum(adaptive_k) / len(adaptive_k)) if adaptive_k else 0.0,
        "fixed_global_k": int(fixed_k),
        "per_query_recall_delta_fixed_minus_adaptive": {
            "mean": float(sum(deltas) / len(deltas)) if deltas else 0.0,
            "min": min(deltas) if deltas else 0.0,
            "max": max(deltas) if deltas else 0.0,
        },
        "per_query_outcomes": dict(outcomes),
        "per_query": [
            {
                "query_id": adaptive_item["query_id"],
                "adaptive_k": adaptive_item["k"],
                "fixed_k": fixed_k,
                "adaptive_page_recall_at_returned_k": adaptive_item["page_recall_at_returned_k"],
                "fixed_page_recall_at_returned_k": fixed_item["page_recall_at_returned_k"],
                "fixed_minus_adaptive_page_recall": delta,
                "outcome": "fixed_wins" if delta > epsilon else "adaptive_wins" if delta < -epsilon else "tie",
            }
            for adaptive_item, fixed_item, delta in zip(
                adaptive["per_query"], fixed["per_query"], deltas
            )
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    pages = _read_jsonl(args.cache_dir / "parsing/pages.jsonl")
    queries = _read_jsonl(ROOT / "data/raw/BENCHMARK/queries.jsonl")
    qrels_rows = _read_jsonl(args.cache_dir / "qrels_page_level.jsonl")
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_rows:
        qrels.setdefault(str(row["query_id"]), {})[str(row["page_id"])] = int(row.get("relevance", 1))

    page_ids = [str(row["page_id"]) for row in pages]
    page_docs = [str(row["doc_id"]) for row in pages]
    page_texts = [str(row.get("text") or "") for row in pages]
    vectors = np.load(args.embedding_root / "embeddings/page_embeddings.npy", mmap_mode="r")
    query_vectors = np.load(args.embedding_root / "embeddings/query_embeddings.npy", mmap_mode="r")
    if vectors.shape[0] != len(pages) or query_vectors.shape[0] != len(queries):
        raise RuntimeError("Dense embedding/page/query count mismatch")

    query_positions = {str(row["query_id"]): index for index, row in enumerate(queries)}
    positions_by_file: dict[str, list[int]] = {}
    for position, doc_id in enumerate(page_docs):
        positions_by_file.setdefault(doc_id, []).append(position)
    policies = {
        "hier_largest_gap": args.embedding_root / "per_query_hier_largest_gap.jsonl",
        "hier_largest_gap_buffer5": args.embedding_root / "per_query_hier_largest_gap_buffer5.jsonl",
    }
    report_policies: dict[str, Any] = {}
    output_dir = args.output_dir.resolve()

    for policy, adaptive_path in policies.items():
        adaptive_rows = _read_jsonl(adaptive_path)
        adaptive_k = [len(row["returned_pages"]) for row in adaptive_rows]
        fixed_k = _rounded_mean(adaptive_k)
        fixed_rows: list[dict[str, Any]] = []
        for adaptive_row in adaptive_rows:
            query_id = str(adaptive_row["query_id"])
            scores = np.asarray(vectors @ query_vectors[query_positions[query_id]], dtype=np.float32)
            for position, text in enumerate(page_texts):
                if not text.strip():
                    scores[position] = -1.0
            selected_files = [str(file_id) for file_id in adaptive_row["selected_files"]]
            positions = [
                position
                for file_id in selected_files
                for position in positions_by_file[file_id]
            ]
            positions.sort(key=lambda position: (-float(scores[position]), position))
            fixed_rows.append(
                {
                    "query_id": query_id,
                    "selected_files": selected_files,
                    "returned_pages": [page_ids[position] for position in positions[:fixed_k]],
                    "k": min(fixed_k, len(positions)),
                    "policy": f"fixed_global_k{fixed_k}",
                }
            )

        adaptive_metrics, adaptive_per_query = _evaluate(adaptive_rows, qrels)
        fixed_metrics, fixed_per_query = _evaluate(fixed_rows, qrels)
        pair = _pair_summary(
            {"metrics": adaptive_metrics, "per_query": adaptive_per_query},
            {"metrics": fixed_metrics, "per_query": fixed_per_query},
            adaptive_k=adaptive_k,
            fixed_k=fixed_k,
        )
        report_policies[policy] = {
            "adaptive": adaptive_metrics,
            "fixed_global_mean_budget": fixed_metrics,
            "comparison": {
                "adaptive_mean_k": pair["adaptive_mean_k"],
                "fixed_global_k": pair["fixed_global_k"],
                "per_query_recall_delta_fixed_minus_adaptive": pair["per_query_recall_delta_fixed_minus_adaptive"],
                "per_query_outcomes": pair["per_query_outcomes"],
            },
        }
        _write_jsonl(output_dir / f"per_query_{policy}_adaptive.jsonl", adaptive_per_query)
        _write_jsonl(output_dir / f"per_query_{policy}_fixed_global_k{fixed_k}.jsonl", fixed_per_query)
        _write_jsonl(output_dir / f"per_query_{policy}_deltas.jsonl", pair["per_query"])

    report = {
        "contract_version": "benchmark-adaptive-global-mean-budget-v1",
        "cache_dir": str(args.cache_dir.resolve()),
        "embedding_root": str(args.embedding_root.resolve()),
        "output_dir": str(output_dir),
        "queries": len(queries),
        "pages": len(pages),
        "ranking_control": {
            "view": "hierarchical",
            "file_scope": "same top-3 files as each Adaptive-k row",
            "page_order": "same dense cosine page ranking",
            "fixed_budget": "one global Kp equal to rounded mean Adaptive-k pages",
            "qrels_used_for_ranking": False,
        },
        "policies": report_policies,
    }
    _write_json(output_dir / "report.json", report)

    lines = [
        "# Adaptive-k versus fixed global mean budget",
        "",
        f"- Corpus: **{len(pages)} pages**, **{len(queries)} queries**.",
        "- Same dense cosine ranking and same top-3 file scope; only page-budget policy differs.",
        "- Fixed Kp is one global integer equal to the rounded mean Adaptive-k Kp.",
        "- No embedding/API call was made; qrels were used only for evaluation.",
        "",
        "| Policy | System | Recall at returned K | Recall@10 | nDCG@10 | File recall@3 | Mean Kp | Median Kp |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, item in report_policies.items():
        adaptive = item["adaptive"]
        fixed = item["fixed_global_mean_budget"]
        outcomes = item["comparison"]["per_query_outcomes"]
        lines.extend(
            [
                f"| {policy} | adaptive | {adaptive['page_recall_at_returned_k']:.2%} | {adaptive['page_recall@10']:.2%} | {adaptive['ndcg@10']:.2f} | {adaptive['file_recall@3']:.2%} | {adaptive['mean_pages']:.2f} | {adaptive['median_pages']:.0f} |",
                f"| {policy} | fixed global Kp={item['comparison']['fixed_global_k']} | {fixed['page_recall_at_returned_k']:.2%} | {fixed['page_recall@10']:.2%} | {fixed['ndcg@10']:.2f} | {fixed['file_recall@3']:.2%} | {fixed['mean_pages']:.2f} | {fixed['median_pages']:.0f} |",
                f"| {policy} | fixed - adaptive delta | {item['comparison']['per_query_recall_delta_fixed_minus_adaptive']['mean']:+.4%} | {(fixed['page_recall@10'] - adaptive['page_recall@10']):+.4%} | {(fixed['ndcg@10'] - adaptive['ndcg@10']):+.4f} | {(fixed['file_recall@3'] - adaptive['file_recall@3']):+.4%} | {(fixed['mean_pages'] - adaptive['mean_pages']):+.2f} | {(fixed['median_pages'] - adaptive['median_pages']):+.0f} |",
                f"- Per-query outcomes: adaptive wins **{outcomes.get('adaptive_wins', 0)}**, fixed wins **{outcomes.get('fixed_wins', 0)}**, ties **{outcomes.get('tie', 0)}**.",
            ]
        )
    lines.extend(
        [
            "",
            "The per-query outcome uses page recall over each system's returned prefix. This is the relevant utility comparison when both systems have the same average page budget but different budget allocation across queries.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
