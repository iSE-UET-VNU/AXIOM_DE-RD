"""Compare Adaptive-k with a per-query fixed-K control on the dense cache.

For a given query, both systems use the same hierarchical dense ranking and
the same top-3 file scope.  The fixed control receives exactly the number of
pages that Adaptive-k selected for that query.  This isolates the stopping
rule from any change in ranking or average context budget.

No embedding/API call is made. Qrels are used only for evaluation.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ROOT = ROOT / "data/work/benchmark_hierarchical_local"
DEFAULT_EMBEDDING_ROOT = DEFAULT_ROOT / "embedding_adaptive_openrouter_te3s_fulltext"
DEFAULT_OUTPUT = DEFAULT_EMBEDDING_ROOT / "matched_k_comparison"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _ndcg(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    gains = {str(page): int(score) for page, score in qrels.items()}
    actual = sum(
        (2.0**gains[page] - 1.0) / math.log2(rank + 1)
        for rank, page in enumerate(ranked[:k], 1)
        if page in gains
    )
    ideal_scores = sorted(gains.values(), reverse=True)[:k]
    ideal = sum(
        (2.0**score - 1.0) / math.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, 1)
    )
    return actual / ideal if ideal else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    metric_k: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    coverage: list[float] = []
    recall10: list[float] = []
    hit10: list[float] = []
    ndcg10: list[float] = []
    file_recall: list[float] = []
    page_counts: list[int] = []
    per_query: list[dict[str, Any]] = []

    for row in rows:
        query_id = str(row["query_id"])
        gold = set(qrels.get(query_id, {}))
        returned = [str(page_id) for page_id in row["returned_pages"]]
        top10 = returned[:metric_k]
        selected = {str(file_id) for file_id in row["selected_files"]}
        gold_files = {page_id.split("#page=", 1)[0] for page_id in gold}
        found = set(returned) & gold
        found10 = set(top10) & gold
        denominator = len(gold)
        coverage_value = len(found) / denominator if denominator else 0.0
        recall10_value = len(found10) / denominator if denominator else 0.0
        file_value = len(selected & gold_files) / len(gold_files) if gold_files else 0.0
        coverage.append(coverage_value)
        recall10.append(recall10_value)
        hit10.append(float(bool(found10)))
        ndcg10.append(_ndcg(top10, qrels.get(query_id, {}), metric_k))
        file_recall.append(file_value)
        page_counts.append(len(returned))
        per_query.append(
            {
                "query_id": query_id,
                "gold_pages": sorted(gold),
                "returned_pages": returned,
                "k": len(returned),
                "selected_files": sorted(selected),
                "page_recall_at_returned_k": coverage_value,
                "page_recall@10": recall10_value,
                "page_hit@10": bool(found10),
                "ndcg@10": _ndcg(top10, qrels.get(query_id, {}), metric_k),
                "file_recall": file_value,
            }
        )

    return (
        {
            "queries": len(rows),
            "page_recall_at_returned_k": _mean(coverage),
            "page_recall@10": _mean(recall10),
            "page_hit@10": _mean(hit10),
            "ndcg@10": 100.0 * _mean(ndcg10),
            "file_recall@3": _mean(file_recall),
            "mean_pages": _mean(page_counts),
            "median_pages": statistics.median(page_counts) if page_counts else 0.0,
            "p90_pages": sorted(page_counts)[min(len(page_counts) - 1, int(0.90 * len(page_counts)))] if page_counts else 0,
        },
        per_query,
    )


def _metric_delta(adaptive: Mapping[str, Any], fixed: Mapping[str, Any]) -> dict[str, float]:
    keys = (
        "page_recall_at_returned_k",
        "page_recall@10",
        "page_hit@10",
        "ndcg@10",
        "file_recall@3",
        "mean_pages",
        "median_pages",
        "p90_pages",
    )
    return {key: float(fixed[key]) - float(adaptive[key]) for key in keys}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    pages = _read_jsonl(args.cache_dir / "parsing/pages.jsonl")
    queries = _read_jsonl(ROOT / "data/raw/BENCHMARK/queries.jsonl")
    qrel_rows = _read_jsonl(args.cache_dir / "qrels_page_level.jsonl")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrel_rows:
        qrels[str(row["query_id"])][str(row["page_id"])] = int(row.get("relevance", 1))

    page_ids = [str(row["page_id"]) for row in pages]
    doc_ids = [str(row["doc_id"]) for row in pages]
    page_texts = [str(row.get("text") or "") for row in pages]
    vectors = np.load(args.embedding_root / "embeddings/page_embeddings.npy", mmap_mode="r")
    query_vectors = np.load(args.embedding_root / "embeddings/query_embeddings.npy", mmap_mode="r")
    if vectors.shape[0] != len(pages) or query_vectors.shape[0] != len(queries):
        raise RuntimeError("Dense embedding/page/query count mismatch")

    query_positions = {str(row["query_id"]): index for index, row in enumerate(queries)}
    page_positions_by_file: dict[str, list[int]] = defaultdict(list)
    for position, doc_id in enumerate(doc_ids):
        page_positions_by_file[doc_id].append(position)

    policies = {
        "hier_largest_gap": args.embedding_root / "per_query_hier_largest_gap.jsonl",
        "hier_largest_gap_buffer5": args.embedding_root / "per_query_hier_largest_gap_buffer5.jsonl",
    }
    report_policies: dict[str, Any] = {}
    output_dir = args.output_dir.resolve()

    for policy, adaptive_path in policies.items():
        adaptive_rows = _read_jsonl(adaptive_path)
        if len(adaptive_rows) != len(queries):
            raise RuntimeError(f"{policy}: expected {len(queries)} rows, found {len(adaptive_rows)}")

        fixed_rows: list[dict[str, Any]] = []
        deltas: list[dict[str, Any]] = []
        exact_matches = 0
        max_page_mismatch = 0
        for adaptive_row in adaptive_rows:
            query_id = str(adaptive_row["query_id"])
            query_position = query_positions[query_id]
            scores = np.asarray(vectors @ query_vectors[query_position], dtype=np.float32)
            for position, text in enumerate(page_texts):
                if not text.strip():
                    scores[position] = -1.0
            selected_files = [str(file_id) for file_id in adaptive_row["selected_files"]]
            selected_positions = [
                position
                for file_id in selected_files
                for position in page_positions_by_file[file_id]
            ]
            selected_positions.sort(key=lambda position: (-float(scores[position]), position))
            k = len(adaptive_row["returned_pages"])
            fixed_returned = [page_ids[position] for position in selected_positions[:k]]
            adaptive_returned = [str(page_id) for page_id in adaptive_row["returned_pages"]]
            mismatch = sum(left != right for left, right in zip(adaptive_returned, fixed_returned))
            mismatch += abs(len(adaptive_returned) - len(fixed_returned))
            max_page_mismatch = max(max_page_mismatch, mismatch)
            if adaptive_returned == fixed_returned:
                exact_matches += 1
            fixed_rows.append(
                {
                    "query_id": query_id,
                    "selected_files": selected_files,
                    "returned_pages": fixed_returned,
                    "k": k,
                    "policy": f"fixed_matched_to_{policy}",
                    "matched_adaptive_k": k,
                }
            )

        adaptive_metrics, adaptive_per_query = _evaluate(adaptive_rows, qrels)
        fixed_metrics, fixed_per_query = _evaluate(fixed_rows, qrels)
        for adaptive_item, fixed_item in zip(adaptive_per_query, fixed_per_query):
            deltas.append(
                {
                    "query_id": adaptive_item["query_id"],
                    "k": adaptive_item["k"],
                    "pages_identical": adaptive_item["returned_pages"] == fixed_item["returned_pages"],
                    "adaptive_page_recall_at_returned_k": adaptive_item["page_recall_at_returned_k"],
                    "fixed_page_recall_at_returned_k": fixed_item["page_recall_at_returned_k"],
                    "fixed_minus_adaptive_page_recall": fixed_item["page_recall_at_returned_k"] - adaptive_item["page_recall_at_returned_k"],
                }
            )
        report_policies[policy] = {
            "adaptive": adaptive_metrics,
            "fixed_matched_k": fixed_metrics,
            "fixed_minus_adaptive": _metric_delta(adaptive_metrics, fixed_metrics),
            "per_query_exact_page_matches": exact_matches,
            "per_query_page_mismatch_max": max_page_mismatch,
            "adaptive_k_distribution": {
                "mean": adaptive_metrics["mean_pages"],
                "median": adaptive_metrics["median_pages"],
                "p90": adaptive_metrics["p90_pages"],
            },
        }
        _write_jsonl(output_dir / f"per_query_{policy}_adaptive.jsonl", adaptive_per_query)
        _write_jsonl(output_dir / f"per_query_{policy}_fixed_matched.jsonl", fixed_per_query)
        _write_jsonl(output_dir / f"per_query_{policy}_deltas.jsonl", deltas)

    report = {
        "contract_version": "benchmark-adaptive-matched-k-v1",
        "cache_dir": str(args.cache_dir.resolve()),
        "embedding_root": str(args.embedding_root.resolve()),
        "output_dir": str(output_dir),
        "queries": len(queries),
        "pages": len(pages),
        "embedding": {
            "dimension": int(vectors.shape[1]),
            "normalized_cache": True,
            "new_api_calls": 0,
        },
        "ranking_control": {
            "view": "hierarchical",
            "file_scope": "same top-3 files as Adaptive-k row",
            "page_order": "same dense cosine page ranking as Adaptive-k row",
            "fixed_control": "per-query Kp exactly equals Adaptive-k returned page count",
            "qrels_used_for_ranking": False,
        },
        "policies": report_policies,
    }
    _write_json(output_dir / "report.json", report)

    lines = [
        "# Per-query matched-K comparison: Adaptive-k versus fixed retrieval",
        "",
        f"- Corpus: **{len(pages)} pages**, **{len(queries)} queries**.",
        "- Dense cosine page embeddings and the same top-3 file scope are used for both systems.",
        "- For every query, fixed retrieval receives exactly the Adaptive-k page count for that query.",
        "- No embedding/API call was made; qrels were used only for evaluation.",
        "",
        "| Policy pair | System | Recall at returned K | Recall@10 | nDCG@10 | File recall@3 | Mean Kp | Median Kp | P90 Kp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, item in report_policies.items():
        for label in ("adaptive", "fixed_matched_k"):
            metrics = item[label]
            lines.append(
                f"| {policy} | {label} | {metrics['page_recall_at_returned_k']:.2%} | {metrics['page_recall@10']:.2%} | {metrics['ndcg@10']:.2f} | {metrics['file_recall@3']:.2%} | {metrics['mean_pages']:.2f} | {metrics['median_pages']:.0f} | {metrics['p90_pages']:.0f} |"
            )
        delta = item["fixed_minus_adaptive"]
        lines.append(
            f"| {policy} | fixed - adaptive delta | {delta['page_recall_at_returned_k']:+.6f} | {delta['page_recall@10']:+.6f} | {delta['ndcg@10']:+.6f} | {delta['file_recall@3']:+.6f} | {delta['mean_pages']:+.6f} | {delta['median_pages']:+.6f} | {delta['p90_pages']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Sanity check",
            "",
            *[
                f"- `{policy}`: exact returned-page match for **{item['per_query_exact_page_matches']}/{len(queries)}** queries; maximum mismatch: **{item['per_query_page_mismatch_max']}**."
                for policy, item in report_policies.items()
            ],
            "",
            "Because both systems use the same ranking and the fixed control is assigned the same Kp per query, identical returned prefixes and identical retrieval metrics are expected. Any Adaptive-k value comes from selecting Kp without qrels, not from improving the ranking at a matched Kp.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
