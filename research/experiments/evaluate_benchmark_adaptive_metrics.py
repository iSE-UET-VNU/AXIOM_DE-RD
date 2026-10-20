"""Evaluate cached dense Adaptive-k runs at page and derived-file level.

File recall is computed from the union of the files appearing in the
returned page list.  The script also reports the first-three-unique-files
variant so it remains comparable with the benchmark's canonical file@3
diagnostic.

This is evaluation only: it reads existing Adaptive-k rows and qrels, with no
embedding, retrieval, OCR, or API call.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ROOT = ROOT / "data/work/benchmark_hierarchical_local"
DEFAULT_EMBEDDING_ROOT = DEFAULT_ROOT / "embedding_adaptive_openrouter_te3s_fulltext"
DEFAULT_OUTPUT = DEFAULT_EMBEDDING_ROOT / "adaptive_metrics"


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


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _file_id(page_id: str) -> str:
    return str(page_id).split("#page=", 1)[0]


def _ndcg(ranked_pages: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    gains = {str(page_id): int(relevance) for page_id, relevance in qrels.items()}
    actual = sum(
        (2.0**gains[page_id] - 1.0) / math.log2(rank + 1)
        for rank, page_id in enumerate(ranked_pages[:k], start=1)
        if page_id in gains
    )
    ideal_relevance = sorted(gains.values(), reverse=True)[:k]
    ideal = sum(
        (2.0**relevance - 1.0) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevance, start=1)
    )
    return actual / ideal if ideal else 0.0


def _evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    page_recall: list[float] = []
    recall10: list[float] = []
    recall20: list[float] = []
    ndcg10: list[float] = []
    ndcg20: list[float] = []
    file_recall_union: list[float] = []
    file_recall_union3: list[float] = []
    selected_file_recall3: list[float] = []
    page_counts: list[int] = []
    unique_file_counts: list[int] = []
    per_query: list[dict[str, Any]] = []

    for row in rows:
        query_id = str(row["query_id"])
        gold_pages = set(qrels.get(query_id, {}))
        gold_files = {_file_id(page_id) for page_id in gold_pages}
        returned_pages = [str(page_id) for page_id in row["returned_pages"]]
        returned_files = list(dict.fromkeys(_file_id(page_id) for page_id in returned_pages))
        returned_files3 = returned_files[:3]
        selected_files = list(dict.fromkeys(str(file_id) for file_id in row.get("selected_files", [])))

        returned_found = len(set(returned_pages) & gold_pages)
        returned_found10 = len(set(returned_pages[:10]) & gold_pages)
        returned_found20 = len(set(returned_pages[:20]) & gold_pages)
        denominator_pages = len(gold_pages)
        denominator_files = len(gold_files)
        page_value = returned_found / denominator_pages if denominator_pages else 0.0
        recall10_value = returned_found10 / denominator_pages if denominator_pages else 0.0
        recall20_value = returned_found20 / denominator_pages if denominator_pages else 0.0
        file_union_value = len(set(returned_files) & gold_files) / denominator_files if denominator_files else 0.0
        file_union3_value = len(set(returned_files3) & gold_files) / denominator_files if denominator_files else 0.0
        selected_file_value = len(set(selected_files[:3]) & gold_files) / denominator_files if denominator_files else 0.0
        ndcg10_value = _ndcg(returned_pages, qrels.get(query_id, {}), 10)
        ndcg20_value = _ndcg(returned_pages, qrels.get(query_id, {}), 20)

        page_recall.append(page_value)
        recall10.append(recall10_value)
        recall20.append(recall20_value)
        ndcg10.append(ndcg10_value)
        ndcg20.append(ndcg20_value)
        file_recall_union.append(file_union_value)
        file_recall_union3.append(file_union3_value)
        selected_file_recall3.append(selected_file_value)
        page_counts.append(len(returned_pages))
        unique_file_counts.append(len(returned_files))
        per_query.append(
            {
                "query_id": query_id,
                "source": str(row.get("source") or query_id.split("::", 1)[0]),
                "adaptive_k": len(returned_pages),
                "gold_pages": len(gold_pages),
                "gold_files": sorted(gold_files),
                "returned_pages": returned_pages,
                "returned_files_union": returned_files,
                "returned_files_union_at3": returned_files3,
                "selected_files_top3": selected_files[:3],
                "page_recall_at_returned_k": page_value,
                "page_recall@10": recall10_value,
                "page_recall@20": recall20_value,
                "ndcg@10": 100.0 * ndcg10_value,
                "ndcg@20": 100.0 * ndcg20_value,
                "file_recall_from_returned_file_union": file_union_value,
                "file_recall@3_from_returned_file_union": file_union3_value,
                "file_recall@3_selected_files": selected_file_value,
            }
        )

    metrics = {
        "queries": len(rows),
        "page_recall_at_returned_k": _mean(page_recall),
        "page_recall@10": _mean(recall10),
        "page_recall@20": _mean(recall20),
        "ndcg@10": 100.0 * _mean(ndcg10),
        "ndcg@20": 100.0 * _mean(ndcg20),
        "file_recall_from_returned_file_union": _mean(file_recall_union),
        "file_recall@3_from_returned_file_union": _mean(file_recall_union3),
        "file_recall@3_selected_files": _mean(selected_file_recall3),
        "mean_adaptive_k": _mean(page_counts),
        "median_adaptive_k": statistics.median(page_counts) if page_counts else 0.0,
        "mean_unique_returned_files": _mean(unique_file_counts),
        "median_unique_returned_files": statistics.median(unique_file_counts) if unique_file_counts else 0.0,
    }
    return metrics, per_query


def _source_breakdown(per_query: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_query:
        grouped[str(row["source"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for source, rows in sorted(grouped.items()):
        result[source] = {
            "queries": len(rows),
            "page_recall_at_returned_k": _mean([float(row["page_recall_at_returned_k"]) for row in rows]),
            "page_recall@10": _mean([float(row["page_recall@10"]) for row in rows]),
            "page_recall@20": _mean([float(row["page_recall@20"]) for row in rows]),
            "ndcg@10": _mean([float(row["ndcg@10"]) for row in rows]),
            "ndcg@20": _mean([float(row["ndcg@20"]) for row in rows]),
            "file_recall_from_returned_file_union": _mean(
                [float(row["file_recall_from_returned_file_union"]) for row in rows]
            ),
            "file_recall@3_from_returned_file_union": _mean(
                [float(row["file_recall@3_from_returned_file_union"]) for row in rows]
            ),
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    qrel_rows = _read_jsonl(args.cache_dir / "qrels_page_level.jsonl")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrel_rows:
        qrels[str(row["query_id"])][str(row["page_id"])] = int(row.get("relevance", 1))

    policy_paths = {
        "hier_largest_gap": args.embedding_root / "per_query_hier_largest_gap.jsonl",
        "hier_largest_gap_buffer5": args.embedding_root / "per_query_hier_largest_gap_buffer5.jsonl",
    }
    report_policies: dict[str, Any] = {}
    output_dir = args.output_dir.resolve()
    for policy, path in policy_paths.items():
        rows = _read_jsonl(path)
        metrics, per_query = _evaluate_rows(rows, qrels)
        report_policies[policy] = {
            "metrics": metrics,
            "source_breakdown": _source_breakdown(per_query),
        }
        _write_jsonl(output_dir / f"per_query_{policy}.jsonl", per_query)

    report = {
        "contract_version": "benchmark-adaptive-metrics-v1",
        "cache_dir": str(args.cache_dir.resolve()),
        "embedding_root": str(args.embedding_root.resolve()),
        "output_dir": str(output_dir),
        "qrels": {
            "records": len(qrel_rows),
            "used_for_evaluation_only": True,
        },
        "file_recall_definition": {
            "union": "deduplicate file IDs from every returned page, preserving page rank order",
            "file_recall_from_returned_file_union": "intersection of all returned unique files with gold files / gold files",
            "file_recall@3_from_returned_file_union": "intersection of first 3 returned unique files with gold files / gold files",
            "file_recall@3_selected_files": "existing hierarchical selected top-3 file list, retained for reference",
        },
        "policies": report_policies,
        "no_new_inference": True,
    }
    _write_json(output_dir / "report.json", report)

    lines = [
        "# Cached Adaptive-k metrics",
        "",
        "- Evaluation-only report over the 220-query benchmark; no embedding, retrieval, OCR, or API call was made.",
        "- Page recall is measured at each query's returned Adaptive-k. `page_recall@10` and `page_recall@20` use the first 10/20 returned pages (or all pages when fewer were returned).",
        "- nDCG is reported at both @10 and @20 on the page ranking, on a 0–100 scale.",
        "- File recall is derived by deduplicating file IDs from the returned page union; the @3 variant keeps only the first 3 unique files.",
        "",
        "## Overall results",
        "",
        "| Policy | Page recall at Adaptive K | Page recall@10 | Page recall@20 | nDCG@10 | nDCG@20 | File recall from all returned files | File recall@3 from returned union | Mean Kp | Mean unique files |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, item in report_policies.items():
        metrics = item["metrics"]
        lines.append(
            f"| {policy} | {metrics['page_recall_at_returned_k']:.2%} | {metrics['page_recall@10']:.2%} | {metrics['page_recall@20']:.2%} | {metrics['ndcg@10']:.2f} | {metrics['ndcg@20']:.2f} | {metrics['file_recall_from_returned_file_union']:.2%} | {metrics['file_recall@3_from_returned_file_union']:.2%} | {metrics['mean_adaptive_k']:.2f} | {metrics['mean_unique_returned_files']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Dataset breakdown",
            "",
            "| Policy | Source | Queries | Page recall at Adaptive K | Page recall@10 | Page recall@20 | nDCG@10 | nDCG@20 | File recall from all returned files | File recall@3 from returned union |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy, item in report_policies.items():
        for source, metrics in item["source_breakdown"].items():
            lines.append(
                f"| {policy} | {source} | {metrics['queries']} | {metrics['page_recall_at_returned_k']:.2%} | {metrics['page_recall@10']:.2%} | {metrics['page_recall@20']:.2%} | {metrics['ndcg@10']:.2f} | {metrics['ndcg@20']:.2f} | {metrics['file_recall_from_returned_file_union']:.2%} | {metrics['file_recall@3_from_returned_file_union']:.2%} |"
            )
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
