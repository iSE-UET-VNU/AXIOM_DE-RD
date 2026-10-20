"""Evaluate score-only iterative stopping on cached hierarchical page rankings.

The experiment deliberately does not re-parse or re-encode documents.  It uses
the score curves emitted by ``analyze_benchmark_adaptive_topk.py`` and tests
three pre-declared stopping rules at the page stage for a fixed Kf=3:

* fixed-10: the current light-retrieval baseline;
* largest-gap: stop after the largest adjacent score drop after rank 10;
* first-q75-gap: scan pages from rank 10 and stop at the first gap above the
  score-only per-rank q75 threshold;
* running-3x: scan pages and stop when a gap is at least three times the
  median of the preceding five gaps.

Qrels are loaded only after a stopping decision has been made.  They are used
for evaluation and diagnostics, never for thresholds or stopping decisions.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import argparse
import json
import math
import statistics
import sys
import time


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


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "q75": _quantile(values, 0.75),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _gaps(ranking: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    scores = [float(item["score"]) for item in ranking]
    # A boundary r means: keep ranks 1..r; the gap is score[r] - score[r+1].
    return {r: max(0.0, scores[r - 1] - scores[r]) for r in range(1, len(scores))}


def _stop_largest_gap(gaps: Mapping[int, float], *, minimum: int, maximum: int) -> tuple[int, float]:
    candidates = [(rank, gap) for rank, gap in gaps.items() if minimum <= rank <= maximum]
    if not candidates:
        return maximum, 0.0
    rank, gap = max(candidates, key=lambda item: (item[1], -item[0]))
    return rank, gap


def _stop_first_q75_gap(
    gaps: Mapping[int, float], thresholds: Mapping[int, float], *, minimum: int, maximum: int
) -> tuple[int, float]:
    for rank in range(minimum, min(maximum, max(gaps, default=minimum)) + 1):
        gap = gaps.get(rank, 0.0)
        threshold = thresholds.get(rank, 0.0)
        if gap > 0.0 and threshold > 0.0 and gap >= threshold:
            return rank, gap
    return min(maximum, max(gaps, default=maximum)), gaps.get(min(maximum, max(gaps, default=maximum)), 0.0)


def _stop_running_jump(
    gaps: Mapping[int, float], *, minimum: int, maximum: int, window: int, ratio: float
) -> tuple[int, float]:
    last = min(maximum, max(gaps, default=maximum))
    for rank in range(minimum, last + 1):
        previous = [gaps[index] for index in range(max(minimum, rank - window), rank) if index in gaps]
        if not previous:
            continue
        baseline = statistics.median(previous)
        gap = gaps.get(rank, 0.0)
        if baseline > 0.0 and gap >= ratio * baseline:
            return rank, gap
    return last, gaps.get(last, 0.0)


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, set[str]],
    *,
    policy: str,
) -> dict[str, Any]:
    returned_recall: list[float] = []
    recall10: list[float] = []
    hit10: list[float] = []
    file_recall: list[float] = []
    page_counts: list[int] = []
    file_counts: list[int] = []
    stop_ranks: list[int] = []
    for row in rows:
        gold = qrels.get(str(row["query_id"]), set())
        returned = list(row["returned_pages"])
        found = set(returned) & gold
        top10 = set(returned[:10]) & gold
        gold_files = {page_id.split("#page=", 1)[0] for page_id in gold}
        returned_files = {page_id.split("#page=", 1)[0] for page_id in returned}
        selected_files = set(row["selected_files"])
        returned_recall.append(len(found) / len(gold) if gold else 0.0)
        recall10.append(len(top10) / len(gold) if gold else 0.0)
        hit10.append(float(bool(top10)))
        file_recall.append(len(selected_files & gold_files) / len(gold_files) if gold_files else 0.0)
        page_counts.append(len(returned))
        file_counts.append(len(returned_files))
        stop_ranks.append(int(row["stop_rank"]))

    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "policy": policy,
        "queries": len(rows),
        "page_recall_at_returned_k": mean(returned_recall),
        "page_recall_at_10": mean(recall10),
        "page_hit_at_10": mean(hit10),
        "file_recall_at_kf3": mean(file_recall),
        "mean_returned_pages": mean(page_counts),
        "median_returned_pages": statistics.median(page_counts) if page_counts else 0.0,
        "p95_returned_pages": _quantile(page_counts, 0.95),
        "mean_returned_files": mean(file_counts),
        "mean_stop_rank": mean(stop_ranks),
        "stop_rank_distribution": {str(rank): stop_ranks.count(rank) for rank in sorted(set(stop_ranks))},
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    started = time.perf_counter()
    traces = _read_jsonl(args.score_curves)
    qrel_rows = _read_jsonl(args.qrels)
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in qrel_rows:
        qrels[str(row["query_id"])].add(str(row["page_id"]))

    # Build thresholds from scores only.  The per-rank threshold is important:
    # gaps near the top naturally have a different scale from gaps near rank 100.
    all_gaps: dict[int, list[float]] = defaultdict(list)
    trace_gaps: dict[str, dict[int, float]] = {}
    rankings: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        qid = str(trace["query_id"])
        ranking = list(trace["page_rankings"][str(args.file_k)])[: args.maximum_page_k]
        rankings[qid] = ranking
        gaps = _gaps(ranking)
        trace_gaps[qid] = gaps
        for rank, gap in gaps.items():
            if rank >= args.minimum_page_k:
                all_gaps[rank].append(gap)
    q75_by_rank = {rank: _quantile(values, 0.75) for rank, values in all_gaps.items()}
    pooled_q75 = _quantile([value for values in all_gaps.values() for value in values], 0.75)

    policy_rows: dict[str, list[dict[str, Any]]] = {"fixed_10": [], "largest_gap": [], "first_q75_gap": [], "running_3x": []}
    for trace in traces:
        qid = str(trace["query_id"])
        ranking = rankings[qid]
        gaps = trace_gaps[qid]
        selected_files = [item["file_id"] for item in trace["file_ranked"][: args.file_k]]
        max_available = min(args.maximum_page_k, len(ranking))
        decisions = {
            "fixed_10": (min(10, max_available), gaps.get(10, 0.0), "fixed"),
            "largest_gap": (*_stop_largest_gap(gaps, minimum=args.minimum_page_k, maximum=max_available), "largest-adjacent-gap"),
            "first_q75_gap": (*_stop_first_q75_gap(gaps, q75_by_rank, minimum=args.minimum_page_k, maximum=max_available), "first-per-rank-q75-gap"),
            "running_3x": (*_stop_running_jump(gaps, minimum=args.minimum_page_k, maximum=max_available, window=5, ratio=3.0), "first-3x-running-median"),
        }
        for policy, (stop_rank, stop_gap, reason) in decisions.items():
            policy_rows[policy].append({
                "query_id": qid,
                "selected_files": selected_files,
                "returned_pages": [item["page_id"] for item in ranking[:stop_rank]],
                "stop_rank": stop_rank,
                "stop_gap": stop_gap,
                "stop_reason": reason,
                "max_available_pages": max_available,
                "pooled_q75_gap": pooled_q75,
            })

    metrics = {policy: _evaluate(rows, qrels, policy=policy) for policy, rows in policy_rows.items()}
    gap_summary = {
        "pooled_q75": pooled_q75,
        "per_rank_q75": {str(rank): value for rank, value in sorted(q75_by_rank.items())},
        "gap_at_rank_10": _summary([trace_gaps[qid].get(10, 0.0) for qid in trace_gaps]),
        "gap_at_rank_20": _summary([trace_gaps[qid].get(20, 0.0) for qid in trace_gaps]),
        "gap_at_rank_50": _summary([trace_gaps[qid].get(50, 0.0) for qid in trace_gaps]),
    }
    report = {
        "contract_version": "benchmark-iterative-stopping-v1",
        "score_curves": str(args.score_curves),
        "qrels": str(args.qrels),
        "queries": len(traces),
        "file_k": args.file_k,
        "minimum_page_k": args.minimum_page_k,
        "maximum_page_k": args.maximum_page_k,
        "visual_mode": "disabled/cache-BM25-only",
        "thresholds": gap_summary,
        "metrics": metrics,
        "policy": {
            "fixed_10": "return the first 10 pages",
            "largest_gap": "scan adjacent page-score gaps from rank 10 to 100 and stop after the largest",
            "first_q75_gap": "scan rank by rank and stop at the first per-rank q75 gap learned from scores across queries",
            "running_3x": "scan rank by rank and stop at the first gap >= 3x the median of the preceding five gaps",
            "qrels_used_for_thresholds": False,
            "query_feedback": False,
        },
        "timing_seconds": {"offline_score_only_analysis": round(time.perf_counter() - started, 3)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "report.json", report)
    _write_json(args.output_dir / "score_gap_summary.json", gap_summary)
    for policy, rows in policy_rows.items():
        _write_jsonl(args.output_dir / f"{policy}_per_query.jsonl", rows)

    lines = [
        "# Iterative / score-gap stopping on benchmark cache",
        "",
        f"- Queries: {len(traces)}; fixed file stage: Kf={args.file_k}; available page ranking: up to {args.maximum_page_k}.",
        "- Cache: PDF-inspector/Tesseract + BM25 hierarchical scores; V-SPLADE disabled.",
        "- All stopping decisions are score-only. Qrels are used only for evaluation.",
        "",
        "## Results",
        "",
        "| Policy | Page recall at returned K | Page recall@10 | Page hit@10 | File recall@Kf3 | Mean pages | Median pages | Mean files |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ("fixed_10", "largest_gap", "first_q75_gap", "running_3x"):
        item = metrics[policy]
        lines.append(
            f"| {policy} | {item['page_recall_at_returned_k']:.2%} | {item['page_recall_at_10']:.2%} | "
            f"{item['page_hit_at_10']:.2%} | {item['file_recall_at_kf3']:.2%} | "
            f"{item['mean_returned_pages']:.2f} | {item['median_returned_pages']:.0f} | {item['mean_returned_files']:.2f} |"
        )
    lines.extend([
        "",
        "## Reading",
        "",
        "- `page recall at returned K` measures the evidence coverage actually sent downstream; it is the relevant metric for a second retrieval stage.",
        "- `page recall@10` is unchanged by returning more than 10 pages: it measures the first ten of the same page ranking, not the adaptive pool coverage.",
        "- `largest_gap` is the closest direct test of the published Adaptive-k idea; the two other policies are diagnostic stopping variants, not tuned final methods.",
        "- A large gap is only a candidate boundary. It does not prove that all evidence is above it, especially when the page ranking is noisy or multi-evidence.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-curves", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file-k", type=int, default=3)
    parser.add_argument("--minimum-page-k", type=int, default=10)
    parser.add_argument("--maximum-page-k", type=int, default=100)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
