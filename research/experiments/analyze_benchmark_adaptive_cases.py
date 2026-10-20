"""Inspect per-query Adaptive-k cases against a fixed global budget.

The two cases are:

* TH1: Adaptive-k returns fewer pages than the fixed global budget. Does it
  lose evidence recall?
* TH2: Adaptive-k returns more pages than the fixed global budget. Do the
  additional pages recover additional gold evidence?

This reads the already-produced dense cosine run rows and normalized qrels;
it does not encode, call an API, or rerun retrieval.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ROOT = ROOT / "data/work/benchmark_hierarchical_local"
DEFAULT_EMBEDDING_ROOT = DEFAULT_ROOT / "embedding_adaptive_openrouter_te3s_fulltext"
DEFAULT_COMPARISON_ROOT = DEFAULT_EMBEDDING_ROOT / "global_mean_budget_comparison"
DEFAULT_OUTPUT = DEFAULT_EMBEDDING_ROOT / "adaptive_case_analysis"


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


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"queries": 0}
    adaptive_k = [int(row["adaptive_k"]) for row in rows]
    fixed_k = [int(row["fixed_k_actual"]) for row in rows]
    adaptive_recall = [float(row["adaptive_recall"]) for row in rows]
    fixed_recall = [float(row["fixed_recall"]) for row in rows]
    recall_delta = [float(row["fixed_minus_adaptive_recall"]) for row in rows]
    adaptive_found = [int(row["adaptive_found"]) for row in rows]
    fixed_found = [int(row["fixed_found"]) for row in rows]
    return {
        "queries": len(rows),
        "mean_adaptive_k": _mean(adaptive_k),
        "mean_fixed_k_actual": _mean(fixed_k),
        "mean_gold_pages": _mean([int(row["gold_pages"]) for row in rows]),
        "mean_adaptive_found": _mean(adaptive_found),
        "mean_fixed_found": _mean(fixed_found),
        "mean_adaptive_recall": _mean(adaptive_recall),
        "mean_fixed_recall": _mean(fixed_recall),
        "mean_fixed_minus_adaptive_recall": _mean(recall_delta),
        "median_fixed_minus_adaptive_recall": statistics.median(recall_delta),
        "max_fixed_minus_adaptive_recall": max(recall_delta),
        "min_fixed_minus_adaptive_recall": min(recall_delta),
        "queries_with_recall_delta": sum(abs(value) > 1e-12 for value in recall_delta),
        "queries_with_identical_recall": sum(abs(value) <= 1e-12 for value in recall_delta),
    }


def _case_summary(rows: Sequence[Mapping[str, Any]], case: str) -> dict[str, Any]:
    result = _summary(rows)
    if not rows:
        return result
    deltas = [float(row["fixed_minus_adaptive_recall"]) for row in rows]
    if case == "adaptive_less":
        losses = [value for value in deltas if value > 1e-12]
        result.update(
            {
                "case": "TH1_adaptive_less_than_fixed",
                "queries_with_recall_loss": len(losses),
                "queries_without_recall_loss": len(rows) - len(losses),
                "recall_preservation_rate": (len(rows) - len(losses)) / len(rows),
                "mean_loss_conditional": _mean(losses),
                "max_loss": max(losses) if losses else 0.0,
                "mean_extra_pages_given_to_fixed": _mean(
                    [int(row["fixed_k_actual"]) - int(row["adaptive_k"]) for row in rows]
                ),
            }
        )
    else:
        gains = [-value for value in deltas if value < -1e-12]
        result.update(
            {
                "case": "TH2_adaptive_more_than_fixed",
                "queries_with_adaptive_recall_gain": len(gains),
                "queries_without_adaptive_recall_gain": len(rows) - len(gains),
                "adaptive_gain_rate": len(gains) / len(rows),
                "mean_gain_conditional": _mean(gains),
                "max_gain": max(gains) if gains else 0.0,
                "mean_extra_pages_given_to_adaptive": _mean(
                    [int(row["adaptive_k"]) - int(row["fixed_k_actual"]) for row in rows]
                ),
                "queries_with_additional_gold_page": sum(
                    int(row["adaptive_found"]) > int(row["fixed_found"]) for row in rows
                ),
            }
        )
    return result


def _source_breakdown(rows: Sequence[Mapping[str, Any]], case: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].append(row)
    return {source: _case_summary(source_rows, case) for source, source_rows in sorted(grouped.items())}


def _case_row(
    adaptive: Mapping[str, Any],
    fixed: Mapping[str, Any],
    *,
    query_meta: Mapping[str, Any],
    fixed_budget: int,
) -> dict[str, Any]:
    adaptive_pages = [str(page_id) for page_id in adaptive["returned_pages"]]
    fixed_pages = [str(page_id) for page_id in fixed["returned_pages"]]
    gold = set(query_meta["gold_pages"])
    adaptive_found = len(set(adaptive_pages) & gold)
    fixed_found = len(set(fixed_pages) & gold)
    gold_count = len(gold)
    adaptive_recall = adaptive_found / gold_count if gold_count else 0.0
    fixed_recall = fixed_found / gold_count if gold_count else 0.0
    if len(adaptive_pages) < len(fixed_pages):
        case = "adaptive_less"
    elif len(adaptive_pages) > len(fixed_pages):
        case = "adaptive_more"
    else:
        case = "same_effective_k"
    return {
        "query_id": str(adaptive["query_id"]),
        "source": str(query_meta.get("source") or "unknown"),
        "query": str(query_meta["query"]),
        "case": case,
        "fixed_budget_nominal": fixed_budget,
        "adaptive_k": len(adaptive_pages),
        "fixed_k_actual": len(fixed_pages),
        "gold_pages": gold_count,
        "adaptive_found": adaptive_found,
        "fixed_found": fixed_found,
        "adaptive_recall": adaptive_recall,
        "fixed_recall": fixed_recall,
        "fixed_minus_adaptive_recall": fixed_recall - adaptive_recall,
        "adaptive_pages": adaptive_pages,
        "fixed_pages": fixed_pages,
        "page_prefixes_match_up_to_shorter": (
            adaptive_pages[: min(len(adaptive_pages), len(fixed_pages))]
            == fixed_pages[: min(len(adaptive_pages), len(fixed_pages))]
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    queries = _read_jsonl(ROOT / "data/raw/BENCHMARK/queries.jsonl")
    query_meta = {str(row["query_id"]): row for row in queries}
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in _read_jsonl(args.cache_dir / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    for query_id, row in query_meta.items():
        row["gold_pages"] = sorted(qrels.get(query_id, set()))

    policy_specs = {
        "hier_largest_gap": {
            "adaptive": args.embedding_root / "per_query_hier_largest_gap.jsonl",
            "fixed": args.comparison_root / "per_query_hier_largest_gap_fixed_global_k27.jsonl",
            "fixed_budget": 27,
        },
        "hier_largest_gap_buffer5": {
            "adaptive": args.embedding_root / "per_query_hier_largest_gap_buffer5.jsonl",
            "fixed": args.comparison_root / "per_query_hier_largest_gap_buffer5_fixed_global_k32.jsonl",
            "fixed_budget": 32,
        },
    }
    report_policies: dict[str, Any] = {}
    output_dir = args.output_dir.resolve()
    for policy, spec in policy_specs.items():
        adaptive_rows = _read_jsonl(spec["adaptive"])
        fixed_rows = _read_jsonl(spec["fixed"])
        if len(adaptive_rows) != len(fixed_rows):
            raise RuntimeError(f"{policy}: adaptive/fixed query count mismatch")
        rows = []
        for adaptive, fixed in zip(adaptive_rows, fixed_rows):
            query_id = str(adaptive["query_id"])
            if query_id != str(fixed["query_id"]):
                raise RuntimeError(f"{policy}: qid ordering mismatch at {query_id}")
            rows.append(
                _case_row(
                    adaptive,
                    fixed,
                    query_meta=query_meta[query_id],
                    fixed_budget=int(spec["fixed_budget"]),
                )
            )
        less = [row for row in rows if row["case"] == "adaptive_less"]
        more = [row for row in rows if row["case"] == "adaptive_more"]
        same = [row for row in rows if row["case"] == "same_effective_k"]
        loss_cases = sorted(less, key=lambda row: (-row["fixed_minus_adaptive_recall"], row["query_id"]))
        gain_cases = sorted(more, key=lambda row: (row["fixed_minus_adaptive_recall"], row["query_id"]))
        report_policies[policy] = {
            "fixed_budget": int(spec["fixed_budget"]),
            "all_queries": _summary(rows),
            "TH1_adaptive_less": _case_summary(less, "adaptive_less"),
            "TH2_adaptive_more": _case_summary(more, "adaptive_more"),
            "same_effective_k": _summary(same),
            "source_breakdown_TH1": _source_breakdown(less, "adaptive_less"),
            "source_breakdown_TH2": _source_breakdown(more, "adaptive_more"),
            "top_recall_losses_TH1": loss_cases[:10],
            "top_recall_gains_TH2": gain_cases[:10],
            "prefix_mismatch_count": sum(not row["page_prefixes_match_up_to_shorter"] for row in rows),
        }
        _write_jsonl(output_dir / f"per_query_{policy}.jsonl", rows)

    report = {
        "contract_version": "benchmark-adaptive-case-analysis-v1",
        "cache_dir": str(args.cache_dir.resolve()),
        "embedding_root": str(args.embedding_root.resolve()),
        "comparison_root": str(args.comparison_root.resolve()),
        "output_dir": str(output_dir),
        "queries": len(queries),
        "ranking_control": {
            "dense_cosine": True,
            "hierarchical_top3_file_scope": True,
            "fixed_control": "global Kp equal to rounded mean Adaptive-k Kp",
            "qrels_used_for_ranking": False,
        },
        "policies": report_policies,
    }
    _write_json(output_dir / "report.json", report)

    lines = [
        "# Per-query Adaptive-k cases versus fixed global budget",
        "",
        f"- Corpus: **{len(queries)} queries**; dense cosine; hierarchical top-3 file scope.",
        "- TH1: Adaptive returns fewer pages than the fixed global budget.",
        "- TH2: Adaptive returns more pages than the fixed global budget.",
        "- Qrels are used only to measure evidence recall after retrieval.",
        "",
        "| Policy | Case | Queries | Mean Adaptive Kp | Mean Fixed Kp | Adaptive recall | Fixed recall | Fixed−Adaptive | Key result |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for policy, item in report_policies.items():
        th1 = item["TH1_adaptive_less"]
        th2 = item["TH2_adaptive_more"]
        lines.extend(
            [
                f"| {policy} | TH1 adaptive < fixed Kp={item['fixed_budget']} | {th1.get('queries', 0)} | {th1.get('mean_adaptive_k', 0.0):.2f} | {th1.get('mean_fixed_k_actual', 0.0):.2f} | {th1.get('mean_adaptive_recall', 0.0):.2%} | {th1.get('mean_fixed_recall', 0.0):.2%} | {th1.get('mean_fixed_minus_adaptive_recall', 0.0):+.2%} | recall preserved {th1.get('recall_preservation_rate', 0.0):.2%} |",
                f"| {policy} | TH2 adaptive > fixed Kp={item['fixed_budget']} | {th2.get('queries', 0)} | {th2.get('mean_adaptive_k', 0.0):.2f} | {th2.get('mean_fixed_k_actual', 0.0):.2f} | {th2.get('mean_adaptive_recall', 0.0):.2%} | {th2.get('mean_fixed_recall', 0.0):.2%} | {th2.get('mean_fixed_minus_adaptive_recall', 0.0):+.2%} | adaptive gain {th2.get('adaptive_gain_rate', 0.0):.2%} |",
                f"| {policy} | same effective K | {item['same_effective_k'].get('queries', 0)} | — | — | {item['same_effective_k'].get('mean_adaptive_recall', 0.0):.2%} | {item['same_effective_k'].get('mean_fixed_recall', 0.0):.2%} | {item['same_effective_k'].get('mean_fixed_minus_adaptive_recall', 0.0):+.2%} | — |",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- TH1 tests whether Adaptive-k cuts below the trivial budget too aggressively; `recall_preservation_rate` is the fraction of those queries with no evidence-recall loss.",
            "- TH2 tests whether spending more pages than the trivial budget buys additional gold evidence; `adaptive_gain_rate` is the fraction with strictly higher recall.",
            "- Since both systems share the same ranking, any difference is caused only by the page budget allocation.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
