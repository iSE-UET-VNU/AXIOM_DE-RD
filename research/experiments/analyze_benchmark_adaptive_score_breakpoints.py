"""Audit dense-score breakpoints behind the benchmark Adaptive-k decisions.

The existing largest-gap policies choose a page budget from the sorted dense
cosine scores.  This script checks whether that decision is supported by a
pronounced score discontinuity or whether the selected gap is only marginally
larger than the rest of the score tail.

It reuses cached embeddings and existing Adaptive-k/fixed-budget rows.  It
does not encode, call an API, rerun OCR, or use qrels to rank pages.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
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
DEFAULT_CASE_ROOT = DEFAULT_EMBEDDING_ROOT / "adaptive_case_analysis"
DEFAULT_OUTPUT = DEFAULT_EMBEDDING_ROOT / "score_breakpoint_analysis"


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


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _distribution(values: Sequence[float | None]) -> dict[str, Any]:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return {"n": 0}
    return {
        "n": len(usable),
        "mean": _mean(usable),
        "median": statistics.median(usable),
        "p10": _quantile(usable, 0.10),
        "p25": _quantile(usable, 0.25),
        "p75": _quantile(usable, 0.75),
        "p90": _quantile(usable, 0.90),
        "p95": _quantile(usable, 0.95),
        "min": min(usable),
        "max": max(usable),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def _auc_positive(positive: Sequence[float | None], negative: Sequence[float | None]) -> float | None:
    """Pairwise AUC: probability a positive row has the larger metric."""
    positive_values = [float(value) for value in positive if value is not None]
    negative_values = [float(value) for value in negative if value is not None]
    if not positive_values or not negative_values:
        return None
    wins = 0.0
    for positive_value in positive_values:
        for negative_value in negative_values:
            if positive_value > negative_value:
                wins += 1.0
            elif positive_value == negative_value:
                wins += 0.5
    return wins / (len(positive_values) * len(negative_values))


def _separation(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    contrasts = (
        ("TH1_loss_vs_preserved", "TH1_recall_loss", "TH1_recall_preserved"),
        ("TH2_gain_vs_no_gain", "TH2_recall_gain", "TH2_no_recall_gain"),
    )
    metrics = (
        ("trigger_gap", "trigger_gap"),
        ("trigger_gap_over_second_largest", "trigger_gap_over_second_largest"),
        ("adaptive_boundary_gap", "adaptive_boundary_gap"),
        ("fixed_boundary_gap", "fixed_boundary_gap"),
    )
    result: dict[str, Any] = {}
    for contrast_name, positive_name, negative_name in contrasts:
        positive_rows = groups.get(positive_name, [])
        negative_rows = groups.get(negative_name, [])
        result[contrast_name] = {
            metric_name: {
                "positive_group": positive_name,
                "negative_group": negative_name,
                "auc": _auc_positive(
                    [row.get(metric_key) for row in positive_rows],
                    [row.get(metric_key) for row in negative_rows],
                ),
            }
            for metric_name, metric_key in metrics
        }
    return result


def _boundary_gap(scores: Sequence[float], cut: int) -> float | None:
    """Return s[cut] - s[cut+1] for a 1-indexed retained-page cut."""
    if cut < 1 or cut >= len(scores):
        return None
    return max(0.0, float(scores[cut - 1]) - float(scores[cut]))


def _gap_profile(scores: Sequence[float], minimum_rank: int, buffer: int) -> dict[str, Any]:
    """Describe the selected gap and the score discontinuity at the final cut."""
    if len(scores) <= minimum_rank:
        return {
            "candidate_gap_count": 0,
            "trigger_rank": len(scores),
            "trigger_gap": None,
            "adaptive_cut": len(scores),
            "adaptive_boundary_gap": None,
            "second_largest_gap": None,
            "median_other_gap": None,
            "mean_other_gap": None,
            "trigger_gap_over_median_other": None,
            "trigger_gap_over_second_largest": None,
            "trigger_gap_z_other": None,
        }

    # rank is 1-indexed: gap rank r is score[r] - score[r+1].
    candidates = [
        (rank, max(0.0, float(scores[rank - 1]) - float(scores[rank])))
        for rank in range(minimum_rank, len(scores))
    ]
    trigger_rank, trigger_gap = max(candidates, key=lambda item: (item[1], -item[0]))
    other_gaps = [gap for rank, gap in candidates if rank != trigger_rank]
    sorted_other = sorted(other_gaps, reverse=True)
    median_other = statistics.median(other_gaps) if other_gaps else None
    mean_other = _mean(other_gaps) if other_gaps else None
    std_other = statistics.pstdev(other_gaps) if len(other_gaps) > 1 else 0.0
    trigger_z = (
        (trigger_gap - mean_other) / std_other
        if mean_other is not None and std_other > 1e-12
        else None
    )
    adaptive_cut = min(len(scores), trigger_rank + buffer)
    return {
        "candidate_gap_count": len(candidates),
        "trigger_rank": int(trigger_rank),
        "trigger_gap": float(trigger_gap),
        "adaptive_cut": int(adaptive_cut),
        "adaptive_boundary_gap": _boundary_gap(scores, adaptive_cut),
        "second_largest_gap": float(sorted_other[0]) if sorted_other else None,
        "median_other_gap": float(median_other) if median_other is not None else None,
        "mean_other_gap": float(mean_other) if mean_other is not None else None,
        "trigger_gap_over_median_other": (
            _safe_ratio(trigger_gap, median_other) if median_other is not None else None
        ),
        "trigger_gap_over_second_largest": (
            _safe_ratio(trigger_gap, sorted_other[0]) if sorted_other else None
        ),
        "trigger_gap_z_other": trigger_z,
    }


def _near_cut_context(
    scores: Sequence[float],
    ranked_page_ids: Sequence[str],
    gold_pages: set[str],
    empty_page_ids: set[str],
    ranks: Sequence[int],
) -> list[dict[str, Any]]:
    usable = sorted({rank for rank in ranks if 1 <= rank <= len(scores)})
    context: list[dict[str, Any]] = []
    for rank in usable:
        next_score = float(scores[rank]) if rank < len(scores) else None
        context.append(
            {
                "rank": rank,
                "page_id": ranked_page_ids[rank - 1],
                "score": float(scores[rank - 1]),
                "gap_after": _boundary_gap(scores, rank),
                "is_gold": ranked_page_ids[rank - 1] in gold_pages,
                "is_empty_page": ranked_page_ids[rank - 1] in empty_page_ids,
                "next_score": next_score,
            }
        )
    return context


def _case_bucket(row: Mapping[str, Any]) -> str:
    case = str(row["case"])
    if case == "adaptive_less":
        return "TH1_recall_loss" if float(row["fixed_minus_adaptive_recall"]) > 1e-12 else "TH1_recall_preserved"
    if case == "adaptive_more":
        return "TH2_recall_gain" if float(row["fixed_minus_adaptive_recall"]) < -1e-12 else "TH2_no_recall_gain"
    return "same_effective_k"


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"queries": 0}
    return {
        "queries": len(rows),
        "mean_adaptive_k": _mean([float(row["adaptive_k"]) for row in rows]),
        "mean_fixed_k": _mean([float(row["fixed_k"]) for row in rows]),
        "trigger_gap": _distribution([row["trigger_gap"] for row in rows]),
        "trigger_gap_when_next_page_nonempty": _distribution(
            [row["trigger_gap"] if not row["trigger_next_page_is_empty"] else None for row in rows]
        ),
        "trigger_score": _distribution([row["trigger_score"] for row in rows]),
        "score_after_trigger": _distribution([row["score_after_trigger"] for row in rows]),
        "adaptive_boundary_gap": _distribution([row["adaptive_boundary_gap"] for row in rows]),
        "score_at_adaptive_cut": _distribution([row["score_at_adaptive_cut"] for row in rows]),
        "score_after_adaptive_cut": _distribution([row["score_after_adaptive_cut"] for row in rows]),
        "fixed_boundary_gap": _distribution([row["fixed_boundary_gap"] for row in rows]),
        "score_at_fixed_cut": _distribution([row["score_at_fixed_cut"] for row in rows]),
        "score_after_fixed_cut": _distribution([row["score_after_fixed_cut"] for row in rows]),
        "trigger_gap_over_median_other": _distribution(
            [row["trigger_gap_over_median_other"] for row in rows]
        ),
        "trigger_gap_over_second_largest": _distribution(
            [row["trigger_gap_over_second_largest"] for row in rows]
        ),
        "trigger_gap_z_other": _distribution([row["trigger_gap_z_other"] for row in rows]),
        "strong_gap_ratio_ge_2": sum(
            row["trigger_gap_over_median_other"] is not None
            and float(row["trigger_gap_over_median_other"]) >= 2.0
            for row in rows
        ),
        "strong_gap_ratio_ge_5": sum(
            row["trigger_gap_over_median_other"] is not None
            and float(row["trigger_gap_over_median_other"]) >= 5.0
            for row in rows
        ),
        "second_ratio_ge_1_5": sum(
            row["trigger_gap_over_second_largest"] is not None
            and float(row["trigger_gap_over_second_largest"]) >= 1.5
            for row in rows
        ),
        "second_ratio_ge_2": sum(
            row["trigger_gap_over_second_largest"] is not None
            and float(row["trigger_gap_over_second_largest"]) >= 2.0
            for row in rows
        ),
        "mean_gold_pages_between_budgets": _mean(
            [float(row["gold_pages_between_budgets"]) for row in rows]
        ),
        "trigger_next_page_empty_count": sum(bool(row["trigger_next_page_is_empty"]) for row in rows),
        "adaptive_cut_next_page_empty_count": sum(bool(row["adaptive_cut_next_page_is_empty"]) for row in rows),
        "fixed_cut_next_page_empty_count": sum(bool(row["fixed_cut_next_page_is_empty"]) for row in rows),
    }


def _compact_stats(summary: Mapping[str, Any], field: str) -> str:
    values = summary.get(field, {})
    if not values or not values.get("n"):
        return "n/a"
    return f"median={values['median']:.5f}; p25={values['p25']:.5f}; p75={values['p75']:.5f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rank", type=int, default=100)
    parser.add_argument("--minimum-rank", type=int, default=10)
    args = parser.parse_args(argv)

    pages = _read_jsonl(args.cache_dir / "parsing/pages.jsonl")
    queries = _read_jsonl(ROOT / "data/raw/BENCHMARK/queries.jsonl")
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in _read_jsonl(args.cache_dir / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))

    page_ids = [str(row["page_id"]) for row in pages]
    page_docs = [str(row["doc_id"]) for row in pages]
    page_texts = [str(row.get("text") or "") for row in pages]
    query_ids = [str(row["query_id"]) for row in queries]
    query_positions = {query_id: index for index, query_id in enumerate(query_ids)}
    page_positions_by_file: dict[str, list[int]] = defaultdict(list)
    for position, doc_id in enumerate(page_docs):
        page_positions_by_file[doc_id].append(position)

    page_vectors = np.load(args.embedding_root / "embeddings/page_embeddings.npy", mmap_mode="r")
    query_vectors = np.load(args.embedding_root / "embeddings/query_embeddings.npy", mmap_mode="r")
    if page_vectors.shape[0] != len(pages) or query_vectors.shape[0] != len(queries):
        raise RuntimeError("Dense embedding/page/query count mismatch")
    empty_positions = [position for position, text in enumerate(page_texts) if not text.strip()]
    empty_page_ids = {page_ids[position] for position in empty_positions}

    policy_specs = {
        "hier_largest_gap": {
            "adaptive": args.embedding_root / "per_query_hier_largest_gap.jsonl",
            "cases": args.case_root / "per_query_hier_largest_gap.jsonl",
            "fixed": args.embedding_root / "global_mean_budget_comparison" / "per_query_hier_largest_gap_fixed_global_k27.jsonl",
            "buffer": 0,
            "fixed_budget": 27,
        },
        "hier_largest_gap_buffer5": {
            "adaptive": args.embedding_root / "per_query_hier_largest_gap_buffer5.jsonl",
            "cases": args.case_root / "per_query_hier_largest_gap_buffer5.jsonl",
            "fixed": args.embedding_root / "global_mean_budget_comparison" / "per_query_hier_largest_gap_buffer5_fixed_global_k32.jsonl",
            "buffer": 5,
            "fixed_budget": 32,
        },
    }

    report_policies: dict[str, Any] = {}
    output_dir = args.output_dir.resolve()
    for policy, spec in policy_specs.items():
        adaptive_rows = {str(row["query_id"]): row for row in _read_jsonl(spec["adaptive"])}
        case_rows = {str(row["query_id"]): row for row in _read_jsonl(spec["cases"])}
        fixed_rows = {str(row["query_id"]): row for row in _read_jsonl(spec["fixed"])}
        if set(adaptive_rows) != set(case_rows) or set(adaptive_rows) != set(fixed_rows):
            raise RuntimeError(f"{policy}: query ID sets do not match")

        enriched: list[dict[str, Any]] = []
        for query_id in query_ids:
            adaptive = adaptive_rows[query_id]
            case = case_rows[query_id]
            fixed = fixed_rows[query_id]
            # Match the original adaptive run exactly: it computed one
            # page-vector/query-vector product at a time.  Batched BLAS can
            # differ by a few ulps and change the stable order of near ties.
            scores = np.asarray(page_vectors @ query_vectors[query_positions[query_id]], dtype=np.float32).copy()
            for position in empty_positions:
                scores[position] = -1.0
            selected_files = [str(file_id) for file_id in adaptive["selected_files"]]
            selected_positions = [
                position
                for file_id in selected_files
                for position in page_positions_by_file[file_id]
            ]
            selected_positions.sort(key=lambda position: (-float(scores[position]), position))
            ranked_positions = selected_positions[: args.max_rank]
            ranked_page_ids = [page_ids[position] for position in ranked_positions]
            ranked_scores = [float(scores[position]) for position in ranked_positions]
            adaptive_k = len(adaptive["returned_pages"])
            fixed_k = len(fixed["returned_pages"])
            expected_adaptive = ranked_page_ids[:adaptive_k]
            expected_fixed = ranked_page_ids[:fixed_k]
            if expected_adaptive != [str(page_id) for page_id in adaptive["returned_pages"]]:
                raise RuntimeError(f"{policy}: reconstructed adaptive prefix mismatch for {query_id}")
            if expected_fixed != [str(page_id) for page_id in fixed["returned_pages"]]:
                raise RuntimeError(f"{policy}: reconstructed fixed prefix mismatch for {query_id}")

            gold_pages = qrels.get(query_id, set())
            profile = _gap_profile(ranked_scores, args.minimum_rank, int(spec["buffer"]))
            trigger_rank = int(profile["trigger_rank"])
            context_ranks = list(range(max(1, adaptive_k - 3), min(len(ranked_scores), adaptive_k + 3) + 1))
            context_ranks += list(range(max(1, fixed_k - 2), min(len(ranked_scores), fixed_k + 2) + 1))
            context_ranks += list(range(max(1, trigger_rank - 2), min(len(ranked_scores), trigger_rank + 2) + 1))
            gold_ranks = [rank for rank, page_id in enumerate(ranked_page_ids, start=1) if page_id in gold_pages]
            adaptive_found = len(set(expected_adaptive) & gold_pages)
            fixed_found = len(set(expected_fixed) & gold_pages)
            if adaptive_found != int(case["adaptive_found"]) or fixed_found != int(case["fixed_found"]):
                raise RuntimeError(f"{policy}: qrel count mismatch for {query_id}")

            bucket = _case_bucket(case)
            enriched.append(
                {
                    "query_id": query_id,
                    "source": str(case["source"]),
                    "query": str(case["query"]),
                    "case": str(case["case"]),
                    "outcome_bucket": bucket,
                    "adaptive_k": adaptive_k,
                    "fixed_k": fixed_k,
                    "adaptive_found": adaptive_found,
                    "fixed_found": fixed_found,
                    "fixed_minus_adaptive_recall": float(case["fixed_minus_adaptive_recall"]),
                    "gold_pages": len(gold_pages),
                    "gold_pages_between_budgets": abs(fixed_found - adaptive_found),
                    "gold_ranks_within_ranked_pool": gold_ranks,
                    "trigger_rank": profile["trigger_rank"],
                    "trigger_gap": profile["trigger_gap"],
                    "adaptive_boundary_gap": profile["adaptive_boundary_gap"],
                    "fixed_boundary_gap": _boundary_gap(ranked_scores, fixed_k),
                    "second_largest_gap": profile["second_largest_gap"],
                    "median_other_gap": profile["median_other_gap"],
                    "trigger_gap_over_median_other": profile["trigger_gap_over_median_other"],
                    "trigger_gap_over_second_largest": profile["trigger_gap_over_second_largest"],
                    "trigger_gap_z_other": profile["trigger_gap_z_other"],
                    "trigger_score": (
                        ranked_scores[trigger_rank - 1]
                        if 1 <= trigger_rank <= len(ranked_scores)
                        else None
                    ),
                    "score_after_trigger": (
                        ranked_scores[trigger_rank]
                        if trigger_rank < len(ranked_scores)
                        else None
                    ),
                    "score_at_adaptive_cut": (
                        ranked_scores[adaptive_k - 1]
                        if 1 <= adaptive_k <= len(ranked_scores)
                        else None
                    ),
                    "score_after_adaptive_cut": (
                        ranked_scores[adaptive_k]
                        if adaptive_k < len(ranked_scores)
                        else None
                    ),
                    "score_at_fixed_cut": (
                        ranked_scores[fixed_k - 1]
                        if 1 <= fixed_k <= len(ranked_scores)
                        else None
                    ),
                    "score_after_fixed_cut": (
                        ranked_scores[fixed_k]
                        if fixed_k < len(ranked_scores)
                        else None
                    ),
                    "trigger_next_page_is_empty": (
                        ranked_page_ids[trigger_rank] in empty_page_ids
                        if trigger_rank < len(ranked_page_ids)
                        else None
                    ),
                    "adaptive_cut_next_page_is_empty": (
                        ranked_page_ids[adaptive_k] in empty_page_ids
                        if adaptive_k < len(ranked_page_ids)
                        else None
                    ),
                    "fixed_cut_next_page_is_empty": (
                        ranked_page_ids[fixed_k] in empty_page_ids
                        if fixed_k < len(ranked_page_ids)
                        else None
                    ),
                    "score_context": _near_cut_context(
                        ranked_scores,
                        ranked_page_ids,
                        gold_pages,
                        empty_page_ids,
                        context_ranks,
                    ),
                }
            )

        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in enriched:
            groups[str(row["outcome_bucket"])].append(row)
        nonempty_trigger_groups = {
            name: [row for row in rows if row["trigger_next_page_is_empty"] is False]
            for name, rows in groups.items()
        }
        report_policies[policy] = {
            "fixed_budget": int(spec["fixed_budget"]),
            "buffer": int(spec["buffer"]),
            "queries": len(enriched),
            "ranked_pool_depth": args.max_rank,
            "minimum_gap_rank": args.minimum_rank,
            "all": _group_summary(enriched),
            "groups": {name: _group_summary(rows) for name, rows in sorted(groups.items())},
            "separation": _separation(groups),
            "separation_nonempty_trigger": _separation(nonempty_trigger_groups),
            "reconstruction_prefix_mismatch_count": 0,
        }
        _write_jsonl(output_dir / f"per_query_{policy}.jsonl", enriched)

    report = {
        "contract_version": "benchmark-adaptive-score-breakpoint-v1",
        "cache_dir": str(args.cache_dir.resolve()),
        "embedding_root": str(args.embedding_root.resolve()),
        "case_root": str(args.case_root.resolve()),
        "output_dir": str(output_dir),
        "queries": len(queries),
        "pages": len(pages),
        "ranking": {
            "score": "dense cosine over cached page/query embeddings",
            "file_scope": "same top-3 files stored by each Adaptive-k row",
            "ranked_pool_depth": args.max_rank,
            "minimum_gap_rank": args.minimum_rank,
            "qrels_used_for_ranking": False,
        },
        "policies": report_policies,
    }
    _write_json(output_dir / "report.json", report)

    lines = [
        "# Adaptive-k score breakpoint audit",
        "",
        f"- Corpus: **{len(queries)} queries**; dense cosine; same hierarchical top-3 file scope as the cached runs.",
        f"- Score pool: top-{args.max_rank} pages per query; candidate gaps start after rank {args.minimum_rank}.",
        "- Qrels are used only to label recall outcomes, never to compute scores or choose pages.",
        "- `trigger_gap` is the largest eligible adjacent score drop. `adaptive_boundary_gap` is the gap at the actual returned-page cutoff.",
        "- Important: the trigger gap is always the largest gap by construction, so within-query percentile is not evidence of a real breakpoint. The useful diagnostics are absolute size and ratios against the other gaps.",
        "",
        "## Outcome-group distributions",
        "",
        "| Policy | Outcome group | Queries | Trigger gap median [p25, p75] | Trigger/2nd-gap median [p25, p75] | Actual-cut gap median | Ratio >=1.5 | Ratio >=2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, item in report_policies.items():
        for group_name, group in item["groups"].items():
            trigger = group.get("trigger_gap", {})
            ratio = group.get("trigger_gap_over_second_largest", {})
            boundary = group.get("adaptive_boundary_gap", {})
            if trigger.get("n"):
                trigger_text = f"{trigger['median']:.5f} [{trigger['p25']:.5f}, {trigger['p75']:.5f}]"
            else:
                trigger_text = "n/a"
            if ratio.get("n"):
                ratio_text = f"{ratio['median']:.2f} [{ratio['p25']:.2f}, {ratio['p75']:.2f}]"
            else:
                ratio_text = "n/a"
            boundary_text = f"{boundary['median']:.5f}" if boundary.get("n") else "n/a"
            lines.append(
                f"| {policy} | {group_name} | {group.get('queries', 0)} | {trigger_text} | {ratio_text} | {boundary_text} | {group.get('second_ratio_ge_1_5', 0)} | {group.get('second_ratio_ge_2', 0)} |"
            )
    lines.extend(
        [
            "",
            "## Reading the result",
            "",
            "- TH1 groups are split into `TH1_recall_preserved` and `TH1_recall_loss`.",
            "- TH2 groups are split into `TH2_recall_gain` and `TH2_no_recall_gain`.",
            "- The report table uses largest-gap / second-largest-gap. A ratio near 1 means the selected gap is not much larger than the runner-up; a high ratio is stronger breakpoint evidence.",
            "- The JSON also contains largest-gap / median-other-gap, but that measure is inflated when most tail gaps are tiny and is not used as the main breakpoint test.",
            "- For `buffer5`, the trigger gap is intentionally five ranks before the actual cutoff, so the actual-cut gap must be inspected separately.",
            "",
            "## Empty-page artifact check",
            "",
            "- Empty/placeholder pages are assigned cosine score `-1.0` by the cached retrieval run. A large gap whose next page is empty is therefore a preparation artifact, not evidence that the query's relevant evidence has ended.",
            "",
            "| Policy | Outcome group | Queries | Trigger next page empty | Non-empty-trigger gap median |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for policy, item in report_policies.items():
        for group_name, group in item["groups"].items():
            nonempty = group.get("trigger_gap_when_next_page_nonempty", {})
            count = group.get("trigger_next_page_empty_count", 0)
            fraction = count / group["queries"] if group.get("queries") else 0.0
            median = f"{nonempty['median']:.5f}" if nonempty.get("n") else "n/a"
            lines.append(
                f"| {policy} | {group_name} | {group.get('queries', 0)} | {count} ({fraction:.1%}) | {median} |"
            )
    lines.extend(
        [
            "",
            "## Score levels around the cut",
            "",
            "| Policy | Outcome group | Trigger score median | Score after trigger median | Actual-cut score median | Score after actual cut median |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for policy, item in report_policies.items():
        for group_name, group in item["groups"].items():
            trigger_score = group.get("trigger_score", {})
            after_trigger = group.get("score_after_trigger", {})
            cut_score = group.get("score_at_adaptive_cut", {})
            after_cut = group.get("score_after_adaptive_cut", {})
            lines.append(
                f"| {policy} | {group_name} | {trigger_score.get('median', 0.0):.5f} | {after_trigger.get('median', 0.0):.5f} | {cut_score.get('median', 0.0):.5f} | {after_cut.get('median', 0.0):.5f} |"
            )
    lines.extend(
        [
            "",
            "## Separation from score features",
            "",
            "- AUC is computed pairwise: probability that a positive outcome group has a larger score feature than the negative group. AUC **0.50** means no separation; values close to 0.50 are consistent with a weak or non-useful signal.",
            "",
            "| Policy | Contrast | Metric | AUC (all) | AUC (non-empty trigger) |",
            "|---|---|---|---:|---:|",
        ]
    )
    for policy, item in report_policies.items():
        for contrast, metrics in item["separation"].items():
            for metric, result in metrics.items():
                auc = result.get("auc")
                clean_auc = item["separation_nonempty_trigger"].get(contrast, {}).get(metric, {}).get("auc")
                auc_text = f"{auc:.3f}" if auc is not None else "n/a"
                clean_auc_text = f"{clean_auc:.3f}" if clean_auc is not None else "n/a"
                lines.append(f"| {policy} | {contrast} | {metric} | {auc_text} | {clean_auc_text} |")
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
