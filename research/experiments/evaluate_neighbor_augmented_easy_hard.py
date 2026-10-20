"""Combine easy/hard routing with OOF semantic-neighbour candidate memory.

This exploratory screen searches for policies with mean K below 70. Query
neighbours are outer-fold training queries saved by the OOF neighbour audit.
The no-gold arm uses only their cached retrieved pages. The optional gold arm
simulates a historical judged-query/click memory and never uses test qrels.
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
ADAPTIVE_DIR = ROOT / "experiments" / "adaptive-k-retrieval"
if str(ADAPTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTIVE_DIR))

from run_canonical_adaptive_methods import load_rows  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "work" / "neighbor_augmented_easy_hard_20261014"
MAX_RETURN_K = 120
RRF_C = 60.0


def recall_curve(ranking: Sequence[str], gold: set[str], max_k: int) -> np.ndarray:
    result = np.zeros(max_k + 1, dtype=float)
    found = 0
    for rank in range(1, max_k + 1):
        if rank <= len(ranking) and ranking[rank - 1] in gold:
            found += 1
        result[rank] = found / max(len(gold), 1)
    return result


def rank_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(len(values) - 1, 1)


def load_risks(qids: list[str]) -> dict[str, np.ndarray]:
    historical_path = ROOT / "data" / "work" / "adaptive_retrieval_20261009" / "score_confidence_gate.json"
    historical = np.asarray(json.loads(historical_path.read_text(encoding="utf-8"))["oof_probability"], dtype=float)

    best_name = "xgboost__score+neighbor_difficulty__high_confidence_50pct"
    best_map: dict[str, float] = {}
    best_path = ROOT / "data" / "work" / "easy_hard_model_ablation_20261014" / "per_query.jsonl"
    for line in best_path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["configuration"] == best_name:
            best_map[str(row["qid"])] = float(row["hard_probability"])

    safe_map: dict[str, float] = {}
    safe_path = ROOT / "data" / "work" / "safe_easy_exit_20261014" / "per_query.jsonl"
    for line in safe_path.open(encoding="utf-8"):
        row = json.loads(line)
        if int(row["cap"]) == 100:
            safe_map[str(row["qid"])] = float(row["risk_safe_exit_extra_trees"])

    if len(historical) != len(qids) or set(best_map) != set(qids) or set(safe_map) != set(qids):
        raise ValueError("OOF risk artifacts do not match current query IDs")
    best = np.asarray([best_map[qid] for qid in qids], dtype=float)
    safe = np.asarray([safe_map[qid] for qid in qids], dtype=float)
    return {
        "historical_hgb": historical,
        "neighbor_xgboost": best,
        "safe_exit_extra_trees": safe,
        "rank_blend_xgb_safe": 0.5 * rank_percentile(best) + 0.5 * rank_percentile(safe),
    }


def load_neighbors(qids: list[str]) -> dict[str, list[tuple[str, float]]]:
    path = ROOT / "data" / "work" / "query_neighbor_statistics_20261014" / "per_query.jsonl"
    result: dict[str, list[tuple[str, float]]] = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["method"] != "knn_10":
            continue
        result[str(row["qid"])] = list(zip(row["neighbor_qids"], row["neighbor_similarities"]))
    if set(result) != set(qids):
        raise ValueError("OOF neighbour artifact does not match current query IDs")
    return result


def candidate_ranking(
    qid: str,
    rows_by_qid: dict[str, dict[str, Any]],
    neighbours: dict[str, list[tuple[str, float]]],
    neighbor_count: int,
    neighbor_depth: int,
    protect: int,
    original_weight: float,
    gold_weight: float,
) -> tuple[list[str], int]:
    row = rows_by_qid[qid]
    selected = neighbours[qid][:neighbor_count]
    similarities = np.asarray([value for _, value in selected], dtype=float)
    if len(similarities):
        weights = np.exp(np.clip((similarities - np.max(similarities)) / 0.05, -50, 0))
        weights /= np.sum(weights) + 1e-12
    else:
        weights = similarities

    candidates = set(row["ranked"])
    original_score: dict[str, float] = {
        page_id: (1.0 / (RRF_C + rank)) / (1.0 / (RRF_C + 1.0))
        for rank, page_id in enumerate(row["ranked"], 1)
    }
    neighbor_score: dict[str, float] = {}
    neighbor_gold_score: dict[str, float] = {}
    for (neighbor_qid, _), weight in zip(selected, weights):
        neighbor = rows_by_qid[neighbor_qid]
        for rank, page_id in enumerate(neighbor["ranked"][:neighbor_depth], 1):
            candidates.add(page_id)
            neighbor_score[page_id] = neighbor_score.get(page_id, 0.0) + float(weight) / (RRF_C + rank)
        neighbor_gold = set(neighbor["gold"])
        for page_id in neighbor_gold:
            candidates.add(page_id)
            neighbor_gold_score[page_id] = neighbor_gold_score.get(page_id, 0.0) + float(weight) / max(len(neighbor_gold), 1)

    max_neighbor = max(neighbor_score.values(), default=1.0)
    max_gold = max(neighbor_gold_score.values(), default=1.0)
    protected = list(row["ranked"][:protect])
    protected_set = set(protected)
    remaining = candidates - protected_set

    def score(page_id: str) -> tuple[float, float, str]:
        retrieval = (
            original_weight * original_score.get(page_id, 0.0)
            + (1.0 - original_weight) * neighbor_score.get(page_id, 0.0) / max_neighbor
        )
        combined = (1.0 - gold_weight) * retrieval + gold_weight * neighbor_gold_score.get(page_id, 0.0) / max_gold
        return combined, original_score.get(page_id, 0.0), page_id

    tail = sorted(remaining, key=lambda page_id: (-score(page_id)[0], -score(page_id)[1], score(page_id)[2]))
    return protected + tail, len(candidates)


def build_policies(risks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    n = len(next(iter(risks.values())))
    for risk_name, risk in risks.items():
        descending = np.argsort(-risk, kind="stable")
        for easy_k in (10, 20, 30):
            for hard_k in (80, 90, 100, 120):
                for hard_fraction in (0.35, 0.40, 0.45, 0.50, 0.55, 0.56, 0.57, 0.58, 0.60, 0.65):
                    hard_count = int(round(hard_fraction * n))
                    choices = np.full(n, easy_k, dtype=int)
                    choices[descending[:hard_count]] = hard_k
                    if float(np.mean(choices)) < 70.0:
                        policies.append({
                            "name": f"{risk_name}__binary_{easy_k}_{hard_k}__hard{hard_fraction:.2f}",
                            "risk": risk_name,
                            "kind": "binary",
                            "easy_k": easy_k,
                            "hard_k": hard_k,
                            "hard_fraction": hard_fraction,
                            "choices": choices,
                            "mean_k": float(np.mean(choices)),
                        })

        for low_k, middle_k, high_k in ((10, 40, 100), (20, 50, 100), (10, 50, 120), (20, 60, 120), (30, 60, 100)):
            for high_fraction in (0.25, 0.35, 0.45, 0.55, 0.56, 0.57, 0.58):
                for middle_fraction in (0.20, 0.24, 0.25, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.35, 0.40):
                    if high_fraction + middle_fraction >= 1.0:
                        continue
                    high_count = int(round(high_fraction * n))
                    middle_count = int(round(middle_fraction * n))
                    choices = np.full(n, low_k, dtype=int)
                    choices[descending[:high_count]] = high_k
                    choices[descending[high_count : high_count + middle_count]] = middle_k
                    if float(np.mean(choices)) < 70.0:
                        policies.append({
                            "name": f"{risk_name}__ternary_{low_k}_{middle_k}_{high_k}__{high_fraction:.2f}_{middle_fraction:.2f}",
                            "risk": risk_name,
                            "kind": "ternary",
                            "levels": [low_k, middle_k, high_k],
                            "high_fraction": high_fraction,
                            "middle_fraction": middle_fraction,
                            "choices": choices,
                            "mean_k": float(np.mean(choices)),
                        })
    return policies


def evaluate_curves(curves: np.ndarray, choices: np.ndarray) -> float:
    return float(np.mean(curves[np.arange(len(choices)), choices]))


def serializable_policy(policy: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in policy.items() if key != "choices"}


def bootstrap_mean_interval(values: np.ndarray, samples: int = 5000) -> list[float]:
    rng = np.random.default_rng(20261014)
    n = len(values)
    means = np.asarray([
        np.mean(values[rng.integers(0, n, n)]) for _ in range(samples)
    ])
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = load_rows("hybrid", 200)
    qids = [str(row["qid"]) for row in rows]
    rows_by_qid = {str(row["qid"]): row for row in rows}
    risks = load_risks(qids)
    neighbours = load_neighbors(qids)
    policies = build_policies(risks)

    original_curves = np.vstack([
        recall_curve(row["ranked"], set(row["gold"]), MAX_RETURN_K) for row in rows
    ])
    original_results = []
    for policy in policies:
        original_results.append({
            "ranking": "original",
            "ranking_config": {},
            "policy": serializable_policy(policy),
            "mean_k": policy["mean_k"],
            "macro_recall": evaluate_curves(original_curves, policy["choices"]),
            "mean_candidate_pool": 200.0,
        })

    results: list[dict[str, Any]] = list(original_results)
    ranking_configs = [
        {
            "neighbor_count": neighbor_count,
            "neighbor_depth": neighbor_depth,
            "protect": protect,
            "original_weight": original_weight,
            "gold_weight": gold_weight,
        }
        for neighbor_count in (3, 5, 10)
        for neighbor_depth in (10, 30, 100)
        for protect in (10, 20, 30)
        for original_weight in (0.25, 0.50, 0.75)
        for gold_weight in (0.0, 0.15, 0.30)
    ]
    for config_index, config in enumerate(ranking_configs):
        curves = []
        pool_sizes = []
        for qid in qids:
            ranking, pool_size = candidate_ranking(
                qid, rows_by_qid, neighbours, **config
            )
            curves.append(recall_curve(ranking, set(rows_by_qid[qid]["gold"]), MAX_RETURN_K))
            pool_sizes.append(pool_size)
        curve_matrix = np.vstack(curves)
        for policy in policies:
            results.append({
                "ranking": "neighbor_augmented",
                "ranking_config": config,
                "policy": serializable_policy(policy),
                "mean_k": policy["mean_k"],
                "macro_recall": evaluate_curves(curve_matrix, policy["choices"]),
                "mean_candidate_pool": float(np.mean(pool_sizes)),
            })
        if (config_index + 1) % 50 == 0:
            print(f"screened {config_index + 1}/{len(ranking_configs)} ranking configurations")

    feasible = [row for row in results if row["mean_k"] < 70.0]
    best_overall = max(feasible, key=lambda row: row["macro_recall"])
    best_no_gold = max(
        (row for row in feasible if float(row["ranking_config"].get("gold_weight", 0.0)) == 0.0),
        key=lambda row: row["macro_recall"],
    )
    best_original = max(
        (row for row in feasible if row["ranking"] == "original"),
        key=lambda row: row["macro_recall"],
    )
    policy_by_name = {policy["name"]: policy for policy in policies}

    def detailed_metrics(result: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        choices = policy_by_name[result["policy"]["name"]]["choices"]
        recalls = []
        cap_recalls = []
        pools = []
        for qid, choice in zip(qids, choices):
            if result["ranking"] == "original":
                ranking = rows_by_qid[qid]["ranked"]
                pool_size = len(ranking)
            else:
                ranking, pool_size = candidate_ranking(
                    qid, rows_by_qid, neighbours, **result["ranking_config"]
                )
            curve = recall_curve(ranking, set(rows_by_qid[qid]["gold"]), MAX_RETURN_K)
            recalls.append(curve[int(choice)])
            cap_recalls.append(curve[100])
            pools.append(pool_size)
        values = np.asarray(recalls, dtype=float)
        cap_values = np.asarray(cap_recalls, dtype=float)
        source_metrics = {}
        for source in sorted({str(row["source"]) for row in rows}):
            mask = np.asarray([str(row["source"]) == source for row in rows])
            source_metrics[source] = {
                "queries": int(np.sum(mask)),
                "mean_k": float(np.mean(choices[mask])),
                "macro_recall": float(np.mean(values[mask])),
            }
        tier_diagnostics = {}
        for cutoff in sorted(set(int(value) for value in choices)):
            mask = choices == cutoff
            losses = cap_values[mask] - values[mask]
            tier_diagnostics[str(cutoff)] = {
                "queries": int(np.sum(mask)),
                "safe_precision_vs_k100": float(np.mean(losses <= 1e-12)),
                "mean_recall_loss_vs_k100": float(np.mean(losses)),
            }
        return {
            "mean_k": float(np.mean(choices)),
            "macro_recall": float(np.mean(values)),
            "mean_candidate_pool": float(np.mean(pools)),
            "by_source": source_metrics,
            "tier_diagnostics": tier_diagnostics,
        }, values

    historical_risk = risks["historical_hgb"]
    historical_choices = np.where(historical_risk >= 0.2, 100, 30)
    historical_values = original_curves[np.arange(len(rows)), historical_choices]
    historical_baseline = {
        "mean_k": float(np.mean(historical_choices)),
        "macro_recall": float(np.mean(historical_values)),
    }
    best_no_gold_details, best_no_gold_values = detailed_metrics(best_no_gold)
    best_overall_details, best_overall_values = detailed_metrics(best_overall)
    same_policy_original = original_curves[
        np.arange(len(rows)), policy_by_name[best_no_gold["policy"]["name"]]["choices"]
    ]
    validation = {
        "historical_baseline": historical_baseline,
        "best_without_gold_memory": best_no_gold_details,
        "best_overall": best_overall_details,
        "best_no_gold_delta_vs_historical": {
            "mean": float(np.mean(best_no_gold_values - historical_values)),
            "bootstrap_95ci": bootstrap_mean_interval(best_no_gold_values - historical_values),
        },
        "best_no_gold_delta_vs_same_policy_original_ranking": {
            "same_policy_original_recall": float(np.mean(same_policy_original)),
            "mean": float(np.mean(best_no_gold_values - same_policy_original)),
            "bootstrap_95ci": bootstrap_mean_interval(best_no_gold_values - same_policy_original),
        },
        "best_overall_delta_vs_historical": {
            "mean": float(np.mean(best_overall_values - historical_values)),
            "bootstrap_95ci": bootstrap_mean_interval(best_overall_values - historical_values),
        },
    }
    top_results = sorted(feasible, key=lambda row: (-row["macro_recall"], row["mean_k"]))[:100]
    output = {
        "protocol": {
            "benchmark": "Physics 80 + Industrial 75 + OHR-Bench 65",
            "queries": len(rows),
            "constraint": "mean K < 70",
            "neighbors": "same-source OOF Qwen query-embedding KNN; training-fold neighbours only",
            "no_gold_arm": "candidate pages from neighbour cached rankings only",
            "gold_arm": "adds training-neighbour qrels as historical relevance memory",
            "selection_warning": "Broad same-benchmark screen over ranking and routing hyperparameters; exploratory, not held-out confirmation.",
        },
        "search_space": {
            "ranking_configurations": len(ranking_configs),
            "routing_policies": len(policies),
            "evaluated_combinations": len(results),
        },
        "target_achieved_overall": bool(best_overall["macro_recall"] > 0.87 and best_overall["mean_k"] < 70),
        "target_achieved_without_gold": bool(best_no_gold["macro_recall"] > 0.87 and best_no_gold["mean_k"] < 70),
        "best_original_ranking": best_original,
        "best_without_gold_memory": best_no_gold,
        "best_overall": best_overall,
        "validation": validation,
        "top_results": top_results,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Neighbor-augmented easy/hard screen",
        "",
        "All reported rows satisfy mean K < 70. Neighbours are OOF training-fold queries; the gold-memory arm uses only neighbour training qrels.",
        "",
        "| Arm | Mean K | Recall | Candidate pool | Risk/policy | Ranking configuration |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, row in (
        ("Best original ranking", best_original),
        ("Best neighbor ranking, no gold", best_no_gold),
        ("Best overall", best_overall),
    ):
        lines.append(
            f"| {name} | {row['mean_k']:.2f} | {row['macro_recall']:.2%} | "
            f"{row['mean_candidate_pool']:.1f} | `{row['policy']['name']}` | "
            f"`{json.dumps(row['ranking_config'], sort_keys=True)}` |"
        )
    lines += [
        "",
        f"Target achieved overall: **{output['target_achieved_overall']}**.",
        f"Target achieved without historical gold memory: **{output['target_achieved_without_gold']}**.",
        "",
        "## Best no-gold arm by source",
        "",
        "| Source | Queries | Mean K | Recall |",
        "|---|---:|---:|---:|",
    ]
    for source, metric in best_no_gold_details["by_source"].items():
        lines.append(
            f"| {source} | {metric['queries']} | {metric['mean_k']:.2f} | {metric['macro_recall']:.2%} |"
        )
    lines += [
        "",
        "## Easy/medium/hard tier diagnostics",
        "",
        "| Returned K | Queries | Safe precision versus K=100 | Mean recall loss versus K=100 |",
        "|---:|---:|---:|---:|",
    ]
    for cutoff, metric in best_no_gold_details["tier_diagnostics"].items():
        lines.append(
            f"| {cutoff} | {metric['queries']} | {metric['safe_precision_vs_k100']:.2%} | "
            f"{metric['mean_recall_loss_vs_k100']:.2%} |"
        )
    delta = validation["best_no_gold_delta_vs_historical"]
    ranking_delta = validation["best_no_gold_delta_vs_same_policy_original_ranking"]
    lines += [
        "",
        f"Versus the historical 72.64/86.42 policy, the best no-gold arm changes recall by {delta['mean']:+.2%} (paired bootstrap 95% CI [{delta['bootstrap_95ci'][0]:+.2%}, {delta['bootstrap_95ci'][1]:+.2%}]).",
        f"With the same routing decisions on the original ranking, recall is {ranking_delta['same_policy_original_recall']:.2%}; neighbor candidate augmentation contributes {ranking_delta['mean']:+.2%}.",
        "",
        "## Guardrail",
        "",
        "This is a broad OOF screen, not nested selection. A row above 87% must be frozen and rerun with outer-fold-only hyperparameter selection or a new query set before being treated as a validated improvement.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "best_original": best_original,
        "best_no_gold": best_no_gold,
        "best_overall": best_overall,
        "search_space": output["search_space"],
    }, indent=2))
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
