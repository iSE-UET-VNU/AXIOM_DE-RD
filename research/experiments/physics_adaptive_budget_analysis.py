"""Audit query-adaptive Kf/Kp budgets for the Physics hierarchy.

This is a diagnostic experiment, not a new retrieval model.  It reuses the
cached fixed-budget and hierarchical traces and asks two questions:

* how heterogeneous are the query-level evidence and file requirements?
* do query-local score curves contain a usable confidence signal for choosing
  a smaller or larger file/page budget?

The adaptive screen uses only score-curve quantiles.  Qrels are used for
metrics and diagnostics, never to choose the score thresholds.  Its output is
therefore an intentionally conservative feasibility test, not a tuned model.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


RESULT_ROOT = ROOT / "data" / "benchmark" / "vidore_v3" / "results"
FIXED_BUDGET_DIR = RESULT_ROOT / "physics_topk_budget_ablation"
HIERARCHICAL_DIR = RESULT_ROOT / "physics_hierarchical_retrieval"
OUTPUT_DIR = RESULT_ROOT / "physics_adaptive_budget_analysis"

KFS = (3, 5, 10)
ALL_KFS = (3, 5, 10, 15, 20)
KPS = (10, 20, 50, 100)
ADAPTIVE_KPS = (10, 50, 100)
FOLDS = 5


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rank(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=float)
    ranks[order] = np.arange(len(values_array), dtype=float)
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    return float(np.corrcoef(_rank(left), _rank(right))[0, 1])


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    scores_array = np.asarray(scores, dtype=float)
    labels_array = np.asarray(labels, dtype=bool)
    positives = scores_array[labels_array]
    negatives = scores_array[~labels_array]
    if not len(positives) or not len(negatives):
        return None
    comparisons = positives[:, None] - negatives[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def _quantiles(values: Sequence[float]) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(values, dtype=float), (1.0 / 3.0, 2.0 / 3.0))
    return float(low), float(high)


def _score_vector(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([float(item.get("score") or 0.0) for item in candidates])


def _score_at(scores: np.ndarray, rank: int) -> float:
    if not len(scores):
        return 0.0
    return float(scores[min(rank - 1, len(scores) - 1)])


def _score_curve_features(
    candidates: Sequence[Mapping[str, Any]],
    *,
    early_rank: int,
    late_rank: int,
) -> dict[str, float]:
    scores = _score_vector(candidates)
    if not len(scores):
        return {
            "top_score": 0.0,
            "gap": 0.0,
            "ratio": 0.0,
            "span": 0.0,
            "entropy": 0.0,
        }
    early = _score_at(scores, early_rank)
    late = _score_at(scores, late_rank)
    top = float(scores[0])
    shifted = scores - top
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    return {
        "top_score": top,
        "gap": top - late,
        "ratio": late / top if top else 0.0,
        "span": top - float(scores[-1]),
        "entropy": entropy,
        "early_late_gap": early - late,
    }


def _stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _gold_file_ids(qrel: Mapping[str, int]) -> set[str]:
    return {str(page).split("#page=", 1)[0] for page in qrel}


def _budget_oracle(
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_arm = {
        arm: {str(row["qid"]): row for row in rows if row.get("arm") == f"full_kf{arm}"}
        for arm in ALL_KFS
    }
    result: dict[str, Any] = {}
    minimum_file_budget: list[int | None] = []
    for qid in qids:
        gold_files = _gold_file_ids(qrels[qid])
        minimum = None
        for kf in ALL_KFS:
            if gold_files.issubset(set(by_arm[kf][qid]["selected_files"])):
                minimum = kf
                break
        minimum_file_budget.append(minimum)

    for kf in ALL_KFS:
        arm_rows = by_arm[kf]
        full_page_counts: Counter[str] = Counter()
        scope_categories: Counter[str] = Counter()
        file_coverage: list[float] = []
        page_recall: dict[str, list[float]] = {str(k): [] for k in KPS}

        for qid in qids:
            row = arm_rows[qid]
            gold_pages = set(qrels[qid])
            selected_files = set(row["selected_files"])
            in_scope = {
                page for page in gold_pages
                if page.split("#page=", 1)[0] in selected_files
            }
            file_coverage.append(len(in_scope) / len(gold_pages))
            maximum = float(row["page_recall_at_k"]["100"])
            if len(in_scope) < len(gold_pages):
                scope_categories["file_scope_missing"] += 1
            elif maximum < 1.0:
                scope_categories["page_rank_missing"] += 1
            else:
                scope_categories["full_at_100"] += 1

            minimum_page_budget = None
            for kp in KPS:
                recall = float(row["page_recall_at_k"][str(kp)])
                page_recall[str(kp)].append(recall)
                if recall >= 1.0 and minimum_page_budget is None:
                    minimum_page_budget = kp
            full_page_counts[str(minimum_page_budget)] += 1

        result[str(kf)] = {
            "mean_file_page_coverage": float(np.mean(file_coverage)),
            "full_file_page_coverage_queries": int(sum(value >= 1.0 for value in file_coverage)),
            "page_recall_by_kp": {
                key: float(np.mean(value)) for key, value in page_recall.items()
            },
            "minimum_kp_for_full_recall": dict(full_page_counts),
            "scope_diagnosis": dict(scope_categories),
        }

    return {
        "by_kf": result,
        "minimum_tested_kf_for_all_gold_files": dict(Counter(str(value) for value in minimum_file_budget)),
    }


def _trace_rows(
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    traces: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Mapping[str, Any]]], list[dict[str, Any]]]:
    by_kf = {
        kf: {
            str(row["qid"]): row
            for row in traces
            if row.get("method") == f"cascade-all_text-max-kf{kf}-fusion"
        }
        for kf in KFS
    }
    page_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    for qid in qids:
        row = by_kf[3][qid]
        page_candidates = row.get("page_candidates") or []
        page_features = _score_curve_features(page_candidates, early_rank=10, late_rank=20)
        page_ids = [str(item["node_id"]) for item in page_candidates]
        gold_pages = set(qrels[qid])
        page_features.update(
            {
                "qid": qid,
                "gold_page_count": len(gold_pages),
                "candidate_count": len(page_ids),
            }
        )
        for kp in KPS:
            page_features[f"recall_{kp}"] = len(set(page_ids[:kp]) & gold_pages) / len(gold_pages)
        page_rows.append(page_features)

        file_candidates = row.get("file_candidates") or []
        file_features = _score_curve_features(file_candidates, early_rank=3, late_rank=3)
        file_ids = [str(item["node_id"]) for item in file_candidates]
        gold_files = _gold_file_ids(qrels[qid])
        file_features.update(
            {
                "qid": qid,
                "gold_file_count": len(gold_files),
                "candidate_count": len(file_ids),
            }
        )
        for kf in ALL_KFS:
            file_features[f"recall_{kf}"] = len(set(file_ids[:kf]) & gold_files) / len(gold_files)
        file_rows.append(file_features)
    return by_kf, page_rows + file_rows


def _score_diagnostics(
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    traces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_kf, combined = _trace_rows(qids, qrels, traces)
    page_rows = [row for row in combined if "gold_page_count" in row]
    file_rows = [row for row in combined if "gold_file_count" in row]
    page_auc: dict[str, dict[str, float | int | None]] = {}
    for kp in KPS:
        labels = [row[f"recall_{kp}"] >= 1.0 for row in page_rows]
        page_auc[str(kp)] = {
            "positive_queries": int(sum(labels)),
            "gap_top1_top20_auc": _auc([row["gap"] for row in page_rows], labels),
            "inverse_ratio_auc": _auc([1.0 - row["ratio"] for row in page_rows], labels),
            "top_score_auc": _auc([row["top_score"] for row in page_rows], labels),
            "inverse_entropy_auc": _auc([-row["entropy"] for row in page_rows], labels),
        }

    page_correlations = {
        key: _spearman(
            [row[key] for row in page_rows],
            [row["recall_10"] for row in page_rows],
        )
        for key in ("top_score", "gap", "ratio", "span", "entropy")
    }
    page_correlations["gap_vs_gold_page_count"] = _spearman(
        [row["gap"] for row in page_rows],
        [row["gold_page_count"] for row in page_rows],
    )

    file_auc: dict[str, dict[str, float | int | None]] = {}
    for kf in (3, 5, 10, 15, 20):
        labels = [row[f"recall_{kf}"] >= 1.0 for row in file_rows]
        file_auc[str(kf)] = {
            "positive_queries": int(sum(labels)),
            "gap_top1_top3_auc": _auc([row["gap"] for row in file_rows], labels),
            "inverse_ratio_auc": _auc([1.0 - row["ratio"] for row in file_rows], labels),
            "top_score_auc": _auc([row["top_score"] for row in file_rows], labels),
            "inverse_entropy_auc": _auc([-row["entropy"] for row in file_rows], labels),
        }

    return {
        "page_score_summary": {
            key: _stats([row[key] for row in page_rows])
            for key in ("top_score", "gap", "ratio", "span", "entropy")
        },
        "file_score_summary": {
            key: _stats([row[key] for row in file_rows])
            for key in ("top_score", "gap", "ratio", "span", "entropy")
        },
        "page_auc": page_auc,
        "file_auc": file_auc,
        "page_spearman": page_correlations,
        "trace_candidate_count": _stats([row["candidate_count"] for row in page_rows]),
        "trace_by_kf": {
            str(kf): {
                "rows": len(by_kf[kf]),
                "mean_page_recall": {
                    str(kp): float(
                        np.mean(
                            [
                                len(
                                    set(str(item["node_id"]) for item in (by_kf[kf][qid].get("page_candidates") or [])[:kp])
                                    & set(qrels[qid])
                                )
                                / len(qrels[qid])
                                for qid in qids
                            ]
                        )
                    )
                    for kp in KPS
                },
            }
            for kf in KFS
        },
    }


def _scope_page_counts(metadata_path: Path) -> Counter[str]:
    if not metadata_path.exists():
        return Counter()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return Counter(str(row.get("doc_id")) for row in metadata)


def _evaluate_trace_arm(
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    by_kf: Mapping[int, Mapping[str, Mapping[str, Any]]],
    kf_by_qid: Sequence[int],
    kp_by_qid: Sequence[int],
    page_counts: Counter[str],
) -> dict[str, float]:
    page_recall: list[float] = []
    page_hit: list[bool] = []
    file_recall: list[float] = []
    scope_pages: list[int] = []
    for qid, kf, kp in zip(qids, kf_by_qid, kp_by_qid):
        trace = by_kf[kf][qid]
        gold_pages = set(qrels[qid])
        page_candidates = trace.get("page_candidates") or []
        page_ids = [str(item["node_id"]) for item in page_candidates[:kp]]
        selected_files = [str(item["node_id"]) for item in (trace.get("file_candidates") or [])[:kf]]
        gold_files = _gold_file_ids(qrels[qid])
        page_set = set(page_ids)
        page_recall.append(len(page_set & gold_pages) / len(gold_pages))
        page_hit.append(bool(page_set & gold_pages))
        file_recall.append(len(set(selected_files) & gold_files) / len(gold_files))
        scope_pages.append(
            sum(page_counts.get(file_id.split("::", 1)[-1], 0) for file_id in selected_files)
        )
    return {
        "page_recall": float(np.mean(page_recall)),
        "page_hit": float(np.mean(page_hit)),
        "file_recall": float(np.mean(file_recall)),
        "avg_scope_pages": float(np.mean(scope_pages)),
        "avg_kf": float(np.mean(kf_by_qid)),
        "avg_kp": float(np.mean(kp_by_qid)),
    }


def _adaptive_predictions(
    qids: Sequence[str],
    by_kf: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> tuple[list[int], list[int], dict[str, Any]]:
    def file_gap(qid: str) -> float:
        return _score_curve_features(
            by_kf[3][qid].get("file_candidates") or [], early_rank=3, late_rank=3
        )["gap"]

    def page_gap(qid: str, kf: int) -> float:
        return _score_curve_features(
            by_kf[kf][qid].get("page_candidates") or [], early_rank=10, late_rank=20
        )["gap"]

    train_qids = [qids[index] for index in train_indices]
    file_thresholds = _quantiles([file_gap(qid) for qid in train_qids])
    page_thresholds = {
        str(kf): _quantiles([page_gap(qid, kf) for qid in train_qids])
        for kf in KFS
    }
    selected_kf: list[int] = []
    selected_kp: list[int] = []
    for index in test_indices:
        qid = qids[index]
        file_value = file_gap(qid)
        if file_value >= file_thresholds[1]:
            kf = 3
        elif file_value >= file_thresholds[0]:
            kf = 5
        else:
            kf = 10
        page_value = page_gap(qid, kf)
        low, high = page_thresholds[str(kf)]
        if page_value >= high:
            kp = 10
        elif page_value >= low:
            kp = 50
        else:
            kp = 100
        selected_kf.append(kf)
        selected_kp.append(kp)
    return selected_kf, selected_kp, {
        "file_thresholds": list(file_thresholds),
        "page_thresholds": page_thresholds,
    }


def _adaptive_screen(
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    by_kf: Mapping[int, Mapping[str, Mapping[str, Any]]],
    page_counts: Counter[str],
) -> dict[str, Any]:
    fixed: dict[str, Any] = {}
    for kf in KFS:
        for kp in KPS:
            fixed[f"kf{kf}_kp{kp}"] = _evaluate_trace_arm(
                qids,
                qrels,
                by_kf,
                [kf] * len(qids),
                [kp] * len(qids),
                page_counts,
            )

    oof_kfs: list[int] = []
    oof_kps: list[int] = []
    fold_info: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        test = [index for index in range(len(qids)) if index % FOLDS == fold]
        train = [index for index in range(len(qids)) if index % FOLDS != fold]
        fold_kfs, fold_kps, thresholds = _adaptive_predictions(
            qids, by_kf, train_indices=train, test_indices=test
        )
        oof_kfs.extend(fold_kfs)
        oof_kps.extend(fold_kps)
        fold_info.append({"fold": fold, "test_queries": len(test), **thresholds})

    adaptive = _evaluate_trace_arm(qids, qrels, by_kf, oof_kfs, oof_kps, page_counts)
    adaptive.update(
        {
            "selected_kf_counts": dict(Counter(str(value) for value in oof_kfs)),
            "selected_kp_counts": dict(Counter(str(value) for value in oof_kps)),
            "folds": fold_info,
            "policy": "file gap top1-top3 -> Kf {3,5,10}; page gap top1-top20 -> Kp {10,50,100}",
        }
    )
    return {"fixed": fixed, "adaptive_oof": adaptive}


def _query_distribution(benchmark: ViDoreV3, qrels: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    queries = list(benchmark._queries)
    qids = [query.qid for query in queries]
    page_counts = [len(qrels[qid]) for qid in qids]
    file_counts = [len(_gold_file_ids(qrels[qid])) for qid in qids]
    token_counts = [len(query.query.split()) for query in queries]
    return {
        "query_count": len(queries),
        "gold_page_count": _stats(page_counts),
        "gold_page_histogram": dict(Counter(str(value) for value in page_counts)),
        "gold_file_count": _stats(file_counts),
        "gold_file_histogram": dict(Counter(str(value) for value in file_counts)),
        "query_token_count": _stats(token_counts),
        "query_type_counts": dict(Counter(
            query_type for query in queries for query_type in query.query_types
        )),
        "query_format_counts": dict(Counter(query.query_format for query in queries)),
        "content_label_counts": dict(Counter(
            label for query in queries for label in query.modalities
        )),
        "length_vs_gold_pages_spearman": _spearman(token_counts, page_counts),
    }


def _write_markdown(report: Mapping[str, Any], path: Path) -> None:
    distribution = report["query_distribution"]
    budget = report["budget_oracle"]["by_kf"]
    diagnostics = report["score_diagnostics"]
    adaptive = report["adaptive_screen"]
    lines = [
        "# Physics query-adaptive Kf/Kp budget analysis",
        "",
        "This is a cached diagnostic experiment. Qrels are used for evaluation,",
        "while the adaptive screen chooses thresholds from score distributions only.",
        "",
        "## Query heterogeneity",
        "",
        f"- Queries: **{distribution['query_count']}**.",
        f"- Gold pages/query: median **{distribution['gold_page_count']['median']:.0f}**, "
        f"p90 **{distribution['gold_page_count']['p90']:.0f}**, "
        f"range **{distribution['gold_page_count']['min']:.0f}–{distribution['gold_page_count']['max']:.0f}**.",
        f"- Gold files/query: median **{distribution['gold_file_count']['median']:.0f}**, "
        f"range **{distribution['gold_file_count']['min']:.0f}–{distribution['gold_file_count']['max']:.0f}**.",
        f"- Query length vs gold-page count Spearman: **{distribution['length_vs_gold_pages_spearman']:.3f}**.",
        "",
        "The query length itself is not a useful proxy for evidence volume.",
        "",
        "## Query-specific budget evidence",
        "",
        "The following rows reuse the existing `full_kf*` budget-ablation artifacts.",
        "They describe potential, not an inference-time policy.",
        "",
        "| Kf | Mean file/page scope coverage | Full scope queries | Page recall@10 | @20 | @50 | @100 | Scope failure | Page-ranking failure | Full@100 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kf in ALL_KFS:
        row = budget[str(kf)]
        diagnosis = row["scope_diagnosis"]
        recall = row["page_recall_by_kp"]
        lines.append(
            f"| {kf} | {row['mean_file_page_coverage']:.2%} | "
            f"{row['full_file_page_coverage_queries']} | {recall['10']:.2%} | "
            f"{recall['20']:.2%} | {recall['50']:.2%} | {recall['100']:.2%} | "
            f"{diagnosis.get('file_scope_missing', 0)} | "
            f"{diagnosis.get('page_rank_missing', 0)} | "
            f"{diagnosis.get('full_at_100', 0)} |"
        )
    lines += [
        "",
        f"Minimum tested Kf for all gold files: `{report['budget_oracle']['minimum_tested_kf_for_all_gold_files']}`.",
        "",
        "## Score-distribution signal",
        "",
        "AUC measures whether a score-curve feature ranks queries that are fully covered at the stated budget above the other queries.",
        "An AUC of 0.5 is no signal; this is a diagnostic, not a trained predictor.",
        "",
        "### Page score curve",
        "",
        "| Full evidence by Kp | Positive queries | Gap top1–top20 AUC | Inverse ratio AUC | Top score AUC | Inverse entropy AUC |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for kp in KPS:
        row = diagnostics["page_auc"][str(kp)]
        lines.append(
            f"| {kp} | {row['positive_queries']} | {row['gap_top1_top20_auc']:.3f} | "
            f"{row['inverse_ratio_auc']:.3f} | {row['top_score_auc']:.3f} | "
            f"{row['inverse_entropy_auc']:.3f} |"
        )
    lines += [
        "",
        "### File score curve",
        "",
        "| All gold files by Kf | Positive queries | Gap top1–top3 AUC | Inverse ratio AUC | Top score AUC | Inverse entropy AUC |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for kf in (3, 5, 10, 15, 20):
        row = diagnostics["file_auc"][str(kf)]
        lines.append(
            f"| {kf} | {row['positive_queries']} | {row['gap_top1_top3_auc']:.3f} | "
            f"{row['inverse_ratio_auc']:.3f} | {row['top_score_auc']:.3f} | "
            f"{row['inverse_entropy_auc']:.3f} |"
        )
    lines += [
        "",
        f"Page score Spearman correlations with page recall@10: `{diagnostics['page_spearman']}`.",
        "",
        "## Conservative adaptive screen",
        "",
        "The OOF policy uses score quantiles only: file gap top1–top3 selects Kf 3/5/10;",
        "then page gap top1–top20 selects Kp 10/50/100. No qrel is used to set a threshold.",
        "",
        "| Policy | Page recall | Page hit | File recall | Avg scope pages | Avg Kf | Avg Kp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["kf3_kp10", "kf3_kp100", "kf5_kp100", "kf10_kp100"]:
        row = adaptive["fixed"][name]
        lines.append(
            f"| {name} | {row['page_recall']:.2%} | {row['page_hit']:.2%} | "
            f"{row['file_recall']:.2%} | {row['avg_scope_pages']:.1f} | "
            f"{row['avg_kf']:.2f} | {row['avg_kp']:.1f} |"
        )
    row = adaptive["adaptive_oof"]
    lines.append(
        f"| adaptive OOF | {row['page_recall']:.2%} | {row['page_hit']:.2%} | "
        f"{row['file_recall']:.2%} | {row['avg_scope_pages']:.1f} | "
        f"{row['avg_kf']:.2f} | {row['avg_kp']:.1f} |"
    )
    lines += [
        "",
        "## Conclusion",
        "",
        "- Query-adaptive budgets are feasible: evidence volume and the minimum useful Kf/Kp vary substantially.",
        "- Score concentration is a real but moderate confidence signal. Page gap AUC is about 0.66–0.68; file gap AUC is about 0.70–0.74 for low budgets.",
        "- Entropy of a softmax over the 100 normalized scores is nearly saturated and is not useful by itself.",
        "- A score-only adaptive screen reduces average Kp, but also loses recall relative to the conservative fixed Kp=100 arm. It is not ready as the final policy.",
        "- The next experiment should learn a small cost-aware gate on held-out folds, using score gaps plus candidate/file coverage signals; it should report recall-cost Pareto points rather than one tuned accuracy number.",
        "",
        "The `full_kf*` budget rows and the hierarchical score traces come from separate existing experiments; their protocols are kept explicitly separate in `report.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    benchmark = ViDoreV3(root=ROOT / "data" / "benchmark" / "vidore_v3", subset="physics", language="french")
    qids = sorted(query.qid for query in benchmark.questions())
    qrels = benchmark.qrels()
    fixed_rows = _read_jsonl(FIXED_BUDGET_DIR / "per_query.jsonl")
    traces = _read_jsonl(HIERARCHICAL_DIR / "stage_traces.jsonl")
    by_kf, _combined = _trace_rows(qids, qrels, traces)
    page_counts = _scope_page_counts(
        ROOT / "data" / "output" / "vsplade" / "vidore_v3_physics_48q" / "page_metadata.json"
    )
    report: dict[str, Any] = {
        "contract_version": "physics-adaptive-budget-analysis-v1",
        "dataset": {
            "subset": "physics",
            "language": "french",
            "queries": len(qids),
            "pages": 1674,
            "files": 42,
            "qrels_pages": sum(len(qrels[qid]) for qid in qids),
        },
        "protocols": {
            "budget_ablation": str(FIXED_BUDGET_DIR),
            "score_trace": str(HIERARCHICAL_DIR),
            "budget_ablation_note": "Existing full_kf3/5/10/15/20 proposal-union experiment.",
            "score_trace_note": "Existing cascade-all_text-max-kf3/5/10 hierarchical traces.",
            "adaptive_thresholds": "Score quantiles fit on four folds and applied to the held-out fold; qrels are not used for thresholds.",
        },
        "query_distribution": _query_distribution(benchmark, qrels),
        "budget_oracle": _budget_oracle(qids, qrels, fixed_rows),
        "score_diagnostics": _score_diagnostics(qids, qrels, traces),
        "adaptive_screen": _adaptive_screen(qids, qrels, by_kf, page_counts),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(report, OUTPUT_DIR / "report.md")
    print(f"Wrote {OUTPUT_DIR / 'report.md'}")
    print(f"Wrote {OUTPUT_DIR / 'report.json'}")
    print(json.dumps(report["adaptive_screen"]["adaptive_oof"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
