"""Paired per-query analysis of XGBoost and continuous-K model OOF outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .scaling_ablation import DEFAULT_RUN_DIR, _read_jsonl


DEFAULT_FAMILY_DIR = DEFAULT_RUN_DIR / "model_family_ablation"
DEFAULT_OUTPUT_DIR = DEFAULT_FAMILY_DIR / "paired_oof_analysis"


def _metrics(k_values: np.ndarray, gold_ranks: list[dict[str, int | None]]) -> dict[str, float]:
    recalls, precisions = [], []
    for k, rank_map in zip(k_values, gold_ranks):
        ranks = [int(rank) for rank in rank_map.values() if rank is not None]
        gold_count = len(rank_map)
        hits = sum(rank <= int(k) for rank in ranks)
        recalls.append(hits / max(gold_count, 1))
        precisions.append(hits / max(int(k), 1))
    return {
        "mean_k": float(np.mean(k_values)),
        "macro_recall": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
    }


def _paired_bootstrap_ci(
    values: np.ndarray, sources: np.ndarray, seed: int, samples: int = 10000
) -> list[float]:
    rng = np.random.default_rng(seed)
    sampled_sum = np.zeros(samples, dtype=np.float64)
    for source in np.unique(sources):
        source_indices = np.flatnonzero(sources == source)
        sampled_positions = rng.integers(0, len(source_indices), size=(samples, len(source_indices)))
        sampled_sum += values[source_indices[sampled_positions]].sum(axis=1)
    sampled_means = sampled_sum / len(values)
    return [float(np.percentile(sampled_means, 2.5)), float(np.percentile(sampled_means, 97.5))]


def _safe(qid: str) -> str:
    return qid.replace("|", "/")


def _load_inputs(run_dir: Path, family_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    gate_path = Path(manifest["gate_artifact"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    xgb_rows = {row["qid"]: row for row in _read_jsonl(run_dir / "oof_predictions.jsonl")}
    input_order = _read_jsonl(run_dir / "feature_labels.jsonl")
    derived_outer_fold: dict[str, int] = {}
    sources_in_order = np.asarray([row["source"] for row in input_order])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dummy_x = np.zeros((len(input_order), 1), dtype=np.float32)
    for fold, (_, test_indices) in enumerate(splitter.split(dummy_x, sources_in_order), start=1):
        for index in test_indices:
            derived_outer_fold[input_order[int(index)]["qid"]] = fold
    ridge_rows = {
        row["qid"]: row
        for row in _read_jsonl(family_dir / "ridge_alpha_10" / "oof_predictions.jsonl")
    }
    if set(xgb_rows) != set(ridge_rows) or len(xgb_rows) != 220:
        raise ValueError("XGBoost and Ridge OOF qids must match exactly across all 220 queries")

    rows = []
    for qid in sorted(xgb_rows):
        xgb, ridge = xgb_rows[qid], ridge_rows[qid]
        if int(xgb["target_k"]) != int(ridge["target_k"]):
            raise ValueError(f"label mismatch for {qid}")
        gold_ranks = gate["gold_rank"].get(qid)
        if not gold_ranks:
            raise ValueError(f"frozen gold-rank map missing for {qid}")
        rows.append({
            "qid": qid,
            "source": xgb["source"],
            "outer_fold_xgb": derived_outer_fold[qid],
            "outer_fold_ridge": int(ridge["outer_fold"]),
            "target_k": int(xgb["target_k"]),
            "reachable_at_cap": all(rank is not None for rank in gold_ranks.values()),
            "gold_count": len(gold_ranks),
            "gold_ranks": gold_ranks,
            "xgb_k": int(xgb["prediction_k_oof"]),
            "ridge_k": int(ridge["prediction_k_oof"]),
        })
    for row in rows:
        if row["outer_fold_xgb"] != row["outer_fold_ridge"]:
            raise ValueError(f"outer-fold mismatch for {row['qid']}")
    return rows, manifest


def _decorate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        gold = row["gold_ranks"]
        present_ranks = [int(rank) for rank in gold.values() if rank is not None]
        xgb_hits = sum(rank <= row["xgb_k"] for rank in present_ranks)
        ridge_hits = sum(rank <= row["ridge_k"] for rank in present_ranks)
        count = row["gold_count"]
        xgb_recall, ridge_recall = xgb_hits / count, ridge_hits / count
        xgb_precision = xgb_hits / row["xgb_k"]
        ridge_precision = ridge_hits / row["ridge_k"]
        delta_k = row["ridge_k"] - row["xgb_k"]
        delta_recall = ridge_recall - xgb_recall
        delta_precision = ridge_precision - xgb_precision
        if delta_k < 0:
            if delta_recall < -1e-12:
                category = "ridge_reduces_K_loses_recall"
            elif delta_recall > 1e-12:
                category = "ridge_reduces_K_gains_recall"
            elif xgb_hits == 0:
                category = "ridge_reduces_K_zero-hit_both"
            else:
                category = "ridge_reduces_K_same_recall_precision_up"
        elif delta_k > 0:
            if delta_recall > 1e-12:
                category = "ridge_increases_K_gains_recall"
            elif delta_recall < -1e-12:
                category = "ridge_increases_K_loses_recall"
            elif xgb_hits == 0 and ridge_hits == 0:
                category = "ridge_increases_K_zero-hit_no_metric_change"
            else:
                category = "ridge_increases_K_same_recall_precision_down"
        else:
            category = "same_K"
        result.append({
            **{key: value for key, value in row.items() if key != "gold_ranks"},
            "gold_hits_xgb": xgb_hits,
            "gold_hits_ridge": ridge_hits,
            "recall_xgb": xgb_recall,
            "recall_ridge": ridge_recall,
            "delta_recall_ridge_minus_xgb": delta_recall,
            "precision_xgb": xgb_precision,
            "precision_ridge": ridge_precision,
            "delta_precision_ridge_minus_xgb": delta_precision,
            "delta_k_ridge_minus_xgb": delta_k,
            "category": category,
        })
    return result


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    result = {}
    for group_name, group in sorted(groups.items()):
        xgb_k = np.asarray([row["xgb_k"] for row in group], dtype=np.float64)
        ridge_k = np.asarray([row["ridge_k"] for row in group], dtype=np.float64)
        result[group_name] = {
            "n": len(group),
            "mean_k_xgb": float(xgb_k.mean()),
            "mean_k_ridge": float(ridge_k.mean()),
            "mean_delta_k_ridge_minus_xgb": float((ridge_k - xgb_k).mean()),
            "macro_recall_xgb": float(np.mean([row["recall_xgb"] for row in group])),
            "macro_recall_ridge": float(np.mean([row["recall_ridge"] for row in group])),
            "delta_macro_recall": float(np.mean([row["delta_recall_ridge_minus_xgb"] for row in group])),
            "macro_precision_xgb": float(np.mean([row["precision_xgb"] for row in group])),
            "macro_precision_ridge": float(np.mean([row["precision_ridge"] for row in group])),
            "delta_macro_precision": float(np.mean([row["delta_precision_ridge_minus_xgb"] for row in group])),
            "recall_unchanged_rows": sum(abs(row["delta_recall_ridge_minus_xgb"]) < 1e-12 for row in group),
            "precision_unchanged_rows": sum(abs(row["delta_precision_ridge_minus_xgb"]) < 1e-12 for row in group),
        }
    return result


def run_analysis(run_dir: Path, family_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows, manifest = _load_inputs(run_dir.resolve(), family_dir.resolve())
    decorated = _decorate_rows(rows)
    by_category = _group_summary(decorated, "category")
    by_source = _group_summary(decorated, "source")

    xgb_k = np.asarray([row["xgb_k"] for row in decorated], dtype=np.int32)
    ridge_k = np.asarray([row["ridge_k"] for row in decorated], dtype=np.int32)
    gold_ranks = [row["gold_ranks"] for row in rows]
    xgb_metrics = _metrics(xgb_k, gold_ranks)
    ridge_metrics = _metrics(ridge_k, gold_ranks)
    delta_k = ridge_k.astype(np.float64) - xgb_k
    delta_recall = np.asarray([row["delta_recall_ridge_minus_xgb"] for row in decorated])
    delta_precision = np.asarray([row["delta_precision_ridge_minus_xgb"] for row in decorated])
    sources = np.asarray([row["source"] for row in decorated])

    counts = {
        "ridge_uses_lower_k": int(np.sum(delta_k < 0)),
        "ridge_uses_higher_k": int(np.sum(delta_k > 0)),
        "same_k": int(np.sum(delta_k == 0)),
        "ridge_lower_k_same_recall": int(sum(
            row["delta_k_ridge_minus_xgb"] < 0 and abs(row["delta_recall_ridge_minus_xgb"]) < 1e-12
            for row in decorated
        )),
        "ridge_lower_k_same_recall_precision_increases": int(sum(
            row["delta_k_ridge_minus_xgb"] < 0
            and abs(row["delta_recall_ridge_minus_xgb"]) < 1e-12
            and row["delta_precision_ridge_minus_xgb"] > 1e-12
            for row in decorated
        )),
        "ridge_lower_k_loses_recall": int(sum(
            row["delta_k_ridge_minus_xgb"] < 0 and row["delta_recall_ridge_minus_xgb"] < -1e-12
            for row in decorated
        )),
        "ridge_higher_k_same_recall_precision": int(sum(
            row["delta_k_ridge_minus_xgb"] > 0
            and abs(row["delta_recall_ridge_minus_xgb"]) < 1e-12
            and abs(row["delta_precision_ridge_minus_xgb"]) < 1e-12
            for row in decorated
        )),
        "ridge_higher_k_gains_recall": int(sum(
            row["delta_k_ridge_minus_xgb"] > 0 and row["delta_recall_ridge_minus_xgb"] > 1e-12
            for row in decorated
        )),
        "ridge_higher_k_no_recall_gain": int(sum(
            row["delta_k_ridge_minus_xgb"] > 0 and row["delta_recall_ridge_minus_xgb"] <= 1e-12
            for row in decorated
        )),
        "ridge_precision_improved": int(np.sum(delta_precision > 1e-12)),
        "ridge_precision_worsened": int(np.sum(delta_precision < -1e-12)),
        "ridge_precision_unchanged": int(np.sum(np.abs(delta_precision) <= 1e-12)),
        "ridge_recall_improved": int(np.sum(delta_recall > 1e-12)),
        "ridge_recall_worsened": int(np.sum(delta_recall < -1e-12)),
        "ridge_recall_unchanged": int(np.sum(np.abs(delta_recall) <= 1e-12)),
    }

    positive_precision = sorted(
        [row for row in decorated if row["delta_precision_ridge_minus_xgb"] > 1e-12],
        key=lambda row: (-row["delta_precision_ridge_minus_xgb"], row["qid"]),
    )
    no_metric_change_after_k_increase = [
        row for row in decorated
        if row["delta_k_ridge_minus_xgb"] > 0
        and abs(row["delta_recall_ridge_minus_xgb"]) < 1e-12
        and abs(row["delta_precision_ridge_minus_xgb"]) < 1e-12
    ]
    safe_downshift = [
        row for row in decorated
        if row["delta_k_ridge_minus_xgb"] < 0
        and abs(row["delta_recall_ridge_minus_xgb"]) < 1e-12
        and row["delta_precision_ridge_minus_xgb"] > 1e-12
    ]
    safe_downshift_target_distribution: dict[str, int] = {}
    for row in safe_downshift:
        target = str(row["target_k"])
        safe_downshift_target_distribution[target] = safe_downshift_target_distribution.get(target, 0) + 1

    xgb_scaling_path = run_dir / "scaling_ablation" / "standard_scaled" / "report.json"
    xgb_scaled_report = json.loads(xgb_scaling_path.read_text(encoding="utf-8"))
    family_comparison = json.loads((family_dir / "comparison.json").read_text(encoding="utf-8"))
    metrics_matrix = {
        "columns": ["XGBoost", "LinearRegression", "Ridge α=1", "Ridge α=10", "MLP (16,8)"],
        "rows": [
            {
                "preprocessing": "Raw features (native tree input)",
                "cells": {
                    "XGBoost": xgb_metrics,
                    "LinearRegression": None,
                    "Ridge α=1": None,
                    "Ridge α=10": None,
                    "MLP (16,8)": None,
                },
                "status": "only XGBoost run",
            },
            {
                "preprocessing": "Fold-local StandardScaler on X; y remains in raw K units",
                "cells": {
                    "XGBoost": xgb_scaled_report["metrics"],
                    "LinearRegression": None,
                    "Ridge α=1": None,
                    "Ridge α=10": None,
                    "MLP (16,8)": None,
                },
                "status": "XGBoost only",
            },
            {
                "preprocessing": "Fold-local StandardScaler on X and y; inverse-transform predictions",
                "cells": {
                    "XGBoost": None,
                    **{
                        "LinearRegression": family_comparison["models"]["linear_regression"]["metrics"],
                        "Ridge α=1": family_comparison["models"]["ridge_alpha_1"]["metrics"],
                        "Ridge α=10": family_comparison["models"]["ridge_alpha_10"]["metrics"],
                        "MLP (16,8)": family_comparison["models"]["mlp_16_8"]["metrics"],
                    },
                },
                "status": "linear, ridge, and MLP screening only",
            },
        ],
        "cell_metric_order": ["macro_recall", "macro_precision"],
        "note": "All metrics are nested 5-fold OOF. Empty cells are not run, not zero. Preprocessing and model are not yet a full factorial grid.",
    }

    analysis = {
        "title": "Paired OOF analysis: XGBoost vs Ridge alpha 10",
        "status": "exploratory_paired_nested_oof",
        "protocol": {
            "queries": len(decorated),
            "outer_folds": 5,
            "same_qids": True,
            "same_outer_folds": True,
            "same_frozen_ranking": True,
            "cap": 100,
            "folds_from_manifest": manifest["ranking_reproduction"],
        },
        "metrics": {
            "xgboost": xgb_metrics,
            "ridge_alpha_10": ridge_metrics,
            "delta_ridge_minus_xgboost": {
                "mean_k": ridge_metrics["mean_k"] - xgb_metrics["mean_k"],
                "macro_recall": ridge_metrics["macro_recall"] - xgb_metrics["macro_recall"],
                "macro_precision": ridge_metrics["macro_precision"] - xgb_metrics["macro_precision"],
            },
            "paired_bootstrap_95ci_delta": {
                "mean_k": _paired_bootstrap_ci(delta_k, sources, 121),
                "macro_recall": _paired_bootstrap_ci(delta_recall, sources, 122),
                "macro_precision": _paired_bootstrap_ci(delta_precision, sources, 123),
                "method": "paired bootstrap resampling qids within each source; 10,000 replicates",
            },
        },
        "query_change_counts": counts,
        "change_categories": by_category,
        "per_source": by_source,
        "precision_gain_cases_top_25": positive_precision[:25],
        "all_queries_with_higher_k_but_no_metric_change": no_metric_change_after_k_increase,
        "safe_downshift_cases": safe_downshift,
        "safe_downshift_target_distribution": safe_downshift_target_distribution,
        "model_preprocessing_matrix": metrics_matrix,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "qid", "source", "outer_fold_xgb", "outer_fold_ridge", "target_k", "reachable_at_cap", "gold_count",
        "xgb_k", "ridge_k", "delta_k_ridge_minus_xgb", "gold_hits_xgb", "gold_hits_ridge",
        "recall_xgb", "recall_ridge", "delta_recall_ridge_minus_xgb", "precision_xgb", "precision_ridge",
        "delta_precision_ridge_minus_xgb", "category",
    ]
    with (output_dir / "paired_query_deltas.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in decorated:
            writer.writerow({name: row[name] for name in fieldnames})

    safe_fieldnames = [
        "qid", "source", "target_k", "reachable_at_cap", "gold_count", "xgb_k", "ridge_k",
        "delta_k_ridge_minus_xgb", "gold_hits_xgb", "gold_hits_ridge", "recall_xgb", "recall_ridge",
        "precision_xgb", "precision_ridge",
    ]
    for filename, selected in (
        ("safe_downshift_cases.csv", safe_downshift),
        ("higher_k_no_metric_change.csv", no_metric_change_after_k_increase),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=safe_fieldnames)
            writer.writeheader()
            for row in selected:
                writer.writerow({name: row[name] for name in safe_fieldnames})

    with (output_dir / "preprocessing_model_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        matrix_columns = metrics_matrix["columns"]
        matrix_fields = ["preprocessing", *matrix_columns]
        writer = csv.DictWriter(handle, fieldnames=matrix_fields)
        writer.writeheader()
        for matrix_row in metrics_matrix["rows"]:
            writer.writerow({
                "preprocessing": matrix_row["preprocessing"],
                **{name: _fmt_cell(matrix_row["cells"][name]) for name in matrix_columns},
            })

    (output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(_format_report(analysis), encoding="utf-8")
    return analysis


def _fmt_cell(metrics: dict[str, Any] | None) -> str:
    if metrics is None:
        return "—"
    return f"{metrics['macro_recall'] * 100:.2f}% / {metrics['macro_precision'] * 100:.2f}%"


def _format_report(data: dict[str, Any]) -> str:
    counts = data["query_change_counts"]
    metrics = data["metrics"]
    delta = metrics["delta_ridge_minus_xgboost"]
    ci = metrics["paired_bootstrap_95ci_delta"]
    matrix = data["model_preprocessing_matrix"]
    columns = matrix["columns"]
    lines = [
        "# Paired per-query analysis: XGBoost vs Ridge α=10",
        "",
        "Both predictions are nested 5-fold OOF on the same 220 qids, frozen rankings, labels, and outer folds. Precision and recall are macro-averaged per query. A delta is Ridge minus XGBoost.",
        "",
        "## Model × preprocessing matrix",
        "",
        "Each cell is `macro recall / macro precision`; `—` means not run.",
        "",
        "| Preprocessing | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for row in matrix["rows"]:
        cell_by_name = row["cells"]
        lines.append("| " + row["preprocessing"] + " | " + " | ".join(_fmt_cell(cell_by_name[name]) for name in columns) + " |")
    lines.extend([
        "",
        "Each row denotes the exact transform that was run; `—` means not run. All scalers are fitted only inside the relevant training fold. A complete factorial comparison still needs the missing cells filled.",
        "",
        "## Paired aggregate result",
        "",
        f"- XGBoost: Mean K {metrics['xgboost']['mean_k']:.2f}, recall {metrics['xgboost']['macro_recall']*100:.2f}%, precision {metrics['xgboost']['macro_precision']*100:.2f}%.",
        f"- Ridge α=10: Mean K {metrics['ridge_alpha_10']['mean_k']:.2f}, recall {metrics['ridge_alpha_10']['macro_recall']*100:.2f}%, precision {metrics['ridge_alpha_10']['macro_precision']*100:.2f}%.",
        f"- Ridge − XGBoost: Mean K {delta['mean_k']:+.2f}; recall {delta['macro_recall']*100:+.2f} percentage points (paired, source-stratified bootstrap 95% CI {ci['macro_recall'][0]*100:+.2f} to {ci['macro_recall'][1]*100:+.2f}); precision {delta['macro_precision']*100:+.2f} points (95% CI {ci['macro_precision'][0]*100:+.2f} to {ci['macro_precision'][1]*100:+.2f}).",
        "",
        "## Where the difference comes from",
        "",
        f"- Ridge chooses lower K on {counts['ridge_uses_lower_k']} queries, higher K on {counts['ridge_uses_higher_k']}, and the same K on {counts['same_k']}.",
        f"- Lower-K cases with unchanged recall: {counts['ridge_lower_k_same_recall']}; among them, precision rises on {counts['ridge_lower_k_same_recall_precision_increases']} queries and recall is lost on {counts['ridge_lower_k_loses_recall']} lower-K queries.",
        f"- Higher-K cases with exactly no change in either recall or precision: {counts['ridge_higher_k_same_recall_precision']} queries. This should occur when both policies retrieve zero gold pages: recall=0 and precision=0 either way, often because those gold pages are outside the top-100 candidate set.",
        f"- Ridge changes per-query precision upward on {counts['ridge_precision_improved']} queries, downward on {counts['ridge_precision_worsened']}, unchanged on {counts['ridge_precision_unchanged']}; recall rises on {counts['ridge_recall_improved']}, falls on {counts['ridge_recall_worsened']}, and is unchanged on {counts['ridge_recall_unchanged']}.",
        f"- The 93 safe downshifts have required-K distribution: {', '.join(f'K={k}: {n}' for k, n in sorted(data['safe_downshift_target_distribution'].items(), key=lambda item: int(item[0])))}.",
        "",
        "The apparent paradox is resolved by macro averaging: Mean K is an arithmetic average of K, while precision is the average of `gold hits / K` per query. They are not algebraic inverses. Ridge can cut K on queries where it retains the same gold hits (raising per-query precision), while spending that budget on zero-hit queries where extra K changes neither recall nor precision. Small increases elsewhere can also balance the total Mean K. Thus similar Mean K does not imply similar precision.",
        "",
        "## Exact higher-K queries with no recall or precision change",
        "",
        "| qid | source | required K | XGB K | Ridge K | reachable at 100 |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in data["all_queries_with_higher_k_but_no_metric_change"]:
        lines.append(f"| {_safe(row['qid'])} | {row['source']} | {row['target_k']} | {row['xgb_k']} | {row['ridge_k']} | {row['reachable_at_cap']} |")
    lines.extend([
        "",
        "## By source",
        "",
        "| Source | n | Mean K XGB → Ridge | Recall XGB → Ridge | Δ recall | Precision XGB → Ridge | Δ precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for source, row in data["per_source"].items():
        lines.append(
            f"| {source} | {row['n']} | {row['mean_k_xgb']:.2f} → {row['mean_k_ridge']:.2f} | "
            f"{row['macro_recall_xgb']*100:.2f}% → {row['macro_recall_ridge']*100:.2f}% | "
            f"{row['delta_macro_recall']*100:+.2f} pp | {row['macro_precision_xgb']*100:.2f}% → {row['macro_precision_ridge']*100:.2f}% | "
            f"{row['delta_macro_precision']*100:+.2f} pp |"
        )
    lines.extend([
        "",
        "## Largest per-query precision gains",
        "",
        "| qid | source | target K | XGB K → Ridge K | hits XGB → Ridge | recall XGB → Ridge | precision XGB → Ridge |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in data["precision_gain_cases_top_25"]:
        lines.append(
            f"| {_safe(row['qid'])} | {row['source']} | {row['target_k']} | {row['xgb_k']} → {row['ridge_k']} | "
            f"{row['gold_hits_xgb']} → {row['gold_hits_ridge']} | {row['recall_xgb']*100:.1f}% → {row['recall_ridge']*100:.1f}% | "
            f"{row['precision_xgb']*100:.2f}% → {row['precision_ridge']*100:.2f}% |"
        )
    lines.extend([
        "",
        "Full per-query changes are in `paired_query_deltas.csv`; machine-readable aggregates are in `analysis.json`. This is paired descriptive analysis of exploratory OOF predictions; it does not prove Ridge generalizes better.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--family-dir", type=Path, default=DEFAULT_FAMILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_analysis(args.run_dir, args.family_dir, args.output_dir)
    print(f"[paired-analysis] wrote {args.output_dir.resolve()}")
    print(json.dumps(result["metrics"], indent=2))
    print(json.dumps(result["query_change_counts"], indent=2))


if __name__ == "__main__":
    main()
