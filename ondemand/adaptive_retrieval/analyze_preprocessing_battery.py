"""Post-hoc paired analysis for the preprocessing battery."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .full_factorial_ablation import MODEL_LABELS, MODELS
from .preprocessing_battery import DEFAULT_OUTPUT_DIR
from .scaling_ablation import DEFAULT_RUN_DIR, _read_jsonl


BEST_BY_MODEL = {
    "xgboost": "expanded_standard_xy",
    "linear_regression": "mutual_info_top10_xy",
    "ridge_alpha_1": "expanded_standard_xy",
    "ridge_alpha_10": "power_xy",
    "mlp_16_8": "mutual_info_top10_xy",
}
BASELINE_METHOD = "standard_xy_baseline"


def _per_query(k: int, rank_map: dict[str, int | None]) -> tuple[float, float]:
    ranks = [int(rank) for rank in rank_map.values() if rank is not None]
    hits = sum(rank <= k for rank in ranks)
    return hits / max(len(rank_map), 1), hits / max(k, 1)


def _paired_stratified_ci(values: np.ndarray, sources: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = 10000
    totals = np.zeros(samples, dtype=np.float64)
    for source in np.unique(sources):
        indices = np.flatnonzero(sources == source)
        draws = rng.integers(0, len(indices), size=(samples, len(indices)))
        totals += values[indices[draws]].sum(axis=1)
    distribution = totals / len(values)
    return [float(np.percentile(distribution, 2.5)), float(np.percentile(distribution, 97.5))]


def _load_prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    return {row["qid"]: row for row in _read_jsonl(path)}


def run_analysis(run_dir: Path, battery_dir: Path) -> dict[str, Any]:
    comparison = json.loads((battery_dir / "comparison.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    gate = json.loads(Path(manifest["gate_artifact"]).read_text(encoding="utf-8"))
    qid_order = [row["qid"] for row in _read_jsonl(run_dir / "feature_labels.jsonl")]
    sources = np.asarray([qid.split("::", 1)[0] for qid in qid_order])
    paired: dict[str, Any] = {}
    for model_index, model in enumerate(MODELS):
        best_method = BEST_BY_MODEL[model]
        baseline_key = f"{BASELINE_METHOD}__{model}"
        best_key = f"{best_method}__{model}"
        baseline = _load_prediction_map(battery_dir / baseline_key / "oof_predictions.jsonl")
        best = _load_prediction_map(battery_dir / best_key / "oof_predictions.jsonl")
        delta_k, delta_recall, delta_precision = [], [], []
        for qid in qid_order:
            baseline_k = int(baseline[qid]["prediction_k_oof"])
            best_k = int(best[qid]["prediction_k_oof"])
            baseline_recall, baseline_precision = _per_query(baseline_k, gate["gold_rank"][qid])
            best_recall, best_precision = _per_query(best_k, gate["gold_rank"][qid])
            delta_k.append(best_k - baseline_k)
            delta_recall.append(best_recall - baseline_recall)
            delta_precision.append(best_precision - baseline_precision)
        arrays = [np.asarray(values, dtype=np.float64) for values in (delta_k, delta_recall, delta_precision)]
        baseline_metrics = comparison["cells"][baseline_key]["metrics"]
        best_metrics = comparison["cells"][best_key]["metrics"]
        paired[model] = {
            "baseline_method": BASELINE_METHOD,
            "best_method": best_method,
            "baseline_metrics": baseline_metrics,
            "best_metrics": best_metrics,
            "delta_best_minus_baseline": {
                "mean_k": float(arrays[0].mean()),
                "macro_recall": float(arrays[1].mean()),
                "macro_precision": float(arrays[2].mean()),
            },
            "paired_source_stratified_bootstrap_95ci": {
                "mean_k": _paired_stratified_ci(arrays[0], sources, 900 + model_index * 3),
                "macro_recall": _paired_stratified_ci(arrays[1], sources, 901 + model_index * 3),
                "macro_precision": _paired_stratified_ci(arrays[2], sources, 902 + model_index * 3),
            },
            "query_counts": {
                "recall_improved": int(np.sum(arrays[1] > 1e-12)),
                "recall_worsened": int(np.sum(arrays[1] < -1e-12)),
                "recall_unchanged": int(np.sum(np.abs(arrays[1]) <= 1e-12)),
                "k_lower": int(np.sum(arrays[0] < 0)),
                "k_higher": int(np.sum(arrays[0] > 0)),
                "k_same": int(np.sum(arrays[0] == 0)),
            },
            "baseline_per_source": comparison["cells"][baseline_key]["per_source"],
            "best_per_source": comparison["cells"][best_key]["per_source"],
        }

    mi_report = json.loads(
        (battery_dir / "mutual_info_top10_xy__linear_regression" / "report.json").read_text(encoding="utf-8")
    )
    selection_frequency: Counter[str] = Counter()
    for fold in mi_report["folds"]:
        selection_frequency.update(fold["outer_fit"]["transform"]["kept_feature_names"])

    result = {
        "status": "exploratory_paired_posthoc",
        "multiple_comparison_warning": "Best method per model was selected after screening 12 methods on the same OOF benchmark; paired intervals are descriptive and do not correct selection bias.",
        "best_by_model": BEST_BY_MODEL,
        "paired_best_vs_baseline": paired,
        "mutual_information_outer_fold_selection_frequency": dict(selection_frequency.most_common()),
    }
    (battery_dir / "paired_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (battery_dir / "analysis.md").write_text(_format_report(result), encoding="utf-8")
    return result


def _format_report(result: dict[str, Any]) -> str:
    lines = [
        "# Paired analysis of the best preprocessing arm per model",
        "",
        "Each best arm is compared query-by-query with that model's StandardScaler(X,y) baseline. CIs use 10,000 paired bootstrap replicates resampled within source.",
        "",
        "| Model | Best method | Δ Mean K | Δ recall | Recall 95% CI | Δ precision | Precision 95% CI | Recall + / − queries |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = result["paired_best_vs_baseline"][model]
        delta, ci, counts = row["delta_best_minus_baseline"], row["paired_source_stratified_bootstrap_95ci"], row["query_counts"]
        lines.append(
            f"| {MODEL_LABELS[model]} | {row['best_method']} | {delta['mean_k']:+.2f} | "
            f"{delta['macro_recall']*100:+.2f} pp | [{ci['macro_recall'][0]*100:+.2f}, {ci['macro_recall'][1]*100:+.2f}] | "
            f"{delta['macro_precision']*100:+.2f} pp | [{ci['macro_precision'][0]*100:+.2f}, {ci['macro_precision'][1]*100:+.2f}] | "
            f"{counts['recall_improved']} / {counts['recall_worsened']} |"
        )
    lines.extend([
        "",
        "These intervals are descriptive, not confirmatory: the best arm was chosen after looking at the same OOF benchmark, so selection bias remains.",
        "",
        "## Mutual-information top-10 selection stability",
        "",
        "Number of outer folds (out of five) selecting each feature:",
        "",
    ])
    lines.extend(
        f"- `{name}`: {count}/5" for name, count in result["mutual_information_outer_fold_selection_frequency"].items()
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--battery-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_analysis(args.run_dir.resolve(), args.battery_dir.resolve())
    print(json.dumps(result["paired_best_vs_baseline"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
