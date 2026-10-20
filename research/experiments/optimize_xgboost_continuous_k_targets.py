"""Ablate K80/K90/K100 continuous targets for score-only XGBoost regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "research" / "experiments"
ADAPTIVE_DIR = ROOT / "experiments" / "adaptive-k-retrieval"
for path in (EXPERIMENT_DIR, ADAPTIVE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_canonical_adaptive_methods import load_rows  # noqa: E402
from optimize_xgboost_continuous_k import (  # noqa: E402
    CAP,
    SEED,
    build_matrices,
    required_k,
    summarize_config,
)


DEFAULT_OUTPUT = ROOT / "data" / "work" / "xgboost_continuous_k_targets_20261016"
TARGETS = (0.80, 0.90, 1.00)
FEATURE_SETS = ("raw", "raw+shape")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = load_rows("hybrid", CAP)
    sources = np.asarray([str(row["source"]) for row in rows])
    matrices, _ = build_matrices(rows)
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            matrices["raw"], sources
        )
    )

    results = []
    for target_fraction in TARGETS:
        target = np.asarray(
            [required_k(row, target_fraction) for row in rows], dtype=float
        )
        for feature_set in FEATURE_SETS:
            config = {
                "name": f"k{int(target_fraction * 100)}_{feature_set.replace('+', '_')}_squared",
                "features": feature_set,
                "objective": "squared",
            }
            print(f"running {config['name']}", flush=True)
            result = summarize_config(
                config,
                matrices[feature_set],
                rows,
                target,
                sources,
                folds,
            )
            result["target_fraction"] = target_fraction
            result["target_distribution"] = {
                "mean": float(np.mean(target)),
                "median": float(np.median(target)),
                "p90": float(np.percentile(target, 90)),
                "at_cap": int(np.sum(target == CAP)),
            }
            results.append(result)

    def best_key(item: dict) -> tuple[int, float, float]:
        metric = item["policies"]["mean_k_70"]
        achieved = int(metric["mean_k"] < 70 and metric["macro_recall"] > 0.87)
        return achieved, metric["macro_recall"], -metric["mean_k"]

    best = max(results, key=best_key)
    payload = {
        "protocol": {
            "cap": CAP,
            "output_k": "continuous integer 1..200",
            "model": "XGBoost squared-error regressor, depth 2",
            "features": "score only",
            "outer_folds": 5,
            "inner_folds": 4,
            "label_ablation": [0.80, 0.90, 1.00],
            "warning": "Exploratory target-label selection on the same outer OOF benchmark.",
        },
        "results": results,
        "best_under_mean_k_70": {
            "name": best["name"],
            "target_fraction": best["target_fraction"],
            "features": best["features"],
            "mean_k": best["policies"]["mean_k_70"]["mean_k"],
            "macro_recall": best["policies"]["mean_k_70"]["macro_recall"],
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# XGBoost continuous-K label ablation",
        "",
        "The model and nested calibration are fixed; only the target K label and raw versus normalized-shape inputs change.",
        "",
        "| Target label | Features | Policy | Mean K | Recall | Target mean | Target MAE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        for mode in ("mean_k_70", "recall_87"):
            metric = result["policies"][mode]
            lines.append(
                f"| K{int(result['target_fraction'] * 100)} | {result['features']} | {mode} | "
                f"{metric['mean_k']:.2f} | {metric['macro_recall']:.2%} | "
                f"{result['target_distribution']['mean']:.2f} | {result['target_mae']:.2f} |"
            )
    chosen = payload["best_under_mean_k_70"]
    lines.extend(
        [
            "",
            f"Best screened: `{chosen['name']}` at mean K={chosen['mean_k']:.2f}, recall={chosen['macro_recall']:.2%}.",
            "",
            "This is exploratory label selection, not an independent held-out confirmation.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["best_under_mean_k_70"], indent=2))
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
