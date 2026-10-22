"""Add retrieval F1 and query-level accuracy diagnostics to the preprocessing battery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .full_factorial_ablation import MODEL_LABELS, MODELS
from .preprocessing_battery import DEFAULT_OUTPUT_DIR
from .scaling_ablation import DEFAULT_RUN_DIR, _read_jsonl


def _query_outcomes(
    rows: list[dict[str, Any]], gold_rank: dict[str, dict[str, int | None]]
) -> dict[str, float]:
    full_coverage: list[float] = []
    exact_k: list[float] = []
    within_5: list[float] = []
    for row in rows:
        chosen_k = int(row["prediction_k_oof"])
        target_k = int(row["target_k"])
        ranks = list(gold_rank[row["qid"]].values())
        hits = sum(rank is not None and int(rank) <= chosen_k for rank in ranks)
        full_coverage.append(float(hits == len(ranks)))
        exact_k.append(float(chosen_k == target_k))
        within_5.append(float(abs(chosen_k - target_k) <= 5))
    return {
        "full_coverage_accuracy": float(np.mean(full_coverage)),
        "exact_k_accuracy": float(np.mean(exact_k)),
        "within_5_k_accuracy": float(np.mean(within_5)),
    }


def run_report(run_dir: Path, battery_dir: Path) -> dict[str, Any]:
    comparison = json.loads((battery_dir / "comparison.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    gold_rank = json.loads(Path(manifest["gate_artifact"]).read_text(encoding="utf-8"))["gold_rank"]
    cells: dict[str, Any] = {}
    for method in comparison["methods"]:
        for model in MODELS:
            key = f"{method}__{model}"
            predictions = _read_jsonl(battery_dir / key / "oof_predictions.jsonl")
            if len(predictions) != 220 or len({row["qid"] for row in predictions}) != 220:
                raise ValueError(f"{key}: expected one OOF prediction for each of 220 qids")
            cells[key] = {
                "method": method,
                "model": model,
                "macro_f1": comparison["cells"][key]["metrics"]["macro_f1"],
                **_query_outcomes(predictions, gold_rank),
            }
    result = {
        "definitions": {
            "macro_f1": "Mean over queries of 2 * page_precision * page_recall / (page_precision + page_recall) at each query's chosen K.",
            "full_coverage_accuracy": "Fraction of queries for which chosen K retrieves every labelled gold page (per-query recall == 1).",
            "exact_k_accuracy": "Fraction of queries for which predicted integer K equals the required-K label exactly. Diagnostic only; K=100 includes censored/unreachable rows.",
            "within_5_k_accuracy": "Fraction of queries whose predicted K is within ±5 of the required-K label. Diagnostic only.",
        },
        "cells": cells,
    }
    (battery_dir / "f1_accuracy.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (battery_dir / "f1_accuracy.md").write_text(_format_report(comparison, result), encoding="utf-8")
    return result


def _format_report(comparison: dict[str, Any], result: dict[str, Any]) -> str:
    lines = [
        "# F1 and accuracy diagnostics",
        "",
        "Cell format is `macro F1 / full-coverage accuracy / exact-K accuracy`. Full-coverage accuracy is the primary retrieval accuracy. Exact-K accuracy is a strict regression diagnostic and is affected by the censored K=100 label.",
        "",
        "| Data method | " + " | ".join(MODEL_LABELS[model] for model in MODELS) + " |",
        "|---|" + "---:|" * len(MODELS),
    ]
    for method, method_spec in comparison["methods"].items():
        values = []
        for model in MODELS:
            row = result["cells"][f"{method}__{model}"]
            values.append(
                f"{row['macro_f1']*100:.2f}% / {row['full_coverage_accuracy']*100:.2f}% / {row['exact_k_accuracy']*100:.2f}%"
            )
        lines.append(f"| {method_spec['description']} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## Within-±5 K diagnostic",
        "",
        "| Data method | " + " | ".join(MODEL_LABELS[model] for model in MODELS) + " |",
        "|---|" + "---:|" * len(MODELS),
    ])
    for method, method_spec in comparison["methods"].items():
        values = [
            f"{result['cells'][f'{method}__{model}']['within_5_k_accuracy']*100:.2f}%" for model in MODELS
        ]
        lines.append(f"| {method_spec['description']} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--battery-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_report(args.run_dir.resolve(), args.battery_dir.resolve())
    print(f"[f1-accuracy] wrote {(args.battery_dir / 'f1_accuracy.md').resolve()}")


if __name__ == "__main__":
    main()
