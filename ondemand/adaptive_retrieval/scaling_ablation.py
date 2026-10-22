"""Compare the current raw-feature XGBoost against fold-local standard scaling."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .dataset import _target_from_gold_ranks
from .features import FEATURE_NAMES
from .model import train_evaluate


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "data/work/adaptive_retrieval_20261020_router_te3s_top100"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prepare_rows(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    gate_path = Path(manifest["artifact_root"]) / "light_prep" / "gate_ppocrv5_ungated.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    stored_rows = _read_jsonl(run_dir / "feature_labels.jsonl")
    rows: list[dict[str, Any]] = []
    for row in stored_rows:
        qid = row["qid"]
        gold_ranks = gate["gold_rank"].get(qid)
        if not gold_ranks:
            raise ValueError(f"frozen gold-rank label missing for {qid}")
        expected_k, expected_reachable = _target_from_gold_ranks(gold_ranks, cap=100)
        if int(row["target_k"]) != expected_k or bool(row["reachable_at_cap"]) != expected_reachable:
            raise ValueError(f"stored target/reachability do not match frozen gold ranks for {qid}")
        if set(row["features"]) != set(FEATURE_NAMES):
            raise ValueError(f"feature schema differs for {qid}")
        rows.append({**row, "gold_ranks": gold_ranks})
    if len(rows) != 220 or len({row["qid"] for row in rows}) != 220:
        raise ValueError(f"expected 220 unique training rows, found {len(rows)}")
    return rows, manifest


def _metric_delta(scaled: dict[str, Any], raw: dict[str, Any], section: str) -> dict[str, float]:
    keys = (
        "mean_k", "macro_recall", "macro_precision", "macro_f1",
        "macro_ndcg_at_chosen_k",
    )
    return {key: float(scaled[section][key] - raw[section][key]) for key in keys}


def run_ablation(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if (output_dir / "comparison.json").exists():
        raise FileExistsError(f"{output_dir / 'comparison.json'} already exists; choose another --output-dir")
    rows, manifest = _prepare_rows(run_dir)
    ranking = manifest.get("settings", {})

    results: dict[str, Any] = {}
    for name, scaling in (("unscaled", "none"), ("standard_scaled", "standard")):
        arm_dir = output_dir / name
        started = time.perf_counter()
        report = train_evaluate(
            rows,
            cap=100,
            max_mean_k=60.0,
            output_dir=arm_dir,
            ranking_settings=ranking,
            scaling=scaling,
        )
        report["protocol"]["ranking_reproduction"] = manifest["ranking_reproduction"]
        (arm_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results[name] = {
            "scaling": scaling,
            "elapsed_seconds": time.perf_counter() - started,
            "report": report,
        }
        metrics = report["metrics"]
        print(
            f"[scaling-ablation] {name}: Mean K={metrics['mean_k']:.2f}, "
            f"recall={metrics['macro_recall'] * 100:.2f}%, "
            f"nDCG={metrics['macro_ndcg_at_chosen_k'] * 100:.2f}"
        )

    raw_report = results["unscaled"]["report"]
    standard_report = results["standard_scaled"]["report"]
    original_report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    comparison = {
        "title": "Fold-local StandardScaler ablation for router-matched adaptive-K XGBoost",
        "status": "exploratory_nested_oof",
        "dataset_rows": len(rows),
        "feature_count": len(FEATURE_NAMES),
        "same_outer_and_inner_fold_protocol": True,
        "same_labels_features_model_parameters_and_calibration_grid": True,
        "scaling_method": "StandardScaler: subtract training-fold feature mean and divide by training-fold standard deviation",
        "leakage_control": "Each inner scaler fits only on that inner training partition; each outer scaler fits only on that outer training partition.",
        "raw_arm": {
            "metrics": raw_report["metrics"],
            "regression_diagnostics": raw_report["regression_diagnostics"],
            "elapsed_seconds": results["unscaled"]["elapsed_seconds"],
            "report_path": str(output_dir / "unscaled" / "report.json"),
        },
        "standard_scaled_arm": {
            "metrics": standard_report["metrics"],
            "regression_diagnostics": standard_report["regression_diagnostics"],
            "elapsed_seconds": results["standard_scaled"]["elapsed_seconds"],
            "report_path": str(output_dir / "standard_scaled" / "report.json"),
            "scaler_path": str(output_dir / "standard_scaled" / "feature_scaler.json"),
        },
        "delta_standard_minus_raw": {
            "metrics": _metric_delta(standard_report, raw_report, "metrics"),
            "rmse": float(
                standard_report["regression_diagnostics"]["oof_rmse"]
                - raw_report["regression_diagnostics"]["oof_rmse"]
            ),
            "mae": float(
                standard_report["regression_diagnostics"]["oof_mae"]
                - raw_report["regression_diagnostics"]["oof_mae"]
            ),
        },
        "comparison_to_frozen_original_report": {
            "raw_mean_k_delta": float(raw_report["metrics"]["mean_k"] - original_report["metrics"]["mean_k"]),
            "raw_recall_delta": float(raw_report["metrics"]["macro_recall"] - original_report["metrics"]["macro_recall"]),
            "expected_original_mean_k": original_report["metrics"]["mean_k"],
            "expected_original_recall": original_report["metrics"]["macro_recall"],
        },
        "target_distribution": raw_report["label_distribution"],
        "ranking_settings": ranking,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_format_report(comparison), encoding="utf-8")
    return comparison


def _format_report(comparison: dict[str, Any]) -> str:
    arms = [
        ("Unscaled baseline", "none", comparison["raw_arm"]),
        ("StandardScaler", "standard", comparison["standard_scaled_arm"]),
    ]
    lines = [
        "# Adaptive-K feature scaling ablation",
        "",
        "Exploratory, source-stratified nested 5-fold OOF comparison. Only the input transform changes. The scaler is fitted inside each training fold to avoid leakage.",
        "",
        "| Arm | Scaling | Mean K | Macro recall | Precision | F1 | nDCG@chosen-K | OOF RMSE | OOF MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, scaling, arm in arms:
        metrics = arm["metrics"]
        diagnostic = arm["regression_diagnostics"]
        lines.append(
            f"| {name} | {scaling} | {metrics['mean_k']:.2f} | "
            f"{metrics['macro_recall'] * 100:.2f}% | {metrics['macro_precision'] * 100:.2f}% | "
            f"{metrics['macro_f1'] * 100:.2f}% | {metrics['macro_ndcg_at_chosen_k'] * 100:.2f} | "
            f"{diagnostic['oof_rmse']:.2f} | {diagnostic['oof_mae']:.2f} |"
        )
    delta = comparison["delta_standard_minus_raw"]
    lines.extend([
        "",
        "## Difference (scaled minus unscaled)",
        "",
        f"- Mean K: {delta['metrics']['mean_k']:+.2f}",
        f"- Macro recall: {delta['metrics']['macro_recall'] * 100:+.2f} percentage points",
        f"- Macro nDCG@chosen-K: {delta['metrics']['macro_ndcg_at_chosen_k'] * 100:+.2f} points",
        f"- OOF RMSE / MAE: {delta['rmse']:+.2f} / {delta['mae']:+.2f}",
        "",
        "Feature scaling does not alter the original feature matrix or labels. XGBoost parameters, seeds, outer/inner folds, and the calibration search are held fixed.",
        "",
        f"Raw arm reproduction delta versus the original run: Mean K {comparison['comparison_to_frozen_original_report']['raw_mean_k_delta']:+.4f}; recall {comparison['comparison_to_frozen_original_report']['raw_recall_delta'] * 100:+.4f} percentage points.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR / "scaling_ablation")
    args = parser.parse_args()
    comparison = run_ablation(args.run_dir, args.output_dir)
    print(f"[scaling-ablation] completed: {args.output_dir.resolve()}")
    print(json.dumps(comparison["delta_standard_minus_raw"], indent=2))


if __name__ == "__main__":
    main()
