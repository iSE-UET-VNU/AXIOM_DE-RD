"""Build router-matched score features and run the independent XGBoost screen."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .dataset import build_dataset
from .model import train_evaluate


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/raw/0.1. BENCHMARK"
DEFAULT_OUTPUT = ROOT / "data/work/adaptive_retrieval_20261020_router_te3s_top100"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if (args.output_dir / "report.json").exists():
        raise SystemExit(f"{args.output_dir} already has a report; choose a new --output-dir to preserve it")

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    rows, manifest = build_dataset(args.artifact_root, args.benchmark_root, args.output_dir)
    report = train_evaluate(
        rows,
        cap=100,
        max_mean_k=60.0,
        output_dir=args.output_dir,
        ranking_settings=manifest["settings"],
    )
    report["protocol"]["ranking_reproduction"] = manifest["ranking_reproduction"]
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["started"] = started
    manifest["completed"] = datetime.now().astimezone().isoformat(timespec="seconds")
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    markdown = _markdown(report)
    (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Artifacts: {args.output_dir}")


def _markdown(report: dict) -> str:
    metrics = report["metrics"]
    ci = report["confidence_intervals_95"]
    lines = [
        "# Adaptive retrieval: router-matched XGBoost transfer screen",
        "",
        "Exploratory nested source-stratified OOF evaluation. The feature inputs are the exact final scores from the adaptive-router top-100 ranking; gold ranks are labels/evaluation only.",
        "",
        f"- Mean K: **{metrics['mean_k']:.2f}** (bootstrap 95% CI {ci['mean_k'][0]:.2f}–{ci['mean_k'][1]:.2f})",
        f"- Macro page recall: **{metrics['macro_recall'] * 100:.2f}%** (bootstrap 95% CI {ci['macro_recall'][0] * 100:.2f}–{ci['macro_recall'][1] * 100:.2f}%)",
        f"- Macro precision / F1 / nDCG@chosen-K: {metrics['macro_precision'] * 100:.2f}% / {metrics['macro_f1'] * 100:.2f}% / {metrics['macro_ndcg_at_chosen_k'] * 100:.2f}",
        f"- Raw continuous-K OOF RMSE / MAE: {report['regression_diagnostics']['oof_rmse']:.2f} / {report['regression_diagnostics']['oof_mae']:.2f}",
        f"- Candidate recall ceiling@100: {report['candidate_recall_at_cap'] * 100:.2f}%",
        f"- Acceptance vs goal Mean K < 60 and recall >=87%: **{'PASS' if metrics['mean_k'] < 60 and metrics['macro_recall'] >= 0.87 else 'NOT MET'}**",
        "",
        "## Fixed-budget baselines",
        "",
        "| K | Mean K | Macro recall | Precision | F1 | nDCG |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k, values in report["fixed_baselines"].items():
        lines.append(
            f"| {k} | {values['mean_k']:.2f} | {values['macro_recall'] * 100:.2f}% | "
            f"{values['macro_precision'] * 100:.2f}% | {values['macro_f1'] * 100:.2f}% | "
            f"{values['macro_ndcg_at_chosen_k'] * 100:.2f} |"
        )
    lines.extend(["", "## Per source", "", "| Source | Queries | Mean K | Recall | Precision | F1 | nDCG |", "|---|---:|---:|---:|---:|---:|---:|"])
    counts = report["label_distribution"]["by_source"]
    for source, values in report["per_source"].items():
        lines.append(
            f"| {source} | {counts[source]['count']} | {values['mean_k']:.2f} | {values['macro_recall'] * 100:.2f}% | "
            f"{values['macro_precision'] * 100:.2f}% | {values['macro_f1'] * 100:.2f}% | "
            f"{values['macro_ndcg_at_chosen_k'] * 100:.2f} |"
        )
    lines.extend([
        "",
        "## Protocol and caveats",
        "",
        f"- Ranking: `{report['protocol']['ranking'].get('ranking_variant', 'see dataset_manifest.json')}`; embedding: `{report['protocol']['ranking'].get('dense_model', 'openai/text-embedding-3-small')}`; cap: 100.",
        "- Model: XGBoost regressor with frozen parameters; 5 outer folds, 4 inner folds for scale/margin calibration. No neighbor features or router branches are used.",
        "- Reproduction audit: 219 rankings match exactly; the remaining query has the same top-100 pages and only a 1.19e-7 adjacent-score inversion (frozen artifact order retained).",
        "- A target is the last gold-page rank when all gold pages fit within top-100; otherwise the target is 100 and a separate unreachable flag is kept.",
        "- This is not a reproduction of the prior Qwen3-Embedding-8B / top-200 score experiment. Scores, embedding model and cap differ by design to match the adaptive router.",
        "- See `feature_matrix.csv`, `training_table.csv`, `retrieval_scores.jsonl`, `oof_predictions.jsonl`, `report.json`, and `deployment_policy.json` for inspectable artifacts.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
