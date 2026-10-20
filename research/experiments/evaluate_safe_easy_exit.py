"""Evaluate easy-query detection as a selective early-exit problem.

The target is action-aligned: a query is safe to stop at easy_k iff doing so
preserves all recall attainable at the chosen hard cap. Test-fold qrels are
used only for evaluation; labels and neighbour statistics for a test query
come from the outer training fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_DIR = ROOT / "experiments" / "adaptive-k-retrieval"
if str(ADAPTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTIVE_DIR))

from run_canonical_adaptive_methods import load_rows  # noqa: E402
from run_risk_label_statistics import features as score_features  # noqa: E402
from evaluate_easy_hard_model_ablation import (  # noqa: E402
    make_model,
    neighbour_feature_matrix,
)


DEFAULT_OUTPUT = ROOT / "data" / "work" / "safe_easy_exit_20261014"
EASY_K = 30
CAPS = (100, 200)
SEED = 42


def recall_at(row: dict[str, Any], cutoff: int) -> float:
    gold = set(row["gold"])
    return len(set(row["ranked"][:cutoff]) & gold) / max(len(gold), 1)


def saturation_rank(row: dict[str, Any], cap: int) -> int:
    gold = set(row["gold"])
    ranks = [rank for rank, page_id in enumerate(row["ranked"][:cap], 1) if page_id in gold]
    return max(ranks, default=1)


def decisions_at_coverage(risk: np.ndarray, coverage: float) -> np.ndarray:
    count = int(round(len(risk) * coverage))
    easy = np.zeros(len(risk), dtype=bool)
    easy[np.argsort(risk, kind="stable")[:count]] = True
    return easy


def policy_metrics(
    rows: Sequence[dict[str, Any]],
    easy: np.ndarray,
    safe: np.ndarray,
    cap: int,
) -> dict[str, Any]:
    choices = np.where(easy, EASY_K, cap)
    recall = np.asarray([recall_at(row, int(k)) for row, k in zip(rows, choices)])
    cap_recall = np.asarray([recall_at(row, cap) for row in rows])
    selected = int(np.sum(easy))
    return {
        "easy_queries": selected,
        "easy_coverage": float(np.mean(easy)),
        "mean_k": float(np.mean(choices)),
        "macro_recall": float(np.mean(recall)),
        "fixed_cap_recall": float(np.mean(cap_recall)),
        "recall_loss_vs_cap": float(np.mean(cap_recall - recall)),
        "safe_easy_precision": float(np.mean(safe[easy])) if selected else 1.0,
        "unsafe_queries_sent_easy": int(np.sum(easy & ~safe)),
        "unsafe_sent_easy_fraction_all": float(np.mean(easy & ~safe)),
    }


def load_prior_risks(qids: list[str]) -> dict[str, np.ndarray]:
    historical_path = (
        ROOT / "data" / "work" / "adaptive_retrieval_20261009" / "score_confidence_gate.json"
    )
    historical = np.asarray(
        json.loads(historical_path.read_text(encoding="utf-8"))["oof_probability"], dtype=float
    )
    previous_path = ROOT / "data" / "work" / "easy_hard_model_ablation_20261014" / "per_query.jsonl"
    previous_name = "xgboost__score+neighbor_difficulty__high_confidence_50pct"
    mapped: dict[str, float] = {}
    for line in previous_path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["configuration"] == previous_name:
            mapped[str(row["qid"])] = float(row["hard_probability"])
    if len(historical) != len(qids) or set(mapped) != set(qids):
        raise ValueError("Prior OOF risk artifacts do not match the current 220-query order")
    return {
        "historical_k95_hgb": historical,
        "previous_k95_xgboost": np.asarray([mapped[qid] for qid in qids], dtype=float),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, vectors = load_rows("hybrid", max(CAPS))
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    qids = [str(row["qid"]) for row in rows]
    sources = np.asarray([str(row["source"]) for row in rows])
    _, first = score_features(rows[0]["scores"])
    score_matrix = np.vstack([score_features(row["scores"])[1] for row in rows])
    if score_matrix.shape[1] != len(first):
        raise AssertionError(score_matrix.shape)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(score_matrix, sources))
    prior_risks = load_prior_risks(qids)
    report_caps: dict[str, Any] = {}
    per_query: list[dict[str, Any]] = []

    for cap in CAPS:
        cap_recall = np.asarray([recall_at(row, cap) for row in rows])
        easy_recall = np.asarray([recall_at(row, EASY_K) for row in rows])
        loss = cap_recall - easy_recall
        safe = loss <= 1e-12
        unsafe = (~safe).astype(int)
        ksat = np.asarray([saturation_rank(row, cap) for row in rows], dtype=int)

        learned_risks: dict[str, np.ndarray] = {
            "safe_exit_xgboost": np.zeros(len(rows), dtype=float),
            "safe_exit_extra_trees": np.zeros(len(rows), dtype=float),
        }
        for train, test in folds:
            train_neighbor = neighbour_feature_matrix(
                train, train, rows, vectors, ksat, unsafe, sources
            )[:, :10]
            test_neighbor = neighbour_feature_matrix(
                test, train, rows, vectors, ksat, unsafe, sources
            )[:, :10]
            x_train = np.hstack([score_matrix[train], train_neighbor])
            x_test = np.hstack([score_matrix[test], test_neighbor])
            for model_name, output_name in (
                ("xgboost", "safe_exit_xgboost"),
                ("extra_trees", "safe_exit_extra_trees"),
            ):
                model = make_model(model_name, unsafe[train])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x_train, unsafe[train])
                learned_risks[output_name][test] = model.predict_proba(x_test)[:, 1]

        risks = {**prior_risks, **learned_risks}
        coverages = sorted(
            set([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
                + [max(0.0, min(1.0, (cap - 62.49) / (cap - EASY_K)))]
                + [max(0.0, min(1.0, (cap - 72.64) / (cap - EASY_K)))])
        )
        curves: dict[str, list[dict[str, Any]]] = {}
        summaries: dict[str, Any] = {}
        for name, risk in risks.items():
            curves[name] = [
                policy_metrics(rows, decisions_at_coverage(risk, coverage), safe, cap)
                for coverage in coverages
            ]
            summaries[name] = {
                "unsafe_roc_auc": float(roc_auc_score(unsafe, risk)),
                "unsafe_average_precision": float(average_precision_score(unsafe, risk)),
                "at_mean_k_62.49": policy_metrics(
                    rows,
                    decisions_at_coverage(risk, max(0.0, min(1.0, (cap - 62.49) / (cap - EASY_K)))),
                    safe,
                    cap,
                ),
                "at_mean_k_72.64": policy_metrics(
                    rows,
                    decisions_at_coverage(risk, max(0.0, min(1.0, (cap - 72.64) / (cap - EASY_K)))),
                    safe,
                    cap,
                ),
            }

        oracle_easy = safe.copy()
        saturation_choices = ksat
        report_caps[str(cap)] = {
            "fixed_cap_recall": float(np.mean(cap_recall)),
            "safe_easy_queries_at_k30": int(np.sum(safe)),
            "safe_easy_fraction_at_k30": float(np.mean(safe)),
            "two_action_oracle_k30_or_cap": policy_metrics(rows, oracle_easy, safe, cap),
            "variable_saturation_oracle": {
                "mean_k": float(np.mean(saturation_choices)),
                "macro_recall": float(np.mean([
                    recall_at(row, int(k)) for row, k in zip(rows, saturation_choices)
                ])),
                "median_k": float(np.median(saturation_choices)),
                "p90_k": float(np.percentile(saturation_choices, 90)),
            },
            "model_summaries": summaries,
            "risk_coverage_curves": curves,
        }
        for index, row in enumerate(rows):
            per_query.append({
                "qid": row["qid"],
                "source": row["source"],
                "cap": cap,
                "safe_at_k30": bool(safe[index]),
                "recall_loss_k30": float(loss[index]),
                "saturation_rank": int(ksat[index]),
                **{f"risk_{name}": float(risk[index]) for name, risk in risks.items()},
            })

    payload = {
        "protocol": {
            "benchmark": "Physics 80 + Industrial 75 + OHR-Bench 65",
            "easy_action": EASY_K,
            "safe_label": "recall@30 equals recall@cap",
            "outer_folds": 5,
            "test_qrels": "evaluation only",
            "model": "OOF XGBoost/ExtraTrees on score shape + train-fold semantic-neighbour risk",
            "warning": "Risk-coverage operating points are inspected on the same OOF set; calibrate on nested/held-out data before deployment.",
        },
        "caps": report_caps,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Safe easy-exit analysis (exploratory OOF)",
        "",
        "A query is labelled safe for K=30 only when stopping at 30 preserves all recall attainable at the selected cap.",
        "",
        "| Cap | Fixed-cap recall | Safe at K=30 | Two-action oracle mean K | Saturation-oracle mean K |",
        "|---:|---:|---:|---:|---:|",
    ]
    for cap in CAPS:
        item = report_caps[str(cap)]
        lines.append(
            f"| {cap} | {item['fixed_cap_recall']:.2%} | {item['safe_easy_queries_at_k30']}/220 "
            f"({item['safe_easy_fraction_at_k30']:.2%}) | "
            f"{item['two_action_oracle_k30_or_cap']['mean_k']:.2f} | "
            f"{item['variable_saturation_oracle']['mean_k']:.2f} |"
        )
    lines += [
        "",
        "## At requested mean K=62.49",
        "",
        "| Cap | Risk model | Easy coverage | Safe precision | Recall | Loss vs fixed cap |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for cap in CAPS:
        for name, summary in report_caps[str(cap)]["model_summaries"].items():
            metric = summary["at_mean_k_62.49"]
            lines.append(
                f"| {cap} | {name} | {metric['easy_coverage']:.2%} | "
                f"{metric['safe_easy_precision']:.2%} | {metric['macro_recall']:.2%} | "
                f"-{metric['recall_loss_vs_cap']:.2%} |"
            )
    lines += [
        "",
        "## Notes",
        "",
        "- The old `k95 > 50` label is not aligned with the K=30 early-exit action.",
        "- The two-action oracle shows what K=30-versus-cap routing can achieve even with perfect labels.",
        "- The variable saturation oracle uses the last reachable relevant rank and preserves cap recall; it is diagnostic and cannot be computed online.",
        "- A deployable no-hurt claim requires held-out or conformal calibration; OOF screening alone cannot guarantee zero recall loss.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report_caps, indent=2)[:20000])
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
