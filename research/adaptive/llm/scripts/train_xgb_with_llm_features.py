"""Nested-OOF test: add inference-time pagewise-LLM signals to continuous-K XGB.

This is an exploratory research runner. It deliberately reads only trajectory
fields available at inference; post-hoc qrels/evaluation fields are ignored.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ondemand.adaptive_retrieval.features import FEATURE_NAMES
from ondemand.adaptive_retrieval.model import (
    _choose_calibration,
    _integer_k,
    _new_model,
    _row_metrics,
    _transform_fold,
)


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data/work/adaptive_retrieval_20261020_router_te3s_top100/feature_labels.jsonl"
GATE = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/light_prep/gate_ppocrv5_ungated.json"
TRAJECTORIES = ROOT / "research/adaptive/llm/results/general_all_220_b1_cap20_xgbfallback_20261021/trajectories.jsonl"
OUT = ROOT / "research/adaptive/llm/results/xgb_llm_features_nested_oof_20261021"
CAP = 100
CALIBRATION_BUDGETS = (50.0, 52.0, 55.0, 60.0)

LLM_FEATURES = (
    "llm_stopped",
    "llm_observed_k",
    "llm_calls",
    "llm_last_confidence",
    "llm_any_answer",
    "llm_any_citation",
    "llm_max_citations",
    "llm_max_resolved_claims",
    "llm_last_missing_count",
    "llm_last_contradiction_count",
    "llm_parse_error_fraction",
    "llm_accepted_stop_fraction",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def trajectory_features(row: dict[str, Any]) -> dict[str, float]:
    steps = row.get("steps") or []
    finals = [step.get("result") or {} for step in steps]
    requests = [step.get("request") or {} for step in steps]
    last = finals[-1] if finals else {}
    parse_errors = sum(bool(step.get("parse_error")) for step in steps)
    accepted = sum(bool(step.get("accepted_stop")) for step in steps)
    citations = [item for result in finals for item in (result.get("cited_page_ids") or [])]
    claims = [item for result in finals for item in (result.get("resolved_claims") or [])]
    # The trajectory's qrels-derived fields (recall, safety, gap, etc.) are
    # intentionally not read. chosen_k is an observable stopping depth.
    return {
        "llm_stopped": float(bool(row.get("stopped_by_llm"))),
        "llm_observed_k": float(row.get("chosen_k", row.get("cap", 20))),
        "llm_calls": float(row.get("calls", len(steps))),
        "llm_last_confidence": float(last.get("confidence", 0.0) or 0.0),
        "llm_any_answer": float(any(bool(result.get("provisional_answer")) for result in finals)),
        "llm_any_citation": float(bool(citations)),
        "llm_max_citations": float(max((len(result.get("cited_page_ids") or []) for result in finals), default=0)),
        "llm_max_resolved_claims": float(max((len(result.get("resolved_claims") or []) for result in finals), default=0)),
        "llm_last_missing_count": float(len(last.get("missing_information") or [])),
        "llm_last_contradiction_count": float(len(last.get("contradictions") or [])),
        "llm_parse_error_fraction": float(parse_errors / max(len(steps), 1)),
        "llm_accepted_stop_fraction": float(accepted / max(len(steps), 1)),
    }


def main() -> None:
    data = {row["qid"]: row for row in read_jsonl(DATA)}
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    trajectories = {row["qid"]: row for row in read_jsonl(TRAJECTORIES)}
    if len(data) != 220 or len(trajectories) != 220 or set(data) != set(trajectories):
        raise ValueError(f"expected aligned 220 rows; got data={len(data)}, trajectories={len(trajectories)}")

    rows = [dict(data[qid], gold_ranks=gate["gold_rank"][qid]) for qid in sorted(data)]
    qids = [row["qid"] for row in rows]
    sources = np.asarray([row["source"] for row in rows])
    y = np.asarray([row["target_k"] for row in rows], dtype=np.float32)
    gold = [row["gold_ranks"] for row in rows]
    base_x = np.asarray([[row["features"][name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    extra = [trajectory_features(trajectories[qid]) for qid in qids]
    llm_x = np.asarray([[feature[name] for name in LLM_FEATURES] for feature in extra], dtype=np.float32)
    matrices = {"xgb_score_only": base_x, "xgb_plus_llm_trajectory": np.column_stack((base_x, llm_x))}
    if not np.isfinite(base_x).all() or not np.isfinite(llm_x).all():
        raise ValueError("feature matrix contains NaN/Inf")

    OUT.mkdir(parents=True, exist_ok=True)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_predictions: dict[str, dict[float, np.ndarray]] = {
        arm: {budget: np.full(220, np.nan, dtype=float) for budget in CALIBRATION_BUDGETS}
        for arm in matrices
    }
    fold_log: list[dict[str, Any]] = []
    for fold, (train_i, test_i) in enumerate(outer.split(base_x, sources), start=1):
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=100 + fold)
        for arm, matrix in matrices.items():
            inner_oof = np.full(len(train_i), np.nan, dtype=float)
            for inner_fold, (inner_train, inner_valid) in enumerate(inner.split(matrix[train_i], sources[train_i])):
                model = _new_model(100 + fold + inner_fold)
                train_x, valid_x, _ = _transform_fold(matrix[train_i[inner_train]], matrix[train_i[inner_valid]], "none")
                model.fit(train_x, y[train_i[inner_train]])
                inner_oof[inner_valid] = model.predict(valid_x)
            model = _new_model(1000 + fold)
            train_x, test_x, _ = _transform_fold(matrix[train_i], matrix[test_i], "none")
            model.fit(train_x, y[train_i])
            pred = model.predict(test_x).astype(float)
            for budget in CALIBRATION_BUDGETS:
                calibration = _choose_calibration(inner_oof, [gold[i] for i in train_i], CAP, budget)
                k = _integer_k(pred, calibration["scale"], calibration["margin"], CAP)
                all_predictions[arm][budget][test_i] = k
                metrics = _row_metrics(k, [gold[i] for i in test_i])
                fold_log.append({"arm": arm, "fold": fold, "budget": budget, "calibration": calibration, "test_metrics": metrics})
            print(f"[xgb-llm] fold {fold}/5 {arm} done")

    if any(not np.isfinite(values).all() for arms in all_predictions.values() for values in arms.values()):
        raise RuntimeError("outer OOF predictions incomplete")
    summaries: dict[str, Any] = {}
    per_source: dict[str, Any] = {}
    query_rows: list[dict[str, Any]] = []
    for arm, budget_predictions in all_predictions.items():
        summaries[arm] = {}
        per_source[arm] = {}
        for budget, pred in budget_predictions.items():
            metrics = _row_metrics(pred.astype(int), gold)
            summaries[arm][str(int(budget))] = metrics
            per_source[arm][str(int(budget))] = {
                source: _row_metrics(pred[sources == source].astype(int), [gold[i] for i in np.flatnonzero(sources == source)])
                for source in sorted(set(sources))
            }
            for i, qid in enumerate(qids):
                query_rows.append({"arm": arm, "calibration_budget": int(budget), "qid": qid, "source": sources[i], "prediction_k_oof": int(pred[i]), "target_k": int(y[i])})

    report = {
        "title": "Nested OOF XGBoost with pagewise LLM trajectory features",
        "protocol": {
            "queries": 220,
            "outer_folds": "StratifiedKFold by source, 5 folds, random_state=42",
            "inner_folds": "StratifiedKFold by source, 4 folds; calibration fit only on inner OOF train rows",
            "label": "frozen target_k from PP-OCRv5 / text-embedding-3-small top-100 dataset",
            "ranking_and_base_features_frozen": True,
            "llm_trajectory": "one page per call, max 20, only observable prompt/response fields",
            "excluded_from_features": ["gold ranks", "recall", "safe_30_vs_100", "safe_30_vs_200", "gap_predicted_minus_gold", "deployed_recall"],
            "calibration_budgets": list(CALIBRATION_BUDGETS),
            "selection_metric": "inner-OOF max recall under mean-K budget; outer folds never used for selection",
            "llm_features": list(LLM_FEATURES),
        },
        "summary": summaries,
        "per_source": per_source,
        "folds": fold_log,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "calibration_budget", "qid", "source", "prediction_k_oof", "target_k"])
        writer.writeheader()
        writer.writerows(query_rows)
    with (OUT / "llm_features.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["qid", "source", *LLM_FEATURES])
        writer.writeheader()
        writer.writerows([{"qid": qid, "source": sources[i], **extra[i]} for i, qid in enumerate(qids)])

    md = ["# XGBoost + LLM trajectory features (nested OOF)", "", "All figures are nested OOF over the same 220 queries. Calibration selection is inside outer-training folds only.", "", "| Arm | MeanK budget | Mean K | Recall | Precision | F1 | nDCG |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, budgets in summaries.items():
        for budget, metrics in budgets.items():
            md.append(f"| {arm} | {budget} | {metrics['mean_k']:.2f} | {metrics['macro_recall']*100:.2f}% | {metrics['macro_precision']*100:.2f}% | {metrics['macro_f1']*100:.2f}% | {metrics['macro_ndcg_at_chosen_k']*100:.2f}% |")
    md += ["", "## By source", ""]
    for arm, budgets in per_source.items():
        md.extend([f"### {arm}", "", "| Budget | Source | n | Mean K | Recall |", "|---:|---|---:|---:|---:|"])
        for budget, source_rows in budgets.items():
            for source, metrics in source_rows.items():
                md.append(f"| {budget} | {source} | {int(np.sum(sources == source))} | {metrics['mean_k']:.2f} | {metrics['macro_recall']*100:.2f}% |")
        md.append("")
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[xgb-llm] wrote {OUT}")


if __name__ == "__main__":
    main()
