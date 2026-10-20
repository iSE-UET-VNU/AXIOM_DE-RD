"""Post-hoc diagnostic of an ExtraTrees safe-exit rescue on historical HGB hard routes."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "research" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from adaptive_k_optimization_core import load_frozen_data  # noqa: E402
from compute_adaptive_k_full_metrics import policy_metrics  # noqa: E402


OUTPUT = ROOT / "data" / "work" / "extratrees_false_hard_rescue_20261016"


def row_recall(row: dict, k: int) -> float:
    gold = set(row["gold"])
    return len(gold & set(row["ranked"][:k])) / max(len(gold), 1)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_frozen_data()
    historical = json.loads(
        (ROOT / "data/work/adaptive_retrieval_20261009/score_confidence_gate.json").read_text(encoding="utf-8")
    )
    probability = np.asarray(historical["oof_probability"], dtype=float)
    hard = np.zeros(len(data.rows), dtype=bool)
    hard[np.argsort(-probability, kind="stable")[:134]] = True

    records = {}
    source = ROOT / "data/work/safe_easy_exit_20261014/per_query.jsonl"
    for line in source.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if int(item["cap"]) == 100:
            records[str(item["qid"])] = item
    risk = np.asarray([
        records[str(row["qid"])]["risk_safe_exit_extra_trees"] for row in data.rows
    ])
    recall30 = np.asarray([row_recall(row, 30) for row in data.rows])
    recall100 = np.asarray([row_recall(row, 100) for row in data.rows])
    unsafe = ~np.isclose(recall30, recall100)

    base_choices = np.where(hard, 100, 30)
    candidates = np.flatnonzero(hard)
    rescue_order = candidates[np.argsort(risk[candidates], kind="stable")]
    curve = []
    for rescued in range(len(candidates) + 1):
        choices = base_choices.copy()
        choices[rescue_order[:rescued]] = 30
        curve.append({"rescued": rescued, **policy_metrics(data.rows, choices)})

    base = curve[0]
    thresholds = [base["macro_recall"], 0.86, 0.855, 0.8537, 0.8522584932359281, 0.85]
    operating_points = {}
    for threshold in thresholds:
        eligible = [item for item in curve if item["macro_recall"] >= threshold]
        operating_points[f"recall_ge_{threshold:.9f}"] = max(
            eligible, key=lambda item: item["rescued"]
        ) if eligible else None

    oracle_choices = base_choices.copy()
    oracle_choices[hard & ~unsafe] = 30
    payload = {
        "protocol": {
            "base": "historical HGB OOF, top 134 hard routes, actions 30/100",
            "rescue_model": "existing fold-safe ExtraTrees unsafe-risk from safe_easy_exit_20261014; score plus train-neighbour features",
            "operation": "rank HGB-hard queries by lowest unsafe risk and change K=100 to K=30",
            "warning": "Rescue count is selected post-hoc on the OOF curve; diagnostic, not nested validation.",
        },
        "hard_routes": int(np.sum(hard)),
        "actually_safe_among_hard": int(np.sum(hard & ~unsafe)),
        "hard_subset_unsafe_roc_auc": float(roc_auc_score(unsafe[hard], risk[hard])),
        "hard_subset_unsafe_average_precision": float(average_precision_score(unsafe[hard], risk[hard])),
        "base": base,
        "oracle_false_hard_rescue": policy_metrics(data.rows, oracle_choices),
        "operating_points": operating_points,
        "curve": curve,
    }
    (OUTPUT / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "curve"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
