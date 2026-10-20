"""OOF learned page budgets over the cached mixed-text file ranking."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DATASET = ROOT / "data/raw/BENCHMARK"
OUTPUT = CACHE / "mixed_learned_budget"
K_VALUES = (4, 8, 12, 16, 20)


def metric(qids: list[str], pages: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict[str, float]:
    result, r10, r20 = [], [], []
    for qid in qids:
        gold = qrels[qid]
        selected = pages[qid]
        denom = max(len(gold), 1)
        result.append(len(set(selected) & gold) / denom)
        r10.append(len(set(selected[:10]) & gold) / denom)
        r20.append(len(set(selected[:20]) & gold) / denom)
    return {"recall_at_returned_k": float(np.mean(result)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean([len(pages[qid]) for qid in qids]))}


def feature_rows(queries: list[dict[str, Any]], ranking_rows: dict[str, dict[str, Any]]) -> np.ndarray:
    values = []
    for query in queries:
        qid = str(query["query_id"])
        row = ranking_rows[qid]
        scores = np.asarray(row.get("scores", [])[:20], dtype=float)
        if len(scores) < 20:
            scores = np.pad(scores, (0, 20 - len(scores)), mode="edge" if len(scores) else "constant")
        probs = np.maximum(scores, 0) / (np.sum(np.maximum(scores, 0)) + 1e-9)
        values.append([
            float(len(str(query["query"]).split())), float(len(str(query["query"]))),
            float(scores[0]), float(scores[0] - scores[1]), float(scores[0] - scores[9]),
            float(scores[11] - scores[12]), float(-np.sum(probs * np.log(probs + 1e-9))),
            *[float(scores[i]) for i in (1, 4, 9, 19)],
        ])
    return np.asarray(values, dtype=float)


def main() -> int:
    rows = [json.loads(line) for line in (CACHE / "adaptive_mixed_text/all_policy_rows.jsonl").open(encoding="utf-8")]
    queries = read_jsonl(DATASET / "queries.jsonl")
    qids = [str(query["query_id"]) for query in queries]
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in [json.loads(line) for line in (CACHE / "qrels_page_level.jsonl").open(encoding="utf-8")]:
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    fixed_rows = {row["query_id"]: row for row in rows if row["policy"] == "file_fixed_k12"}
    ranking_rows = {row["query_id"]: row for row in rows if row["policy"] == "file_largest_gap_plus15"}
    if set(ranking_rows) != set(qids):
        raise RuntimeError("Missing mixed-text ranking rows")
    pages = {qid: list(ranking_rows[qid]["pages"]) for qid in qids}
    # The offset-15 row is deep enough to expose the first 20 pages for nearly
    # every query; retain only the available prefix if a candidate list ends.
    x = feature_rows(queries, ranking_rows)
    y = np.asarray([[len(set(pages[qid][:k]) & qrels[qid]) / max(len(qrels[qid]), 1) for k in K_VALUES] for qid in qids], dtype=float)
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    groups = np.asarray([str(query["source"]) for query in queries])
    fold_rows = []
    oof_k = np.zeros(len(qids), dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, groups), start=1):
        scaler = StandardScaler().fit(x[train_idx])
        train_x, test_x = scaler.transform(x[train_idx]), scaler.transform(x[test_idx])
        models = []
        for column in range(len(K_VALUES)):
            model = ExtraTreesRegressor(n_estimators=250, min_samples_leaf=6, max_features=0.8, random_state=500 + column)
            model.fit(train_x, y[train_idx, column])
            models.append(model)
        pred_train = np.column_stack([model.predict(train_x) for model in models])
        pred_test = np.column_stack([model.predict(test_x) for model in models])
        # Deployment calibration uses score predictions only, not test qrels.
        best = None
        for penalty in np.linspace(-0.03, 0.05, 801):
            train_choice = np.argmax(pred_train - penalty * np.asarray(K_VALUES)[None, :], axis=1)
            mean_k = float(np.mean(np.asarray(K_VALUES)[train_choice]))
            key = (abs(mean_k - 12.0), -float(np.mean(pred_train[np.arange(len(train_choice)), train_choice])))
            if best is None or key < best[0]:
                best = (key, penalty)
        penalty = float(best[1])
        test_choice = np.argmax(pred_test - penalty * np.asarray(K_VALUES)[None, :], axis=1)
        selected_k = np.asarray([K_VALUES[int(index)] for index in test_choice], dtype=int)
        oof_k[test_idx] = selected_k
        test_qids = [qids[i] for i in test_idx]
        selected_pages = {qid: pages[qid][:int(k)] for qid, k in zip(test_qids, selected_k)}
        fold_rows.append({"fold": fold, "metrics": metric(test_qids, selected_pages, qrels), "mean_train_k": float(np.mean(np.asarray(K_VALUES)[np.argmax(pred_train - penalty * np.asarray(K_VALUES)[None, :], axis=1)])), "choice_histogram": {str(k): int(np.sum(selected_k == k)) for k in K_VALUES}})
    oof_pages = {qid: pages[qid][:int(k)] for qid, k in zip(qids, oof_k)}
    fixed_pages = {qid: fixed_rows[qid]["pages"] for qid in qids}
    report = {"queries": len(qids), "features": int(x.shape[1]), "qrels_used_for_test_selection": False, "fixed_file_k12": metric(qids, fixed_pages, qrels), "oof_learned_budget": metric(qids, oof_pages, qrels), "folds": fold_rows, "notes": ["The ranker/budget model is trained only on train-fold qrels; penalty calibration is score-prediction-only.", "This is an exploratory OOF screen, not a final model-selection claim."]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Mixed-text learned adaptive budget", "", "| Method | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    for name, value in (("Fixed file K=12", report["fixed_file_k12"]), ("OOF learned budget", report["oof_learned_budget"])):
        lines.append(f"| {name} | {value['recall_at_returned_k']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} | {value['mean_k']:.2f} |")
    lines += ["", "| Fold | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---:|---:|---:|---:|---:|"]
    for row in fold_rows:
        value = row["metrics"]
        lines.append(f"| {row['fold']} | {value['recall_at_returned_k']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} | {value['mean_k']:.2f} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    from research.data_discovery.raw_hierarchical import read_jsonl
    raise SystemExit(main())
