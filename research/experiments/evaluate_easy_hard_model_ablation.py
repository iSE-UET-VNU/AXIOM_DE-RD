"""OOF ablation for easy/hard adaptive-retrieval models.

This is an exploratory experiment on the active merged 220-query benchmark.
It compares:

* score-curve features versus score-curve + semantic-neighbour features;
* all training queries versus only the clearest easy/hard training examples;
* several small tabular classifiers, including XGBoost and LightGBM.

All neighbour labels and neighbour qrels used to construct a test query's
features come from that query's outer training fold. Test-fold qrels are used
only to create the evaluation label and retrieval metrics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_DIR = ROOT / "experiments" / "adaptive-k-retrieval"
if str(ADAPTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTIVE_DIR))

from run_canonical_adaptive_methods import load_rows  # noqa: E402
from run_risk_label_statistics import features as score_features  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "work" / "easy_hard_model_ablation_20261014"
MAX_K = 200
EASY_K = 30
HARD_KS = (100, 120)
PRIOR_HARD_COUNT = 134  # 0.60909 * 220; reproduces the earlier threshold-0.2 budget.
HARD_LABEL_K = 50
NEIGHBOURS = 5
SEED = 42


def k95(row: dict[str, Any]) -> int:
    gold = set(row["gold"])
    for rank, page_id in enumerate(row["ranked"][:MAX_K], start=1):
        if page_id in gold:
            found = len(set(row["ranked"][:rank]) & gold)
            if found / max(len(gold), 1) >= 0.95:
                return rank
    return MAX_K + 1


def recall_at(row: dict[str, Any], cutoff: int) -> float:
    gold = set(row["gold"])
    return len(set(row["ranked"][:cutoff]) & gold) / max(len(gold), 1)


def file_id(page_id: str) -> str:
    return page_id.rsplit("#page=", 1)[0]


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.sum(values * weights))
    variance = float(np.sum(weights * (values - mean) ** 2))
    return mean, math.sqrt(max(variance, 0.0))


NEIGHBOUR_NAMES = [
    "global_sim_top1",
    "global_sim_mean5",
    "global_sim_std5",
    "global_hard_rate5",
    "global_k95_mean5",
    "global_k95_std5",
    "same_source_sim_top1",
    "same_source_sim_mean5",
    "same_source_hard_rate5",
    "same_source_k95_mean5",
    "neighbor_page_vote_top10",
    "neighbor_page_vote_top20",
    "neighbor_page_vote_top50",
    "neighbor_page_vote_top100",
    "neighbor_page_vote_top200",
    "neighbor_page_vote_max_top50",
    "neighbor_page_first_rank_score",
    "neighbor_file_vote_top10",
    "neighbor_file_vote_top50",
    "neighbor_file_vote_top200",
]


def _top_neighbours(
    target_index: int,
    memory_indices: np.ndarray,
    similarities: np.ndarray,
    sources: np.ndarray,
    same_source: bool,
) -> tuple[np.ndarray, np.ndarray]:
    allowed = np.ones(len(memory_indices), dtype=bool)
    allowed &= memory_indices != target_index
    if same_source:
        allowed &= sources[memory_indices] == sources[target_index]
    candidates = np.flatnonzero(allowed)
    if len(candidates) == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    order = candidates[np.argsort(-similarities[candidates], kind="stable")[:NEIGHBOURS]]
    neighbour_indices = memory_indices[order]
    neighbour_similarities = similarities[order].astype(float)
    return neighbour_indices, neighbour_similarities


def _weights(similarities: np.ndarray) -> np.ndarray:
    if len(similarities) == 0:
        return similarities
    shifted = (similarities - np.max(similarities)) / 0.05
    values = np.exp(np.clip(shifted, -50, 0))
    return values / (np.sum(values) + 1e-12)


def neighbour_feature_matrix(
    target_indices: np.ndarray,
    memory_indices: np.ndarray,
    rows: Sequence[dict[str, Any]],
    vectors: np.ndarray,
    k95_values: np.ndarray,
    hard_labels: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    result: list[list[float]] = []
    memory_vectors = vectors[memory_indices]
    for target_index in target_indices:
        similarities = np.asarray(vectors[target_index] @ memory_vectors.T, dtype=float)
        global_ids, global_sims = _top_neighbours(
            int(target_index), memory_indices, similarities, sources, same_source=False
        )
        source_ids, source_sims = _top_neighbours(
            int(target_index), memory_indices, similarities, sources, same_source=True
        )

        global_w = _weights(global_sims)
        source_w = _weights(source_sims)
        if len(global_ids):
            global_k_mean, global_k_std = weighted_stats(
                k95_values[global_ids].astype(float), global_w
            )
            global_values = [
                float(global_sims[0]),
                float(np.mean(global_sims)),
                float(np.std(global_sims)),
                float(np.sum(global_w * hard_labels[global_ids])),
                global_k_mean,
                global_k_std,
            ]
        else:
            global_values = [0.0] * 6

        if len(source_ids):
            source_k_mean, _ = weighted_stats(k95_values[source_ids].astype(float), source_w)
            source_values = [
                float(source_sims[0]),
                float(np.mean(source_sims)),
                float(np.sum(source_w * hard_labels[source_ids])),
                source_k_mean,
            ]
        else:
            source_values = [0.0] * 4

        page_votes: defaultdict[str, float] = defaultdict(float)
        file_votes: defaultdict[str, float] = defaultdict(float)
        for neighbour_index, weight in zip(source_ids, source_w):
            gold_pages = set(rows[int(neighbour_index)]["gold"])
            for page_id in gold_pages:
                page_votes[page_id] += float(weight) / max(len(gold_pages), 1)
            gold_files = {file_id(page_id) for page_id in gold_pages}
            for gold_file in gold_files:
                file_votes[gold_file] += float(weight) / max(len(gold_files), 1)

        ranked = list(rows[int(target_index)]["ranked"])
        page_mass = [
            float(sum(page_votes.get(page_id, 0.0) for page_id in ranked[:cutoff]))
            for cutoff in (10, 20, 50, 100, 200)
        ]
        max_vote_top50 = float(max((page_votes.get(page_id, 0.0) for page_id in ranked[:50]), default=0.0))
        first_rank = next((rank for rank, page_id in enumerate(ranked, 1) if page_votes.get(page_id, 0.0) > 0), MAX_K + 1)
        first_rank_score = float(1.0 - min(first_rank, MAX_K + 1) / (MAX_K + 1))
        file_mass = []
        for cutoff in (10, 50, 200):
            seen_files = {file_id(page_id) for page_id in ranked[:cutoff]}
            file_mass.append(float(sum(file_votes.get(item, 0.0) for item in seen_files)))

        result.append(
            global_values
            + source_values
            + page_mass
            + [max_vote_top50, first_rank_score]
            + file_mass
        )
    matrix = np.asarray(result, dtype=float)
    if matrix.shape != (len(target_indices), len(NEIGHBOUR_NAMES)):
        raise AssertionError(f"Unexpected neighbour feature shape: {matrix.shape}")
    return matrix


def confident_subset(
    train_indices: np.ndarray,
    labels: np.ndarray,
    k95_values: np.ndarray,
    retain_fraction: float,
) -> np.ndarray:
    if retain_fraction >= 1.0:
        return np.arange(len(train_indices), dtype=int)
    selected: list[int] = []
    local_labels = labels[train_indices]
    local_k95 = k95_values[train_indices]
    for label in (0, 1):
        positions = np.flatnonzero(local_labels == label)
        count = max(2, int(math.ceil(retain_fraction * len(positions))))
        if label == 0:
            order = positions[np.argsort(local_k95[positions], kind="stable")]
        else:
            order = positions[np.argsort(-local_k95[positions], kind="stable")]
        selected.extend(order[:count].tolist())
    return np.asarray(sorted(selected), dtype=int)


def make_model(name: str, y: np.ndarray) -> Any:
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=4,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=7, random_state=SEED
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        positive = max(int(np.sum(y == 1)), 1)
        negative = max(int(np.sum(y == 0)), 1)
        return XGBClassifier(
            n_estimators=220,
            max_depth=3,
            learning_rate=0.035,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negative / positive,
            random_state=SEED,
            n_jobs=4,
            tree_method="hist",
            verbosity=0,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=220,
            num_leaves=7,
            learning_rate=0.035,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=4,
            verbosity=-1,
        )
    raise ValueError(name)


def normalized_importance(model: Any, feature_count: int) -> np.ndarray | None:
    fitted = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(fitted, "feature_importances_"):
        values = np.asarray(fitted.feature_importances_, dtype=float)
    elif hasattr(fitted, "coef_"):
        values = np.abs(np.asarray(fitted.coef_[0], dtype=float))
    else:
        return None
    if len(values) != feature_count:
        return None
    return values / (np.sum(values) + 1e-12)


def bootstrap_delta_interval(deltas: np.ndarray, samples: int = 2000) -> list[float]:
    rng = np.random.default_rng(20261014)
    n = len(deltas)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = float(np.mean(deltas[rng.integers(0, n, n)]))
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def policy_metrics(
    rows: Sequence[dict[str, Any]],
    choices: np.ndarray,
    actual_hard: np.ndarray,
    predicted_hard: np.ndarray,
) -> dict[str, Any]:
    recalls = np.asarray([recall_at(row, int(k)) for row, k in zip(rows, choices)], dtype=float)
    mean_k = float(np.mean(choices))
    fixed_k = int(round(mean_k))
    fixed_recalls = np.asarray([recall_at(row, fixed_k) for row in rows], dtype=float)
    delta = recalls - fixed_recalls
    hard_mask = actual_hard == 1
    easy_mask = ~hard_mask
    false_easy = hard_mask & (predicted_hard == 0)
    return {
        "mean_k": mean_k,
        "adaptive_recall": float(np.mean(recalls)),
        "matched_fixed_k": fixed_k,
        "matched_fixed_recall": float(np.mean(fixed_recalls)),
        "delta_vs_matched_fixed": float(np.mean(delta)),
        "delta_bootstrap_95ci": bootstrap_delta_interval(delta),
        "hard_query_recall": float(np.mean(recalls[hard_mask])) if np.any(hard_mask) else 0.0,
        "easy_query_recall": float(np.mean(recalls[easy_mask])) if np.any(easy_mask) else 0.0,
        "hard_sent_easy_joint_fraction": float(np.mean(false_easy)),
        "hard_false_negative_rate": float(np.mean(predicted_hard[hard_mask] == 0)) if np.any(hard_mask) else 0.0,
        "predicted_hard_fraction": float(np.mean(predicted_hard)),
    }


def top_budget_predictions(probabilities: np.ndarray, hard_count: int) -> np.ndarray:
    prediction = np.zeros(len(probabilities), dtype=int)
    order = np.argsort(-probabilities, kind="stable")
    prediction[order[:hard_count]] = 1
    return prediction


def selected_report_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    baseline = next(
        row
        for row in results
        if row["model"] == "hist_gradient_boosting"
        and row["feature_set"] == "score"
        and row["training_regime"] == "all"
    )
    selected.append(baseline)
    for model in sorted({row["model"] for row in results}):
        candidates = [
            row
            for row in results
            if row["model"] == model and row["feature_set"] == "score+neighbors"
        ]
        selected.append(max(candidates, key=lambda item: item["budget_30_100"]["adaptive_recall"]))
    unique: dict[str, dict[str, Any]] = {row["name"]: row for row in selected}
    return list(unique.values())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, query_vectors = load_rows("hybrid", MAX_K)
    vectors = np.asarray(query_vectors, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    qids = [str(row["qid"]) for row in rows]
    sources = np.asarray([str(row["source"]) for row in rows])
    k95_values = np.asarray([k95(row) for row in rows], dtype=int)
    score_names, _ = score_features(rows[0]["scores"])
    base_matrix = np.vstack([score_features(row["scores"])[1] for row in rows])

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(base_matrix, sources))
    fold_ids = np.zeros(len(rows), dtype=int)
    actual_hard = (k95_values > HARD_LABEL_K).astype(int)
    fold_matrices: dict[int, tuple[np.ndarray, ...]] = {}
    for fold, (train, test) in enumerate(folds):
        train_neighbor = neighbour_feature_matrix(
            train, train, rows, vectors, k95_values, actual_hard, sources
        )
        test_neighbor = neighbour_feature_matrix(
            test, train, rows, vectors, k95_values, actual_hard, sources
        )
        fold_matrices[fold] = (
            base_matrix[train],
            base_matrix[test],
            np.hstack([base_matrix[train], train_neighbor[:, :10]]),
            np.hstack([base_matrix[test], test_neighbor[:, :10]]),
            np.hstack([base_matrix[train], train_neighbor]),
            np.hstack([base_matrix[test], test_neighbor]),
        )
        fold_ids[test] = fold

    model_names = [
        "hist_gradient_boosting",
        "logistic",
        "extra_trees",
        "xgboost",
        "lightgbm",
    ]
    feature_sets = ["score", "score+neighbor_difficulty", "score+neighbors"]
    regimes = {"all": 1.0, "high_confidence_70pct": 0.70, "high_confidence_50pct": 0.50}
    results: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    budget_predictions_by_name: dict[str, np.ndarray] = {}

    for model_name in model_names:
        for feature_set in feature_sets:
            if feature_set == "score":
                feature_names = score_names
            elif feature_set == "score+neighbor_difficulty":
                feature_names = score_names + NEIGHBOUR_NAMES[:10]
            else:
                feature_names = score_names + NEIGHBOUR_NAMES
            for regime_name, retain_fraction in regimes.items():
                probabilities = np.zeros(len(rows), dtype=float)
                importances: list[np.ndarray] = []
                train_counts: list[int] = []
                train_class_counts: list[dict[str, int]] = []
                for fold, (train, test) in enumerate(folds):
                    (
                        base_train,
                        base_test,
                        difficulty_train,
                        difficulty_test,
                        joined_train,
                        joined_test,
                    ) = fold_matrices[fold]
                    if feature_set == "score":
                        x_train, x_test = base_train, base_test
                    elif feature_set == "score+neighbor_difficulty":
                        x_train, x_test = difficulty_train, difficulty_test
                    else:
                        x_train, x_test = joined_train, joined_test
                    local_positions = confident_subset(
                        train, actual_hard, k95_values, retain_fraction
                    )
                    y_train_all = actual_hard[train]
                    y_fit = y_train_all[local_positions]
                    train_counts.append(int(len(local_positions)))
                    train_class_counts.append(
                        {"easy": int(np.sum(y_fit == 0)), "hard": int(np.sum(y_fit == 1))}
                    )
                    model = make_model(model_name, y_fit)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(x_train[local_positions], y_fit)
                    probabilities[test] = model.predict_proba(x_test)[:, 1]
                    importance = normalized_importance(model, len(feature_names))
                    if importance is not None:
                        importances.append(importance)

                budget_prediction = top_budget_predictions(probabilities, PRIOR_HARD_COUNT)
                threshold_prediction = (probabilities >= 0.2).astype(int)
                config_name = f"{model_name}__{feature_set}__{regime_name}"
                budget_predictions_by_name[config_name] = budget_prediction.copy()
                result: dict[str, Any] = {
                    "name": config_name,
                    "model": model_name,
                    "feature_set": feature_set,
                    "training_regime": regime_name,
                    "retain_fraction_target": retain_fraction,
                    "mean_training_rows": float(np.mean(train_counts)),
                    "fold_training_rows": train_counts,
                    "fold_training_class_counts": train_class_counts,
                    "roc_auc": float(roc_auc_score(actual_hard, probabilities)),
                    "average_precision": float(average_precision_score(actual_hard, probabilities)),
                    "brier_score": float(brier_score_loss(actual_hard, probabilities)),
                }
                if importances:
                    mean_importance = np.mean(np.vstack(importances), axis=0)
                    order = np.argsort(-mean_importance)[:12]
                    result["top_features"] = [
                        {"feature": feature_names[int(index)], "importance": float(mean_importance[index])}
                        for index in order
                    ]

                for hard_k in HARD_KS:
                    budget_choices = np.where(budget_prediction == 1, hard_k, EASY_K)
                    threshold_choices = np.where(threshold_prediction == 1, hard_k, EASY_K)
                    result[f"budget_30_{hard_k}"] = policy_metrics(
                        rows, budget_choices, actual_hard, budget_prediction
                    )
                    result[f"threshold_0.2_30_{hard_k}"] = policy_metrics(
                        rows, threshold_choices, actual_hard, threshold_prediction
                    )
                results.append(result)

                for index, row in enumerate(rows):
                    per_query_rows.append(
                        {
                            "configuration": config_name,
                            "qid": row["qid"],
                            "source": row["source"],
                            "fold": int(fold_ids[index]),
                            "k95": int(k95_values[index]),
                            "label_hard": int(actual_hard[index]),
                            "hard_probability": float(probabilities[index]),
                            "budget_predicted_hard": int(budget_prediction[index]),
                            "threshold_0.2_predicted_hard": int(threshold_prediction[index]),
                        }
                    )

    refit_baseline = next(
        row
        for row in results
        if row["model"] == "hist_gradient_boosting"
        and row["feature_set"] == "score"
        and row["training_regime"] == "all"
    )
    historical_path = ROOT / "data" / "work" / "adaptive_retrieval_20261009" / "score_confidence_gate.json"
    historical_payload = json.loads(historical_path.read_text(encoding="utf-8"))
    historical_probabilities = np.asarray(historical_payload["oof_probability"], dtype=float)
    if len(historical_probabilities) != len(rows):
        raise ValueError("Historical score-confidence probability count no longer matches 220-query order")
    historical_prediction = (historical_probabilities >= 0.2).astype(int)
    if int(np.sum(historical_prediction)) != PRIOR_HARD_COUNT:
        raise ValueError("Historical threshold-0.2 policy no longer matches the expected 134 hard slots")
    historical_choices_100 = np.where(historical_prediction == 1, 100, EASY_K)
    historical_recalls_100 = np.asarray(
        [recall_at(row, int(k)) for row, k in zip(rows, historical_choices_100)], dtype=float
    )
    historical_baseline = {
        "source": str(historical_path.relative_to(ROOT)),
        "model": "HistGradientBoostingClassifier (parameters not preserved in a checked-in runner)",
        "feature_set": "ranked fused score distribution",
        "target": "k95 > 50",
        "threshold": 0.2,
        "roc_auc": float(roc_auc_score(actual_hard, historical_probabilities)),
        "average_precision": float(average_precision_score(actual_hard, historical_probabilities)),
        "budget_30_100": policy_metrics(
            rows, historical_choices_100, actual_hard, historical_prediction
        ),
        "budget_30_120": policy_metrics(
            rows, np.where(historical_prediction == 1, 120, EASY_K), actual_hard, historical_prediction
        ),
    }
    for result in results:
        prediction = budget_predictions_by_name[result["name"]]
        choices = np.where(prediction == 1, 100, EASY_K)
        recalls = np.asarray(
            [recall_at(row, int(k)) for row, k in zip(rows, choices)], dtype=float
        )
        paired_delta = recalls - historical_recalls_100
        result["budget_30_100"]["delta_vs_historical_baseline"] = float(np.mean(paired_delta))
        result["budget_30_100"]["delta_vs_historical_bootstrap_95ci"] = bootstrap_delta_interval(paired_delta)

    best = max(results, key=lambda item: item["budget_30_100"]["adaptive_recall"])
    best_by_feature_set = {
        feature_set: max(
            (row for row in results if row["feature_set"] == feature_set),
            key=lambda item: item["budget_30_100"]["adaptive_recall"],
        )
        for feature_set in feature_sets
    }
    report = {
        "protocol": {
            "benchmark": "Physics 80 + Industrial 75 + OHR-Bench 65",
            "queries": len(rows),
            "pages": 5091,
            "ranking": "0.2 normalized BM25 + 0.8 Qwen3-Embedding-8B cosine; cached top 200",
            "folds": 5,
            "split": "source-stratified outer folds, random_state=42",
            "label": f"hard iff k95 > {HARD_LABEL_K}, matching the earlier score-confidence gate",
            "neighbor_memory": "top-5 Qwen query-embedding neighbors from outer training fold only",
            "neighbor_evidence": "only training-fold neighbor qrels; same-source neighbors for page/file overlap",
            "selective_training": "retain the lowest-k95 easy and highest-k95 hard examples separately",
            "budget_policy": f"globally allocate exactly {PRIOR_HARD_COUNT}/220 hard slots; no test qrels",
            "easy_k": EASY_K,
            "hard_ks": list(HARD_KS),
            "exploratory_warning": "30 configurations were screened on one 220-query benchmark; this is not an independent held-out confirmation.",
        },
        "k95_distribution": {
            "min": int(np.min(k95_values)),
            "median": float(np.median(k95_values)),
            "p75": float(np.percentile(k95_values, 75)),
            "p90": float(np.percentile(k95_values, 90)),
            "unreached_at_200": int(np.sum(k95_values == MAX_K + 1)),
        },
        "historical_baseline_reproduction": historical_baseline,
        "refit_score_only_baseline": refit_baseline,
        "best_budget_30_100": best,
        "best_by_feature_set": best_by_feature_set,
        "selected_results": selected_report_rows(results),
        "results": results,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in per_query_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Easy/hard model ablation (exploratory OOF)",
        "",
        "The primary comparison gives every configuration exactly 134 hard slots, so all rows use mean K=72.64 for the 30/100 policy. Test-fold qrels are evaluation-only; neighbour labels and evidence come only from outer training folds.",
        "",
        "| Model | Features | Training rows | ROC-AUC | AP | Recall | Fixed K=73 | Delta | 95% CI | Hard FN rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    historical_metric = historical_baseline["budget_30_100"]
    historical_ci = historical_metric["delta_bootstrap_95ci"]
    lines.append(
        f"| historical HGB artifact | score / all | n/a | {historical_baseline['roc_auc']:.3f} | "
        f"{historical_baseline['average_precision']:.3f} | {historical_metric['adaptive_recall']:.2%} | "
        f"{historical_metric['matched_fixed_recall']:.2%} | {historical_metric['delta_vs_matched_fixed']:+.2%} | "
        f"[{historical_ci[0]:+.2%}, {historical_ci[1]:+.2%}] | {historical_metric['hard_false_negative_rate']:.2%} |"
    )
    for row in selected_report_rows(results):
        metric = row["budget_30_100"]
        ci = metric["delta_bootstrap_95ci"]
        lines.append(
            f"| {row['model']} | {row['feature_set']} / {row['training_regime']} | "
            f"{row['mean_training_rows']:.1f} | {row['roc_auc']:.3f} | {row['average_precision']:.3f} | "
            f"{metric['adaptive_recall']:.2%} | {metric['matched_fixed_recall']:.2%} | "
            f"{metric['delta_vs_matched_fixed']:+.2%} | [{ci[0]:+.2%}, {ci[1]:+.2%}] | "
            f"{metric['hard_false_negative_rate']:.2%} |"
        )

    base_metric = historical_baseline["budget_30_100"]
    best_metric = best["budget_30_100"]
    lines += [
        "",
        "## Baseline reproduction",
        "",
        f"The saved original HistGradientBoosting score-only probabilities at threshold 0.2 reproduce mean K={base_metric['mean_k']:.2f} and recall={base_metric['adaptive_recall']:.2%}.",
        "Its exact training hyperparameters were not preserved in a runner, so this report treats the saved OOF probabilities as the frozen historical baseline.",
        "",
        "## Feature ablation",
        "",
        "| Feature family | Best configuration | Recall | Delta vs historical | Paired 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for feature_set, row in best_by_feature_set.items():
        metric = row["budget_30_100"]
        ci = metric["delta_vs_historical_bootstrap_95ci"]
        lines.append(
            f"| {feature_set} | `{row['name']}` | {metric['adaptive_recall']:.2%} | "
            f"{metric['delta_vs_historical_baseline']:+.2%} | [{ci[0]:+.2%}, {ci[1]:+.2%}] |"
        )
    lines += [
        "",
        "## Best screened configuration",
        "",
        f"`{best['name']}` gives recall={best_metric['adaptive_recall']:.2%} at mean K={best_metric['mean_k']:.2f}; matched fixed K={best_metric['matched_fixed_k']} gives {best_metric['matched_fixed_recall']:.2%}. Its paired delta versus the historical easy/hard model is {best_metric['delta_vs_historical_baseline']:+.2%}.",
        f"With hard K=120, the same decisions give {best['budget_30_120']['adaptive_recall']:.2%} at mean K={best['budget_30_120']['mean_k']:.2f}, versus {historical_baseline['budget_30_120']['adaptive_recall']:.2%} for the historical model.",
        "",
        "Top averaged feature importances for that configuration:",
        "",
    ]
    for item in best.get("top_features", []):
        lines.append(f"- `{item['feature']}`: {item['importance']:.4f}")
    lines += [
        "",
        "## Interpretation guardrail",
        "",
        "This is a model-selection screen, not a final result: feature sets, training filters and five model families were compared on the same 220 queries. A promising row must be frozen and tested on a new query set (or by nested CV) before it is promoted.",
        "",
        "Full results are in `report.json`; all OOF probabilities and decisions are in `per_query.jsonl`.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "historical_baseline_threshold_0.2": base_metric,
        "best": {"name": best["name"], **best_metric},
        "selected": [
            {
                "name": row["name"],
                "auc": row["roc_auc"],
                "recall": row["budget_30_100"]["adaptive_recall"],
                "delta": row["budget_30_100"]["delta_vs_matched_fixed"],
            }
            for row in selected_report_rows(results)
        ],
    }, indent=2))
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
