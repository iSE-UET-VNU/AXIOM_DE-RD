"""Build a human-readable audit bundle for the score-only continuous-K model.

This script does not train or tune a model.  It materializes the exact rows,
score curves, features and labels consumed by optimize_xgboost_continuous_k.py
so they can be inspected independently of the experiment code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_DIR = ROOT / "experiments" / "adaptive-k-retrieval"
if str(ADAPTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTIVE_DIR))

from run_canonical_adaptive_methods import load_rows  # noqa: E402
from run_risk_label_statistics import features as score_features  # noqa: E402


CAP = 200
DEFAULT_OUTPUT = ROOT / "data" / "work" / "xgboost_continuous_k_audit_20261016"


FEATURE_DESCRIPTIONS = {
    **{
        f"score_mean_top_{k}": (
            f"mean(scores[1:{k}])",
            f"Mean hybrid retrieval score among the first {k} ranked pages.",
            "Absolute scale depends on per-query fusion normalization; top means are strongly correlated.",
        )
        for k in (1, 5, 10, 20, 50, 100)
    },
    "score_mean_tail": (
        "mean(scores[101:200])",
        "Mean score in ranks 101 through 200.",
        "This is only the observed top-200 tail, not the corpus score tail.",
    ),
    "score_std": (
        "std(scores[1:200], ddof=0)",
        "Population standard deviation of the top-200 scores.",
        "Mixes absolute scale and curve shape.",
    ),
    "score_range": (
        "score(rank=1) - score(rank=200)",
        "Range of the top-200 score curve.",
        "Highly coupled to score_std and several slopes.",
    ),
    **{
        f"gap_at_{k}": (
            f"score(rank={k}) - score(rank={k + 1})",
            f"Adjacent score drop immediately after rank {k}.",
            "A single local difference can be unstable and does not indicate relevance by itself.",
        )
        for k in (1, 5, 10, 20, 50)
    },
    "max_gap": (
        "max(score(rank=r) - score(rank=r+1)), r=1..199",
        "Largest adjacent score drop in the top-200 curve.",
        "May capture an arbitrary outlier or fusion boundary rather than a relevance boundary.",
    ),
    "max_gap_rank": (
        "argmax(adjacent gaps) / 200",
        "Normalized rank where the largest adjacent drop occurs.",
        "Discontinuous and sensitive when several gaps are nearly tied.",
    ),
    **{
        f"slope_top_{k}": (
            f"(score(rank={k}) - score(rank=1)) / {k - 1}",
            f"Average signed score change per rank from rank 1 to rank {k}.",
            "Always non-positive for a sorted curve and correlated with range/gaps.",
        )
        for k in (10, 20, 50, 100)
    },
    "softmax_entropy": (
        "entropy(softmax((scores-max(scores))/0.1)) / log(200)",
        "Normalized entropy of a fixed-temperature softmax over top-200 scores.",
        "Temperature 0.1 is hand-set and interacts with the hybrid score scale.",
    ),
}


def recall_at(row: dict[str, Any], cutoff: int) -> float:
    gold = set(row["gold"])
    return len(set(row["ranked"][:cutoff]) & gold) / max(len(gold), 1)


def required_k(row: dict[str, Any], target: float = 1.0) -> int:
    gold = set(row["gold"])
    ranks = sorted(
        rank for rank, page_id in enumerate(row["ranked"][:CAP], 1) if page_id in gold
    )
    need = int(math.ceil(target * len(gold) - 1e-12))
    return int(ranks[need - 1]) if need > 0 and len(ranks) >= need else CAP


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = load_rows("hybrid", CAP)
    feature_names, _ = score_features(rows[0]["scores"])
    x = np.vstack([score_features(row["scores"])[1] for row in rows])
    targets = np.asarray([required_k(row) for row in rows], dtype=int)

    query_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        gold = set(row["gold"])
        gold_ranks = [
            rank for rank, page_id in enumerate(row["ranked"], 1) if page_id in gold
        ]
        item: dict[str, Any] = {
            "row_index": i,
            "qid": row["qid"],
            "source": row["source"],
            "query": row["query"],
            "gold_page_count": len(gold),
            "gold_retrieved_in_top200": len(gold_ranks),
            "candidate_recall_at_200": recall_at(row, 200),
            "recall_at_10": recall_at(row, 10),
            "recall_at_20": recall_at(row, 20),
            "recall_at_50": recall_at(row, 50),
            "recall_at_70": recall_at(row, 70),
            "recall_at_100": recall_at(row, 100),
            "target_k": int(targets[i]),
            "target_is_cap": int(targets[i] == CAP),
            "all_gold_reachable_top200": int(len(gold_ranks) == len(gold)),
            "gold_ranks_in_top200": "|".join(str(rank) for rank in gold_ranks),
        }
        item.update({name: float(x[i, j]) for j, name in enumerate(feature_names)})
        query_rows.append(item)
        for rank, (page_id, score) in enumerate(zip(row["ranked"], row["scores"]), 1):
            curve_rows.append(
                {
                    "row_index": i,
                    "qid": row["qid"],
                    "source": row["source"],
                    "rank": rank,
                    "page_id": page_id,
                    "hybrid_score": float(score),
                    "is_gold": int(page_id in gold),
                }
            )

    query_fields = list(query_rows[0])
    curve_fields = list(curve_rows[0])
    write_csv(args.output_dir / "query_features_labels.csv", query_fields, query_rows)
    write_csv(args.output_dir / "score_curves_long.csv", curve_fields, curve_rows)

    summary_rows: list[dict[str, Any]] = []
    for j, name in enumerate(feature_names):
        values = x[:, j]
        pearson = float(np.corrcoef(values, targets)[0, 1]) if np.std(values) else float("nan")
        ranks_x = np.argsort(np.argsort(values, kind="stable"), kind="stable")
        ranks_y = np.argsort(np.argsort(targets, kind="stable"), kind="stable")
        spearman = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
        summary_rows.append(
            {
                "feature": name,
                "missing_count": int(np.sum(~np.isfinite(values))),
                "unique_count": int(len(np.unique(values))),
                "min": float(np.min(values)),
                "p01": percentile(values, 1),
                "p10": percentile(values, 10),
                "p25": percentile(values, 25),
                "median": percentile(values, 50),
                "mean": float(np.mean(values)),
                "p75": percentile(values, 75),
                "p90": percentile(values, 90),
                "p99": percentile(values, 99),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
                "pearson_with_target_k": pearson,
                "spearman_with_target_k": spearman,
            }
        )
    write_csv(args.output_dir / "feature_summary.csv", list(summary_rows[0]), summary_rows)

    dictionary_rows = []
    for name in feature_names:
        formula, meaning, caveat = FEATURE_DESCRIPTIONS[name]
        dictionary_rows.append(
            {"feature": name, "formula": formula, "meaning": meaning, "risk_or_caveat": caveat}
        )
    write_csv(
        args.output_dir / "feature_dictionary.csv",
        ["feature", "formula", "meaning", "risk_or_caveat"],
        dictionary_rows,
    )

    label_rows: list[dict[str, Any]] = []
    bins = [(1, 10), (11, 20), (21, 40), (41, 70), (71, 100), (101, 199), (200, 200)]
    sources = ["all", *sorted({str(row["source"]) for row in rows})]
    for source in sources:
        mask = np.ones(len(rows), dtype=bool) if source == "all" else np.asarray(
            [row["source"] == source for row in rows], dtype=bool
        )
        values = targets[mask]
        for low, high in bins:
            count = int(np.sum((values >= low) & (values <= high)))
            label_rows.append(
                {
                    "source": source,
                    "target_k_bin": f"{low}-{high}",
                    "count": count,
                    "fraction_within_source": count / len(values),
                }
            )
    write_csv(
        args.output_dir / "label_distribution.csv",
        ["source", "target_k_bin", "count", "fraction_within_source"],
        label_rows,
    )

    source_rows: list[dict[str, Any]] = []
    for source in sources:
        indices = np.arange(len(rows)) if source == "all" else np.asarray(
            [i for i, row in enumerate(rows) if row["source"] == source], dtype=int
        )
        source_scores = np.vstack([np.asarray(rows[int(i)]["scores"], dtype=float) for i in indices])
        source_targets = targets[indices]
        source_rows.append(
            {
                "source": source,
                "queries": len(indices),
                "mean_gold_pages": float(np.mean([len(rows[int(i)]["gold"]) for i in indices])),
                "target_mean": float(np.mean(source_targets)),
                "target_median": float(np.median(source_targets)),
                "target_cap_fraction": float(np.mean(source_targets == CAP)),
                "candidate_recall_at_200": float(
                    np.mean([recall_at(rows[int(i)], CAP) for i in indices])
                ),
                "score_top1_mean": float(np.mean(source_scores[:, 0])),
                "score_top100_mean": float(np.mean(source_scores[:, :100])),
                "score_rank200_zero_fraction": float(np.mean(source_scores[:, -1] == 0.0)),
                "all_score_zero_fraction": float(np.mean(source_scores == 0.0)),
                "median_positive_scores_in_top200": float(
                    np.median(np.sum(source_scores > 0.0, axis=1))
                ),
            }
        )
    write_csv(args.output_dir / "source_summary.csv", list(source_rows[0]), source_rows)

    gold_count_rows: list[dict[str, Any]] = []
    for gold_count in sorted({len(row["gold"]) for row in rows}):
        indices = np.asarray(
            [i for i, row in enumerate(rows) if len(row["gold"]) == gold_count], dtype=int
        )
        values = targets[indices]
        gold_count_rows.append(
            {
                "gold_page_count": gold_count,
                "queries": len(indices),
                "target_mean": float(np.mean(values)),
                "target_median": float(np.median(values)),
                "target_cap_fraction": float(np.mean(values == CAP)),
                "candidate_recall_at_200": float(
                    np.mean([recall_at(rows[int(i)], CAP) for i in indices])
                ),
            }
        )
    write_csv(
        args.output_dir / "label_by_gold_page_count.csv",
        list(gold_count_rows[0]),
        gold_count_rows,
    )

    corr = np.corrcoef(x, rowvar=False)
    corr_rows = []
    for i, left in enumerate(feature_names):
        for j in range(i + 1, len(feature_names)):
            corr_rows.append(
                {"feature_1": left, "feature_2": feature_names[j], "pearson": float(corr[i, j])}
            )
    corr_rows.sort(key=lambda item: abs(item["pearson"]), reverse=True)
    write_csv(
        args.output_dir / "feature_correlations.csv",
        ["feature_1", "feature_2", "pearson"],
        corr_rows,
    )

    all_scores = np.concatenate([np.asarray(row["scores"], dtype=float) for row in rows])
    descending_violations = sum(
        int(np.sum(np.diff(np.asarray(row["scores"], dtype=float)) > 1e-12)) for row in rows
    )
    checks = {
        "queries": len(rows),
        "features": len(feature_names),
        "score_rows": len(curve_rows),
        "all_finite_features": bool(np.all(np.isfinite(x))),
        "all_finite_scores": bool(np.all(np.isfinite(all_scores))),
        "descending_score_violations": descending_violations,
        "duplicate_qids": len(rows) - len({str(row["qid"]) for row in rows}),
        "score_min": float(np.min(all_scores)),
        "score_max": float(np.max(all_scores)),
        "score_zero_fraction": float(np.mean(all_scores == 0.0)),
        "target_mean": float(np.mean(targets)),
        "target_median": float(np.median(targets)),
        "target_at_cap": int(np.sum(targets == CAP)),
        "target_at_cap_but_all_gold_reachable": int(
            sum(
                target == CAP and recall_at(row, CAP) >= 1.0 - 1e-12
                for row, target in zip(rows, targets)
            )
        ),
        "unreachable_all_gold_at_200": int(sum(recall_at(row, CAP) < 1.0 - 1e-12 for row in rows)),
    }
    (args.output_dir / "quality_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    model_config = {
        "model": "xgboost.XGBRegressor",
        "objective": "reg:squarederror",
        "n_estimators": 320,
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "random_state": 42,
        "tree_method": "hist",
        "training_protocol": "5 outer folds stratified by source; 4 inner folds for scale/margin calibration",
        "tuning_status": "Small hand-written screen only; no systematic hyperparameter study yet.",
    }
    (args.output_dir / "model_config.json").write_text(
        json.dumps(model_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt

        display_sources = [source for source in sources if source != "all"]
        display_bins = [f"{low}-{high}" for low, high in bins]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        global_counts = [
            next(
                int(item["count"])
                for item in label_rows
                if item["source"] == "all" and item["target_k_bin"] == bin_name
            )
            for bin_name in display_bins
        ]
        axes[0].bar(display_bins, global_counts, color="#4472C4")
        axes[0].set_title("Continuous-K label distribution (220 queries)")
        axes[0].set_xlabel("target_k bin")
        axes[0].set_ylabel("query count")
        axes[0].tick_params(axis="x", rotation=35)
        width = 0.25
        positions = np.arange(len(display_bins))
        for source_index, source in enumerate(display_sources):
            fractions = [
                next(
                    float(item["fraction_within_source"])
                    for item in label_rows
                    if item["source"] == source and item["target_k_bin"] == bin_name
                )
                for bin_name in display_bins
            ]
            axes[1].bar(
                positions + (source_index - 1) * width,
                fractions,
                width=width,
                label=source,
            )
        axes[1].set_xticks(positions, display_bins, rotation=35)
        axes[1].set_title("Label distribution differs strongly by source")
        axes[1].set_xlabel("target_k bin")
        axes[1].set_ylabel("fraction within source")
        axes[1].legend(fontsize=8)
        fig.savefig(args.output_dir / "label_distribution.png", dpi=160)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
        image = axis.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(np.arange(len(feature_names)), feature_names, rotation=90, fontsize=7)
        axis.set_yticks(np.arange(len(feature_names)), feature_names, fontsize=7)
        axis.set_title("Pearson correlation among the 21 input features")
        fig.colorbar(image, ax=axis, shrink=0.8)
        fig.savefig(args.output_dir / "feature_correlation_heatmap.png", dpi=180)
        plt.close(fig)

        score_matrix = np.vstack([np.asarray(row["scores"], dtype=float) for row in rows])
        fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
        groups = [
            ("target 1-10", targets <= 10),
            ("target 11-70", (targets >= 11) & (targets <= 70)),
            ("target 71-199", (targets >= 71) & (targets <= 199)),
            ("target 200 / unreachable", targets == 200),
        ]
        ranks = np.arange(1, CAP + 1)
        for label, mask in groups:
            if np.any(mask):
                axis.plot(ranks, np.median(score_matrix[mask], axis=0), label=f"{label} (n={int(np.sum(mask))})")
        axis.set_title("Median top-200 score curves grouped by regression label")
        axis.set_xlabel("rank")
        axis.set_ylabel("fused hybrid score")
        axis.legend()
        axis.grid(alpha=0.2)
        fig.savefig(args.output_dir / "score_curves_by_label.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass

    top_correlations = corr_rows[:12]
    readme = f"""# XGBoost continuous-K data audit — 2026-10-16

This folder materializes the exact 220 rows used by the current score-only continuous-K experiment. It is an inspection bundle, not a new model result.

## Read these files first

- `query_features_labels.csv`: one query per row, query text, label, recalls, gold ranks and all 21 model inputs.
- `feature_dictionary.csv`: exact formula, intended meaning and caveat for each feature.
- `feature_summary.csv`: percentiles, missingness, cardinality and correlation with `target_k`.
- `label_distribution.csv`: label histogram globally and by source.
- `source_summary.csv`: source/domain shift in labels and score curves.
- `label_by_gold_page_count.csv`: how strongly the regression label changes with qrel cardinality.
- `score_curves_long.csv`: all 44,000 ranked query-page scores; filter by qid to see the curve and gold locations.
- `feature_correlations.csv`: pairwise redundancy diagnostics.
- `quality_checks.json`: basic data-integrity checks.
- `model_config.json`: the actual XGBoost configuration and what has or has not been tuned.
- `label_distribution.png`, `feature_correlation_heatmap.png`, `score_curves_by_label.png`: quick visual diagnostics.

## Exact score provenance

The current loader does **not** read the official merged language-routed/file-K=3 hybrid run. It reconstructs a global 5,091-page ranking from the `_repro_retrieval_baseline` artifacts:

```text
lexical_raw = BM25(PP-OCR page text, query)
dense_raw   = cosine(Qwen3-Embedding-8B page vector, query vector)
lexical_norm = min-max using the 100 highest lexical scores, clipped below at 0
dense_norm   = min-max using the 100 highest dense scores, clipped below at 0
hybrid_score = 0.2 * lexical_norm + 0.8 * dense_norm
ranking      = global top 200 hybrid_score over 5,091 pages
```

Therefore the word `raw` in the experiment means the final fused score as stored in the top-200 curve. It does not mean raw BM25 or raw dense similarity.

## Label

`target_k` is the smallest rank containing every gold page when all gold pages are present in the top 200. If at least one gold page is absent, the label is forced to 200. Consequently, `target_k=200` mixes two different cases: a gold page genuinely at rank 200 and an unreachable gold set.

Current label summary: mean={checks['target_mean']:.2f}, median={checks['target_median']:.2f}, target K=200 count={checks['target_at_cap']}, unreachable count={checks['unreachable_all_gold_at_200']}.

## Immediate audit findings

1. The dataset is very small for boosted trees: 220 independent query rows for 21 features.
2. The six top-score means, range, standard deviation and slopes are deliberately overlapping summaries; inspect `feature_correlations.csv` before treating all 21 as independent evidence.
3. Local gaps can be noisy: each `gap_at_k` is based on only two adjacent pages.
4. `softmax_entropy` uses a fixed temperature of 0.1 on an already normalized fusion score; its semantics depend on that scale.
5. The label has structural censoring at 200, not ordinary continuous regression noise.
6. Hyperparameters were hand-set and only six feature/objective configurations were screened. There has not yet been systematic depth, regularization, sampling or early-stopping tuning.
7. Rank 200 has score zero for every query. Therefore `score_mean_top_1` and `score_range` are exactly identical, and the observed tail contains normalization-induced zero padding.
8. Gold cardinality is a major hidden driver of the label: OHR has one gold page per query on average and median target K=1, while Industrial and Physics have many more gold pages. Gold count is not a model input. Random folds stratified by source do not test transfer to an unseen domain.

## Integrity checks

```json
{json.dumps(checks, ensure_ascii=False, indent=2)}
```

## Strongest absolute feature correlations

```json
{json.dumps(top_correlations, ensure_ascii=False, indent=2)}
```
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
