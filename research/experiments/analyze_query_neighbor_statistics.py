"""OOF statistics for semantic query-neighbour transfer.

For each held-out query, neighbours are selected only from the outer training
fold using cached query embeddings. The analysis measures whether neighbour
cutoffs predict the test cutoff and how much test evidence is covered by
neighbour gold pages or neighbour retrieved-page pools.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[2]
ADAPTIVE_DIR = ROOT / "experiments" / "adaptive-k-retrieval"
if str(ADAPTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTIVE_DIR))

from run_canonical_adaptive_methods import load_rows  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "work" / "query_neighbor_statistics_20261014"
MAX_K = 200
K_VALUES = (1, 3, 5, 10)
ALPHAS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
RETRIEVED_DEPTHS = (10, 30, 100, 200)
SEED = 42


def file_id(page_id: str) -> str:
    return page_id.rsplit("#page=", 1)[0]


def recall(pool: set[str], gold: set[str]) -> float:
    return len(pool & gold) / max(len(gold), 1)


def saturation_rank(row: dict[str, Any]) -> int:
    gold = set(row["gold"])
    ranks = [rank for rank, page_id in enumerate(row["ranked"][:MAX_K], 1) if page_id in gold]
    return max(ranks, default=1)


def k95_capped(row: dict[str, Any]) -> int:
    gold = set(row["gold"])
    for rank, page_id in enumerate(row["ranked"][:MAX_K], 1):
        if page_id in gold:
            found = len(set(row["ranked"][:rank]) & gold)
            if found / max(len(gold), 1) >= 0.95:
                return rank
    return MAX_K


def safe_corr(kind: str, x: list[float], y: list[float]) -> float:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    value = spearmanr(x, y).statistic if kind == "spearman" else pearsonr(x, y).statistic
    return float(0.0 if np.isnan(value) else value)


def aggregate(records: list[dict[str, Any]], total_queries: int) -> dict[str, Any]:
    valid = [record for record in records if record["neighbor_count"] > 0]
    target_sat = [float(record["target_saturation_k"]) for record in valid]
    predicted_sat = [float(record["neighbor_mean_saturation_k"]) for record in valid]
    target_k95 = [float(record["target_k95_capped"]) for record in valid]
    predicted_k95 = [float(record["neighbor_mean_k95_capped"]) for record in valid]
    baseline_sat_error = [abs(record["target_saturation_k"] - record["fold_mean_saturation_k"]) for record in valid]
    baseline_k95_error = [abs(record["target_k95_capped"] - record["fold_mean_k95_capped"]) for record in valid]
    return {
        "queries": total_queries,
        "queries_with_neighbors": len(valid),
        "query_coverage": len(valid) / max(total_queries, 1),
        "neighbor_count_mean_all": float(np.mean([record["neighbor_count"] for record in records])),
        "neighbor_count_mean_valid": float(np.mean([record["neighbor_count"] for record in valid])) if valid else 0.0,
        "similarity_top1_mean_valid": float(np.mean([record["similarity_top1"] for record in valid])) if valid else 0.0,
        "saturation_k": {
            "target_mean": float(np.mean(target_sat)) if valid else 0.0,
            "neighbor_mean_prediction": float(np.mean(predicted_sat)) if valid else 0.0,
            "mae": float(np.mean(np.abs(np.asarray(target_sat) - predicted_sat))) if valid else 0.0,
            "fold_global_mean_mae": float(np.mean(baseline_sat_error)) if valid else 0.0,
            "pearson": safe_corr("pearson", target_sat, predicted_sat),
            "spearman": safe_corr("spearman", target_sat, predicted_sat),
            "within_10": float(np.mean(np.abs(np.asarray(target_sat) - predicted_sat) <= 10)) if valid else 0.0,
            "within_25": float(np.mean(np.abs(np.asarray(target_sat) - predicted_sat) <= 25)) if valid else 0.0,
        },
        "k95_capped": {
            "target_mean": float(np.mean(target_k95)) if valid else 0.0,
            "neighbor_mean_prediction": float(np.mean(predicted_k95)) if valid else 0.0,
            "mae": float(np.mean(np.abs(np.asarray(target_k95) - predicted_k95))) if valid else 0.0,
            "fold_global_mean_mae": float(np.mean(baseline_k95_error)) if valid else 0.0,
            "pearson": safe_corr("pearson", target_k95, predicted_k95),
            "spearman": safe_corr("spearman", target_k95, predicted_k95),
        },
        "neighbor_gold_pages": {
            "pool_size_mean": float(np.mean([record["gold_page_pool_size"] for record in records])),
            "page_recall": float(np.mean([record["gold_page_recall"] for record in records])),
            "page_hit": float(np.mean([record["gold_page_hit"] for record in records])),
            "file_recall": float(np.mean([record["gold_file_recall"] for record in records])),
            "union_with_test_top30_recall": float(np.mean([record["gold_union_test_top30_recall"] for record in records])),
            "marginal_over_test_top30": float(np.mean([record["gold_union_test_top30_recall"] - record["test_top30_recall"] for record in records])),
        },
        "neighbor_retrieved_pages": {
            str(depth): {
                "pool_size_mean": float(np.mean([record[f"retrieved_{depth}_pool_size"] for record in records])),
                "page_recall": float(np.mean([record[f"retrieved_{depth}_page_recall"] for record in records])),
                "page_hit": float(np.mean([record[f"retrieved_{depth}_page_hit"] for record in records])),
                "file_recall": float(np.mean([record[f"retrieved_{depth}_file_recall"] for record in records])),
                "union_with_test_top30_recall": float(np.mean([record[f"retrieved_{depth}_union_test_top30_recall"] for record in records])),
                "marginal_over_test_top30": float(np.mean([record[f"retrieved_{depth}_union_test_top30_recall"] - record["test_top30_recall"] for record in records])),
            }
            for depth in RETRIEVED_DEPTHS
        },
    }


def make_record(
    test_index: int,
    neighbor_indices: np.ndarray,
    similarities: np.ndarray,
    rows: Sequence[dict[str, Any]],
    saturation: np.ndarray,
    k95: np.ndarray,
    fold_mean_saturation: float,
    fold_mean_k95: float,
) -> dict[str, Any]:
    row = rows[test_index]
    gold = set(row["gold"])
    gold_files = {file_id(page_id) for page_id in gold}
    test_top30 = set(row["ranked"][:30])
    record: dict[str, Any] = {
        "qid": row["qid"],
        "source": row["source"],
        "neighbor_count": int(len(neighbor_indices)),
        "neighbor_qids": [str(rows[int(index)]["qid"]) for index in neighbor_indices],
        "neighbor_similarities": [float(value) for value in similarities],
        "similarity_top1": float(similarities[0]) if len(similarities) else 0.0,
        "target_saturation_k": int(saturation[test_index]),
        "target_k95_capped": int(k95[test_index]),
        "neighbor_mean_saturation_k": float(np.mean(saturation[neighbor_indices])) if len(neighbor_indices) else fold_mean_saturation,
        "neighbor_mean_k95_capped": float(np.mean(k95[neighbor_indices])) if len(neighbor_indices) else fold_mean_k95,
        "fold_mean_saturation_k": fold_mean_saturation,
        "fold_mean_k95_capped": fold_mean_k95,
        "test_top30_recall": recall(test_top30, gold),
    }
    neighbor_gold_pages: set[str] = set()
    for index in neighbor_indices:
        neighbor_gold_pages.update(rows[int(index)]["gold"])
    neighbor_gold_files = {file_id(page_id) for page_id in neighbor_gold_pages}
    record.update({
        "gold_page_pool_size": len(neighbor_gold_pages),
        "gold_page_recall": recall(neighbor_gold_pages, gold),
        "gold_page_hit": float(recall(neighbor_gold_pages, gold) > 0),
        "gold_file_recall": recall(neighbor_gold_files, gold_files),
        "gold_union_test_top30_recall": recall(neighbor_gold_pages | test_top30, gold),
    })
    for depth in RETRIEVED_DEPTHS:
        pool: set[str] = set()
        for index in neighbor_indices:
            pool.update(rows[int(index)]["ranked"][:depth])
        pool_files = {file_id(page_id) for page_id in pool}
        record.update({
            f"retrieved_{depth}_pool_size": len(pool),
            f"retrieved_{depth}_page_recall": recall(pool, gold),
            f"retrieved_{depth}_page_hit": float(recall(pool, gold) > 0),
            f"retrieved_{depth}_file_recall": recall(pool_files, gold_files),
            f"retrieved_{depth}_union_test_top30_recall": recall(pool | test_top30, gold),
        })
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, vectors = load_rows("hybrid", MAX_K)
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    sources = np.asarray([str(row["source"]) for row in rows])
    saturation = np.asarray([saturation_rank(row) for row in rows], dtype=int)
    k95 = np.asarray([k95_capped(row) for row in rows], dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    records_by_method: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    per_query_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20261014)

    for fold, (train, test) in enumerate(splitter.split(vectors, sources)):
        fold_mean_saturation = float(np.mean(saturation[train]))
        fold_mean_k95 = float(np.mean(k95[train]))
        for test_index in test:
            same_source_train = train[sources[train] == sources[test_index]]
            similarities = np.asarray(vectors[test_index] @ vectors[same_source_train].T, dtype=float)
            order = np.argsort(-similarities, kind="stable")
            ordered_indices = same_source_train[order]
            ordered_similarities = similarities[order]

            for neighbor_count in K_VALUES:
                selected_indices = ordered_indices[:neighbor_count]
                selected_similarities = ordered_similarities[:neighbor_count]
                method = f"knn_{neighbor_count}"
                record = make_record(
                    int(test_index), selected_indices, selected_similarities, rows,
                    saturation, k95, fold_mean_saturation, fold_mean_k95,
                )
                record.update({"method": method, "fold": fold})
                records_by_method[method].append(record)
                per_query_rows.append(record)

            for alpha in ALPHAS:
                keep = ordered_similarities >= alpha
                selected_indices = ordered_indices[keep]
                selected_similarities = ordered_similarities[keep]
                method = f"radius_{alpha:.2f}"
                record = make_record(
                    int(test_index), selected_indices, selected_similarities, rows,
                    saturation, k95, fold_mean_saturation, fold_mean_k95,
                )
                record.update({"method": method, "fold": fold})
                records_by_method[method].append(record)
                per_query_rows.append(record)

            # Same-source random K=5 baseline, averaged as per-query statistics
            # over 50 draws to expose how much semantic proximity adds.
            random_records = []
            for _ in range(50):
                selected_indices = rng.choice(same_source_train, size=min(5, len(same_source_train)), replace=False)
                random_records.append(make_record(
                    int(test_index), selected_indices, np.zeros(len(selected_indices)), rows,
                    saturation, k95, fold_mean_saturation, fold_mean_k95,
                ))
            numeric_keys = [
                key for key, value in random_records[0].items()
                if isinstance(value, (int, float)) and key not in {"neighbor_count"}
            ]
            random_record = dict(random_records[0])
            for key in numeric_keys:
                random_record[key] = float(np.mean([record[key] for record in random_records]))
            random_record.update({"method": "random_same_source_5", "fold": fold, "neighbor_count": 5})
            records_by_method["random_same_source_5"].append(random_record)

    summaries = {
        method: aggregate(records, len(rows))
        for method, records in records_by_method.items()
    }
    by_source = {
        source: {
            method: aggregate(
                [record for record in records_by_method[method] if record["source"] == source],
                int(np.sum(sources == source)),
            )
            for method in ("knn_5", "knn_10")
        }
        for source in sorted(set(sources.tolist()))
    }
    baseline_top30 = float(np.mean([
        recall(set(row["ranked"][:30]), set(row["gold"])) for row in rows
    ]))
    payload = {
        "protocol": {
            "benchmark": "Physics 80 + Industrial 75 + OHR-Bench 65",
            "queries": len(rows),
            "split": "5-fold source-stratified OOF; neighbours from training fold only",
            "similarity": "cosine over cached normalized Qwen3-Embedding-8B query vectors",
            "document_transfer": "same-source neighbours only",
            "saturation_k": "last reachable gold-page rank within top 200; 1 if no reachable gold",
            "k95_capped": "first rank reaching 95% of all gold; unreachable mapped to 200",
            "test_qrels": "evaluation only",
            "warning": "Descriptive exploratory analysis; alpha and k choices are screened on this benchmark.",
        },
        "baseline_test_top30_recall": baseline_top30,
        "summaries": summaries,
        "by_source": by_source,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in per_query_rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    lines = [
        "# Query-neighbour statistics (OOF)",
        "",
        f"The test query's own top-30 baseline recall is {baseline_top30:.2%}. All neighbours are same-source outer-training-fold queries.",
        "",
        "## Does neighbour mean K predict test K?",
        "",
        "| Selection | Valid queries | Mean neighbors | Saturation-K Spearman | MAE | Fold-mean MAE | k95 Spearman | k95 MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered_methods = [f"knn_{k}" for k in K_VALUES] + [f"radius_{alpha:.2f}" for alpha in ALPHAS] + ["random_same_source_5"]
    for method in ordered_methods:
        item = summaries[method]
        lines.append(
            f"| {method} | {item['queries_with_neighbors']}/220 | {item['neighbor_count_mean_all']:.2f} | "
            f"{item['saturation_k']['spearman']:.3f} | {item['saturation_k']['mae']:.2f} | "
            f"{item['saturation_k']['fold_global_mean_mae']:.2f} | {item['k95_capped']['spearman']:.3f} | "
            f"{item['k95_capped']['mae']:.2f} |"
        )
    lines += [
        "",
        "## Page/file coverage from neighbours",
        "",
        "| Selection | Pool type | Mean pool pages | Page recall | Page hit | File recall | Recall after union with test top-30 | Marginal gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    coverage_methods = [f"knn_{k}" for k in K_VALUES] + ["radius_0.50", "radius_0.60", "radius_0.70", "random_same_source_5"]
    for method in coverage_methods:
        item = summaries[method]
        gold = item["neighbor_gold_pages"]
        lines.append(
            f"| {method} | neighbor gold | {gold['pool_size_mean']:.1f} | {gold['page_recall']:.2%} | "
            f"{gold['page_hit']:.2%} | {gold['file_recall']:.2%} | {gold['union_with_test_top30_recall']:.2%} | "
            f"{gold['marginal_over_test_top30']:+.2%} |"
        )
        for depth in (30, 100, 200):
            retrieved = item["neighbor_retrieved_pages"][str(depth)]
            lines.append(
                f"| {method} | neighbor retrieved@{depth} | {retrieved['pool_size_mean']:.1f} | "
                f"{retrieved['page_recall']:.2%} | {retrieved['page_hit']:.2%} | {retrieved['file_recall']:.2%} | "
                f"{retrieved['union_with_test_top30_recall']:.2%} | {retrieved['marginal_over_test_top30']:+.2%} |"
            )
    lines += [
        "",
        "## KNN transfer by source",
        "",
        "| Source | Selection | Saturation-K Spearman | k95 Spearman | Gold-page recall | Gold-file recall | Retrieved@30 recall | Test top-30 + retrieved@30 | Marginal gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, methods in by_source.items():
        for method, item in methods.items():
            gold = item["neighbor_gold_pages"]
            retrieved = item["neighbor_retrieved_pages"]["30"]
            lines.append(
                f"| {source} | {method} | {item['saturation_k']['spearman']:.3f} | "
                f"{item['k95_capped']['spearman']:.3f} | {gold['page_recall']:.2%} | "
                f"{gold['file_recall']:.2%} | {retrieved['page_recall']:.2%} | "
                f"{retrieved['union_with_test_top30_recall']:.2%} | {retrieved['marginal_over_test_top30']:+.2%} |"
            )
    lines += [
        "",
        "## Guardrail",
        "",
        "Gold-page transfer simulates a historical judged-query memory. Retrieved-page unions may be very large; recall must be interpreted together with pool size. Exact page/file overlap is meaningful only inside the same corpus/source.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
