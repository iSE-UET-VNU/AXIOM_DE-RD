"""Quick offline screen of research-inspired fusion strategies.

This experiment is intentionally separate from the existing fusion scripts.
It reuses cached Physics BM25/V-SPLADE runs and the feature artifact produced
by physics_adaptive_evidence_fusion.py.  It tests:

* robust quantile score calibration;
* a query-level adaptive gate for lexical versus visual weight;
* residual visual novelty fusion;
* existing hierarchical file-to-page results when available.

No rendering, encoding, API call or new model is used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
for _path in (ROOT, EXPERIMENT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.evaluation.benchmarks.vidore_v3 import ViDoreV3
from evaluate_physics_file_level import _evaluate, _load_physics_questions_and_qrels
from physics_adaptive_evidence_fusion import FEATURE_NAMES


PAIR_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"
ADAPTIVE_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_adaptive_evidence_fusion"
FILE_POOL_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_file_pool_ablation"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_fusion_research_screen"
BM25_PATH = PAIR_DIR / "bm25_french_bm25-french_vs-english.jsonl"
VISUAL_PATH = PAIR_DIR / "vsplade_french_bm25-french_vs-english.jsonl"


def _load_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["qid"])] = list(row["chunks"])
    return result


def _load_features(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["qid"])][str(row["page_id"])] = row
    return dict(result)


def _norm(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    maximum = max(values.values())
    if maximum <= 0:
        return {key: 0.0 for key in values}
    return {key: value / maximum for key, value in values.items()}


def _quantile_norm(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(list(values.values()), dtype=np.float64)
    low, high = np.quantile(array, [0.05, 0.95])
    if high <= low:
        return _norm(values)
    return {
        key: float(np.clip((value - low) / (high - low), 0.0, 1.0))
        for key, value in values.items()
    }


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _ranking_to_run(
    rankings: dict[str, list[tuple[str, float, str]]],
    depth: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    return {
        qid: [
            {
                "chunk_id": page_id,
                "doc_id": page_id,
                "score": float(score),
                "rank": rank,
                "reason": reason,
            }
            for rank, (page_id, score, reason) in enumerate(items[:depth], 1)
        ]
        for qid, items in rankings.items()
    }


def _weighted_rank(
    bm_chunks: list[dict[str, Any]],
    visual_chunks: list[dict[str, Any]],
    alpha: float,
    *,
    calibration: str = "max",
) -> list[tuple[str, float, str]]:
    bm_raw = {str(item["chunk_id"]): float(item.get("score", 0.0)) for item in bm_chunks}
    visual_raw = {str(item["chunk_id"]): float(item.get("score", 0.0)) for item in visual_chunks}
    if calibration == "quantile":
        bm_score = _quantile_norm(bm_raw)
        visual_score = _quantile_norm(visual_raw)
    else:
        bm_score = _norm(bm_raw)
        visual_score = _norm(visual_raw)
    pages = set(bm_raw) | set(visual_raw)
    ranked = [
        (
            page_id,
            alpha * bm_score.get(page_id, 0.0)
            + (1.0 - alpha) * visual_score.get(page_id, 0.0),
            "calibrated_weighted" if calibration == "quantile" else "weighted_alpha",
        )
        for page_id in pages
    ]
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _query_features(
    qids: list[str],
    bm25_run: dict[str, list[dict[str, Any]]],
    visual_run: dict[str, list[dict[str, Any]]],
    benchmark: ViDoreV3,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    features: list[list[float]] = []
    names: dict[str, list[str]] = {}
    for qid in qids:
        bm = [float(item.get("score", 0.0)) for item in bm25_run[qid][:100]]
        visual = [float(item.get("score", 0.0)) for item in visual_run[qid][:100]]
        bm_norm = np.asarray(list(_norm({str(i): value for i, value in enumerate(bm)}).values()))
        visual_norm = np.asarray(list(_norm({str(i): value for i, value in enumerate(visual)}).values()))
        query = benchmark._by_qid[qid].query
        bm_top = set(str(item["chunk_id"]) for item in bm25_run[qid][:10])
        visual_top = set(str(item["chunk_id"]) for item in visual_run[qid][:10])
        names[qid] = [
            "query_length",
            "bm25_top1",
            "bm25_top5_mean",
            "bm25_top10_mean",
            "bm25_gap",
            "bm25_nonzero_fraction",
            "vsplade_top1",
            "vsplade_top5_mean",
            "vsplade_top10_mean",
            "vsplade_gap",
            "vsplade_nonzero_fraction",
            "top10_overlap",
            "top100_overlap",
        ]
        features.append(
            [
                float(len(query.split())),
                float(bm_norm[0]) if len(bm_norm) else 0.0,
                float(np.mean(bm_norm[:5])) if len(bm_norm) else 0.0,
                float(np.mean(bm_norm[:10])) if len(bm_norm) else 0.0,
                float(bm_norm[0] - bm_norm[1]) if len(bm_norm) > 1 else 0.0,
                float(np.count_nonzero(bm) / max(len(bm), 1)),
                float(visual_norm[0]) if len(visual_norm) else 0.0,
                float(np.mean(visual_norm[:5])) if len(visual_norm) else 0.0,
                float(np.mean(visual_norm[:10])) if len(visual_norm) else 0.0,
                float(visual_norm[0] - visual_norm[1]) if len(visual_norm) > 1 else 0.0,
                float(np.count_nonzero(visual) / max(len(visual), 1)),
                len(bm_top & visual_top) / 10.0,
                len(
                    set(str(item["chunk_id"]) for item in bm25_run[qid][:100])
                    & set(str(item["chunk_id"]) for item in visual_run[qid][:100])
                ) / 100.0,
            ]
        )
    return np.asarray(features, dtype=np.float64), names


def _page_recall(qid: str, ranking: list[tuple[str, float, str]], qrels: dict[str, dict[str, int]]) -> float:
    gold = set(qrels.get(qid, {}))
    top = set(page_id for page_id, _, _ in ranking[:10])
    return len(gold & top) / len(gold) if gold else 0.0


def _fit_ridge_gate(
    train_qids: list[str],
    qids: list[str],
    query_matrix: np.ndarray,
    bm25_run: dict[str, list[dict[str, Any]]],
    visual_run: dict[str, list[dict[str, Any]]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    q_index = {qid: index for index, qid in enumerate(qids)}
    alpha_grid = np.round(np.linspace(0.0, 1.0, 11), 2)
    targets: list[float] = []
    for qid in train_qids:
        choices = []
        for alpha in alpha_grid:
            ranking = _weighted_rank(bm25_run[qid], visual_run[qid], float(alpha))
            choices.append((_page_recall(qid, ranking, qrels), -abs(float(alpha) - 0.7), float(alpha)))
        targets.append(max(choices)[2])
    train_matrix = query_matrix[[q_index[qid] for qid in train_qids]]
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    scaled = (train_matrix - mean) / std
    design = np.column_stack([np.ones(len(scaled)), scaled])
    target = np.asarray(targets, dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * 0.1
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return {
        "weights": weights.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "alpha_grid": alpha_grid.tolist(),
        "train_qids": len(train_qids),
        "train_alpha_mean": float(np.mean(target)) if len(target) else 0.7,
    }


def _gate_alpha(model: dict[str, Any], vector: np.ndarray) -> float:
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    scaled = (vector - mean) / std
    return float(np.clip(np.r_[1.0, scaled] @ weights, 0.0, 1.0))


def _fit_residual_rule(
    train_qids: list[str],
    feature_rows: dict[str, dict[str, dict[str, Any]]],
    bm25_run: dict[str, list[dict[str, Any]]],
    visual_run: dict[str, list[dict[str, Any]]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    index = {name: FEATURE_NAMES.index(name) for name in (
        "bm25_score_norm", "vsplade_score_norm", "lexical_term_coverage",
        "visual_percentile", "both_top10",
    )}
    best_params = {"beta": 0.0, "visual_threshold": 0.75, "lexical_threshold": 0.10, "agreement_bonus": 0.0}
    best_key = (-1.0, -1.0)
    for beta in (0.00, 0.02, 0.05, 0.10, 0.20):
        for visual_threshold in (0.70, 0.80, 0.90):
            for lexical_threshold in (0.05, 0.10, 0.20):
                for agreement_bonus in (0.00, 0.01, 0.03):
                    rankings: dict[str, list[tuple[str, float, str]]] = {}
                    for qid in train_qids:
                        bm = {str(item["chunk_id"]): float(item.get("score", 0.0)) for item in bm25_run[qid]}
                        visual = {str(item["chunk_id"]): float(item.get("score", 0.0)) for item in visual_run[qid]}
                        bm_norm = _norm(bm)
                        visual_norm = _norm(visual)
                        scored = []
                        for page_id in set(bm) | set(visual):
                            row = feature_rows[qid][page_id]
                            values = row["features"]
                            lexical = float(values[index["lexical_term_coverage"]])
                            visual_conf = float(values[index["visual_percentile"]])
                            residual = (
                                beta * visual_norm.get(page_id, 0.0)
                                * max(0.0, 1.0 - bm_norm.get(page_id, 0.0))
                                * max(0.0, 1.0 - lexical)
                            )
                            rescue = visual_conf >= visual_threshold and lexical <= lexical_threshold
                            score = (
                                0.7 * bm_norm.get(page_id, 0.0)
                                + 0.3 * visual_norm.get(page_id, 0.0)
                                + residual
                                + (agreement_bonus if values[index["both_top10"]] > 0 else 0.0)
                            )
                            scored.append((page_id, score, "residual_visual" if rescue else "weighted_backbone"))
                        rankings[qid] = sorted(scored, key=lambda item: (-item[1], item[0]))
                    recall = float(np.mean([_page_recall(qid, rankings[qid], qrels) for qid in train_qids]))
                    hit = float(np.mean([
                        bool(set(page for page, _, _ in rankings[qid][:10]) & set(qrels.get(qid, {})))
                        for qid in train_qids
                    ]))
                    if (recall, hit) > best_key:
                        best_key = (recall, hit)
                        best_params = {
                            "beta": beta,
                            "visual_threshold": visual_threshold,
                            "lexical_threshold": lexical_threshold,
                            "agreement_bonus": agreement_bonus,
                        }
    return best_params


def _residual_rank(
    qid: str,
    params: dict[str, float],
    feature_rows: dict[str, dict[str, dict[str, Any]]],
    bm25_run: dict[str, list[dict[str, Any]]],
    visual_run: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, float, str]]:
    index = {name: FEATURE_NAMES.index(name) for name in (
        "bm25_score_norm", "vsplade_score_norm", "lexical_term_coverage",
        "visual_percentile", "both_top10",
    )}
    bm = {str(item["chunk_id"]): float(item.get("score", 0.0)) for item in bm25_run[qid]}
    visual = {str(item["chunk_id"]): float(item.get("score", 0.0)) for item in visual_run[qid]}
    bm_norm = _norm(bm)
    visual_norm = _norm(visual)
    scored = []
    for page_id in set(bm) | set(visual):
        values = feature_rows[qid][page_id]["features"]
        lexical = float(values[index["lexical_term_coverage"]])
        visual_conf = float(values[index["visual_percentile"]])
        rescue = visual_conf >= params["visual_threshold"] and lexical <= params["lexical_threshold"]
        residual = (
            params["beta"] * visual_norm.get(page_id, 0.0)
            * max(0.0, 1.0 - bm_norm.get(page_id, 0.0))
            * max(0.0, 1.0 - lexical)
        )
        score = (
            0.7 * bm_norm.get(page_id, 0.0)
            + 0.3 * visual_norm.get(page_id, 0.0)
            + residual
            + (params["agreement_bonus"] if values[index["both_top10"]] > 0 else 0.0)
        )
        scored.append((page_id, score, "residual_visual" if rescue else "weighted_backbone"))
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def _summary(run: dict[str, list[dict[str, Any]]], qids: list[str], qrels: dict[str, dict[str, int]]) -> dict[str, Any]:
    metrics = _evaluate(run, qids, qrels)
    metrics.pop("per_query", None)
    return metrics


def _metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "page_recall@10": candidate["page_recall@10"] - baseline["page_recall@10"],
        "page_hit@10": candidate["page_hit@10"] - baseline["page_hit@10"],
        "ndcg@10": candidate["ndcg@10"] - baseline["ndcg@10"],
        "file_recall@3": (
            candidate["file_metrics_by_k"]["3"]["file_recall"]
            - baseline["file_metrics_by_k"]["3"]["file_recall"]
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_related_reports() -> dict[str, Any]:
    result: dict[str, Any] = {}
    adaptive_report = ADAPTIVE_DIR / "report.json"
    if adaptive_report.is_file():
        payload = json.loads(adaptive_report.read_text(encoding="utf-8"))
        result["adaptive_evidence_existing"] = {
            "oof_metrics": payload.get("oof_metrics"),
            "deltas_vs_baseline": payload.get("deltas_vs_baseline"),
        }
    file_report = FILE_POOL_DIR / "report.json"
    if file_report.is_file():
        payload = json.loads(file_report.read_text(encoding="utf-8"))
        result["hierarchical_file_page_existing"] = {
            "summary": payload.get("summary"),
            "source": str(file_report),
        }
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Physics fusion research screen",
        "",
        "Offline screen over 302 French Physics queries. All new methods use the "
        "same union of cached top-100 BM25 and V-SPLADE candidates. V-SPLADE "
        "uses English query vectors against French qrels.",
        "",
        "## OOF metrics",
        "",
        "| Method | nDCG@10 | page hit@10 | page recall@10 | file recall@3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, metrics in report["oof_metrics"].items():
        lines.append(
            f"| {method} | {metrics['ndcg@10']:.2f} | "
            f"{metrics['page_hit@10']:.2%} | {metrics['page_recall@10']:.2%} | "
            f"{metrics['file_metrics_by_k']['3']['file_recall']:.2%} |"
        )
    lines += [
        "",
        "## Delta versus weighted alpha=0.7",
        "",
        "| Method | page recall delta | page hit delta | nDCG delta | file recall@3 delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, delta in report["deltas_vs_baseline"].items():
        lines.append(
            f"| {method} | {100.0 * delta['page_recall@10']:+.2f} pp | "
            f"{100.0 * delta['page_hit@10']:+.2f} pp | "
            f"{delta['ndcg@10']:+.2f} | "
            f"{100.0 * delta['file_recall@3']:+.2f} pp |"
        )
    lines += [
        "",
        "## Fold results",
        "",
        "| Fold | Method | page recall@10 | delta versus baseline |",
        "|---:|---|---:|---:|",
    ]
    for fold in report["fold_metrics"]:
        for method, metrics in fold["methods"].items():
            delta = fold.get("delta_vs_baseline", {}).get(method, {}).get("page_recall@10", 0.0)
            lines.append(
                f"| {fold['fold']} | {method} | {metrics['page_recall@10']:.2%} | "
                f"{100.0 * delta:+.2f} pp |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Quantile calibration tests whether robust query-local score scaling is enough.",
        "- The query gate predicts one lexical/visual alpha per query from score-curve and overlap features.",
        "- Residual fusion keeps weighted BM25 as the backbone and gives V-SPLADE only a novelty bonus when lexical coverage is weak.",
        "- Hierarchical file-to-page numbers are imported from the existing controlled ablation and are not silently mixed into the new OOF ranking table.",
        "",
        "Late interaction and GQR were not run: the repository has no compatible "
        "cached multi-vector checkpoint for this offline screen. They remain "
        "future model branches rather than comparable cached baselines.",
        "",
        "See per_query.jsonl for query-level alpha, rankings and score reasons.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir
    runs_dir = output_dir / "runs"
    started = time.perf_counter()
    timings: dict[str, float] = {}

    load_started = time.perf_counter()
    qids, qrels = _load_physics_questions_and_qrels()
    qids = sorted(qids, key=lambda qid: int(qid.rsplit("::", 1)[1]))
    benchmark = ViDoreV3(root=ROOT / "data/benchmark/vidore_v3", subset="physics", language="french")
    bm25_run = _load_run(BM25_PATH)
    visual_run = _load_run(VISUAL_PATH)
    feature_rows = _load_features(ADAPTIVE_DIR / "features.jsonl")
    query_matrix, query_feature_names = _query_features(qids, bm25_run, visual_run, benchmark)
    timings["load_cached_artifacts"] = time.perf_counter() - load_started

    if len(qids) != 302 or set(bm25_run) != set(qids) or set(visual_run) != set(qids):
        raise RuntimeError("Expected exactly 302 qids in both cached runs")
    if any(len(bm25_run[qid]) != 100 or len(visual_run[qid]) != 100 for qid in qids):
        raise RuntimeError("Both cached inputs must contain exactly 100 candidates per qid")
    if set(feature_rows) != set(qids):
        raise RuntimeError("Adaptive feature artifact does not cover all qids")
    if any(len(feature_rows[qid]) == 0 for qid in qids):
        raise RuntimeError("Adaptive feature artifact contains an empty candidate query")
    if query_matrix.shape != (302, 13):
        raise RuntimeError(f"Unexpected query feature shape: {query_matrix.shape}")
    if any(len(set(str(item["chunk_id"]) for item in bm25_run[qid])) != 100 for qid in qids):
        raise RuntimeError("BM25 input contains duplicate candidates")
    if any(len(set(str(item["chunk_id"]) for item in visual_run[qid])) != 100 for qid in qids):
        raise RuntimeError("V-SPLADE input contains duplicate candidates")

    folds = {
        qid: index % 5
        for index, qid in enumerate(qids)
    }
    fold_metrics: list[dict[str, Any]] = []
    baseline_oof: dict[str, list[tuple[str, float, str]]] = {}
    calibrated_oof: dict[str, list[tuple[str, float, str]]] = {}
    gate_oof: dict[str, list[tuple[str, float, str]]] = {}
    residual_oof: dict[str, list[tuple[str, float, str]]] = {}
    gate_models: dict[str, Any] = {}
    residual_models: dict[str, Any] = {}
    baseline_all: dict[str, list[tuple[str, float, str]]] = {}
    calibrated_all: dict[str, list[tuple[str, float, str]]] = {}
    for qid in qids:
        baseline_all[qid] = _weighted_rank(bm25_run[qid], visual_run[qid], 0.7)
        calibrated_all[qid] = _weighted_rank(
            bm25_run[qid], visual_run[qid], 0.7, calibration="quantile"
        )

    fit_started = time.perf_counter()
    for fold in range(5):
        train_qids = [qid for qid in qids if folds[qid] != fold]
        test_qids = [qid for qid in qids if folds[qid] == fold]
        gate = _fit_ridge_gate(
            train_qids, qids, query_matrix, bm25_run, visual_run, qrels
        )
        residual_params = _fit_residual_rule(
            train_qids, feature_rows, bm25_run, visual_run, qrels
        )
        gate_models[str(fold)] = gate
        residual_models[str(fold)] = residual_params
        gate_test: dict[str, list[tuple[str, float, str]]] = {}
        residual_test: dict[str, list[tuple[str, float, str]]] = {}
        for qid in test_qids:
            query_index = qids.index(qid)
            alpha = _gate_alpha(gate, query_matrix[query_index])
            gate_test[qid] = _weighted_rank(
                bm25_run[qid], visual_run[qid], alpha
            )
            residual_test[qid] = _residual_rank(
                qid, residual_params, feature_rows, bm25_run, visual_run
            )
        baseline_test = {qid: baseline_all[qid] for qid in test_qids}
        calibrated_test = {qid: calibrated_all[qid] for qid in test_qids}
        baseline_oof.update(baseline_test)
        calibrated_oof.update(calibrated_test)
        gate_oof.update(gate_test)
        residual_oof.update(residual_test)
        test_runs = {
            "baseline_weighted07": _ranking_to_run(baseline_test),
            "calibrated_quantile07": _ranking_to_run(calibrated_test),
            "query_adaptive_gate": _ranking_to_run(gate_test),
            "residual_novelty": _ranking_to_run(residual_test),
        }
        fold_method_metrics = {
            method: _summary(run, test_qids, qrels)
            for method, run in test_runs.items()
        }
        fold_metrics.append(
            {
                "fold": fold,
                "train_queries": len(train_qids),
                "test_queries": len(test_qids),
                "methods": fold_method_metrics,
                "delta_vs_baseline": {
                    method: _metric_delta(
                        fold_method_metrics[method],
                        fold_method_metrics["baseline_weighted07"],
                    )
                    for method in (
                        "calibrated_quantile07",
                        "query_adaptive_gate",
                        "residual_novelty",
                    )
                },
                "gate_fit": gate,
                "residual_fit": residual_params,
                "test_fold_qrels_used_during_fit": False,
            }
        )
    timings["fit_and_oof_generation"] = time.perf_counter() - fit_started

    full_gate = _fit_ridge_gate(qids, qids, query_matrix, bm25_run, visual_run, qrels)
    full_residual = _fit_residual_rule(qids, feature_rows, bm25_run, visual_run, qrels)
    gate_full: dict[str, list[tuple[str, float, str]]] = {}
    residual_full: dict[str, list[tuple[str, float, str]]] = {}
    for query_index, qid in enumerate(qids):
        alpha = _gate_alpha(full_gate, query_matrix[query_index])
        gate_full[qid] = _weighted_rank(bm25_run[qid], visual_run[qid], alpha)
        residual_full[qid] = _residual_rank(
            qid, full_residual, feature_rows, bm25_run, visual_run
        )
    oof_rankings = {
        "baseline_weighted07": baseline_oof,
        "calibrated_quantile07": calibrated_oof,
        "query_adaptive_gate": gate_oof,
        "residual_novelty": residual_oof,
    }
    oof_runs = {method: _ranking_to_run(ranking) for method, ranking in oof_rankings.items()}
    oof_metrics = {
        method: _summary(run, qids, qrels)
        for method, run in oof_runs.items()
    }
    baseline_metrics = oof_metrics["baseline_weighted07"]
    deltas = {
        method: _metric_delta(oof_metrics[method], baseline_metrics)
        for method in (
            "calibrated_quantile07",
            "query_adaptive_gate",
            "residual_novelty",
        )
    }

    per_query: list[dict[str, Any]] = []
    for qid in qids:
        source_gold = set(qrels.get(qid, {}))
        payload: dict[str, Any] = {
            "qid": qid,
            "fold": folds[qid],
            "query": benchmark._by_qid[qid].query,
            "gold_pages": sorted(source_gold),
            "gold_files": sorted({_file_id(page) for page in source_gold}),
            "candidate_count": len(feature_rows[qid]),
            "methods": {},
        }
        for method, run in oof_runs.items():
            top10 = [str(item["chunk_id"]) for item in run[qid][:10]]
            metric = _evaluate(run, [qid], qrels)["per_query"][0]
            payload["methods"][method] = {
                "top10_pages": top10,
                "top3_files": list(dict.fromkeys(_file_id(page) for page in [
                    str(item["chunk_id"]) for item in run[qid][:100]
                ]))[:3],
                "page_hit@10": metric["page_hit@10"],
                "page_recall@10": metric["page_recall@10"],
                "file_recall@3": (
                    len(set(list(dict.fromkeys(_file_id(page) for page in [
                        str(item["chunk_id"]) for item in run[qid][:100]
                    ]))[:3]) & {
                        _file_id(page) for page in qrels.get(qid, {})
                    })
                    / len({_file_id(page) for page in qrels.get(qid, {})})
                    if qrels.get(qid) else 0.0
                ),
                "reasons": [str(item.get("reason", "")) for item in run[qid][:10]],
            }
        query_index = qids.index(qid)
        payload["query_gate_alpha"] = _gate_alpha(gate_models[str(folds[qid])], query_matrix[query_index])
        payload["residual_params"] = residual_models[str(folds[qid])]
        per_query.append(payload)

    report = {
        "experiment": "physics_fusion_research_screen",
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": 1674,
        "candidate_pool": "union of cached top-100 BM25 and top-100 V-SPLADE",
        "primary_metric": "page_recall@10",
        "oof_metrics": oof_metrics,
        "deltas_vs_baseline": deltas,
        "fold_metrics": fold_metrics,
        "query_gate": {
            "feature_names": query_feature_names[qids[0]],
            "models_by_fold": gate_models,
        },
        "residual_novelty": {
            "models_by_fold": residual_models,
        },
        "full_fit_qualitative": {
            "query_gate_model": full_gate,
            "residual_params": full_residual,
            "query_gate_metrics": _summary(_ranking_to_run(gate_full), qids, qrels),
            "residual_metrics": _summary(_ranking_to_run(residual_full), qids, qrels),
        },
        "related_existing_results": _load_related_reports(),
        "timing_seconds": {
            **timings,
            "total": time.perf_counter() - started,
        },
        "validation": {
            "qids": len(qids),
            "bm25_candidates_per_qid": 100,
            "vsplade_candidates_per_qid": 100,
            "folds_disjoint": True,
            "test_fold_qrels_used_during_fit": False,
            "no_render_encode_api_or_new_model": True,
            "vsplade_query_language": "english",
            "evaluation_language": "french",
        },
        "sources": {
            "bm25_run": str(BM25_PATH),
            "vsplade_run": str(VISUAL_PATH),
            "feature_rows": str(ADAPTIVE_DIR / "features.jsonl"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", {
        "experiment": report["experiment"],
        "folds": 5,
        "seed": 42,
        "fixed_baseline_alpha": 0.7,
        "candidate_depth": 100,
        "primary_metric": "page_recall@10",
        "methods": [
            "baseline_weighted07",
            "calibrated_quantile07",
            "query_adaptive_gate",
            "residual_novelty",
        ],
        "sources": report["sources"],
    })
    _write_json(output_dir / "report.json", report)
    _write_json(output_dir / "fold_metrics.json", fold_metrics)
    _write_jsonl(output_dir / "per_query.jsonl", per_query)
    for method, run in oof_runs.items():
        _write_jsonl(
            runs_dir / f"{method}_oof.jsonl",
            [{"qid": qid, "chunks": chunks} for qid, chunks in run.items()],
        )
    _write_jsonl(
        runs_dir / "query_adaptive_gate_full_fit.jsonl",
        [{"qid": qid, "chunks": chunks} for qid, chunks in _ranking_to_run(gate_full).items()],
    )
    output_md = output_dir / "report.md"
    output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "baseline_page_recall@10": baseline_metrics["page_recall@10"],
        "methods": {
            method: {
                "page_recall@10": metrics["page_recall@10"],
                "file_recall@3": metrics["file_metrics_by_k"]["3"]["file_recall"],
            }
            for method, metrics in oof_metrics.items()
        },
        "deltas_vs_baseline": deltas,
        "total_seconds": report["timing_seconds"]["total"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
