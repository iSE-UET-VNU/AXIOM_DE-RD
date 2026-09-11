"""Fold-fitted nonlinear reranking for the cached Physics hybrid candidates.

This is an offline research experiment.  It reuses the feature rows produced
by ``physics_adaptive_evidence_fusion.py`` and the released Physics qrels.  No
rendering, encoding, network call or new document model is used.

The candidate set is fixed to the union of the cached top-100 BM25 and
V-SPLADE results.  ExtraTrees and HistGradientBoosting are tested as small
nonlinear rankers because the earlier linear ranker could not model
interactions such as ``visual confidence x low lexical coverage`` well.
All fitting is query-folded; held-out qrels are never used for fitting.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
for path in (ROOT, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_physics_file_level import _evaluate, _ndcg_at_k  # noqa: E402
from physics_adaptive_evidence_fusion import FEATURE_NAMES  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


FEATURE_PATH = (
    ROOT
    / "data/benchmark/vidore_v3/results/physics_adaptive_evidence_fusion/features.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/physics_fusion_ml_screen"
)
WEIGHTED_BASELINE = 0.7
SEED = 20260904


def _benchmark() -> ViDoreV3:
    candidates = (
        ROOT / "data/benchmark/vidore_v3",
        ROOT / "data/raw/benchmarks/vidore_v3",
    )
    for root in candidates:
        try:
            return ViDoreV3(root=root, subset="physics", language="french")
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Could not locate the Physics ViDoRe V3 parquet files")


def _load_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature artifact not found: {path}")
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                output[str(row["qid"])].append(row)
    return dict(output)


def _feature_vector(values: list[float]) -> np.ndarray:
    base = np.asarray(values, dtype=np.float64)
    if base.shape != (len(FEATURE_NAMES),):
        raise RuntimeError(f"Expected {len(FEATURE_NAMES)} features, got {base.shape}")
    # Explicit interactions encode the research hypothesis and also make the
    # linear part of a tree split easier to discover with a small dataset.
    idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    bm = base[idx["bm25_score_norm"]]
    vis = base[idx["vsplade_score_norm"]]
    lex = base[idx["lexical_term_coverage"]]
    anchor = base[idx["anchor_coverage"]]
    agreement = base[idx["both_top10"]]
    conflict = base[idx["conflict_signal"]]
    rescue = base[idx["semantic_rescue_signal"]]
    return np.concatenate(
        [
            base,
            np.asarray(
                [
                    bm * lex,
                    vis * max(0.0, 1.0 - lex),
                    vis * anchor,
                    agreement * bm,
                    agreement * vis,
                    rescue * vis,
                    conflict * abs(bm - vis),
                    bm - vis,
                ],
                dtype=np.float64,
            ),
        ]
    )


def _prepare(
    rows_by_qid: Mapping[str, list[dict[str, Any]]],
    qrels: Mapping[str, Mapping[str, int]],
    qids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, tuple[int, int]]]:
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    baseline: list[float] = []
    page_ids: list[str] = []
    slices: dict[str, tuple[int, int]] = {}
    for qid in qids:
        start = len(vectors)
        seen: set[str] = set()
        for row in rows_by_qid[qid]:
            page_id = str(row["page_id"])
            if page_id in seen:
                raise RuntimeError(f"Duplicate candidate {qid}/{page_id}")
            seen.add(page_id)
            values = [float(value) for value in row["features"]]
            vectors.append(_feature_vector(values))
            labels.append(int(qrels.get(qid, {}).get(page_id, 0) > 0))
            idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
            baseline.append(
                WEIGHTED_BASELINE * values[idx["bm25_score_norm"]]
                + (1.0 - WEIGHTED_BASELINE) * values[idx["vsplade_score_norm"]]
            )
            page_ids.append(page_id)
        slices[qid] = (start, len(vectors))
    return (
        np.asarray(vectors, dtype=np.float64),
        np.asarray(labels, dtype=np.int8),
        np.asarray(baseline, dtype=np.float64),
        page_ids,
        slices,
    )


def _run_from_scores(
    qids: list[str],
    page_ids: list[str],
    scores: np.ndarray,
    slices: Mapping[str, tuple[int, int]],
    rows_by_qid: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for qid in qids:
        start, end = slices[qid]
        indices = list(range(start, end))
        indices.sort(key=lambda pos: (-float(scores[pos]), page_ids[pos]))
        output[qid] = [
            {
                "chunk_id": page_ids[pos],
                "doc_id": page_ids[pos],
                "score": float(scores[pos]),
                "rank": rank,
                "text": "",
            }
            for rank, pos in enumerate(indices[:100], 1)
        ]
        if len(output[qid]) != min(100, len(rows_by_qid[qid])):
            raise RuntimeError(f"Unexpected output depth for {qid}")
    return output


def _metric_summary(
    run: Mapping[str, list[dict[str, Any]]],
    qids: list[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    payload = _evaluate(dict(run), qids, dict(qrels))
    payload.pop("per_query", None)
    return payload


def _delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        "page_recall@10": float(candidate["page_recall@10"])
        - float(baseline["page_recall@10"]),
        "page_hit@10": float(candidate["page_hit@10"])
        - float(baseline["page_hit@10"]),
        "ndcg@10": float(candidate["ndcg@10"]) - float(baseline["ndcg@10"]),
        "file_recall@3": float(candidate["file_metrics_by_k"]["3"]["file_recall"])
        - float(baseline["file_metrics_by_k"]["3"]["file_recall"]),
    }


def _make_models() -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

    return {
        "extra_trees_d8_leaf2": ExtraTreesClassifier(
            n_estimators=260,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "extra_trees_d12_leaf2": ExtraTreesClassifier(
            n_estimators=260,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED + 1,
            n_jobs=-1,
        ),
        "extra_trees_unbounded_leaf4": ExtraTreesClassifier(
            n_estimators=260,
            max_depth=None,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=SEED + 2,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=SEED + 3,
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Physics nonlinear hybrid reranking",
        "",
        "Fold-fitted nonlinear rerankers over the fixed union of cached top-100 BM25 and V-SPLADE candidates.",
        "V-SPLADE uses English query vectors while qrels and PDF text are French.",
        "",
        "## OOF results",
        "",
        "| Method | nDCG@10 | Page hit@10 | Page recall@10 | File recall@3 | Delta page recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["oof_metrics"].items():
        delta = (
            report["deltas_vs_baseline"].get(name, {}).get("page_recall@10", 0.0)
            * 100.0
        )
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.2f} | {metrics['page_hit@10']:.2%} | "
            f"{metrics['page_recall@10']:.2%} | "
            f"{metrics['file_metrics_by_k']['3']['file_recall']:.2%} | {delta:+.2f} pp |"
        )
    lines += [
        "",
        "## Fold results",
        "",
        "| Fold | Method | Page recall@10 | Delta vs fold baseline |",
        "|---:|---|---:|---:|",
    ]
    for fold in report["fold_metrics"]:
        for name, metrics in fold["methods"].items():
            delta = fold["deltas_vs_baseline"].get(name, {}).get("page_recall@10", 0.0)
            lines.append(
                f"| {fold['fold']} | {name} | {metrics['page_recall@10']:.2%} | "
                f"{delta * 100.0:+.2f} pp |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "The candidate pool is fixed before fitting, so this experiment tests ranking rather than candidate generation.",
        "The models use only query/page features and do not use modality or qrel labels as features.",
        "",
        "See `per_query.jsonl` for per-query rankings and `runs/` for OOF runs.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=Path, default=FEATURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    started = time.perf_counter()

    benchmark = _benchmark()
    questions = sorted(
        list(benchmark.questions()),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    qids = [question.qid for question in questions]
    qrels = benchmark.qrels()
    rows_by_qid = _load_rows(args.feature_path)
    if len(qids) != 302 or set(rows_by_qid) != set(qids):
        raise RuntimeError("Expected exactly 302 Physics qids in benchmark and features")
    if any(len(rows_by_qid[qid]) < 100 for qid in qids):
        raise RuntimeError("Every query must have at least 100 union candidates")

    features, labels, baseline_scores, page_ids, slices = _prepare(
        rows_by_qid, qrels, qids
    )
    if features.shape[0] < 40000:
        raise RuntimeError(f"Unexpectedly small candidate matrix: {features.shape}")
    baseline_run = _run_from_scores(
        qids, page_ids, baseline_scores, slices, rows_by_qid
    )
    baseline_metrics = _metric_summary(baseline_run, qids, qrels)

    folds = {qid: index % 5 for index, qid in enumerate(qids)}
    model_names = list(_make_models())
    oof_scores = {name: np.zeros(len(page_ids), dtype=np.float64) for name in model_names}
    fold_metrics: list[dict[str, Any]] = []
    fit_started = time.perf_counter()
    for fold in range(5):
        train_qids = [qid for qid in qids if folds[qid] != fold]
        test_qids = [qid for qid in qids if folds[qid] == fold]
        train_positions = np.asarray(
            [pos for qid in train_qids for pos in range(*slices[qid])], dtype=np.int64
        )
        test_positions = np.asarray(
            [pos for qid in test_qids for pos in range(*slices[qid])], dtype=np.int64
        )
        fold_models = _make_models()
        fold_method_runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
        fold_method_runs["baseline_weighted07"] = _run_from_scores(
            test_qids, page_ids, baseline_scores, slices, rows_by_qid
        )
        for name, model in fold_models.items():
            model.fit(features[train_positions], labels[train_positions])
            probabilities = model.predict_proba(features[test_positions])[:, 1]
            oof_scores[name][test_positions] = probabilities
            # Keep the fold's test score map isolated; no test labels are
            # consulted in fitting or model selection.
            fold_method_runs[name] = _run_from_scores(
                test_qids, page_ids, oof_scores[name], slices, rows_by_qid
            )
        fold_methods = {
            name: _metric_summary(run, test_qids, qrels)
            for name, run in fold_method_runs.items()
        }
        fold_baseline = fold_methods["baseline_weighted07"]
        fold_metrics.append(
            {
                "fold": fold,
                "train_queries": len(train_qids),
                "test_queries": len(test_qids),
                "methods": fold_methods,
                "deltas_vs_baseline": {
                    name: _delta(metrics, fold_baseline)
                    for name, metrics in fold_methods.items()
                    if name != "baseline_weighted07"
                },
                "test_fold_qrels_used_during_fit": False,
            }
        )
        print(f"[fold {fold}] train={len(train_qids)} test={len(test_qids)}")
    fit_seconds = time.perf_counter() - fit_started

    oof_runs = {
        "baseline_weighted07": baseline_run,
        **{
            name: _run_from_scores(qids, page_ids, scores, slices, rows_by_qid)
            for name, scores in oof_scores.items()
        },
    }
    oof_metrics = {
        name: _metric_summary(run, qids, qrels) for name, run in oof_runs.items()
    }
    deltas = {
        name: _delta(metrics, oof_metrics["baseline_weighted07"])
        for name, metrics in oof_metrics.items()
        if name != "baseline_weighted07"
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name, run in oof_runs.items():
        with (runs_dir / f"{name}_oof.jsonl").open("w", encoding="utf-8") as handle:
            for qid in qids:
                handle.write(
                    json.dumps(
                        {"qid": qid, "query": benchmark._by_qid[qid].query, "chunks": run[qid]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    per_query: list[dict[str, Any]] = []
    for qid in qids:
        row: dict[str, Any] = {
            "qid": qid,
            "fold": folds[qid],
            "query": benchmark._by_qid[qid].query,
            "gold_pages": sorted(qrels.get(qid, {})),
            "methods": {},
        }
        for name, run in oof_runs.items():
            top10 = [str(item["chunk_id"]) for item in run[qid][:10]]
            metric = _evaluate(run, [qid], qrels)["per_query"][0]
            row["methods"][name] = {
                "top10_pages": top10,
                "page_hit@10": metric["page_hit@10"],
                "page_recall@10": metric["page_recall@10"],
                "nDCG@10": _ndcg_at_k(top10, qrels.get(qid, {}), 10),
            }
        per_query.append(row)

    report = {
        "experiment": "physics_fusion_ml_screen",
        "dataset": "vidore_v3/physics",
        "evaluation_language": "french",
        "queries": len(qids),
        "candidate_pool": "fixed union of cached top-100 BM25 and top-100 V-SPLADE candidates",
        "feature_count": int(features.shape[1]),
        "candidate_rows": int(features.shape[0]),
        "positive_candidate_rows": int(labels.sum()),
        "folds": 5,
        "fold_assignment": "sorted ordinal qid modulo five",
        "baseline": {
            "name": "baseline_weighted07",
            "alpha_bm25": WEIGHTED_BASELINE,
            "metrics": baseline_metrics,
        },
        "oof_metrics": oof_metrics,
        "deltas_vs_baseline": deltas,
        "fold_metrics": fold_metrics,
        "sources": {
            "features": str(args.feature_path),
            "benchmark_root": str(benchmark.root),
        },
        "validation": {
            "qids": len(qids),
            "candidate_min_per_qid": min(len(rows_by_qid[qid]) for qid in qids),
            "candidate_max_per_qid": max(len(rows_by_qid[qid]) for qid in qids),
            "qrels_unreachable": benchmark.unreachable_n,
            "test_fold_qrels_used_during_fit": False,
            "qrel_modality_features_used": False,
            "no_render_encode_api_or_new_document_model": True,
        },
        "timing_seconds": {
            "fit_oof_models": round(fit_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        },
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment": report["experiment"],
                "seed": SEED,
                "folds": 5,
                "features": str(args.feature_path),
                "models": model_names,
                "baseline_alpha_bm25": WEIGHTED_BASELINE,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "fold_metrics.json").write_text(
        json.dumps(fold_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "oof_metrics": {
                    name: {
                        "page_recall@10": metrics["page_recall@10"],
                        "file_recall@3": metrics["file_metrics_by_k"]["3"]["file_recall"],
                    }
                    for name, metrics in oof_metrics.items()
                },
                "deltas_vs_baseline": deltas,
                "timing_seconds": report["timing_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
