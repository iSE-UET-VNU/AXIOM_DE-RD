"""Held-out adaptive page-budget transfer on the 220-query benchmark.

This experiment learns a query's recall-vs-K curve from two source datasets and
chooses K for the held-out dataset.  It is deliberately retrieval-only:
features come from the query and cached BM25/dense scores, while qrels are
used only for fitting the source folds and evaluating the held-out fold.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.raw_hierarchical import RawPage, read_jsonl  # noqa: E402
from src.chunking_embedding.lexical import analyze  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DEFAULT_DATASET = ROOT / "data/raw/BENCHMARK"
DEFAULT_EMBEDDING = DEFAULT_CACHE / "embedding_adaptive_openrouter_te3s_fulltext/embeddings"
DEFAULT_OUTPUT = DEFAULT_CACHE / "adaptive_transfer"
K_VALUES = (4, 8, 12, 16, 20)


def _normalise(values: Mapping[str, float]) -> dict[str, float]:
    maximum = max(values.values(), default=0.0)
    return {key: value / maximum if maximum > 0 else 0.0 for key, value in values.items()}


def _features(query: str, rankings: Mapping[str, Sequence[tuple[str, float]]]) -> list[float]:
    bm = list(rankings["bm25"])
    de = list(rankings["dense"])
    hy = list(rankings["hybrid"])
    bm_scores = np.asarray([score for _pid, score in bm[:20]], dtype=float)
    de_scores = np.asarray([score for _pid, score in de[:20]], dtype=float)
    hy_scores = np.asarray([score for _pid, score in hy[:20]], dtype=float)

    def curve(scores: np.ndarray) -> list[float]:
        if not len(scores):
            return [0.0] * 12
        top = float(scores[0])
        tail = float(scores[-1])
        mass = float(np.sum(np.maximum(scores, 0.0)))
        entropy = float(-np.sum((scores / (np.sum(scores) + 1e-9)) * np.log(scores / (np.sum(scores) + 1e-9) + 1e-9)))
        return [
            top, float(scores[1] if len(scores) > 1 else top),
            float(scores[4] if len(scores) > 4 else tail),
            float(scores[9] if len(scores) > 9 else tail), tail,
            float(top - (scores[1] if len(scores) > 1 else top)),
            float(top - (scores[9] if len(scores) > 9 else tail)),
            mass, entropy,
            float(np.mean(scores[:5])), float(np.mean(scores[:10])), float(np.mean(scores)),
        ]

    bm_ids = {pid for pid, _score in bm[:10]}
    de_ids = {pid for pid, _score in de[:10]}
    overlap10 = len(bm_ids & de_ids) / 10.0
    bm20 = {pid for pid, _score in bm[:20]}
    de20 = {pid for pid, _score in de[:20]}
    tokens = list(analyze(query))
    return [
        float(len(tokens)), float(len(query)), overlap10,
        len(bm20 & de20) / 20.0,
        *_normalise({str(i): float(v) for i, v in enumerate(curve(bm_scores))}).values(),
        *_normalise({str(i): float(v) for i, v in enumerate(curve(de_scores))}).values(),
        *_normalise({str(i): float(v) for i, v in enumerate(curve(hy_scores))}).values(),
    ]


def _load() -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, dict[str, list[tuple[str, float]]]]]:
    pages_all = [RawPage(**row) for row in read_jsonl(DEFAULT_CACHE / "parsing/pages.jsonl")]
    pages = [page for page in pages_all if page.text_source == "pdf_inspector"]
    queries = read_jsonl(DEFAULT_DATASET / "queries.jsonl")
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(DEFAULT_CACHE / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    bm25 = BM25Index(analyzer_name="auto").build(
        {"chunk_id": page.page_id, "doc_id": page.doc_id, "text": page.text} for page in pages
    )
    page_ids = [page.page_id for page in pages]
    page_vectors = np.load(DEFAULT_EMBEDDING / "page_embeddings.npy", mmap_mode="r")
    query_vectors = np.load(DEFAULT_EMBEDDING / "query_embeddings.npy", mmap_mode="r")
    all_page_ids = [page.page_id for page in pages_all]
    positions = np.asarray([all_page_ids.index(page_id) for page_id in page_ids], dtype=np.int64)
    dense_pages = page_vectors[positions]
    dense_norms = np.linalg.norm(dense_pages, axis=1)
    query_norms = np.linalg.norm(query_vectors, axis=1)
    query_index = {str(row["query_id"]): i for i, row in enumerate(queries)}
    rankings: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for query in queries:
        qid = str(query["query_id"])
        bm = [(bm25.chunk_ids[i], float(s)) for i, s in bm25.search(str(query["query"]), 100)]
        dense_scores = (dense_pages @ query_vectors[query_index[qid]]) / (dense_norms * query_norms[query_index[qid]] + 1e-9)
        dense_by_id = {pid: float(score) for pid, score in zip(page_ids, dense_scores)}
        de = sorted(dense_by_id.items(), key=lambda x: (-x[1], x[0]))[:100]
        bm_norm = _normalise(dict(bm))
        de_norm = _normalise(dense_by_id)
        hy_scores = {pid: 0.10 * bm_norm.get(pid, 0.0) + 0.90 * de_norm.get(pid, 0.0) for pid in page_ids}
        hy = sorted(hy_scores.items(), key=lambda x: (-x[1], x[0]))[:100]
        rankings[qid] = {"bm25": bm, "dense": de, "hybrid": hy}
    return queries, qrels, rankings


def _recall_curve(qid: str, ranking: Sequence[tuple[str, float]], qrels: Mapping[str, set[str]]) -> np.ndarray:
    gold = qrels.get(qid, set())
    denom = max(len(gold), 1)
    return np.asarray([len({pid for pid, _score in ranking[:k]} & gold) / denom for k in K_VALUES], dtype=float)


def _choose(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, target_mean_k: float = 12.0) -> np.ndarray:
    from sklearn.ensemble import ExtraTreesRegressor

    # Predict the full recall curve, then select K under a train-calibrated
    # linear page-cost penalty.  The penalty is selected on source data only.
    models = []
    for col in range(train_y.shape[1]):
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.75, random_state=100 + col)
        model.fit(train_x, train_y[:, col])
        models.append(model)
    predicted_train = np.column_stack([model.predict(train_x) for model in models])
    predicted_test = np.column_stack([model.predict(test_x) for model in models])
    penalties = np.linspace(-0.01, 0.03, 401)
    best = None
    for penalty in penalties:
        choices = np.argmax(predicted_train - penalty * np.asarray(K_VALUES)[None, :], axis=1)
        mean_k = float(np.mean(np.asarray(K_VALUES)[choices]))
        key = (abs(mean_k - target_mean_k), -float(np.mean(predicted_train[np.arange(len(choices)), choices])))
        if best is None or key < best[0]:
            best = (key, penalty)
    # Calibrate the deployment budget using predictions only.  This avoids the
    # source/target dataset shift seen when a source-fitted cost penalty is
    # transferred directly (Industrial otherwise receives almost all K=20).
    test_best = None
    for penalty in penalties:
        choices = np.argmax(predicted_test - penalty * np.asarray(K_VALUES)[None, :], axis=1)
        mean_k = float(np.mean(np.asarray(K_VALUES)[choices]))
        key = (abs(mean_k - target_mean_k), -float(np.mean(predicted_test[np.arange(len(choices)), choices])))
        if test_best is None or key < test_best[0]:
            test_best = (key, choices)
    return np.asarray([K_VALUES[i] for i in test_best[1]], dtype=int)


def _metrics(qids: Sequence[str], rankings: Mapping[str, Sequence[tuple[str, float]]], qrels: Mapping[str, set[str]], choices: Sequence[int]) -> dict[str, float]:
    values = []
    r10 = []
    r20 = []
    for qid, k in zip(qids, choices):
        gold = qrels[qid]
        ranking = rankings[qid]
        denom = max(len(gold), 1)
        values.append(len({pid for pid, _s in ranking[:k]} & gold) / denom)
        r10.append(len({pid for pid, _s in ranking[:10]} & gold) / denom)
        r20.append(len({pid for pid, _s in ranking[:20]} & gold) / denom)
    return {"recall_at_returned_k": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean(choices))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    queries, qrels, all_rankings = _load()
    qids = [str(row["query_id"]) for row in queries]
    groups = [str(row["source"]) for row in queries]
    x = np.asarray([_features(str(row["query"]), all_rankings[qid]) for row, qid in zip(queries, qids)], dtype=float)
    y = {name: np.asarray([_recall_curve(qid, [*all_rankings[qid][name]], qrels) for qid in qids]) for name in ("bm25", "dense", "hybrid")}
    output: dict[str, Any] = {"benchmark": {"queries": len(qids), "features": int(x.shape[1]), "k_values": list(K_VALUES)}, "methods": {}}
    for name in ("bm25", "dense", "hybrid"):
        ranking = {qid: all_rankings[qid][name] for qid in qids}
        rows = []
        for held_out in sorted(set(groups)):
            train = np.asarray([g != held_out for g in groups])
            test = ~train
            choices = _choose(x[train], y[name][train], x[test])
            test_qids = [qid for qid, flag in zip(qids, test) if flag]
            rows.append({"held_out": held_out, "metrics": _metrics(test_qids, ranking, qrels, choices), "choice_histogram": {str(k): int(np.sum(choices == k)) for k in K_VALUES}})
        fixed = _metrics(qids, ranking, qrels, [12] * len(qids))
        output["methods"][name] = {"fixed_k12": fixed, "leave_one_dataset_out": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Held-out adaptive transfer", "", "Qrel-fitted budget policy is trained on two datasets and evaluated on the third.", "", "| Method | Fold | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---|---:|---:|---:|---:|"]
    for name, result in output["methods"].items():
        fixed = result["fixed_k12"]
        lines.append(f"| {name} | fixed K=12 | {fixed['recall_at_returned_k']:.2%} | {fixed['recall@10']:.2%} | {fixed['recall@20']:.2%} | {fixed['mean_k']:.2f} |")
        for fold in result["leave_one_dataset_out"]:
            m = fold["metrics"]
            lines.append(f"| {name} | held out {fold['held_out']} | {m['recall_at_returned_k']:.2%} | {m['recall@10']:.2%} | {m['recall@20']:.2%} | {m['mean_k']:.2f} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
