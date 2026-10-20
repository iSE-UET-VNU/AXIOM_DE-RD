"""OOF validation of file-evidence aggregation plus coverage-aware K.

The beta used to combine page score and file top-2 score is selected on each
training fold only.  The test fold receives K=0 for a source with no indexed
PDF-inspector pages and K=17 otherwise.  This validates the best current
candidate without selecting beta on the full qrel set.
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
from src.retrieval.sparse import BM25Index  # noqa: E402


CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DATASET = ROOT / "data/raw/BENCHMARK"
EMBEDDING = CACHE / "embedding_adaptive_openrouter_te3s_fulltext/embeddings"
OUTPUT = CACHE / "file_adaptive_oof"
BETAS = (0.10, 0.20, 0.30, 0.50)


def _metrics(qids: Sequence[str], rankings: Mapping[str, Mapping[float, Sequence[tuple[str, float]]]], qrels: Mapping[str, set[str]], beta: float, coverage_aware: bool) -> dict[str, float]:
    values, r10, r20, ks = [], [], [], []
    for qid in qids:
        source = qid.split("::", 1)[0]
        k = 0 if coverage_aware and source == "mpdocvqa" else (17 if coverage_aware else 12)
        ranking = rankings[qid][beta]
        pages = [page_id for page_id, _score in ranking[:k]]
        gold = qrels.get(qid, set())
        denom = max(len(gold), 1)
        values.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
        ks.append(len(pages))
    return {"recall_at_returned_k": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean(ks))}


def _build_rankings() -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, dict[float, list[tuple[str, float]]]]]:
    pages_all = [RawPage(**row) for row in read_jsonl(CACHE / "parsing/pages.jsonl")]
    pages = [page for page in pages_all if page.text_source == "pdf_inspector"]
    queries = read_jsonl(DATASET / "queries.jsonl")
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(CACHE / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    index = BM25Index(analyzer_name="auto").build({"chunk_id": p.page_id, "doc_id": p.doc_id, "text": p.text} for p in pages)
    page_ids = [page.page_id for page in pages]
    all_page_ids = [page.page_id for page in pages_all]
    page_position = {page_id: position for position, page_id in enumerate(all_page_ids)}
    positions = np.asarray([page_position[page_id] for page_id in page_ids], dtype=np.int64)
    page_vectors = np.load(EMBEDDING / "page_embeddings.npy", mmap_mode="r")[positions]
    query_vectors = np.load(EMBEDDING / "query_embeddings.npy", mmap_mode="r")
    page_norms = np.linalg.norm(page_vectors, axis=1)
    rankings: dict[str, dict[float, list[tuple[str, float]]]] = {}
    query_index = {str(row["query_id"]): i for i, row in enumerate(queries)}
    for query in queries:
        qid = str(query["query_id"])
        bm_hits = index.search(str(query["query"]), 100)
        bm_scores = {index.chunk_ids[position]: float(score) for position, score in bm_hits}
        bm_max = max(bm_scores.values(), default=0.0)
        qv = query_vectors[query_index[qid]]
        dense_scores = (page_vectors @ qv) / (page_norms * np.linalg.norm(qv) + 1e-9)
        dense_by_id = {page_id: float(score) for page_id, score in zip(page_ids, dense_scores)}
        dense_max = max(dense_by_id.values(), default=0.0)
        hybrid_scores = {page_id: 0.20 * bm_scores.get(page_id, 0.0) / max(bm_max, 1e-9) + 0.80 * dense_score / max(dense_max, 1e-9) for page_id, dense_score in dense_by_id.items()}
        candidate = sorted(hybrid_scores.items(), key=lambda item: (-item[1], item[0]))[:100]
        by_file: dict[str, list[float]] = defaultdict(list)
        for page_id, score in candidate:
            by_file[page_id.split("#page=", 1)[0]].append(score)
        file_scores = {file_id: sum(sorted(scores, reverse=True)[:2]) for file_id, scores in by_file.items()}
        file_max = max(file_scores.values(), default=1.0)
        page_max = max((score for _page_id, score in candidate), default=1.0)
        rankings[qid] = {}
        for beta in BETAS:
            scored = {
                page_id: (1.0 - beta) * score / max(page_max, 1e-9) + beta * file_scores.get(page_id.split("#page=", 1)[0], 0.0) / max(file_max, 1e-9)
                for page_id, score in candidate
            }
            rankings[qid][beta] = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    return queries, qrels, rankings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    queries, qrels, rankings = _build_rankings()
    qids = [str(query["query_id"]) for query in queries]
    groups = np.asarray([str(query["source"]) for query in queries])
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=2026)
    folds = []
    oof_choices = []
    oof_fixed_choices = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(qids, groups), start=1):
        train_qids = [qids[i] for i in train_idx]
        test_qids = [qids[i] for i in test_idx]
        train_metrics = {beta: _metrics(train_qids, rankings, qrels, beta, True)["recall_at_returned_k"] for beta in BETAS}
        train_fixed_metrics = {beta: _metrics(train_qids, rankings, qrels, beta, False)["recall_at_returned_k"] for beta in BETAS}
        selected = max(BETAS, key=lambda beta: (train_metrics[beta], -beta))
        selected_fixed = max(BETAS, key=lambda beta: (train_fixed_metrics[beta], -beta))
        test_metric = _metrics(test_qids, rankings, qrels, selected, True)
        test_fixed_metric = _metrics(test_qids, rankings, qrels, selected_fixed, False)
        folds.append({"fold": fold, "selected_beta": selected, "selected_fixed_beta": selected_fixed, "train_recall_at_returned_k": train_metrics, "train_fixed_recall_at_k12": train_fixed_metrics, "test_metrics": test_metric, "test_fixed_metrics": test_fixed_metric})
        oof_choices.extend((qids[i], selected) for i in test_idx)
        oof_fixed_choices.extend((qids[i], selected_fixed) for i in test_idx)
    oof_values = []
    oof_r10 = []
    oof_r20 = []
    oof_ks = []
    for qid, beta in oof_choices:
        source = qid.split("::", 1)[0]
        k = 0 if source == "mpdocvqa" else 17
        ranking = rankings[qid][beta]
        pages = [page_id for page_id, _score in ranking[:k]]
        gold = qrels[qid]
        denom = max(len(gold), 1)
        oof_values.append(len(set(pages) & gold) / denom)
        oof_r10.append(len(set(pages[:10]) & gold) / denom)
        oof_r20.append(len(set(pages[:20]) & gold) / denom)
        oof_ks.append(len(pages))
    oof = {"recall_at_returned_k": float(np.mean(oof_values)), "recall@10": float(np.mean(oof_r10)), "recall@20": float(np.mean(oof_r20)), "mean_k": float(np.mean(oof_ks))}
    fixed_oof_values, fixed_oof_r10, fixed_oof_r20 = [], [], []
    for qid, beta in oof_fixed_choices:
        ranking = rankings[qid][beta]
        pages = [page_id for page_id, _score in ranking[:12]]
        gold = qrels[qid]
        denom = max(len(gold), 1)
        fixed_oof_values.append(len(set(pages) & gold) / denom)
        fixed_oof_r10.append(len(set(pages[:10]) & gold) / denom)
        fixed_oof_r20.append(len(set(pages[:20]) & gold) / denom)
    fixed_oof = {"recall_at_returned_k": float(np.mean(fixed_oof_values)), "recall@10": float(np.mean(fixed_oof_r10)), "recall@20": float(np.mean(fixed_oof_r20)), "mean_k": 12.0}
    fixed_beta = {beta: _metrics(qids, rankings, qrels, beta, False) for beta in BETAS}
    report = {"queries": len(qids), "betas": list(BETAS), "method": "hybrid BM25=.20/dense=.80 top-2 file score; beta selected on train folds; coverage K=0/17", "qrels_used_for_ranking": False, "oof_adaptive": oof, "oof_fixed_control": fixed_oof, "fixed_k12_by_beta": fixed_beta, "folds": folds}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# OOF validation: file aggregation + coverage-aware K", "", "| Method | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    lines.append(f"| OOF adaptive beta | {oof['recall_at_returned_k']:.2%} | {oof['recall@10']:.2%} | {oof['recall@20']:.2%} | {oof['mean_k']:.2f} |")
    lines.append(f"| OOF fixed K=12 beta | {fixed_oof['recall_at_returned_k']:.2%} | {fixed_oof['recall@10']:.2%} | {fixed_oof['recall@20']:.2%} | {fixed_oof['mean_k']:.2f} |")
    for beta, metric in fixed_beta.items():
        lines.append(f"| Fixed K=12 beta={beta:.2f} | {metric['recall_at_returned_k']:.2%} | {metric['recall@10']:.2%} | {metric['recall@20']:.2%} | {metric['mean_k']:.2f} |")
    lines += ["", "| Fold | Adaptive beta | Fixed beta | Adaptive test recall | Fixed test recall |", "|---:|---:|---:|---:|---:|"]
    for row in folds:
        metric = row["test_metrics"]
        lines.append(f"| {row['fold']} | {row['selected_beta']:.2f} | {row['selected_fixed_beta']:.2f} | {metric['recall_at_returned_k']:.2%} | {row['test_fixed_metrics']['recall_at_returned_k']:.2%} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
