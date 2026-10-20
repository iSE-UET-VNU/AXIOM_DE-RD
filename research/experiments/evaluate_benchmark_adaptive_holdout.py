"""Clean source-stratified holdout for hybrid file aggregation + adaptive budget."""

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

from research.data_discovery.raw_hierarchical import RawPage, read_jsonl  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DATASET = ROOT / "data/raw/BENCHMARK"
OUTPUT = CACHE / "adaptive_clean_holdout"
ALPHAS = (0.10, 0.20, 0.30)
BETAS = (0.00, 0.20, 0.30, 0.50)


def _metrics(qids: list[str], pages_by_qid: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict[str, float]:
    values, r10, r20 = [], [], []
    for qid in qids:
        gold = qrels[qid]
        pages = pages_by_qid[qid]
        denom = max(len(gold), 1)
        values.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
    return {"recall_at_returned_k": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean([len(pages_by_qid[qid]) for qid in qids]))}


def _build_rankings(queries: list[dict[str, Any]], pages: list[RawPage], index: BM25Index, page_vectors: np.ndarray, query_vectors: np.ndarray) -> dict[str, dict[tuple[float, float], list[str]]]:
    page_ids = [page.page_id for page in pages]
    page_norms = np.linalg.norm(page_vectors, axis=1)
    output: dict[str, dict[tuple[float, float], list[str]]] = {}
    for q_index, query in enumerate(queries):
        qid = str(query["query_id"])
        bm_hits = index.search(str(query["query"]), 100)
        bm_scores = {index.chunk_ids[position]: float(score) for position, score in bm_hits}
        bm_max = max(bm_scores.values(), default=1.0)
        bm_scores = {page_id: score / bm_max for page_id, score in bm_scores.items()}
        qv = query_vectors[q_index]
        dense_scores = (page_vectors @ qv) / (page_norms * np.linalg.norm(qv) + 1e-9)
        dense_by_id = {page_id: float(score) for page_id, score in zip(page_ids, dense_scores)}
        dense_max = max(dense_by_id.values(), default=1.0)
        dense_by_id = {page_id: score / dense_max for page_id, score in dense_by_id.items()}
        output[qid] = {}
        for alpha in ALPHAS:
            hybrid_scores = {page_id: alpha * bm_scores.get(page_id, 0.0) + (1.0 - alpha) * dense_by_id[page_id] for page_id in page_ids}
            base = sorted(hybrid_scores.items(), key=lambda item: (-item[1], item[0]))[:100]
            page_max = max((score for _page_id, score in base), default=1.0)
            page_norm = {page_id: score / page_max for page_id, score in base}
            by_file: dict[str, list[float]] = defaultdict(list)
            for page_id, score in base:
                by_file[page_id.split("#page=", 1)[0]].append(score)
            file_scores = {file_id: sum(sorted(scores, reverse=True)[:2]) for file_id, scores in by_file.items()}
            file_max = max(file_scores.values(), default=1.0)
            file_scores = {file_id: score / file_max for file_id, score in file_scores.items()}
            for beta in BETAS:
                combined = {page_id: (1.0 - beta) * page_norm[page_id] + beta * file_scores.get(page_id.split("#page=", 1)[0], 0.0) for page_id in page_norm}
                ranking = sorted(combined.items(), key=lambda item: (-item[1], item[0]))
                output[qid][(alpha, beta)] = [page_id for page_id, _score in ranking[:17]]
    return output


def main() -> int:
    pages = [RawPage(**row) for row in read_jsonl(CACHE / "parsing/pages.jsonl") if row.get("text_source") == "pdf_inspector"]
    queries = read_jsonl(DATASET / "queries.jsonl")
    qids = [str(query["query_id"]) for query in queries]
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(CACHE / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    index = BM25Index(analyzer_name="auto").build({"chunk_id": page.page_id, "doc_id": page.doc_id, "text": page.text} for page in pages)
    all_page_ids = [page.page_id for page in [RawPage(**row) for row in read_jsonl(CACHE / "parsing/pages.jsonl")]]
    positions = np.asarray([all_page_ids.index(page.page_id) for page in pages], dtype=np.int64)
    page_vectors = np.load(CACHE / "embedding_adaptive_openrouter_te3s_fulltext/embeddings/page_embeddings.npy", mmap_mode="r")[positions]
    query_vectors = np.load(CACHE / "embedding_adaptive_openrouter_te3s_fulltext/embeddings/query_embeddings.npy", mmap_mode="r")
    rankings = _build_rankings(queries, pages, index, page_vectors, query_vectors)

    source_by_qid = {qid: qid.split("::", 1)[0] for qid in qids}
    fold_by_qid: dict[str, int] = {}
    for source in sorted(set(source_by_qid.values())):
        source_qids = sorted(qid for qid in qids if source_by_qid[qid] == source)
        for position, qid in enumerate(source_qids):
            fold_by_qid[qid] = position % 5
    parameter_keys = [(alpha, beta) for alpha in ALPHAS for beta in BETAS]
    oof_choice: dict[str, tuple[float, float]] = {}
    fold_rows = []
    for fold in range(5):
        test_qids = [qid for qid in qids if fold_by_qid[qid] == fold]
        train_qids = [qid for qid in qids if fold_by_qid[qid] != fold]
        train_scores = {}
        for key in parameter_keys:
            train_pages = {qid: ([] if qid.startswith("mpdocvqa::") else rankings[qid][key]) for qid in train_qids}
            train_scores[key] = _metrics(train_qids, train_pages, qrels)["recall_at_returned_k"]
        chosen = max(parameter_keys, key=lambda key: train_scores[key])
        for qid in test_qids:
            oof_choice[qid] = chosen
        test_pages = {qid: ([] if qid.startswith("mpdocvqa::") else rankings[qid][chosen]) for qid in test_qids}
        fold_rows.append({"fold": fold + 1, "train_queries": len(train_qids), "test_queries": len(test_qids), "chosen": {"alpha": chosen[0], "beta": chosen[1]}, "train_score": train_scores[chosen], "test_metrics": _metrics(test_qids, test_pages, qrels)})

    oof_pages = {qid: ([] if qid.startswith("mpdocvqa::") else rankings[qid][oof_choice[qid]]) for qid in qids}
    frozen_pages = {qid: ([] if qid.startswith("mpdocvqa::") else rankings[qid][(0.10, 0.30)]) for qid in qids}
    frozen_a20_pages = {qid: ([] if qid.startswith("mpdocvqa::") else rankings[qid][(0.20, 0.30)]) for qid in qids}
    bm25_fixed_pages = {}
    for query in queries:
        qid = str(query["query_id"])
        # Keep the trivial control global and fixed K=12.  The adaptive
        # candidate intentionally drops MPDocVQA because no inspector pages
        # exist there; this distinction is part of the budget experiment.
        bm25_fixed_pages[qid] = [index.chunk_ids[pos] for pos, _score in index.search(str(query["query"]), 12)]
    report = {"protocol": {"queries": len(qids), "indexed_pages": len(pages), "budget": "K=0 for MPDocVQA with no PDF-inspector pages, K=17 otherwise", "qrels_used_for_test_selection": False}, "fixed_bm25": _metrics(qids, bm25_fixed_pages, qrels), "frozen_a10_candidate": _metrics(qids, frozen_pages, qrels), "frozen_a20_candidate": _metrics(qids, frozen_a20_pages, qrels), "clean_oof_candidate": _metrics(qids, oof_pages, qrels), "folds": fold_rows}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Clean adaptive holdout", "", "| Method | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    for name, value in (("BM25 fixed K=12", report["fixed_bm25"]), ("Frozen alpha=.10 beta=.30", report["frozen_a10_candidate"]), ("Frozen alpha=.20 beta=.30", report["frozen_a20_candidate"]), ("Clean OOF alpha/beta selection", report["clean_oof_candidate"])):
        lines.append(f"| {name} | {value['recall_at_returned_k']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} | {value['mean_k']:.2f} |")
    lines += ["", "| Fold | Train-selected alpha/beta | Test recall at K | Test recall@10 | Test recall@20 |", "|---:|---|---:|---:|---:|"]
    for row in fold_rows:
        value = row["test_metrics"]
        lines.append(f"| {row['fold']} | {row['chosen']['alpha']:.2f}/{row['chosen']['beta']:.2f} | {value['recall_at_returned_k']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
