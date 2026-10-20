"""Evaluate file aggregation and score-gap adaptive K on the full text cache.

Unlike the PDF-inspector-only screen, this index includes both PDF-inspector
and successful Tesseract text pages.  Every source therefore has indexed text,
so no source-specific zero-budget shortcut is used.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.raw_hierarchical import RawPage, read_jsonl  # noqa: E402
from research.experiments.run_benchmark_adaptive_k import adaptive_k  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DATASET = ROOT / "data/raw/BENCHMARK"
EMBEDDING = CACHE / "embedding_adaptive_openrouter_te3s_fulltext/embeddings"
OUTPUT = CACHE / "fulltext_adaptive"
BETAS = (0.10, 0.20, 0.30, 0.50)


def _metrics(rows: Sequence[Mapping[str, Any]], qrels: Mapping[str, set[str]]) -> dict[str, float]:
    values, r10, r20, ks = [], [], [], []
    for row in rows:
        gold = qrels.get(str(row["query_id"]), set())
        pages = list(row["pages"])
        denom = max(len(gold), 1)
        values.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
        ks.append(len(pages))
    return {"recall_at_returned_k": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean(ks))}


def _load() -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, dict[float, list[tuple[str, float]]]]]:
    pages_all = [RawPage(**row) for row in read_jsonl(CACHE / "parsing/pages.jsonl")]
    pages = [page for page in pages_all if page.text_source in {"pdf_inspector", "tesseract"} and page.text.strip()]
    queries = read_jsonl(DATASET / "queries.jsonl")
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(CACHE / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    index = BM25Index(analyzer_name="auto").build({"chunk_id": p.page_id, "doc_id": p.doc_id, "text": p.text} for p in pages)
    page_ids = [page.page_id for page in pages]
    all_page_ids = [page.page_id for page in pages_all]
    positions = {page_id: position for position, page_id in enumerate(all_page_ids)}
    selected_positions = np.asarray([positions[page_id] for page_id in page_ids], dtype=np.int64)
    page_vectors = np.load(EMBEDDING / "page_embeddings.npy", mmap_mode="r")[selected_positions]
    query_vectors = np.load(EMBEDDING / "query_embeddings.npy", mmap_mode="r")
    page_norms = np.linalg.norm(page_vectors, axis=1)
    qindex = {str(query["query_id"]): i for i, query in enumerate(queries)}
    rankings: dict[str, dict[float, list[tuple[str, float]]]] = {}
    for query in queries:
        qid = str(query["query_id"])
        bm = {index.chunk_ids[position]: float(score) for position, score in index.search(str(query["query"]), 100)}
        bm_max = max(bm.values(), default=1.0)
        qv = query_vectors[qindex[qid]]
        dense_scores = (page_vectors @ qv) / (page_norms * np.linalg.norm(qv) + 1e-9)
        dense = {page_id: float(score) for page_id, score in zip(page_ids, dense_scores)}
        dense_max = max(dense.values(), default=1.0)
        hybrid = {page_id: 0.20 * bm.get(page_id, 0.0) / max(bm_max, 1e-9) + 0.80 * score / max(dense_max, 1e-9) for page_id, score in dense.items()}
        candidate = sorted(hybrid.items(), key=lambda item: (-item[1], item[0]))[:100]
        by_file: dict[str, list[float]] = defaultdict(list)
        for page_id, score in candidate:
            by_file[page_id.split("#page=", 1)[0]].append(score)
        file_scores = {file_id: sum(sorted(scores, reverse=True)[:2]) for file_id, scores in by_file.items()}
        file_max = max(file_scores.values(), default=1.0)
        page_max = max((score for _page_id, score in candidate), default=1.0)
        rankings[qid] = {}
        for beta in BETAS:
            scored = {page_id: (1.0 - beta) * score / max(page_max, 1e-9) + beta * file_scores.get(page_id.split("#page=", 1)[0], 0.0) / max(file_max, 1e-9) for page_id, score in candidate}
            rankings[qid][beta] = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    return queries, qrels, rankings


def main() -> int:
    queries, qrels, rankings = _load()
    qids = [str(query["query_id"]) for query in queries]
    policies: dict[str, dict[str, float]] = {}
    for beta in BETAS:
        fixed = [{"query_id": qid, "pages": [page_id for page_id, _score in rankings[qid][beta][:12]]} for qid in qids]
        policies[f"fixed_k12_beta{beta:.2f}"] = _metrics(fixed, qrels)
        for extra in (8, 9, 10, 11, 12, 13, 14):
            adaptive_rows = []
            for qid in qids:
                ranking = rankings[qid][beta]
                decision = adaptive_k([score for _page_id, score in ranking], ignore_extreme=0, ignore_extreme_tail=0.1, retrieve_more=extra)
                k = int(decision["k"])
                adaptive_rows.append({"query_id": qid, "pages": [page_id for page_id, _score in ranking[:k]], "k": k})
            policies[f"gap_plus{extra}_beta{beta:.2f}"] = _metrics(adaptive_rows, qrels)
    output = {"pages_indexed": len([page for page in [RawPage(**row) for row in read_jsonl(CACHE / "parsing/pages.jsonl")] if page.text_source in {"pdf_inspector", "tesseract"} and page.text.strip()]), "queries": len(qids), "protocol": "global full-text page index; PDF-inspector + successful Tesseract; no source-specific budget shortcut", "qrels_used_for_ranking": False, "policies": policies}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Full-text adaptive retrieval", "", "| Policy | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    for name, metric in policies.items():
        lines.append(f"| {name} | {metric['recall_at_returned_k']:.2%} | {metric['recall@10']:.2%} | {metric['recall@20']:.2%} | {metric['mean_k']:.2f} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
