"""Rerank a fixed light candidate pool with the locally cached cross-encoder."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys
from typing import Any

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
OUTPUT = CACHE / "cross_encoder_rerank"
MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _metrics(rows: list[dict[str, Any]], qrels: dict[str, set[str]]) -> dict[str, float]:
    returned, r10, r20, ks = [], [], [], []
    for row in rows:
        gold = qrels.get(row["query_id"], set())
        denom = max(len(gold), 1)
        pages = row["pages"]
        returned.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
        ks.append(len(pages))
    return {"queries": len(rows), "recall_at_returned_k": float(np.mean(returned)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean(ks))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-depth", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()

    pages_all = [RawPage(**row) for row in read_jsonl(CACHE / "parsing/pages.jsonl")]
    pages = [page for page in pages_all if page.text_source == "pdf_inspector"]
    page_by_id = {page.page_id: page for page in pages}
    queries = read_jsonl(DATASET / "queries.jsonl")
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(CACHE / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    bm25 = BM25Index(analyzer_name="auto").build({"chunk_id": p.page_id, "doc_id": p.doc_id, "text": p.text} for p in pages)
    page_ids = [p.page_id for p in pages]
    all_page_ids = [p.page_id for p in pages_all]
    positions = np.asarray([all_page_ids.index(pid) for pid in page_ids], dtype=np.int64)
    page_vectors = np.load(EMBEDDING / "page_embeddings.npy", mmap_mode="r")[positions]
    query_vectors = np.load(EMBEDDING / "query_embeddings.npy", mmap_mode="r")
    page_norms = np.linalg.norm(page_vectors, axis=1)

    from sentence_transformers import CrossEncoder
    model = CrossEncoder(MODEL, device=args.device, max_length=512)
    reranked: dict[str, list[tuple[str, float]]] = {}
    query_index = {str(q["query_id"]): i for i, q in enumerate(queries)}
    for number, query in enumerate(queries, start=1):
        qid = str(query["query_id"])
        text = str(query["query"])
        bm = [bm25.chunk_ids[pos] for pos, _score in bm25.search(text, args.candidate_depth)]
        qv = query_vectors[query_index[qid]]
        dense_scores = (page_vectors @ qv) / (page_norms * np.linalg.norm(qv) + 1e-9)
        dense = [page_ids[pos] for pos in np.argsort(-dense_scores, kind="stable")[:args.candidate_depth]]
        candidate_ids = list(dict.fromkeys(bm + dense))
        pairs = [(text, page_by_id[pid].text[:12000]) for pid in candidate_ids]
        scores = np.asarray(model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False), dtype=float)
        order = np.argsort(-scores, kind="stable")
        reranked[qid] = [(candidate_ids[int(i)], float(scores[int(i)])) for i in order]
        if number % 10 == 0:
            print(f"reranked {number}/{len(queries)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "reranked.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for qid, ranking in reranked.items():
            handle.write(json.dumps({"query_id": qid, "ranking": ranking}, ensure_ascii=False, separators=(",", ":")) + "\n")

    policies: dict[str, dict[str, float]] = {}
    fixed_rows = [{"query_id": str(q["query_id"]), "pages": [pid for pid, _s in reranked[str(q["query_id"])][:12]]} for q in queries]
    policies["cross_encoder_fixed_k12"] = _metrics(fixed_rows, qrels)
    for extra in (8, 9, 10):
        rows = []
        for q in queries:
            qid = str(q["query_id"])
            ranking = reranked[qid]
            decision = adaptive_k([score for _pid, score in ranking], ignore_extreme=0, ignore_extreme_tail=0.1, retrieve_more=extra)
            k = int(decision["k"])
            rows.append({"query_id": qid, "pages": [pid for pid, _s in ranking[:k]], "k": k})
        policies[f"cross_encoder_largest_gap_plus{extra}"] = _metrics(rows, qrels)
    report = {"model": MODEL, "candidate_depth_per_stream": args.candidate_depth, "candidate_pool": "union of BM25 and dense top-depth pages", "qrels_used_for_ranking": False, "policies": policies}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Cached cross-encoder rerank", "", "| Policy | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    for name, metric in policies.items():
        lines.append(f"| {name} | {metric['recall_at_returned_k']:.2%} | {metric['recall@10']:.2%} | {metric['recall@20']:.2%} | {metric['mean_k']:.2f} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
