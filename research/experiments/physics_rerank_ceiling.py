"""Oracle-rerank ceiling: how much NDCG@10 is recoverable by reordering alone?

The candidate pool is fixed at the production DEPTH=100 alpha0.7 fusion, so every
rerank depth reorders a prefix of the *same* served ranking. Reordering is graded:
qrels carry relevance 1 and 2, and NDCG's ideal ranking sorts by grade descending,
so a gold-first-in-arbitrary-order prefix would understate the ceiling.

A high ceiling means the gold pages are already in hand and merely mis-ordered,
which is what a cross-encoder reranker fixes. Costs no API calls -- the embeddings
are read from the ladder's disk cache.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytrec_eval

from src.utils.env import load_dotenv_file

load_dotenv_file(ROOT)

from src.evaluation.benchmarks import load
from src.evaluation.retrieval import alpha_fuse
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.retrieval.sparse import BM25Index

POOL, ALPHA = 100, 0.7
DEPTHS = (10, 20, 50, 100)
CACHE = ROOT / "data/work/vidore_physics_emb"

bench = load("vidore_v3", subset="physics", language="french")
qrels = bench.qrels()
questions = [q for q in bench.questions() if qrels.get(q.qid)]
pages = {d.doc_id: (d.text or "") for d in bench.corpus()}

embedder = OpenRouterEmbedder(cache_dir=CACHE, batch_size=64)
query_vectors = np.asarray(embedder.embed([q.query for q in questions]), dtype=np.float32)
query_vectors /= np.clip(np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12, None)

ids = [u for u, text in pages.items() if text.strip()]
bm25 = BM25Index(analyzer_name="plain").build(
    [{"chunk_id": u, "doc_id": u, "text": pages[u]} for u in ids])
matrix = np.asarray(embedder.embed([pages[u] for u in ids]), dtype=np.float32)
matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

served: dict[str, list[str]] = {}
for question, vector in zip(questions, query_vectors):
    lexical = bm25.search(question.query, POOL)
    scores = matrix @ vector
    top = np.argpartition(-scores, min(POOL, len(scores) - 1))[:POOL]
    dense = sorted(((int(i), float(scores[i])) for i in top), key=lambda p: -p[1])
    served[question.qid] = [ids[p] for p, _ in alpha_fuse(lexical, dense, ALPHA, POOL)]

POOL_OUT = ROOT / "data/benchmark/vidore_v3/results/physics_served_pool.json"
POOL_OUT.parent.mkdir(parents=True, exist_ok=True)
POOL_OUT.write_text(json.dumps(
    {"index": "vidore_page", "arm": f"alpha{ALPHA:g}", "depth": POOL,
     "queries": {q.qid: {"query": q.query, "candidates": served[q.qid]} for q in questions}},
    indent=2), encoding="utf-8")
print(f"served pool -> {POOL_OUT}")

evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})


def score(run: dict[str, dict[str, float]]) -> float:
    scored = evaluator.evaluate(run)
    return 100 * sum(v["ndcg_cut_10"] for v in scored.values()) / len(scored)


baseline = score({q.qid: {c: float(len(served[q.qid]) - i)
                          for i, c in enumerate(served[q.qid])} for q in questions})
print(f"served alpha0.7 (pool={POOL})                  NDCG@10 = {baseline:.2f}")

for depth in DEPTHS:
    run = {}
    for question in questions:
        candidates = served[question.qid]
        head, tail = candidates[:depth], candidates[depth:]
        grade = qrels[question.qid]
        # ideal ordering within the reranked prefix: by relevance grade, descending
        head = sorted(head, key=lambda c: -grade.get(c, 0))
        order = head + tail
        run[question.qid] = {c: float(len(order) - i) for i, c in enumerate(order)}
    print(f"  oracle reorder of served top-{depth:<3d}          NDCG@10 = {score(run):.2f}"
          f"   (+{score(run) - baseline:.2f})")
