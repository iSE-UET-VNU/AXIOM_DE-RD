"""Serve the chandra2 α=0.7 top-100 pool, so the rerank arm can run on *our*
pipeline rather than on ViDoRe's supplied page text.

`physics_served_pool.json` holds `vidore_page` candidates -- ViDoRe's own text
extraction, which involves none of our parsing. Comparing that against the paper's
retriever+reranker pipelines measures their text against their text. The arm that
represents our work is chandra2.

Writes the same shape as `physics_rerank_ceiling.py`, so `physics_rerank_voyage.py
--pool` consumes it unchanged. Costs no API calls; embeddings come from the
ladder's disk cache.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.utils.env import load_dotenv_file

load_dotenv_file(ROOT)

from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks
from src.evaluation.retrieval import alpha_fuse
from src.retrieval.sparse import BM25Index

RUN = str(ROOT / "data_vidore_parsed_physics/output/benchmarks/vidore-v3-physics-chandra2/c6e049e4fd9c4b9b")
SUBSET, LANG = "physics", "french"
POOL, ALPHA = 100, 0.7
CACHE = ROOT / "data/work/vidore_physics_emb"
OUT = ROOT / "data/benchmark/vidore_v3/results/physics_served_pool_chandra.json"
TEXTS = ROOT / "data/benchmark/vidore_v3/results/physics_chandra_page_texts.json"


def main() -> None:
    bench = load("vidore_v3", subset=SUBSET, language=LANG)
    qrels = bench.qrels()
    questions = [q for q in bench.questions() if qrels.get(q.qid)]

    blocks_by_unit = {}
    for document in documents(RUN):
        doc = canonical_doc(document.get("document", {}).get("file_name"))
        for page, blocks in page_blocks(document).items():
            blocks_by_unit[unit_id(SUBSET, doc, page)] = blocks
    pages = {unit: "\n".join(b["text"] for b in blocks if b["text"].strip())
             for unit, blocks in blocks_by_unit.items()}
    ids = [u for u, text in pages.items() if text.strip()]
    print(f"chandra2 pages indexed = {len(ids)}   queries = {len(questions)}")

    embedder = OpenRouterEmbedder(cache_dir=CACHE, batch_size=64)
    queries = np.asarray(embedder.embed([q.query for q in questions]), dtype=np.float32)
    queries /= np.clip(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12, None)
    matrix = np.asarray(embedder.embed([pages[u] for u in ids]), dtype=np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

    bm25 = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": u, "doc_id": u, "text": pages[u]} for u in ids])

    served = {}
    for question, vector in zip(questions, queries):
        lexical = bm25.search(question.query, POOL)
        scores = matrix @ vector
        top = np.argpartition(-scores, min(POOL, len(scores) - 1))[:POOL]
        dense = sorted(((int(i), float(scores[i])) for i in top), key=lambda p: -p[1])
        served[question.qid] = [ids[p] for p, _ in alpha_fuse(lexical, dense, ALPHA, POOL)]

    OUT.write_text(json.dumps(
        {"index": "chandra_page", "arm": f"alpha{ALPHA:g}", "depth": POOL,
         "queries": {q.qid: {"query": q.query, "candidates": served[q.qid]} for q in questions}},
        indent=2), encoding="utf-8")
    # The rerank pass needs the chandra2 text, which is not in bench.corpus().
    TEXTS.write_text(json.dumps(pages), encoding="utf-8")
    print(f"-> {OUT}\n-> {TEXTS}")


if __name__ == "__main__":
    main()
