"""Recall@k curve for the physics ladder's two headline indexes.

The ladder reports NDCG@10 and recall@10 only, which cannot separate a candidate
generation failure from an ordering failure. This walks k out to 200 over the
cached embeddings, so it costs no API calls.

High recall@100 with low NDCG@10 means the gold pages are being retrieved and
mis-ordered -- a reranker recovers that. Low recall@100 means they are never
retrieved at all, and only a better first-stage representation recovers it.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.utils.env import load_dotenv_file

load_dotenv_file(Path(__file__).resolve().parents[2])

from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks
from src.evaluation.retrieval import alpha_fuse, rrf
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.retrieval.sparse import BM25Index

RUN = str(Path(__file__).resolve().parents[2] / "data_vidore_parsed_physics/output/benchmarks/vidore-v3-physics-chandra2/c6e049e4fd9c4b9b")
SUBSET, LANG = "physics", "french"
DEPTH, ALPHA = 200, 0.7
CUTS = (5, 10, 20, 50, 100, 200)
CACHE = Path(__file__).resolve().parents[2] / "data/work/vidore_physics_emb"
OUT = Path(__file__).resolve().parents[2] / "data/benchmark/vidore_v3/results" / "physics_recall_curve.json"

bench = load("vidore_v3", subset=SUBSET, language=LANG)
qrels = bench.qrels()
questions = [q for q in bench.questions() if qrels.get(q.qid)]
gold = {q.qid: {d for d, rel in qrels[q.qid].items() if rel > 0} for q in questions}
vidore_pages = {d.doc_id: (d.text or "") for d in bench.corpus()}

blocks_by_unit = {}
for document in documents(RUN):
    doc = canonical_doc(document.get("document", {}).get("file_name"))
    for page, blocks in page_blocks(document).items():
        blocks_by_unit[unit_id(SUBSET, doc, page)] = blocks

chandra_pages = {unit: "\n".join(b["text"] for b in blocks if b["text"].strip())
                 for unit, blocks in blocks_by_unit.items()}

sizes = [len(g) for g in gold.values()]
print(f"queries={len(questions)}  gold pages/query: mean={np.mean(sizes):.2f} "
      f"median={np.median(sizes):.0f} max={max(sizes)}")

embedder = OpenRouterEmbedder(cache_dir=CACHE, batch_size=64)
query_vectors = np.asarray(embedder.embed([q.query for q in questions]), dtype=np.float32)
query_vectors /= np.clip(np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12, None)

INDEXES = {"vidore_page": vidore_pages, "chandra_page": chandra_pages}
results = []

for name, units in INDEXES.items():
    items = {u: t for u, t in units.items() if t.strip()}
    ids = list(items)
    texts = [items[u] for u in ids]

    bm25 = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": u, "doc_id": u, "text": items[u]} for u in ids])
    matrix = np.asarray(embedder.embed(texts), dtype=np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

    ranked = defaultdict(dict)
    for question, vector in zip(questions, query_vectors):
        lexical = bm25.search(question.query, DEPTH)
        scores = matrix @ vector
        top = np.argpartition(-scores, min(DEPTH, len(scores) - 1))[:DEPTH]
        dense = sorted(((int(i), float(scores[i])) for i in top), key=lambda p: -p[1])
        for arm, hits in (("bm25", lexical), ("dense", dense),
                          ("rrf", rrf([lexical, dense], top_k=DEPTH)),
                          (f"alpha{ALPHA:g}", alpha_fuse(lexical, dense, ALPHA, DEPTH))):
            best = {}
            for position, score in hits:
                unit = ids[position]
                best[unit] = max(best.get(unit, -1e9), float(score))
            ranked[arm][question.qid] = sorted(best, key=best.get, reverse=True)

    for arm, run in ranked.items():
        row = {"index": name, "arm": arm}
        for cut in CUTS:
            recall = np.mean([len(set(run[q.qid][:cut]) & gold[q.qid]) / len(gold[q.qid])
                              for q in questions])
            row[f"recall@{cut}"] = round(100 * float(recall), 2)
        # any-gold hit rate: does the query get at least one gold page at all?
        for cut in (10, 100):
            hit = np.mean([bool(set(run[q.qid][:cut]) & gold[q.qid]) for q in questions])
            row[f"anygold@{cut}"] = round(100 * float(hit), 2)
        results.append(row)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

header = ["index", "arm"] + [f"R@{c}" for c in CUTS] + ["any@10", "any@100"]
print("\n" + " ".join(f"{h:>12s}" for h in header))
for r in results:
    cells = [r["index"], r["arm"]] + [f"{r[f'recall@{c}']:.1f}" for c in CUTS] \
            + [f"{r['anygold@10']:.1f}", f"{r['anygold@100']:.1f}"]
    print(" ".join(f"{c:>12s}" for c in cells))
print(f"\nwrote {OUT}")
