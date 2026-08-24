"""ViDoRe physics, French only: chunk, embed and retrieve over the chandra2 parse.

Gold is page-level, so chunked arms score by their best chunk per page. Every
index is built from the same 1,674 pages and scored on the same French queries.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytrec_eval

from src.utils.env import load_dotenv_file

load_dotenv_file(Path(__file__).resolve().parents[2])

from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.chunking import chunk_corpus
from src.evaluation.pipeline_pages import BOILERPLATE, canonical_doc, documents, page_blocks
from src.evaluation.retrieval import alpha_fuse, rrf
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.retrieval.sparse import BM25Index

RUN = (str(Path(__file__).resolve().parents[2] / "data_vidore_parsed_physics/output/benchmarks/vidore-v3-physics-chandra2/c6e049e4fd9c4b9b"))
SUBSET, LANG = "physics", "french"
PUBLISHED_BM25S = 39.8
DEPTH, K, ALPHA = 100, 10, 0.7
CACHE = Path(__file__).resolve().parents[2] / "data/work/vidore_physics_emb"
OUT = Path(__file__).resolve().parents[2] / "data/benchmark/vidore_v3/results" / "physics_retrieval_ladder.json"
PROSE = {"paragraph", "heading", "table", "caption"}

bench = load("vidore_v3", subset=SUBSET, language=LANG)
qrels = bench.qrels()
questions = [q for q in bench.questions() if qrels.get(q.qid)]
vidore_pages = {d.doc_id: (d.text or "") for d in bench.corpus()}

blocks_by_unit = {}
for document in documents(RUN):
    doc = canonical_doc(document.get("document", {}).get("file_name"))
    for page, blocks in page_blocks(document).items():
        blocks_by_unit[unit_id(SUBSET, doc, page)] = blocks

print(f"queries (french, with gold): {len(questions)}")
print(f"pages  vidore={len(vidore_pages)}  chandra2={len(blocks_by_unit)}  "
      f"identical ids={set(vidore_pages) == set(blocks_by_unit)}")


def joined(keep=None):
    return {unit: "\n".join(b["text"] for b in blocks
                            if b["text"].strip() and (keep is None or b["kind"] in keep))
            for unit, blocks in blocks_by_unit.items()}


def as_documents(units):
    return [{"doc_id": unit, "title": unit.split("::")[1].split("#")[0].replace("_", " "),
             "text": text, "blocks": []} for unit, text in units.items()]


def chunked(units, strategy, params, source_blocks=None):
    docs = as_documents(units)
    if source_blocks is not None:
        for record in docs:
            record["blocks"] = [{"text": b["text"], "kind": b["kind"]}
                                for b in source_blocks.get(record["doc_id"], [])
                                if b["kind"] != BOILERPLATE]
    return {c.chunk_id: (c.doc_id, c.index_text)
            for c in chunk_corpus(docs, strategy, params, prefix=False)}


def as_chunks(units):
    return {unit: (unit, text) for unit, text in units.items()}


INDEXES = {
    "vidore_page":        lambda: as_chunks(vidore_pages),
    "vidore_fixed512":    lambda: chunked(vidore_pages, "fixed_overlap", {"n_words": 512, "overlap": 128}),
    "chandra_page":       lambda: as_chunks(joined()),
    "chandra_no_boiler":  lambda: as_chunks(joined(keep=PROSE | {"figure"})),
    "chandra_no_figure":  lambda: as_chunks(joined(keep=PROSE | {BOILERPLATE})),
    "chandra_prose":      lambda: as_chunks(joined(keep=PROSE)),
    "chandra_fixed512":   lambda: chunked(joined(), "fixed_overlap", {"n_words": 512, "overlap": 128}),
    "chandra_blocks":     lambda: chunked(joined(), "blocks", {"target": 1200, "overlap": 200},
                                          source_blocks=blocks_by_unit),
}

embedder = OpenRouterEmbedder(cache_dir=CACHE, batch_size=64)
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut_{K}"})
query_vectors = np.asarray(embedder.embed([q.query for q in questions]), dtype=np.float32)
query_vectors /= np.clip(np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12, None)

results = []
for name, builder in INDEXES.items():
    items = {cid: pair for cid, pair in builder().items() if pair[1].strip()}
    if not items:
        raise SystemExit(f"{name}: empty index")
    ids = list(items)
    doc_of = [items[c][0] for c in ids]
    texts = [items[c][1] for c in ids]

    bm25 = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": c, "doc_id": items[c][0], "text": items[c][1]} for c in ids])
    matrix = np.asarray(embedder.embed(texts), dtype=np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

    runs = defaultdict(dict)
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
                unit = doc_of[position]
                best[unit] = max(best.get(unit, -1e9), float(score))
            runs[arm][question.qid] = best

    for arm, run in runs.items():
        scored = evaluator.evaluate(run)
        ndcg = 100 * sum(v[f"ndcg_cut_{K}"] for v in scored.values()) / len(scored)
        recall = np.mean([len(set(sorted(run[q.qid], key=run[q.qid].get, reverse=True)[:K])
                              & set(qrels[q.qid])) / len(qrels[q.qid]) for q in questions])
        results.append({"index": name, "arm": arm, "units": len({d for d in doc_of}),
                        "items": len(ids), "chars": sum(len(t) for t in texts),
                        f"ndcg@{K}": round(ndcg, 2), f"recall@{K}": round(100 * recall, 2),
                        "per_question": {q: v[f"ndcg_cut_{K}"] for q, v in scored.items()}})
    print(f"  {name:18s} items={len(ids):6d} " +
          "  ".join(f"{r['arm']}={r[f'ndcg@{K}']:.1f}" for r in results[-4:]))

OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\n{'index':20s} {'items':>7s} {'chars':>10s} " +
      " ".join(f"{a:>9s}" for a in ["bm25", "dense", "rrf", f"alpha{ALPHA:g}"]))
for name in INDEXES:
    rows = {r["arm"]: r for r in results if r["index"] == name}
    first = next(iter(rows.values()))
    print(f"{name:20s} {first['items']:7d} {first['chars']:10d} " +
          " ".join(f"{rows[a][f'ndcg@{K}']:9.1f}" for a in ["bm25", "dense", "rrf", f"alpha{ALPHA:g}"]))
print(f"\npublished BM25S physics French-only = {PUBLISHED_BM25S}")
print("caveat: plain BM25, not the paper's BM25S; French only; 8 public subsets exist, this is 1")
