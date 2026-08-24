"""Independent check: the paper's own BM25S over our exact corpus, queries and qrels.

If BM25S lands near the published 39.8 on the same inputs, the loader and the
scoring path are right and our 37.5 is the scorer. If it lands at 37.5 too, the
gap is in the data and every number in the ladder is suspect.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bm25s
import Stemmer
import numpy as np
import pytrec_eval

from src.evaluation.benchmarks import load
from src.retrieval.sparse import BM25Index

SUBSET, LANG, K = "physics", "french", 10
PUBLISHED = 39.8

bench = load("vidore_v3", subset=SUBSET, language=LANG)
qrels = bench.qrels()
questions = [q for q in bench.questions() if qrels.get(q.qid)]
pages = {d.doc_id: (d.text or "") for d in bench.corpus()}
units = [u for u, t in pages.items() if t.strip()]
texts = [pages[u] for u in units]

print(f"pages {len(pages)} -> indexed {len(units)}   queries {len(questions)}")
print(f"gold/query {sum(len(v) for v in qrels.values())/len(qrels):.2f}   "
      f"graded values {sorted({g for v in qrels.values() for g in v.values()})}")

evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut_{K}"})


def score(run, label):
    result = evaluator.evaluate(run)
    value = 100 * sum(v[f"ndcg_cut_{K}"] for v in result.values()) / len(result)
    print(f"{label:34s} NDCG@{K} = {value:.2f}   (n={len(result)})")
    return value


# -- ours -----------------------------------------------------------------
index = BM25Index(analyzer_name="plain").build(
    [{"chunk_id": u, "doc_id": u, "text": t} for u, t in zip(units, texts)])
ours = {q.qid: {index.doc_ids[i]: float(s) for i, s in index.search(q.query, 100)}
        for q in questions}
a = score(ours, "ours: BM25Index analyzer=plain")

# -- the paper's -----------------------------------------------------------
for label, stem, stop in [("bm25s, no stemmer, no stopwords", None, None),
                          ("bm25s, french stopwords", None, "fr"),
                          ("bm25s, french stemmer + stopwords", Stemmer.Stemmer("french"), "fr")]:
    corpus_tokens = bm25s.tokenize(texts, stopwords=stop, stemmer=stem, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    run = {}
    for q in questions:
        query_tokens = bm25s.tokenize([q.query], stopwords=stop, stemmer=stem, show_progress=False)
        docs, scores = retriever.retrieve(query_tokens, k=100, show_progress=False)
        run[q.qid] = {units[int(d)]: float(s) for d, s in zip(docs[0], scores[0])}
    score(run, label)

print(f"\npublished BM25S physics French-only = {PUBLISHED}")
