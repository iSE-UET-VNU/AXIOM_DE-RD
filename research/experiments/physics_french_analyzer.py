"""E0a: what the `french` analyzer is worth on BM25, decomposed step by step.

The ledger previously attributed our -2.7 against bm25s to "no French stopwords,
no stemming". That was wrong, and this is the evidence: the largest single term is
elision. ``analyze`` joins on the apostrophe, so ``l'énergie`` is one token and a
query for ``énergie`` cannot match it -- and French elides constantly.

Scored the same way as the ladder (``BM25Index.search`` directly, not the fused
pool), so 37.45 here is the same 37.45 in the ledger.

    python research/experiments/physics_french_analyzer.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytrec_eval

from src.chunking_embedding.lexical import analyze
from src.evaluation.benchmarks import load
from src.retrieval import sparse
from src.retrieval.sparse import FRENCH_STOPWORDS, BM25Index, analyze_french

PUBLISHED_BM25S = 39.8


def elide(text: str) -> list[str]:
    tokens: list[str] = []
    for token in analyze(text):
        tokens.extend(part for part in token.split("'") if part)
    return tokens


def elide_stop(text: str) -> list[str]:
    return [t for t in elide(text) if t not in FRENCH_STOPWORDS]


def main() -> None:
    bench = load("vidore_v3", subset="physics", language="french")
    qrels = bench.qrels()
    questions = [q for q in bench.questions() if qrels.get(q.qid)]
    pages = {d.doc_id: (d.text or "") for d in bench.corpus()}
    ids = [u for u, text in pages.items() if text.strip()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})
    steps = [
        ("plain", analyze),
        ("+ split elisions", elide),
        ("+ French stopwords", elide_stop),
        ("+ Snowball stemming", analyze_french),
    ]

    previous = None
    for label, function in steps:
        # Registered under a throwaway name so the index carries a resolved
        # analyzer and queries are tokenized the same way the corpus was.
        sparse.ANALYZERS[label] = function
        index = BM25Index(analyzer_name=label).build(
            [{"chunk_id": u, "doc_id": u, "text": pages[u]} for u in ids])
        run = {q.qid: {index.doc_ids[i]: float(s) for i, s in index.search(q.query, 100)}
               for q in questions}
        scored = evaluator.evaluate(run)
        value = 100 * sum(v["ndcg_cut_10"] for v in scored.values()) / len(scored)
        delta = "" if previous is None else f"   ({value - previous:+.2f})"
        print(f"{label:22s} NDCG@10 = {value:.2f}{delta}")
        previous = value

    print(f"\npublished BM25S physics French-only = {PUBLISHED_BM25S}")
    print("bm25s over our own inputs = 40.15 (research/experiments/bm25s_check.py)")


if __name__ == "__main__":
    main()
