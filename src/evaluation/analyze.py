"""Diagnose why retrieval misses, so the next change is chosen not guessed.

Each miss is attributed to one cause, checked in order of how early it caps the
result: the document can never be found (not in the corpus), it shares no query
term (lexical mismatch, which only a dense view fixes), it was found but ranked
below the cut (a reranking problem), or it lost to distractors at aggregation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import argparse
import json

from .chunking import chunk_corpus
from .retrieval import ANALYZERS, BM25Index, aggregate
from .run import ARMS, DATA, load_corpus, load_eval

CUTOFF = 10
DEEP = 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="fixed_512_ol")
    parser.add_argument("--analyzer", default="plain", choices=sorted(ANALYZERS))
    parser.add_argument("--aggregation", default="maxp")
    parser.add_argument("--modalities", default="text,table")
    parser.add_argument("--out", default=str(DATA / "failures.json"))
    args = parser.parse_args()

    corpus = load_corpus(DATA / "corpus.jsonl")
    questions = load_eval(DATA / "questions.jsonl", {m.strip() for m in args.modalities.split(",")})
    corpus_ids = {record["doc_id"] for record in corpus}
    by_id = {record["doc_id"]: record for record in corpus}

    spec = ARMS[args.arm]
    chunks = chunk_corpus(corpus, spec["strategy"], spec["params"])
    analyzer = ANALYZERS[args.analyzer]
    index = BM25Index(analyzer=analyzer).build(chunks)

    doc_tokens = {
        doc_id: set(analyzer(" ".join(block["text"] for block in record["blocks"][:400])))
        for doc_id, record in by_id.items()
    }

    rows: list[dict[str, Any]] = []
    for question in questions:
        gold = question["gold_doc_ids"]
        hits = index.search(question["question"], top_k=DEEP * 5)
        ranked = [doc for doc, _ in aggregate(hits, index.doc_ids, args.aggregation)]
        found = set(ranked[:CUTOFF]) & set(gold)
        rows.append(
            {
                "qid": question["qid"],
                "level": question["level"],
                "n_gold": len(gold),
                "hit@10": bool(found),
                "cause": _cause(question, gold, ranked, corpus_ids, doc_tokens, analyzer),
                "gold_ranks": [ranked.index(d) + 1 if d in ranked else None for d in gold],
            }
        )

    misses = [row for row in rows if not row["hit@10"]]
    report = {
        "arm": args.arm,
        "questions": len(rows),
        "hit@10": round(sum(1 for r in rows if r["hit@10"]) / len(rows), 4) if rows else 0.0,
        "misses": len(misses),
        "causes": dict(Counter(row["cause"] for row in misses).most_common()),
        "causes_by_level": {
            level: dict(Counter(r["cause"] for r in misses if r["level"] == level).most_common())
            for level in sorted({r["level"] for r in misses})
        },
        "detail": misses,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"arm={args.arm}  questions={report['questions']}  hit@10={report['hit@10']:.3f}  misses={report['misses']}")
    print("\nfailure causes:")
    for cause, count in report["causes"].items():
        print(f"  {count:3d}  {cause}")
    print(f"\nwrote {args.out}")


def _cause(
    question: dict[str, Any],
    gold: list[str],
    ranked: list[str],
    corpus_ids: set[str],
    doc_tokens: dict[str, set[str]],
    analyzer: Any,
) -> str:
    if not set(gold) <= corpus_ids:
        return "gold_not_in_corpus"
    terms = set(analyzer(question["question"]))
    overlaps = [len(terms & doc_tokens.get(doc, set())) for doc in gold]
    positions = [ranked.index(doc) + 1 for doc in gold if doc in ranked]
    if not positions:
        return "gold_never_retrieved"
    best = min(positions)
    if best <= CUTOFF:
        return "hit"
    if max(overlaps) == 0:
        return "no_lexical_overlap"
    if best <= DEEP:
        return "ranked_below_cutoff"
    return "ranked_very_deep"


if __name__ == "__main__":
    main()
