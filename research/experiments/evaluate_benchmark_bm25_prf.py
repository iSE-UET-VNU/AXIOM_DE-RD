"""Offline pseudo-relevance feedback screen for the 220-query benchmark.

The first BM25 pass supplies a small pseudo-relevant set.  High TF-IDF terms
from those pages are appended to the original query and a second BM25 pass
produces the final ranking.  No qrels, answer text, OCR, API, or embedding is
used by the ranker.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
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
from src.chunking_embedding.lexical import analyze  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DATASET = ROOT / "data/raw/BENCHMARK"
OUTPUT = CACHE / "bm25_prf_screen"
STOPWORDS = set(
    "the a an and or of to in on for with from by is are was were be been being "
    "what when where how why who whom which this that these those do does did "
    "should can could would will may might as at it its into about all any each "
    "per their there while during explain list number page pages describe "
    "using use used role show shown give given based related relation"
    " le la les de des du un une et ou en dans pour avec sur est sont que qui "
    "comment pourquoi quel quelle quelles quels aux au par cette ces ce se"
    .split()
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def _metrics(rows: Sequence[Mapping[str, Any]], qrels: Mapping[str, set[str]]) -> dict[str, float]:
    returned, r10, r20, ks = [], [], [], []
    for row in rows:
        gold = qrels.get(str(row["query_id"]), set())
        pages = [str(page_id) for page_id in row["pages"]]
        denom = max(len(gold), 1)
        returned.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
        ks.append(len(pages))
    return {
        "queries": len(rows),
        "recall_at_returned_k": float(np.mean(returned)),
        "recall@10": float(np.mean(r10)),
        "recall@20": float(np.mean(r20)),
        "mean_k": float(np.mean(ks)),
    }


def _feedback_terms(
    feedback_ids: Sequence[str],
    text_by_id: Mapping[str, str],
    index: BM25Index,
    query_terms: set[str],
    *,
    n_terms: int,
) -> list[str]:
    scores: Counter[str] = Counter()
    total_docs = max(len(index.chunk_ids), 1)
    for page_id in feedback_ids:
        tokens = [token for token in analyze(text_by_id[page_id]) if len(token) >= 3 and token not in STOPWORDS]
        counts = Counter(tokens)
        position = index.chunk_ids.index(page_id)
        for term, tf in counts.items():
            posting = index.postings.get(term, [])
            df = len(posting)
            # Term frequency dampening plus inverse document frequency keeps
            # generic words from dominating the feedback vocabulary.
            idf = np.log1p((total_docs + 0.5) / (df + 0.5))
            scores[term] += (1.0 + np.log1p(tf)) * float(idf)
    for term in query_terms:
        scores.pop(term, None)
    return [term for term, _score in scores.most_common(n_terms)]


def _build_rankings(
    queries: Sequence[Mapping[str, Any]],
    index: BM25Index,
    text_by_id: Mapping[str, str],
    *,
    feedback_docs: int,
    expansion_terms: int,
    expansion_repeats: int,
) -> dict[str, list[tuple[str, float]]]:
    rankings: dict[str, list[tuple[str, float]]] = {}
    for query in queries:
        qid = str(query["query_id"])
        original = str(query["query"])
        initial = index.search(original, max(5, feedback_docs))
        feedback_ids = [index.chunk_ids[position] for position, _score in initial]
        terms = _feedback_terms(feedback_ids, text_by_id, index, set(analyze(original)), n_terms=expansion_terms)
        expanded = original + " " + " ".join(terms * max(1, expansion_repeats))
        hits = index.search(expanded, 100)
        rankings[qid] = [(index.chunk_ids[position], float(score)) for position, score in hits]
    return rankings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    pages = [RawPage(**row) for row in read_jsonl(CACHE / "parsing/pages.jsonl") if row.get("text_source") == "pdf_inspector"]
    queries = read_jsonl(DATASET / "queries.jsonl")
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(CACHE / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    text_by_id = {page.page_id: page.text for page in pages}
    index = BM25Index(analyzer_name="auto").build({"chunk_id": p.page_id, "doc_id": p.doc_id, "text": p.text} for p in pages)
    qids = [str(query["query_id"]) for query in queries]
    specs = [(3, 5, 1), (5, 5, 1), (5, 10, 1), (10, 5, 1), (10, 10, 1), (5, 10, 2)]
    policies: dict[str, dict[str, float]] = {}
    all_rows: list[dict[str, Any]] = []
    for feedback_docs, expansion_terms, repeats in specs:
        name = f"prf_docs{feedback_docs}_terms{expansion_terms}_repeat{repeats}"
        rankings = _build_rankings(queries, index, text_by_id, feedback_docs=feedback_docs, expansion_terms=expansion_terms, expansion_repeats=repeats)
        fixed_rows = [{"query_id": qid, "pages": [pid for pid, _score in rankings[qid][:12]]} for qid in qids]
        policies[f"{name}_fixed_k12"] = _metrics(fixed_rows, qrels)
        all_rows.extend({**row, "policy": f"{name}_fixed_k12"} for row in fixed_rows)
        for extra in (8, 9, 10):
            adaptive_rows = []
            for query in queries:
                qid = str(query["query_id"])
                ranking = rankings[qid]
                decision = adaptive_k([score for _pid, score in ranking], ignore_extreme=0, ignore_extreme_tail=0.1, retrieve_more=extra)
                k = int(decision["k"])
                adaptive_rows.append({"query_id": qid, "pages": [pid for pid, _score in ranking[:k]], "k": k})
            adaptive_name = f"{name}_largest_gap_plus{extra}"
            policies[adaptive_name] = _metrics(adaptive_rows, qrels)
            all_rows.extend({**row, "policy": adaptive_name} for row in adaptive_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "report.json", {"queries": len(queries), "indexed_pages": len(pages), "qrels_used_for_ranking": False, "policies": policies})
    _write_jsonl(args.output_dir / "all_policy_rows.jsonl", all_rows)
    lines = ["# BM25 pseudo-relevance feedback screen", "", "PDF-inspector text only; no embeddings, OCR, API, or qrels in ranking.", "", "| Policy | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    for name, metric in policies.items():
        lines.append(f"| {name} | {metric['recall_at_returned_k']:.2%} | {metric['recall@10']:.2%} | {metric['recall@20']:.2%} | {metric['mean_k']:.2f} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir.resolve()), "policies": policies}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
