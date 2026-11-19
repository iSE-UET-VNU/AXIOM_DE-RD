"""Stage 2 - light retrieval: every question -> up to 100 ranked pages inside its top-5 files.

Per question (all scores max-normalised over the corpus):
  bm25   = BM25(enfr(bm25_text))                       bm25_text = SCOPE + TOPIC + ANCHOR (query preparation)
  dense  = cosine(Qwen3-Embedding-8B page vector, vector of the FULL question)
  hybrid = 0.3 * bm25 + 0.7 * dense, with rank-matched language routing (arm D): a page whose language differs from
           the question's borrows a BM25 value by rank (common.rank_matched_hybrid)
  page   = max_norm(0.85 * hybrid + 0.15 * file_prior)  file_prior = best dense page of the page's file
  file   = minmax over files of the mean of the file's 3 best page scores
  final  = page + 0.2 * file
Only pages of the 5 best files are kept, ranked by final score, cut at 100 (OHR files are short: some lists < 100).
"""
from __future__ import annotations

import time

import numpy as np

from . import common as C
from .text import enfr

ALPHA, PARENT, FILE_WEIGHT, TOP_FILES, DEPTH = 0.3, 0.15, 0.2, 5, 100


def score_query(corpus, query: str, bm25_text: str, qvec: np.ndarray, top_files: int | None = TOP_FILES):
    nf, pf = len(corpus.files), corpus.page_file
    bm = C.max_norm(corpus.bm25.scores(enfr(bm25_text)).astype(float))
    dn = C.max_norm(corpus.vectors @ qvec)
    fp = np.zeros(nf); np.maximum.at(fp, pf, dn); fp = C.max_norm(fp)[pf]
    ql = C.lang(query) if C.lang(query) != "unknown" else "en"
    same = (corpus.page_lang == ql) | (corpus.page_lang == "unknown")
    s = C.max_norm((1 - PARENT) * C.rank_matched_hybrid(bm, dn, same, ALPHA) + PARENT * fp)
    top3 = np.array([np.sort(s[pf == f])[::-1][:3].mean() if (pf == f).any() else 0.0 for f in range(nf)])
    fs = C.minmax(top3)
    s = s + FILE_WEIGHT * fs[pf]
    file_order = np.argsort(-fs, kind="stable")
    cand = np.flatnonzero(np.isin(pf, file_order[:top_files])) if top_files else np.arange(len(s))
    order = cand[np.argsort(-s[cand], kind="stable")][:DEPTH]
    return order, s, file_order


def run(corpus, prepared: list[dict], qvecs: dict, qrels: dict, ablations: bool = True, exclude: frozenset = frozenset()):
    rows, per_q, setup = [], {}, 0.0
    t0 = time.perf_counter()
    for r in prepared:
        order, s, forder = score_query(corpus, r["query"], r["bm25_text"], qvecs[r["qid"]])
        ranked = [corpus.page_ids[i] for i in order]
        files = [corpus.files[f] for f in forder[:TOP_FILES]]
        rows.append({"qid": r["qid"], "selected_files": files, "n_candidates": len(ranked),
                     "ranked_pages": ranked, "scores": [round(float(s[i]), 6) for i in order]})
        m = C.page_metrics(ranked, qrels[r["qid"]])
        gold_files = {p.split("#", 1)[0] for p in qrels[r["qid"]]}
        m["file_recall@5"] = len(gold_files & set(files)) / len(gold_files)
        per_q[r["qid"]] = m
    wall = time.perf_counter() - t0
    metrics = {"chosen: BM25 concepts + dense full question + file score + top-5 files": C.by_source(per_q)}
    if exclude:
        metrics[f"chosen, excluding {len(exclude)} questions (inputs/excluded_questions.json)"] = C.by_source(
            {q: v for q, v in per_q.items() if q not in exclude})
    if ablations:   # same formula, no query preparation and/or no file cut (reference rows)
        for name, use_concepts, k in (("ablation: BM25 full question, whole corpus", False, None),
                                      ("ablation: BM25 concepts, whole corpus", True, None),
                                      ("ablation: BM25 full question, top-5 files", False, TOP_FILES)):
            pq = {}
            for r in prepared:
                order, _, _ = score_query(corpus, r["query"], r["bm25_text"] if use_concepts else r["query"], qvecs[r["qid"]], k)
                pq[r["qid"]] = C.page_metrics([corpus.page_ids[i] for i in order], qrels[r["qid"]])
            metrics[name] = C.by_source(pq)
    timing = {"retrieval_wall_s_all_questions": round(wall, 3), "retrieval_ms_per_query": round(1000 * wall / len(prepared), 2),
              "mean_candidates": round(float(np.mean([r["n_candidates"] for r in rows])), 2),
              "lists_shorter_than_100": sum(r["n_candidates"] < DEPTH for r in rows)}
    return rows, metrics, timing
