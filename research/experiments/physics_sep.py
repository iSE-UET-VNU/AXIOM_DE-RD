"""SEP -- Structural Evidence Propagation over the alpha0.7 pool.

The Phase-0 diagnostic (physics_structure_diagnostic.py) showed the physics
failure is almost never "the gold file is missing" (7%) but "the gold file is
already in the top-10 and the wrong pages within it rank" (69%), and that a
rank-11..100 page adjacent to a returned page is ~4.4x more likely to be gold
than a non-adjacent one. SEP turns that free structural prior into a score.

Mechanism, following DISRetrieval stripped of RST: a relevant internal node
promotes the leaves of its own subtree. Our tree is the corpus's own
file -> page structure, already encoded in the unit id, so it needs no parser
and no LLM-summarised nodes (DISRetrieval RQ3 finds summary retrieval *under*-
performs leaves, and RQ4 finds the summariser barely matters -- so the
expensive half is the half its own evidence says to drop).

    s'(c) = lambda * s(c) + (1 - lambda) * [ beta * A_file(c) + (1 - beta) * N(c) ]

    N(c)      distance-decayed evidence from neighbouring pages of the same file
    A_file(c) top-m mean of the file's pool scores (mean, not sum: sum rewards
              long documents, which is the failure distinct_documents exists to
              suppress)

Reorders only the served pool, so recall@100 is unchanged by construction and
every delta is an ordering delta. Costs no API calls.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
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
CACHE = ROOT / "data/work/vidore_physics_emb"
RESULTS = ROOT / "data/benchmark/vidore_v3/results"
UNIT = re.compile(r"^[^:]+::(?P<file>.+)#page=(?P<page>\d+)$")


def split(unit_id: str):
    m = UNIT.match(unit_id)
    return (m["file"], int(m["page"])) if m else None


def minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    span = high - low
    if span <= 1e-12:
        return {k: 1.0 for k in values}
    return {k: (v - low) / span for k, v in values.items()}


def propagate(scores: dict[str, float], lam: float, w: int, gamma: float,
              beta: float, topm: int, radius: int = 0) -> dict[str, float]:
    """One query's pool scores -> SEP-rescored pool scores.

    `radius` bounds the aggregation node: pages within +/-radius of c in the same
    file. radius=0 means the whole file, i.e. the tree's root. A bounded node is
    what DISRetrieval actually aggregates over, and it matters on corpora whose
    documents are large -- averaging over a 194-page file dilutes the signal
    (lift 8.5x within +/-5 pages vs 2.1x beyond 50).
    """
    base = minmax(scores)
    by_file: dict[str, dict[int, float]] = defaultdict(dict)
    for unit, value in base.items():
        if (p := split(unit)):
            by_file[p[0]][p[1]] = value

    neighbour: dict[str, float] = {}
    aggregate: dict[str, float] = {}
    for unit in base:
        p = split(unit)
        if p is None:
            neighbour[unit] = aggregate[unit] = 0.0
            continue
        pages = by_file[p[0]]
        neighbour[unit] = sum(
            (gamma ** abs(d)) * pages[p[1] + d]
            for d in range(-w, w + 1)
            if d and (p[1] + d) in pages)
        scope = (pages.values() if radius <= 0 else
                 [v for pg, v in pages.items() if abs(pg - p[1]) <= radius])
        top = sorted(scope, reverse=True)[:topm]
        aggregate[unit] = sum(top) / len(top) if top else 0.0

    neighbour, aggregate = minmax(neighbour), minmax(aggregate)
    return {u: lam * base[u] + (1 - lam) * (beta * aggregate[u] + (1 - beta) * neighbour[u])
            for u in base}


def build(index: str, subset: str = "physics", language: str = "french"):
    """Recompute the alpha0.7 pool *with* scores (the frozen pool stores order only)."""
    bench = load("vidore_v3", subset=subset, language=language)
    qrels = bench.qrels()
    questions = [q for q in bench.questions() if qrels.get(q.qid)]

    if index == "vidore_page":
        pages = {d.doc_id: (d.text or "") for d in bench.corpus()}
    else:
        pages = json.loads((RESULTS / "physics_chandra_page_texts.json").read_text())

    cache = CACHE if subset == "physics" else ROOT / f"data/work/vidore_{subset}_emb"
    embedder = OpenRouterEmbedder(cache_dir=cache, batch_size=64)
    qv = np.asarray(embedder.embed([q.query for q in questions]), dtype=np.float32)
    qv /= np.clip(np.linalg.norm(qv, axis=1, keepdims=True), 1e-12, None)

    ids = [u for u, text in pages.items() if text.strip()]
    bm25 = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": u, "doc_id": u, "text": pages[u]} for u in ids])
    matrix = np.asarray(embedder.embed([pages[u] for u in ids]), dtype=np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

    pool: dict[str, dict[str, float]] = {}
    for question, vector in zip(questions, qv):
        lexical = bm25.search(question.query, POOL)
        dense_scores = matrix @ vector
        top = np.argpartition(-dense_scores, min(POOL, len(dense_scores) - 1))[:POOL]
        dense = sorted(((int(i), float(dense_scores[i])) for i in top), key=lambda p: -p[1])
        pool[question.qid] = {ids[p]: s for p, s in alpha_fuse(lexical, dense, ALPHA, POOL)}
    return qrels, pool


def ndcg(evaluator, pool: dict[str, dict[str, float]], qids=None) -> tuple[float, dict]:
    run = {q: v for q, v in pool.items() if qids is None or q in qids}
    scored = evaluator.evaluate(run)
    per_q = {q: 100 * v["ndcg_cut_10"] for q, v in scored.items()}
    return sum(per_q.values()) / len(per_q), per_q


def fold(qid: str) -> int:
    return int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="vidore_page", choices=["vidore_page", "chandra_page"])
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    qrels, pool = build(args.index)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})

    baseline, base_per_q = ndcg(evaluator, pool)
    print(f"\n=== {args.index} ===")
    print(f"baseline alpha0.7 (pool={POOL})   NDCG@10 = {baseline:.2f}")

    GRID = [(lam, w, gamma, beta, topm)
            for lam in (0.4, 0.5, 0.6, 0.7, 0.8)
            for w in (1, 2, 3)
            for gamma in (0.5,)
            for beta in (0.0, 0.25, 0.5, 0.75, 1.0)
            for topm in (1, 2, 3, 5)]

    folds = {q: fold(q) for q in pool}
    half = {0: {q for q, f in folds.items() if f == 0}, 1: {q for q, f in folds.items() if f == 1}}

    rows = []
    for lam, w, gamma, beta, topm in GRID:
        rescored = {q: propagate(s, lam, w, gamma, beta, topm) for q, s in pool.items()}
        overall, per_q = ndcg(evaluator, rescored)
        rows.append({"lambda": lam, "w": w, "gamma": gamma, "beta": beta, "topm": topm,
                     "ndcg@10": round(overall, 2), "delta": round(overall - baseline, 2),
                     "fold0": round(ndcg(evaluator, rescored, half[0])[0], 2),
                     "fold1": round(ndcg(evaluator, rescored, half[1])[0], 2),
                     "per_question": {q: round(v, 4) for q, v in per_q.items()}})

    rows.sort(key=lambda r: -r["ndcg@10"])
    print(f"\n{'lam':>4s} {'w':>2s} {'gam':>4s} {'beta':>5s}  {'ndcg':>6s} {'delta':>6s}"
          f"  {'fold0':>6s} {'fold1':>6s}")
    for r in rows[:12]:
        print(f"{r['lambda']:4.1f} {r['w']:2d} {r['gamma']:4.1f} {r['beta']:5.2f}  "
              f"{r['ndcg@10']:6.2f} {r['delta']:+6.2f}  {r['fold0']:6.2f} {r['fold1']:6.2f}")

    # Honest lambda: tune on one fold, report the other. Both directions.
    b0 = ndcg(evaluator, pool, half[0])[0]
    b1 = ndcg(evaluator, pool, half[1])[0]
    best_on_0 = max(rows, key=lambda r: r["fold0"])
    best_on_1 = max(rows, key=lambda r: r["fold1"])
    print(f"\nheld-out (tune on fold0 -> score fold1): "
          f"{best_on_0['fold1']:.2f} vs baseline {b1:.2f}  ({best_on_0['fold1'] - b1:+.2f})"
          f"   [lam={best_on_0['lambda']} w={best_on_0['w']} gamma={best_on_0['gamma']} beta={best_on_0['beta']}]")
    print(f"held-out (tune on fold1 -> score fold0): "
          f"{best_on_1['fold0']:.2f} vs baseline {b0:.2f}  ({best_on_1['fold0'] - b0:+.2f})"
          f"   [lam={best_on_1['lambda']} w={best_on_1['w']} gamma={best_on_1['gamma']} beta={best_on_1['beta']}]")
    print(f"in-sample best {rows[0]['ndcg@10']:.2f} -- selection cost is the gap to the held-out numbers")

    out = Path(args.out) if args.out else RESULTS / f"physics_sep_{args.index}.json"
    out.write_text(json.dumps({
        "index": args.index, "pool": POOL, "alpha": ALPHA,
        "baseline": round(baseline, 2),
        "baseline_per_question": {q: round(v, 4) for q, v in base_per_q.items()},
        "grid": rows}, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
