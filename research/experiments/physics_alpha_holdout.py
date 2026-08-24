"""E0b: is alpha=0.7 real, or was it fitted to the 302 queries we report on?

Selecting alpha on the same queries the number is quoted for inflates it by an
unknown amount. This splits the queries in half by a hash of the qid, tunes on
one half, and reports on the other -- twice, swapping the roles, so both halves
are scored out-of-sample and neither split is itself a choice.

The french analyzer from E0a rides along, because a stronger BM25 leg does not
have to help a fusion that weights it 0.3 and the ladder has no measurement of
that either way.

    python research/experiments/physics_alpha_holdout.py
"""
import json
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytrec_eval

from src.utils.env import load_dotenv_file

load_dotenv_file(ROOT)

from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.evaluation.benchmarks import load
from src.evaluation.retrieval import alpha_fuse
from src.retrieval.sparse import BM25Index

POOL = 100
ALPHAS = tuple(round(0.05 * i, 2) for i in range(21))
ANALYZERS = ("plain", "french")
CACHE = ROOT / "data/work/vidore_physics_emb"
OUT = ROOT / "data/benchmark/vidore_v3/results/physics_alpha_holdout.json"


def fold(qid: str) -> int:
    return sha256(qid.encode()).digest()[0] % 2


def main() -> None:
    bench = load("vidore_v3", subset="physics", language="french")
    qrels = bench.qrels()
    questions = [q for q in bench.questions() if qrels.get(q.qid)]
    pages = {d.doc_id: (d.text or "") for d in bench.corpus()}
    ids = [u for u, text in pages.items() if text.strip()]

    embedder = OpenRouterEmbedder(cache_dir=CACHE, batch_size=64)
    queries = np.asarray(embedder.embed([q.query for q in questions]), dtype=np.float32)
    queries /= np.clip(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12, None)
    matrix = np.asarray(embedder.embed([pages[u] for u in ids]), dtype=np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

    folds = {q.qid: fold(q.qid) for q in questions}
    print(f"queries={len(questions)}  fold sizes: "
          f"{sum(1 for v in folds.values() if v == 0)}/{sum(1 for v in folds.values() if v == 1)}")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})

    def score(run: dict, subset: set[str]) -> float:
        scored = {q: v for q, v in evaluator.evaluate(run).items() if q in subset}
        return 100 * sum(v["ndcg_cut_10"] for v in scored.values()) / len(scored)

    report: dict[str, dict] = {}
    for analyzer in ANALYZERS:
        bm25 = BM25Index(analyzer_name=analyzer).build(
            [{"chunk_id": u, "doc_id": u, "text": pages[u]} for u in ids])

        legs = {}
        for question, vector in zip(questions, queries):
            scores = matrix @ vector
            top = np.argpartition(-scores, min(POOL, len(scores) - 1))[:POOL]
            dense = sorted(((int(i), float(scores[i])) for i in top), key=lambda p: -p[1])
            legs[question.qid] = (bm25.search(question.query, POOL), dense)

        runs = {}
        for alpha in ALPHAS:
            runs[alpha] = {
                qid: {ids[p]: float(POOL - rank)
                      for rank, (p, _) in enumerate(alpha_fuse(lex, dns, alpha, POOL))}
                for qid, (lex, dns) in legs.items()
            }

        full = {q.qid for q in questions}
        insample = max(ALPHAS, key=lambda a: score(runs[a], full))
        print(f"\n[{analyzer}]  best alpha on all 302 = {insample} "
              f"-> {score(runs[insample], full):.2f}  (tuned on test)")

        held = []
        for tune, test in ((0, 1), (1, 0)):
            tune_ids = {q for q, f in folds.items() if f == tune}
            test_ids = {q for q, f in folds.items() if f == test}
            picked = max(ALPHAS, key=lambda a: score(runs[a], tune_ids))
            held.append((picked, score(runs[picked], test_ids), len(test_ids)))
            print(f"  tune on fold {tune} -> alpha={picked:<5} "
                  f"test on fold {test} = {held[-1][1]:.2f}  (n={len(test_ids)})")

        honest = sum(value * n for _, value, n in held) / sum(n for _, _, n in held)
        production = score(runs[0.7], full)
        print(f"  out-of-sample NDCG@10 = {honest:.2f}"
              f"   |  alpha=0.7 on all 302 = {production:.2f}")
        report[analyzer] = {
            "in_sample_alpha": insample,
            "in_sample_ndcg10": score(runs[insample], full),
            "alpha07_ndcg10": production,
            "held_out_ndcg10": honest,
            "folds": [{"alpha": a, "ndcg10": v, "n": n} for a, v, n in held],
            "curve": {str(a): score(runs[a], full) for a in ALPHAS},
        }

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
