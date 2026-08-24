"""Paired permutation test for the rerank arm against the served alpha0.7 pool.

Same test and seed as ``physics_significance.py``. Both arms are scored on the
identical candidate pool, so the pairing is exact and the only difference between
them is the ordering of the top-``depth`` prefix.

Reads the rerank checkpoint, so it runs before the arm finishes and reports on
whatever is scored so far.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytrec_eval

from src.evaluation.benchmarks import load

RESULTS = ROOT / "data/benchmark/vidore_v3/results"
RESAMPLES = 10000


def main() -> None:
    pool = json.loads((RESULTS / "physics_served_pool.json").read_text())["queries"]
    scores = json.loads((RESULTS / "physics_rerank_voyage_scores.json").read_text())

    bench = load("vidore_v3", subset="physics", language="french")
    qrels = {q: g for q, g in bench.qrels().items() if q in scores}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})

    def per_query(order: dict[str, list[str]]) -> dict[str, float]:
        run = {q: {c: float(len(ids) - i) for i, c in enumerate(ids)} for q, ids in order.items()}
        return {q: v["ndcg_cut_10"] for q, v in evaluator.evaluate(run).items()}

    served = {q: pool[q]["candidates"] for q in scores}
    reranked = {
        q: sorted(scores[q], key=lambda c: -scores[q][c])
           + [c for c in pool[q]["candidates"] if c not in scores[q]]
        for q in scores
    }

    base, arm = per_query(served), per_query(reranked)
    qids = sorted(base)
    diffs = [arm[q] - base[q] for q in qids]
    observed = sum(diffs) / len(diffs)

    rng = random.Random(0)
    extreme = sum(
        abs(sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)) >= abs(observed)
        for _ in range(RESAMPLES)
    )
    p = (extreme + 1) / (RESAMPLES + 1)
    better = sum(1 for d in diffs if d > 1e-9)
    worse = sum(1 for d in diffs if d < -1e-9)

    print(f"n={len(qids)} queries scored")
    print(f"served alpha0.7   NDCG@10 = {100 * sum(base.values()) / len(qids):.2f}")
    print(f"+ voyage rerank   NDCG@10 = {100 * sum(arm.values()) / len(qids):.2f}")
    print(f"delta = {100 * observed:+.2f}   p = {p:.4f}   "
          f"better/worse/tied = {better}/{worse}/{len(qids) - better - worse}")

    # A third of queries get worse. If the losers are the ones whose passages were
    # clipped at MAX_DOC_CHARS, that is a budget artefact rather than the model.
    from src.evaluation.voyage_rerank import MAX_DOC_CHARS

    pages = {d.doc_id: (d.text or "") for d in bench.corpus()}
    clipped = {
        q: sum(len(pages.get(c, "")) > MAX_DOC_CHARS for c in list(scores[q])) / len(scores[q])
        for q in qids
    }
    for label, group in (("better", [q for q in qids if arm[q] - base[q] > 1e-9]),
                         ("worse", [q for q in qids if arm[q] - base[q] < -1e-9])):
        if group:
            print(f"  {label:6s} n={len(group):3d}  clipped passages/query = "
                  f"{100 * sum(clipped[q] for q in group) / len(group):.1f}%")


if __name__ == "__main__":
    main()
