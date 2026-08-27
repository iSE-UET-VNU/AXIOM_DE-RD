"""Is the local reranker's -6.59 at depth-100 a depth problem or a model problem?

physics_rerank_local.py scores the full top-100 in one pass and checkpoints it,
so this reads that cache and re-slices to shallower depths at zero extra
compute -- no rescoring, just re-truncating the already-scored candidate list
per query (dict insertion order == candidates[:100] order, same convention
physics_rerank_local.py itself relies on).

    python research/experiments/physics_rerank_local_depth_sweep.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytrec_eval

from research.experiments.physics_sep_test import permutation
from src.evaluation.benchmarks import load

RESULTS = ROOT / "data/benchmark/vidore_v3/results"
POOL = RESULTS / "physics_served_pool.json"
SCORES = RESULTS / "physics_rerank_local_scores_d100.json"
DEPTHS = (10, 20, 50, 100)


def ndcg10(run, qrels):
    s = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"}).evaluate(run)
    per_q = {q: 100 * v["ndcg_cut_10"] for q, v in s.items()}
    return sum(per_q.values()) / len(per_q), per_q


def as_run(order):
    return {q: {c: float(len(ids) - i) for i, c in enumerate(ids)} for q, ids in order.items()}


def main() -> None:
    pool = json.loads(POOL.read_text())["queries"]
    scores = json.loads(SCORES.read_text())
    bench = load("vidore_v3", subset="physics", language="french")
    qrels = {q: g for q, g in bench.qrels().items() if q in pool}
    served = {qid: entry["candidates"] for qid, entry in pool.items()}

    baseline, base_pq = ndcg10(as_run(served), qrels)
    print(f"baseline NDCG@10 = {baseline:.2f}")

    report = {"baseline": round(baseline, 2), "depths": []}
    for depth in DEPTHS:
        reranked = {}
        for qid, candidates in served.items():
            sc = scores[qid]
            head_ids = list(sc.keys())[:depth]
            head_set = set(head_ids)
            head_sorted = sorted(head_ids, key=lambda c: -sc[c])
            reranked[qid] = head_sorted + [c for c in candidates if c not in head_set]
        final, final_pq = ndcg10(as_run(reranked), qrels)
        d, p, better, worse, tied = permutation(base_pq, final_pq)
        flag = "" if p < 0.05 else "  n.s."
        print(f"depth={depth:4d}  NDCG@10={final:6.2f}  delta={d:+6.2f}  p={p:.4f}{flag}  "
              f"{better}/{worse}/{tied} better/worse/tied")
        report["depths"].append({"depth": depth, "ndcg10": round(final, 2), "delta": round(d, 3),
                                 "p": round(p, 5), "better": better, "worse": worse, "tied": tied})

    out = RESULTS / "physics_rerank_local_depth_sweep.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
