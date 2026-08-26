"""Score the ColQwen2 visual arm exported from Colab -- alone, fused, and on
the specific 473-page failure group identified in vidore_why_missed.py.

Reads the (query x page) score matrix produced by ColQwen2_visual_arm_physics.ipynb
(no recompute, no GPU, no API). The text side reuses physics_served_pool.json's
cached alpha0.7 order rather than recomputing embeddings -- no OpenRouter calls.

    python research/experiments/physics_colqwen_eval.py \\
        --export data/work/vidore_physics_colqwen
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytrec_eval

from research.experiments.physics_sep_test import permutation
from src.evaluation.benchmarks import load

RESULTS = ROOT / "data/benchmark/vidore_v3/results"
POOL = RESULTS / "physics_served_pool.json"
minmax = lambda d: ({k: 1.0 for k in d} if max(d.values()) - min(d.values()) <= 1e-12 else
                    {k: (v - min(d.values())) / (max(d.values()) - min(d.values())) for k, v in d.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", default="data/work/vidore_physics_colqwen")
    args = parser.parse_args()

    export = ROOT / args.export
    score_matrix = np.load(export / "physics_colqwen_scores.npy")
    keys = json.loads((export / "physics_colqwen_keys.json").read_text())
    qids = json.loads((export / "physics_colqwen_qids.json").read_text())
    print(f"loaded {score_matrix.shape} score matrix, {len(keys)} pages, {len(qids)} queries")

    bench = load("vidore_v3", subset="physics", language="french")
    qrels = bench.qrels()
    served = json.loads(POOL.read_text())["queries"]
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_10"})

    def report(run):
        s = ev.evaluate(run)
        return (100 * sum(v["ndcg_cut_10"] for v in s.values()) / len(s),
                100 * sum(v["recall_10"] for v in s.values()) / len(s),
                {q: 100 * v["ndcg_cut_10"] for q, v in s.items()})

    key_index = {k: i for i, k in enumerate(keys)}

    # visual-only run
    vis_run = {}
    for row, qid in enumerate(qids):
        vis_run[qid] = {k: float(score_matrix[row, key_index[k]]) for k in keys}
    nv, rv, vpq = report(vis_run)

    # text baseline: served pool order as a rank-based proxy score (order only
    # is cached, not the fused numeric score -- same convention as
    # physics_rerank_local.py and physics_rerank_voyage.py).
    text_run = {qid: {c: float(len(ids) - i) for i, c in enumerate(ids)}
                for qid, ids in ((q, served[q]["candidates"]) for q in qids if q in served)}
    nt, rt, tpq = report(text_run)

    print(f"\n{'arm':40s} {'NDCG@10':>8s} {'R@10':>7s}")
    print(f"{'text alpha=0.7 (served pool)':40s} {nt:8.2f} {rt:7.2f}")
    print(f"{'ColQwen2 visual-only':40s} {nv:8.2f} {rv:7.2f}")

    # complementarity
    gold_of = lambda qid: {u for u, v in qrels[qid].items() if v > 0}
    only_v = only_t = 0
    common = [q for q in qids if q in served]
    for qid in common:
        g = gold_of(qid)
        if not g:
            continue
        t10 = set(list(text_run[qid])[:10])
        v10 = set(sorted(keys, key=lambda k: -vis_run[qid][k])[:10])
        only_v += len(g & (v10 - t10)) / len(g)
        only_t += len(g & (t10 - v10)) / len(g)
    print(f"\ngold@10 found ONLY by visual: {100 * only_v / len(common):.2f}%   "
          f"ONLY by text: {100 * only_t / len(common):.2f}%")

    # fusion sweep
    print(f"\n{'text + ColQwen2 fusion':40s} {'NDCG@10':>8s} {'vs text':>8s} {'p':>8s}")
    for wv in (0.1, 0.2, 0.3, 0.5, 0.7):
        out = {}
        for qid in common:
            tv = minmax(text_run[qid])
            vv = minmax({k: float(score_matrix[qids.index(qid), key_index[k]]) for k in keys})
            out[qid] = {u: (1 - wv) * s + wv * vv.get(u, 0.0) for u, s in tv.items()}
            for u in vv:
                if u not in out[qid]:
                    out[qid][u] = wv * vv[u]
        nf, rf, fpq = report(out)
        d, p, *_ = permutation({q: tpq[q] for q in common}, {q: fpq[q] for q in common})
        flag = "" if p < 0.05 else "  n.s."
        print(f"{'  w_visual=' + str(wv):40s} {nf:8.2f} {nf - nt:+8.2f} {p:8.4f}{flag}")

    # the target group: gold pages absent from the served text top-100
    print("\n--- the 473-page target group (fig-heavy pages text missed entirely) ---")
    deep_total, found10, found100 = 0, 0, 0
    for qid in common:
        served_set = set(served[qid]["candidates"][:100])
        gold = gold_of(qid)
        deep = gold - served_set
        if not deep:
            continue
        ranked = sorted(keys, key=lambda k: -score_matrix[qids.index(qid), key_index[k]])
        top10, top100 = set(ranked[:10]), set(ranked[:100])
        deep_total += len(deep)
        found10 += len(deep & top10)
        found100 += len(deep & top100)
    print(f"gold pages missed entirely by text (rank>=100): {deep_total}")
    print(f"recovered by ColQwen2 in visual top-10:  {found10} ({100 * found10 / max(deep_total, 1):.1f}%)")
    print(f"recovered by ColQwen2 in visual top-100: {found100} ({100 * found100 / max(deep_total, 1):.1f}%)")

    out_path = RESULTS / "physics_colqwen_eval.json"
    out_path.write_text(json.dumps({
        "text_ndcg10": round(nt, 2), "visual_ndcg10": round(nv, 2),
        "only_visual_pct": round(100 * only_v / len(common), 2),
        "only_text_pct": round(100 * only_t / len(common), 2),
        "deep_group_total": deep_total, "deep_group_top10": found10, "deep_group_top100": found100,
    }, indent=2), encoding="utf-8")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
