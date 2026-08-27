"""Compose the three verified retrieval levers on ONE identical pool.

handoff.md priority 2: SEP (+2.41), Voyage rerank-2.5 (+5.08) and ColQwen2
fusion (+3.22) were each measured against a slightly different baseline and
never together. This script stacks them on the `vidore_page` alpha0.7 pool,
every arm scored on the same qrels with a paired permutation test vs the
44.15 baseline.

Everything here is cached -- no API, no GPU:
  - SEP        : research/experiments/physics_sep.propagate (arithmetic)
  - Voyage     : data/.../physics_rerank_voyage_scores.json (302 q, top-20)
  - ColQwen2   : data/work/vidore_physics_colqwen/*.npy (302 x 1674 matrix)

Compose order is retrieve -> rerank -> structural+visual reorder, so Voyage is
applied to the base-pool top-20 (fully cached) and SEP / ColQwen2 reorder the
result. The 3-way where ColQwen2 fusion happens *before* the reranker cannot be
measured from cache (the top-20 fed to Voyage would change) and needs one fresh
Voyage run.

    python research/experiments/physics_stack.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytrec_eval

from research.experiments.physics_sep import build, propagate, minmax
from research.experiments.physics_sep_test import permutation

RESULTS = ROOT / "data/benchmark/vidore_v3/results"
COLQWEN = ROOT / "data/work/vidore_physics_colqwen"
VOYAGE = RESULTS / "physics_rerank_voyage_scores.json"

SEP = dict(lam=0.5, w=2, gamma=0.5, beta=0.75, topm=3)
WEIGHTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def rank_proxy(order: list[str]) -> dict[str, float]:
    return {c: float(len(order) - i) for i, c in enumerate(order)}


def sep_rescore(scores: dict[str, float]) -> dict[str, float]:
    return propagate(scores, SEP["lam"], SEP["w"], SEP["gamma"], SEP["beta"], SEP["topm"])


def voyage_reorder(order: list[str], vscores: dict[str, float]) -> list[str]:
    """Sort the cached-scored prefix by Voyage score, keep the tail as-is.

    Iterate `vscores` (not a filter over `order`) so a duplicate id in the pool
    cannot land in the head twice -- that bug inflated the Voyage arm to 49.50
    against §1b-iii's frozen 49.23.
    """
    head = sorted(vscores, key=lambda c: -vscores[c])
    seen = set(head)
    return head + [c for c in order if c not in seen]


def fuse(text: dict[str, float], vis: dict[str, float], w: float) -> dict[str, float]:
    t, v = minmax(text), minmax(vis)
    out = {u: (1 - w) * s + w * v.get(u, 0.0) for u, s in t.items()}
    for u, s in v.items():
        out.setdefault(u, w * s)
    return out


def main() -> None:
    qrels, pool = build("vidore_page")
    qids = list(pool)
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_10"})

    def rep(run):
        s = ev.evaluate(run)
        n = len(s)
        return (100 * sum(v["ndcg_cut_10"] for v in s.values()) / n,
                100 * sum(v["recall_10"] for v in s.values()) / n,
                {q: 100 * v["ndcg_cut_10"] for q, v in s.items()})

    matrix = np.load(COLQWEN / "physics_colqwen_scores.npy")
    keys = json.loads((COLQWEN / "physics_colqwen_keys.json").read_text())
    cq = json.loads((COLQWEN / "physics_colqwen_qids.json").read_text())
    kidx = {k: i for i, k in enumerate(keys)}
    ridx = {q: i for i, q in enumerate(cq)}
    vis = {q: {c: float(matrix[ridx[q], kidx[c]]) for c in pool[q] if c in kidx}
           for q in qids if q in ridx}

    voyage = json.loads(VOYAGE.read_text())

    base = {q: dict(pool[q]) for q in qids}
    nb, rb, pqb = rep(base)

    rows = []
    ref = {"baseline": pqb}  # per-question dicts to run paired tests against

    def emit(name, run, sweep_w=None, vs="baseline"):
        n, r, pq = rep(run)
        d, p, better, worse, tied = permutation(ref[vs], pq)
        flag = "" if p < 0.05 else "  n.s."
        wtag = f" w={sweep_w}" if sweep_w is not None else ""
        print(f"{name+wtag:40s} {n:7.2f} {r:7.2f} {d:+7.2f} {p:9.4f} vs {vs:8s} {better:3d}/{worse:3d}/{tied:3d}{flag}")
        rows.append(dict(arm=name, w=sweep_w, ndcg10=round(n, 2), recall10=round(r, 2),
                         vs=vs, delta=round(d, 3), p=round(p, 5), better=better, worse=worse, tied=tied))
        return n, pq

    def best_fusion(name, text_scores, vs="baseline"):
        """Report the whole w-band, not the argmax (repo convention, §1b-ii/§18)."""
        band = []
        for w in WEIGHTS:
            run = {q: fuse(text_scores[q], vis[q], w) for q in text_scores if q in vis}
            n, pq = emit(name, run, sweep_w=w, vs=vs)
            band.append((w, n, pq))
        lo, hi = min(b[1] for b in band), max(b[1] for b in band)
        argmax = max(band, key=lambda b: b[1])
        print(f"  -> {name}: band {lo:.2f}-{hi:.2f}, peak w={argmax[0]} ({argmax[1]:.2f})")
        return argmax

    print(f"{'arm':44s} {'NDCG@10':>7s} {'R@10':>7s} {'delta':>7s} {'p':>9s}  b/w/t")
    print(f"{'baseline alpha=0.7':44s} {nb:7.2f} {rb:7.2f} {0.0:+7.2f} {'':>9s}")

    # --- single levers -------------------------------------------------------
    sep_scores = {q: sep_rescore(dict(pool[q])) for q in qids}
    emit("SEP (lam=0.5)", sep_scores)

    best_fusion("ColQwen2 fusion", {q: dict(pool[q]) for q in qids})

    voy_order = {q: voyage_reorder(list(pool[q]), voyage.get(q, {})) for q in qids}
    voy_scores = {q: rank_proxy(voy_order[q]) for q in qids}
    _, voy_pq = emit("Voyage rerank-2.5 top-20", voy_scores)
    ref["voyage"] = voy_pq  # the bar the stack has to clear

    # --- 2-way: does anything beat Voyage alone? --------------------------
    best_fusion("SEP + ColQwen2", sep_scores)
    best_fusion("Voyage + ColQwen2", voy_scores, vs="voyage")

    # rank-proxy input makes SEP-after-rerank not a clean measurement -- SEP's
    # file aggregate is built for a real score decay, not a linear ramp. Kept
    # only to show direction; do not quote the delta.
    voy_then_sep_scores = {q: sep_rescore(voy_scores[q]) for q in qids}
    emit("Voyage + SEP [rank-proxy, not clean]", voy_then_sep_scores, vs="voyage")
    best_fusion("Voyage + SEP + ColQwen2 [not clean]", voy_then_sep_scores, vs="voyage")

    out = RESULTS / "physics_stack.json"
    out.write_text(json.dumps(dict(baseline=round(nb, 2), baseline_recall=round(rb, 2),
                                   sep=SEP, weights=WEIGHTS, rows=rows), indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
