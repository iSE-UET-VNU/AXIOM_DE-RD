"""Out-of-sample check for SEP on a subset it was never tuned on.

Every SEP hyperparameter (w, gamma, beta, topm) was fixed on physics from the
effect sizes in physics_structure_diagnostic.py. This script changes nothing
about them and asks whether the result transfers to a different subset, a
different domain and a different language. That is the only test here that the
configuration has not seen in any form.

Also re-derives the structural diagnostic on the new subset, so a transfer (or
a failure to transfer) can be read against whether the same failure shape --
"gold file already in the top-10, wrong pages within it" -- actually holds there.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytrec_eval

from src.utils.env import load_dotenv_file

load_dotenv_file(ROOT)

from research.experiments.physics_sep import build, propagate, ndcg, split
from research.experiments.physics_sep_test import permutation, W, GAMMA, BETA, TOPM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--subset", default="industrial")
parser.add_argument("--language", default="english")
parser.add_argument("--lambdas", default="0.4,0.5,0.6,0.7,0.8,0.9")
args = parser.parse_args()
lambdas = [float(x) for x in args.lambdas.split(",")]

qrels, pool = build("vidore_page", subset=args.subset, language=args.language)
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})
baseline, base_per_q = ndcg(evaluator, pool)

# --- does the physics failure shape hold here at all? ---
shape, near = Counter(), [0, 0, 0, 0]   # adj_n, adj_gold, far_n, far_gold
for qid, scores in pool.items():
    gold = {u for u, g in qrels.get(qid, {}).items() if g > 0}
    if not gold:
        continue
    ranked = [u for u, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
    gold_files = {p[0] for u in gold if (p := split(u))}
    top = [p for u in ranked[:10] if (p := split(u))]
    topfiles, toppos = {f for f, _ in top}, set(top)
    in_top = sum(1 for u in ranked[:10] if u in gold)
    if not (gold_files & topfiles):
        shape["A: gold file absent from top-10"] += 1
    elif in_top < len(gold):
        shape["B: gold file present, wrong pages within it"] += 1
    else:
        shape["C: all gold already in top-10"] += 1
    for u in ranked[10:100]:
        p = split(u)
        if not p or p[0] not in topfiles:
            continue
        if any((p[0], p[1] + d) in toppos for d in (-2, -1, 1, 2)):
            near[0] += 1; near[1] += u in gold
        else:
            near[2] += 1; near[3] += u in gold

n = sum(shape.values())
print(f"\n=== vidore_v3/{args.subset}/{args.language}  ({n} scored queries) ===")
print(f"baseline alpha0.7   NDCG@10 = {baseline:.2f}")
print("failure shape:")
for label, count in shape.most_common():
    print(f"   {label:45s} {count:4d}  {100*count/n:5.1f}%")
if near[0] and near[2]:
    print(f"   within-file adjacency: {100*near[1]/near[0]:.2f}% vs {100*near[3]/near[2]:.2f}% gold")

print(f"\nfixed config w={W} gamma={GAMMA} beta={BETA} topm={TOPM}  (unchanged from physics)")
print(f"{'lambda':>7s} {'ndcg':>7s} {'delta':>7s} {'p':>8s}   better/worse/tied")
rows = []
for lam in lambdas:
    rescored = {q: propagate(s, lam, W, GAMMA, BETA, TOPM) for q, s in pool.items()}
    overall, per_q = ndcg(evaluator, rescored)
    delta, p, better, worse, tied = permutation(base_per_q, per_q)
    print(f"{lam:7.1f} {overall:7.2f} {delta:+7.2f} {p:8.4f}   "
          f"{better:3d}/{worse:3d}/{tied:3d}{'' if p < 0.05 else '   n.s.'}")
    rows.append({"lambda": lam, "ndcg@10": round(overall, 2), "delta": round(delta, 3),
                 "p": round(p, 5), "better": better, "worse": worse, "tied": tied,
                 "per_question": {q: round(v, 4) for q, v in per_q.items()}})

out = ROOT / f"data/benchmark/vidore_v3/results/sep_holdout_{args.subset}_{args.language}.json"
out.write_text(json.dumps({
    "subset": args.subset, "language": args.language, "queries": n,
    "baseline": round(baseline, 2), "w": W, "gamma": GAMMA, "beta": BETA, "topm": TOPM,
    "shape": dict(shape), "lambdas": rows,
    "baseline_per_question": {q: round(v, 4) for q, v in base_per_q.items()},
}, indent=1), encoding="utf-8")
print(f"\nwrote {out}")
