"""Fixed-configuration significance test for SEP.

The grid sweep in physics_sep.py overfits: 300 configurations against 302
queries gave a +2.06 in-sample peak whose held-out transfer swung from +2.04 to
-0.14 across folds. So this script does not select. It pins one configuration
chosen from the *measured* effect sizes in physics_structure_diagnostic.py --
file membership lifts the gold rate 16.9x while adjacency adds only 1.63x on
top, so beta is set file-dominant at 0.75 -- and tests it once with a paired
permutation test over all 302 queries, on both parses.

lambda is reported as a sweep rather than tuned, so the reader sees the
stability of the effect instead of a cherry-picked peak.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytrec_eval

from src.utils.env import load_dotenv_file

load_dotenv_file(ROOT)

from src.evaluation.benchmarks import load
from research.experiments.physics_sep import build, propagate, ndcg

RESAMPLES = 10_000
# Fixed a priori from the diagnostic, not selected on the metric.
W, GAMMA, BETA, TOPM = 2, 0.5, 0.75, 3
LAMBDAS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def permutation(a: dict[str, float], b: dict[str, float], resamples=RESAMPLES):
    qids = sorted(set(a) & set(b))
    diffs = [b[q] - a[q] for q in qids]
    observed = sum(diffs) / len(diffs)
    rng = random.Random(0)
    extreme = sum(
        abs(sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)) >= abs(observed)
        for _ in range(resamples))
    better = sum(1 for d in diffs if d > 1e-9)
    worse = sum(1 for d in diffs if d < -1e-9)
    return observed, (extreme + 1) / (resamples + 1), better, worse, len(diffs) - better - worse


def main() -> None:
    report = {}
    for index in ("vidore_page", "chandra_page"):
        qrels, pool = build(index)
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})
        baseline, base_per_q = ndcg(evaluator, pool)

        print(f"\n=== {index}  (baseline alpha0.7 = {baseline:.2f}, "
              f"fixed w={W} gamma={GAMMA} beta={BETA} topm={TOPM}) ===")
        print(f"{'lambda':>7s} {'ndcg':>7s} {'delta':>7s} {'p':>8s}   better/worse/tied")
        rows = []
        for lam in LAMBDAS:
            rescored = {q: propagate(s, lam, W, GAMMA, BETA, TOPM) for q, s in pool.items()}
            overall, per_q = ndcg(evaluator, rescored)
            delta, p, better, worse, tied = permutation(base_per_q, per_q)
            flag = "" if p < 0.05 else "   n.s."
            print(f"{lam:7.1f} {overall:7.2f} {delta:+7.2f} {p:8.4f}   "
                  f"{better:3d}/{worse:3d}/{tied:3d}{flag}")
            rows.append({"lambda": lam, "ndcg@10": round(overall, 2), "delta": round(delta, 3),
                         "p": round(p, 5), "better": better, "worse": worse, "tied": tied,
                         "per_question": {q: round(v, 4) for q, v in per_q.items()}})
        report[index] = {"baseline": round(baseline, 2), "w": W, "gamma": GAMMA,
                         "beta": BETA, "topm": TOPM, "lambdas": rows,
                         "baseline_per_question": {q: round(v, 4) for q, v in base_per_q.items()}}

    out = ROOT / "data/benchmark/vidore_v3/results/physics_sep_significance.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
