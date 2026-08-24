"""Paired permutation tests on the physics ladder, for the paper's two claims.

Deltas here are around 1 NDCG point on 302 queries, which is inside the range
where a sign is not a result.
"""
import json
import random
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "data/benchmark/vidore_v3/results" / "physics_retrieval_ladder.json"
RESAMPLES = 10000
PAIRS = [
    ("chunking within pages, ViDoRe text", "vidore_page", "vidore_fixed512"),
    ("chunking within pages, chandra2 flat", "chandra_page", "chandra_fixed512"),
    ("chunking within pages, chandra2 blocks", "chandra_page", "chandra_blocks"),
    ("image descriptions", "chandra_no_figure", "chandra_page"),
    ("page header/footer text", "chandra_no_boiler", "chandra_page"),
    ("simplest vs richest chandra2", "chandra_prose", "chandra_page"),
    ("parser: ViDoRe vs chandra2", "vidore_page", "chandra_page"),
]

rows = json.loads(RESULTS.read_text())
by_key = {(r["index"], r["arm"]): r for r in rows}
arms = sorted({r["arm"] for r in rows})


def permutation(a, b, resamples=RESAMPLES):
    qids = sorted(set(a) & set(b))
    diffs = [b[q] - a[q] for q in qids]
    observed = sum(diffs) / len(diffs)
    rng = random.Random(0)
    extreme = sum(
        abs(sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)) >= abs(observed)
        for _ in range(resamples)
    )
    better = sum(1 for d in diffs if d > 1e-9)
    worse = sum(1 for d in diffs if d < -1e-9)
    return observed, (extreme + 1) / (resamples + 1), better, worse, len(diffs) - better - worse


print(f"{'comparison':40s} {'arm':9s} {'base':>6s} {'+feat':>6s} {'delta':>7s} "
      f"{'p':>7s}  better/worse/tied")
print("-" * 104)
for label, base, feature in PAIRS:
    for arm in arms:
        left, right = by_key.get((base, arm)), by_key.get((feature, arm))
        if not left or not right:
            continue
        delta, p, better, worse, tied = permutation(left["per_question"], right["per_question"])
        flag = "" if p < 0.05 else "  n.s."
        print(f"{label:40s} {arm:9s} {left['ndcg@10']:6.1f} {right['ndcg@10']:6.1f} "
              f"{100*delta:+7.2f} {p:7.4f}  {better:3d}/{worse:3d}/{tied:3d}{flag}")
    print()
