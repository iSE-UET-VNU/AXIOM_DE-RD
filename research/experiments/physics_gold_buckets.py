"""Answer accuracy conditioned on how many gold pages retrieval actually delivered.

The aggregate oracle-minus-retrieved gap is small, which reads as "retrieval does
not matter for QA". Bucketing by delivered gold shows the aggregate is a mixture:
the loss is concentrated in the queries that get zero gold pages.

Note the buckets are self-selected -- queries where retrieval succeeds are also
easier queries -- so the gradient is an upper bound on the causal effect, not a
clean estimate of it. The zero-gold bucket is still where any recoverable loss is.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data/benchmark/vidore_v3/results"
ARMS = ("physics_e2e_retrieved_vidore", "physics_e2e_retrieved_chandra2")


def bucket(row):
    n = sum(1 for hit in row["retrieved"] if hit.get("is_gold"))
    return "0 gold" if n == 0 else ("1-2 gold" if n <= 2 else "3+ gold")


for name in ARMS:
    rows = json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))
    groups: dict[str, list] = {"0 gold": [], "1-2 gold": [], "3+ gold": []}
    for row in rows:
        groups[bucket(row)].append(row)

    overall = Counter(r["judgment"] for r in rows)
    base = 100 * overall["Correct"] / len(rows)
    print(f"=== {name}  n={len(rows)}  correct_only={base:.1f}%")
    for label, members in groups.items():
        if not members:
            continue
        counts = Counter(r["judgment"] for r in members)
        n = len(members)
        correct = counts["Correct"]
        credited = correct + counts["Partially Correct"]
        print(f"  {label:9s} n={n:3d}  correct_only={100 * correct / n:5.1f}%  "
              f"+partial={100 * credited / n:5.1f}%")

    zero = groups["0 gold"]
    rest = [r for r in rows if bucket(r) != "0 gold"]
    if zero and rest:
        zc = 100 * Counter(r["judgment"] for r in zero)["Correct"] / len(zero)
        rc = 100 * Counter(r["judgment"] for r in rest)["Correct"] / len(rest)
        recoverable = len(zero) / len(rows) * (rc - zc)
        print(f"  upper-bound gain if every zero-gold query were fixed: "
              f"+{recoverable:.1f}pp correct_only")
