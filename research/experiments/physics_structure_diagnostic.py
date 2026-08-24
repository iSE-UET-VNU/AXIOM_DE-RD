"""Which structural level carries the reordering signal?

Retrieval on physics fails at ordering, not recall (any-gold@100 = 98.3%). But
"ordering" has two very different shapes, and they call for opposite mechanisms:

  A. the gold *file* is absent from the top-10 -> aggregate evidence over a file
     and promote the whole file (document-level aggregation helps)
  B. the gold file is already in the top-10 and the wrong *pages within it* rank
     -> file-level aggregation is a no-op, and the lever is intra-document
     neighbour propagation instead

This settles which case dominates before any retriever is written. It also sizes
the adjacency term: of the gold pages missed at 10 but held at 100, how many sit
next to a page we already returned. Reads the frozen served pool, so it is exact
and costs nothing.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytrec_eval

from src.evaluation.benchmarks import load

RESULTS = ROOT / "data/benchmark/vidore_v3/results"
POOLS = {"vidore_page": "physics_served_pool.json",
         "chandra_page": "physics_served_pool_chandra.json"}
UNIT = re.compile(r"^(?P<subset>[^:]+)::(?P<file>.+)#page=(?P<page>\d+)$")
NEAR = (1, 2)


def split(unit_id: str) -> tuple[str, int] | None:
    m = UNIT.match(unit_id)
    return (m["file"], int(m["page"])) if m else None


bench = load("vidore_v3", subset="physics", language="french")
qrels = bench.qrels()
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})
report: dict[str, dict] = {}

for index, filename in POOLS.items():
    pool = json.loads((RESULTS / filename).read_text())["queries"]
    ndcg = evaluator.evaluate({
        qid: {c: float(len(e["candidates"]) - i) for i, c in enumerate(e["candidates"])}
        for qid, e in pool.items() if qrels.get(qid)})

    rows, shape = [], Counter()
    for qid, entry in pool.items():
        gold_grades = qrels.get(qid)
        if not gold_grades:
            continue
        gold_pages = {u for u, g in gold_grades.items() if g > 0}
        gold_files = {p[0] for u in gold_pages if (p := split(u))}

        ranked = entry["candidates"]
        top10, parts = ranked[:10], [split(u) for u in ranked]
        top10_files = [p[0] for p in parts[:10] if p]

        first_gold = next((i for i, u in enumerate(ranked) if u in gold_pages), None)
        first_gold_file = next((i for i, p in enumerate(parts) if p and p[0] in gold_files), None)

        # of the gold pages we missed at 10 but still hold at 100, how many sit
        # adjacent to a page we did return in the top-10?
        held = {u for u in ranked[:100] if u in gold_pages}
        missed = [u for u in held if u not in set(top10)]
        returned = {(p[0], p[1]) for p in parts[:10] if p}
        adjacent = sum(
            1 for u in missed
            if (g := split(u)) and any((g[0], g[1] + d) in returned
                                       for d in (*NEAR, *(-n for n in NEAR))))

        row = {
            "qid": qid,
            "ndcg@10": round(100 * ndcg[qid]["ndcg_cut_10"], 2),
            "n_gold_pages": len(gold_pages),
            "n_gold_files": len(gold_files),
            "gold_file_in_top10": bool(gold_files & set(top10_files)),
            "top10_from_gold_file": sum(1 for f in top10_files if f in gold_files),
            "top10_gold": sum(1 for u in top10 if u in gold_pages),
            "rank_first_gold_page": first_gold,
            "rank_first_gold_file": first_gold_file,
            "missed_at10_held_at100": len(missed),
            "missed_adjacent_to_top10": adjacent,
        }
        rows.append(row)
        if not row["gold_file_in_top10"]:
            shape["A: gold file absent from top-10"] += 1
        elif row["top10_gold"] < row["n_gold_pages"]:
            shape["B: gold file present, wrong pages within it"] += 1
        else:
            shape["C: all gold already in top-10"] += 1

    n = len(rows)
    missed_total = sum(r["missed_at10_held_at100"] for r in rows)
    adj_total = sum(r["missed_adjacent_to_top10"] for r in rows)
    summary = {
        "queries": n,
        "ndcg@10": round(sum(r["ndcg@10"] for r in rows) / n, 2),
        "shape": {k: {"n": v, "pct": round(100 * v / n, 1)} for k, v in shape.most_common()},
        "mean_gold_pages": round(sum(r["n_gold_pages"] for r in rows) / n, 2),
        "mean_gold_files": round(sum(r["n_gold_files"] for r in rows) / n, 2),
        "multi_file_gold_pct": round(100 * sum(r["n_gold_files"] > 1 for r in rows) / n, 1),
        "mean_top10_from_gold_file": round(sum(r["top10_from_gold_file"] for r in rows) / n, 2),
        "mean_top10_gold": round(sum(r["top10_gold"] for r in rows) / n, 2),
        "gold_pages_missed_at10_held_at100": missed_total,
        "of_which_adjacent_to_a_top10_page": adj_total,
        "adjacency_yield_pct": round(100 * adj_total / missed_total, 1) if missed_total else 0.0,
    }
    # Control: adjacency only matters if it beats the base rate. Compare the gold
    # rate among rank-11..100 pages adjacent to a returned page against the rate
    # among the rest of the same tail.
    lift = {}
    for w in (1, 2, 3):
        near_n = near_gold = far_n = far_gold = 0
        for qid, entry in pool.items():
            gold_grades = qrels.get(qid)
            if not gold_grades:
                continue
            gold = {u for u, g in gold_grades.items() if g > 0}
            ranked = entry["candidates"]
            returned = {p for u in ranked[:10] if (p := split(u))}
            for unit in ranked[10:100]:
                p = split(unit)
                if not p:
                    continue
                if any((p[0], p[1] + d) in returned for d in range(-w, w + 1) if d):
                    near_n += 1
                    near_gold += unit in gold
                else:
                    far_n += 1
                    far_gold += unit in gold
        near_pct = 100 * near_gold / near_n if near_n else 0.0
        far_pct = 100 * far_gold / far_n if far_n else 0.0
        lift[f"w{w}"] = {
            "adjacent_gold": near_gold, "adjacent_n": near_n,
            "adjacent_gold_pct": round(near_pct, 2),
            "other_gold": far_gold, "other_n": far_n,
            "other_gold_pct": round(far_pct, 2),
            "lift": round(near_pct / far_pct, 2) if far_pct else None,
        }
    summary["adjacency_lift"] = lift

    report[index] = {"summary": summary, "per_query": rows}

    print(f"\n=== {index}  (NDCG@10 {summary['ndcg@10']}, n={n}) ===")
    for label, v in summary["shape"].items():
        print(f"  {label:45s} {v['n']:4d}  {v['pct']:5.1f}%")
    print(f"  mean gold pages/query        {summary['mean_gold_pages']:.2f}"
          f"   in {summary['mean_gold_files']:.2f} file(s)"
          f"   multi-file {summary['multi_file_gold_pct']}%")
    print(f"  mean top-10 from a gold file {summary['mean_top10_from_gold_file']:.2f}"
          f"   of which gold {summary['mean_top10_gold']:.2f}")
    print(f"  gold missed@10 but held@100  {missed_total}"
          f"   adjacent to a returned page {adj_total}"
          f"  ({summary['adjacency_yield_pct']}%)")
    for w, v in lift.items():
        print(f"  adjacency lift {w:3s}  gold@adjacent {v['adjacent_gold_pct']:5.2f}%"
              f"  vs elsewhere {v['other_gold_pct']:5.2f}%   x{v['lift']}")

OUT = RESULTS / "physics_structure_diagnostic.json"
OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
print(f"\nwrote {OUT}")
