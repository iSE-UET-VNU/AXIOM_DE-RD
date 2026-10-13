import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from .arms import ColVec
from .bench import BENCH, FLAWED_QIDS, doc_of, gold as load_gold, load_jsonl, work

GROUPS = ("all", "vidore_industrial", "vidore_physics", "ohrbench:scan", "ohrbench:handwritten")
BUCKETS = ("over_10_gold", "not_in_pool", "wrong_doc_on_top", "right_doc_wrong_page")


def rows_for(gate_name, labels_name, k, bench=BENCH):
    gold = {q: {u for u, g in rel.items() if g > 0} for q, rel in load_gold(bench).items()}
    labels = {r["query_id"]: r for r in load_jsonl(work("routing", bench=bench) / labels_name)}
    gate = json.loads((work("light_prep", bench=bench) / gate_name).read_text(encoding="utf-8"))
    cv = ColVec(bench)
    keys = list(cv.col)
    rows = []
    for q, lab in labels.items():
        if q in FLAWED_QIDS:
            continue
        g = gold[q]
        light = gate["light_run_top100"][q][:k]
        visual = sorted(light, key=lambda u: -cv.score(q, u))
        full = [keys[i] for i in np.argsort(-cv.matrix[cv.row[q]], kind="stable")[:10]]
        rec = {"light": len(g & set(light[:10])) / len(g), "visual": len(g & set(visual[:10])) / len(g)}
        arm = "visual" if rec["visual"] >= rec["light"] else "light"
        top = visual[:10] if arm == "visual" else light[:10]
        ceiling = min(10, len(g)) / len(g)
        missed = sorted(g - set(top))
        gold_docs = {doc_of(u) for u in g}
        wrong = [u for u in top if u not in g]
        same_doc = sum(doc_of(u) in gold_docs for u in wrong)
        kinds = []
        for u in missed:
            if u not in light:
                kinds.append("not_in_pool")
            elif wrong and same_doc * 2 >= len(wrong):
                kinds.append("right_doc_wrong_page")
            else:
                kinds.append("wrong_doc_on_top")
        lost = defaultdict(float)
        lost["over_10_gold"] = 1 - ceiling
        fixable = ceiling - rec[arm]
        for kind, n in Counter(kinds).items():
            lost[kind] += fixable * n / len(missed)
        rows.append({"query_id": q, "query": lab["query"], "group": lab["group"], "n_gold": len(g),
                     "n_gold_docs": len(gold_docs), "light_recall": rec["light"], "visual_recall": rec["visual"],
                     "best_arm": arm, "best": rec[arm], "ceiling": ceiling, "lost": dict(lost), "kinds": kinds,
                     "missed": missed, "pool_recall": len(g & set(light)) / len(g),
                     "full_corpus_recall": len(g & set(full)) / len(g),
                     "out_of_pool_found_by_full": sum(u in full for u in missed if u not in light),
                     "missed_rank_visual": [visual.index(u) + 1 for u in missed if u in light],
                     "missed_rank_light": [light.index(u) + 1 for u in missed if u in light]})
    return rows


def mean(sub, f):
    return round(100 * float(np.mean([f(r) for r in sub])), 2)


def summary(rows):
    for group in GROUPS:
        sub = [r for r in rows if group in ("all", r["group"])]
        ranks = [x for r in sub for x in r["missed_rank_visual"]]
        print(json.dumps({
            "group": group, "n": len(sub),
            "light": mean(sub, lambda r: r["light_recall"]), "visual": mean(sub, lambda r: r["visual_recall"]),
            "best_of_two": mean(sub, lambda r: r["best"]), "ceiling_r10": mean(sub, lambda r: r["ceiling"]),
            "pool_recall": mean(sub, lambda r: r["pool_recall"]),
            "colvec_full_corpus": mean(sub, lambda r: r["full_corpus_recall"]),
            "points_lost": {b: mean(sub, lambda r, b=b: r["lost"].get(b, 0.0)) for b in BUCKETS},
            "queries_perfect": sum(r["best"] == 1 for r in sub),
            "queries_with_fixable_miss": sum(r["best"] < r["ceiling"] for r in sub),
            "queries_zero_recall": sum(r["best"] == 0 for r in sub),
            "gold_pages_per_query": {"median": float(np.median([r["n_gold"] for r in sub])),
                                     "mean": round(float(np.mean([r["n_gold"] for r in sub])), 1),
                                     "max": max(r["n_gold"] for r in sub),
                                     "over_10": sum(r["n_gold"] > 10 for r in sub),
                                     "one": sum(r["n_gold"] == 1 for r in sub)},
            "queries_multi_gold_doc": sum(r["n_gold_docs"] > 1 for r in sub),
            "missed_pages": dict(Counter(k for r in sub for k in r["kinds"])),
            "out_of_pool_found_by_full_colvec": sum(r["out_of_pool_found_by_full"] for r in sub),
            "missed_in_pool_visual_rank": {"11-20": sum(x <= 20 for x in ranks),
                                           "21-50": sum(20 < x <= 50 for x in ranks),
                                           "51-100": sum(x > 50 for x in ranks)}}))


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand.miss_breakdown")
    ap.add_argument("--gate", default="gate_ppocrv5_backfill.json")
    ap.add_argument("--labels", default="labels_ladder_k100.jsonl")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--per-query", action="store_true", help="also print one JSON row per query with a fixable miss")
    args = ap.parse_args()
    rows = rows_for(args.gate, args.labels, args.k)
    summary(rows)
    if args.per_query:
        for r in rows:
            if r["best"] < r["ceiling"]:
                print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
