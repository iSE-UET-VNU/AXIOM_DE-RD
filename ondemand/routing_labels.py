import argparse
import json
from collections import defaultdict

import numpy as np
import pytrec_eval

from .arms import ColVec
from .bench import BENCH, gold as load_gold, group_of, queries as load_queries, work, write_jsonl
from .evaluate import as_run, groups

FOLDS = 5
FOLD_SEED = 42


def recall10(runs, gold):
    per_query = pytrec_eval.RelevanceEvaluator({q: gold[q] for q in runs}, {"recall_10"}).evaluate(
        {q: as_run(r) for q, r in runs.items()})
    return {q: 100 * v["recall_10"] for q, v in per_query.items()}


def branches(gate, k, gold, bench):
    colvec = ColVec(bench)
    light = {q: r[:k] for q, r in gate["light_run_top100"].items() if q in gold}
    short = [q for q, r in light.items() if len(r) < k]
    if short:
        raise SystemExit(f"{len(short)} queries have fewer than {k} light pages, e.g. {short[0]}")
    visual = {q: sorted(r, key=lambda u: -colvec.score(q, u)) for q, r in light.items()}
    return recall10(light, gold), recall10(visual, gold)


def folds(qids, queries):
    by_group = defaultdict(list)
    for q in sorted(qids):
        by_group[group_of(q, queries[q])].append(q)
    rng = np.random.default_rng(FOLD_SEED)
    out = {}
    for group in sorted(by_group):
        for i, q in enumerate(rng.permutation(by_group[group])):
            out[q] = i % FOLDS
    return out


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand.routing_labels")
    ap.add_argument("--gate", default="gate_ppocrv5_backfill.json")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", default=None, help="default: labels_r10_k<k>.jsonl")
    args = ap.parse_args()
    gold, queries = load_gold(BENCH), load_queries(BENCH)
    gate = json.loads((work("light_prep", bench=BENCH) / args.gate).read_text(encoding="utf-8"))
    light, visual = branches(gate, args.k, gold, BENCH)
    qids = sorted(light)
    fold = folds(qids, queries)
    rows = [{"query_id": q, "query": queries[q]["query"], "group": group_of(q, queries[q]), "fold": fold[q],
             "gate": args.gate, "k": args.k, "light_recall@10": light[q], "visual_recall@10": visual[q],
             "visual_helps": visual[q] > light[q]} for q in qids]
    write_jsonl(work("routing", bench=BENCH) / (args.out or f"labels_r10_k{args.k}.jsonl"), rows)
    by_qid = {r["query_id"]: r for r in rows}
    for group, members in groups(qids, queries).items():
        sub = [by_qid[q] for q in members]
        n = len(sub)
        print(json.dumps({"group": group, "n": n,
                          "light": round(sum(r["light_recall@10"] for r in sub) / n, 2),
                          "visual": round(sum(r["visual_recall@10"] for r in sub) / n, 2),
                          "oracle": round(sum(max(r["light_recall@10"], r["visual_recall@10"]) for r in sub) / n, 2),
                          "visual_helps": sum(r["visual_helps"] for r in sub),
                          "light_wins": sum(r["light_recall@10"] > r["visual_recall@10"] for r in sub),
                          "ties": sum(r["light_recall@10"] == r["visual_recall@10"] for r in sub)}))


if __name__ == "__main__":
    main()
