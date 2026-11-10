from collections import defaultdict

import pytrec_eval

from .bench import doc_of, group_of


def evaluate(run, gold, k):
    scored = {q: s for q, s in run.items() if q in gold}
    if not scored:
        return 0.0, 0.0, {}
    per_query = pytrec_eval.RelevanceEvaluator({q: gold[q] for q in scored}, {f"ndcg_cut_{k}", f"recall_{k}"}).evaluate(scored)
    n = len(per_query)
    return (100 * sum(v[f"ndcg_cut_{k}"] for v in per_query.values()) / n,
            100 * sum(v[f"recall_{k}"] for v in per_query.values()) / n, per_query)


def as_run(ranked):
    return {u: float(len(ranked) - i) for i, u in enumerate(ranked)}


def page_recall(units, gold_row):
    positives = {u for u, g in gold_row.items() if g > 0}
    return len(set(units) & positives) / max(len(positives), 1)


def file_hit(units, gold_row):
    positives = {doc_of(u) for u, g in gold_row.items() if g > 0}
    return float(bool({doc_of(u) for u in units} & positives))


def groups(qids, queries):
    out = defaultdict(list)
    for q in qids:
        out["all"].append(q)
        out[group_of(q, queries[q])].append(q)
    return dict(out)


def summarize(run, gold, queries, qids):
    out = {}
    for name, members in groups(qids, queries).items():
        sub = {q: run[q] for q in members}
        n10, r10, _ = evaluate(sub, gold, 10)
        n20, r20, _ = evaluate(sub, gold, 20)
        out[name] = {"n": len(members), "ndcg@10": round(n10, 2), "recall@10": round(r10, 2),
                     "ndcg@20": round(n20, 2), "recall@20": round(r20, 2)}
    _, _, pq10 = evaluate(run, gold, 10)
    _, _, pq20 = evaluate(run, gold, 20)
    per_query = {q: {"ndcg@10": pq10[q]["ndcg_cut_10"], "recall@10": pq10[q]["recall_10"],
                     "ndcg@20": pq20[q]["ndcg_cut_20"], "recall@20": pq20[q]["recall_20"]} for q in pq10}
    return out, per_query
