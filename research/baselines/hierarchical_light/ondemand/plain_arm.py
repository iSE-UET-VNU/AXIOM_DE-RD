import json
from time import perf_counter

import numpy as np

from .arms import ColVec
from .bench import BENCH, fingerprint, gold as load_gold, load_jsonl, queries as load_queries, work, write_jsonl
from .evaluate import as_run, evaluate, file_hit, groups
from .text import PageBM25, plain, real_text

K = 20


def run(bench=BENCH, pages_name="pages_ocr_sparse.jsonl", tag="plain_bm25_colvec"):
    out = work("light_prep", bench=bench)
    rows = load_jsonl(out / pages_name)
    gold, queries = load_gold(bench), load_queries(bench)
    qids = sorted(gold)
    units = [r["page_id"] for r in rows]
    bm25 = PageBM25([plain(real_text(r["text"])) for r in rows])
    ranked, seconds = {}, []
    for q in qids:
        started = perf_counter()
        scores = bm25.scores(plain(queries[q]["query"]))
        ranked[q] = [units[i] for i in np.argsort(-scores, kind="stable")[:100]]
        seconds.append(perf_counter() - started)
    gate = {q: ranked[q][:K] for q in qids}

    colvec = ColVec(bench)
    arm_a = {q: {u: colvec.score(q, u) for u in gate[q]} for q in qids}

    light, accurate = {}, {}
    for group, members in groups(qids, queries).items():
        sub = {q: as_run(ranked[q]) for q in members}
        n10, r10, _ = evaluate(sub, gold, 10)
        n20, r20, _ = evaluate(sub, gold, 20)
        light[group] = {"n": len(members), "file_recall@20": round(100 * float(np.mean([file_hit(gate[q], gold[q]) for q in members])), 2),
                        "ndcg@10": round(n10, 2), "recall@10": round(r10, 2), "ndcg@20": round(n20, 2), "recall@20": round(r20, 2)}
        a = {q: arm_a[q] for q in members}
        an10, ar10, _ = evaluate(a, gold, 10)
        an20, ar20, _ = evaluate(a, gold, 20)
        accurate[group] = {"n": len(members), "ndcg@10": round(an10, 2), "recall@10": round(ar10, 2),
                           "ndcg@20": round(an20, 2), "recall@20": round(ar20, 2)}

    result = {"variant": "bm25_plain_k20", "pages_source": pages_name, "gate_k": K,
              "bundle_fingerprint": fingerprint(bench), "n_queries": len(qids), "n_corpus_pages": len(units),
              "light": light, "accurate_colvec": accurate,
              "light_query_seconds_mean": round(float(np.mean(seconds)), 4),
              "light_query_seconds_total": round(float(np.sum(seconds)), 2)}
    folder = work("results", tag, bench=bench)
    (folder / "metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_jsonl(folder / "per_query.jsonl", [
        {"query_id": q, "source": queries[q]["source"], "query": queries[q]["query"], "answers": queries[q]["answers"],
         "gold": [{"unit": u, "relevance": g} for u, g in gold[q].items()],
         "gate": [{"unit": u, "rank": i + 1} for i, u in enumerate(gate[q])],
         "arms": {"A": {"top": [{"unit": u, "rank": i + 1, "score": round(arm_a[q][u], 6)}
                                for i, u in enumerate(sorted(gate[q], key=lambda x: -arm_a[q][x]))]}}}
        for q in qids])
    for group in light:
        l, a = light[group], accurate[group]
        print(f"{group:22s} n={l['n']:3d} | light file@20 {l['file_recall@20']:6.2f} ndcg@10 {l['ndcg@10']:6.2f} "
              f"r@10 {l['recall@10']:6.2f} ndcg@20 {l['ndcg@20']:6.2f} r@20 {l['recall@20']:6.2f} "
              f"| colvec ndcg@10 {a['ndcg@10']:6.2f} r@10 {a['recall@10']:6.2f} ndcg@20 {a['ndcg@20']:6.2f} r@20 {a['recall@20']:6.2f}")
    print(f"bm25 {result['light_query_seconds_total']}s total, {result['light_query_seconds_mean']}s/query")
    return result


if __name__ == "__main__":
    run()
