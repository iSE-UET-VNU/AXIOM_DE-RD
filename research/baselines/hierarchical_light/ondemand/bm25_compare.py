import numpy as np

from .bench import BENCH, gold as load_gold, load_jsonl, queries as load_queries, work
from .evaluate import as_run, evaluate, file_hit, groups
from .text import PageBM25, enfr, plain, real_text

K = 20


def compare(bench=BENCH):
    rows = load_jsonl(work("light_prep", bench=bench) / "pages_ocr_sparse.jsonl")
    gold, queries = load_gold(bench), load_queries(bench)
    qids = sorted(gold)
    units = [r["page_id"] for r in rows]
    for name, tokenize in (("plain", plain), ("enfr_stem", enfr)):
        bm25 = PageBM25([tokenize(real_text(r["text"])) for r in rows])
        ranked = {}
        for q in qids:
            s = bm25.scores(tokenize(queries[q]["query"]))
            idx = np.argsort(-s, kind="stable")[:100]
            ranked[q] = [units[i] for i in idx]
        print(name)
        for group, members in groups(qids, queries).items():
            sub = {q: as_run(ranked[q]) for q in members}
            n10, r10, _ = evaluate(sub, gold, 10)
            n20, r20, _ = evaluate(sub, gold, 20)
            fr = 100 * np.mean([file_hit(ranked[q][:K], gold[q]) for q in members])
            print(f"  {group:22s} n={len(members):3d} file_r@20 {fr:6.2f} ndcg@10 {n10:6.2f} r@10 {r10:6.2f} ndcg@20 {n20:6.2f} r@20 {r20:6.2f}")


if __name__ == "__main__":
    compare()
