from collections import Counter, defaultdict

import numpy as np

from .bench import BENCH, doc_of, gold as load_gold, group_of, load_jsonl, queries as load_queries, work
from .gate import ALPHA, K, dense_vectors, minmax_top100
from .text import PageBM25, enfr, real_text

CONF = 60


def bucket(rank):
    for low, high in ((21, 30), (31, 50), (51, 100), (101, 200)):
        if rank <= high:
            return f"{low}-{high}"
    return ">200"


def analyze(bench=BENCH):
    rows = load_jsonl(work("light_prep", bench=bench) / "pages_ocr_sparse.jsonl")
    gold, queries = load_gold(bench), load_queries(bench)
    qids = sorted(gold)
    units = [r["page_id"] for r in rows]
    row_of = {r["page_id"]: r for r in rows}
    bm25 = PageBM25([enfr(real_text(r["text"])) for r in rows])
    sims = dense_vectors(rows, qids, queries, bench)

    misses, lost = [], defaultdict(float)
    for i, q in enumerate(qids):
        fused = ALPHA * minmax_top100(bm25.scores(enfr(queries[q]["query"]))) + (1 - ALPHA) * minmax_top100(sims[i])
        order = np.argsort(-fused, kind="stable")
        rank = {units[j]: r + 1 for r, j in enumerate(order)}
        top = [units[j] for j in order[:K]]
        top_docs = {doc_of(u) for u in top}
        top_set = set(top)
        positives = [u for u, g in gold[q].items() if g > 0]
        group = group_of(q, queries[q])
        for u in positives:
            if rank[u] <= K:
                continue
            r = row_of[u]
            doc, page = doc_of(u), r["page_index"]
            doc_best = min(rank[v] for v in units if doc_of(v) == doc) if doc not in top_docs else None
            misses.append({
                "qid": q, "group": group, "unit": u, "rank": rank[u], "bucket": bucket(rank[u]),
                "file_in_top20": doc in top_docs,
                "neighbour_in_top20": any(f"{doc}#page={page + d}" in top_set for d in (-1, 1)),
                "ocr_applied": r["ocr_applied"], "ocr_conf": r["ocr_mean_confidence"], "ocr_words": r["ocr_word_count"],
                "ocr_weak": r["ocr_applied"] and (r["ocr_mean_confidence"] < CONF or r["ocr_word_count"] == 0),
                "chars": len(real_text(r["text"])), "doc_best_rank": doc_best})
            lost[group] += 100 / len(positives) / len(qids)

    n_all = Counter(group_of(q, queries[q]) for q in qids)
    print(f"missed gold pages (rank > {K}): {len(misses)} across {len({m['qid'] for m in misses})} queries")
    print(f"R@20 points lost per group (sum = 100 - R@20): " + ", ".join(f"{g} {v:.2f}" for g, v in sorted(lost.items())))
    for group in sorted({m["group"] for m in misses}):
        sub = [m for m in misses if m["group"] == group]
        print(f"\n== {group}  queries {n_all[group]}  missed pages {len(sub)}")
        print("  rank bucket:", dict(Counter(m["bucket"] for m in sub).most_common()))
        print(f"  file in top20 (right file, wrong page): {sum(m['file_in_top20'] for m in sub)}")
        print(f"  neighbour page (+-1) in top20:          {sum(m['neighbour_in_top20'] for m in sub)}")
        print(f"  ocr applied: {sum(m['ocr_applied'] for m in sub)}  ocr weak (conf<{CONF} or 0 words): {sum(m['ocr_weak'] for m in sub)}")
        print(f"  text chars p25/p50/p75: {np.percentile([m['chars'] for m in sub], [25, 50, 75]).round().tolist()}")
        file_miss = [m["doc_best_rank"] for m in sub if not m["file_in_top20"]]
        if file_miss:
            print(f"  file missed: best page rank of gold file p50 {int(np.median(file_miss))}, <=50: {sum(r <= 50 for r in file_miss)}/{len(file_miss)}")


if __name__ == "__main__":
    analyze()
