import json
from collections import defaultdict
from time import perf_counter

import numpy as np

from .bench import BENCH, doc_of, fingerprint, gold as load_gold, load_jsonl, queries as load_queries, work
from .evaluate import as_run, evaluate, file_hit, groups
from .text import PageBM25, plain

FILE_K, PAGE_K, METRIC_K = 3, 20, 10
BM25_WEIGHT, FILE_DIRECT, PARENT_WEIGHT = 0.70, 0.50, 0.15


def normalise(scores):
    maximum = max((v for v in scores.values() if v > 0), default=0.0)
    if maximum <= 0:
        return {k: 0.0 for k in scores}
    return {k: max(0.0, v) / maximum for k, v in scores.items()}


def sort_scores(scores):
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def their_ndcg(ranked, gold_row, k):
    actual = sum((2.0 ** gold_row[p] - 1.0) / np.log2(r + 1) for r, p in enumerate(ranked[:k], 1) if p in gold_row)
    ideal = sum((2.0 ** s - 1.0) / np.log2(r + 1) for r, s in enumerate(sorted(gold_row.values(), reverse=True)[:k], 1))
    return actual / ideal if ideal else 0.0


def run(bench=BENCH, pages_name="pages_ocr_sparse.jsonl", tag="hier_replica"):
    rows = load_jsonl(work("light_prep", bench=bench) / pages_name)
    gold, queries = load_gold(bench), load_queries(bench)
    qids = sorted(gold)
    units = [r["page_id"] for r in rows]
    page_to_file = {r["page_id"]: doc_of(r["page_id"]) for r in rows}
    texts = [r["text"] for r in rows]

    grouped = defaultdict(list)
    for r in rows:
        grouped[doc_of(r["page_id"])].append(r["text"])
    file_ids = sorted(grouped)
    page_bm25 = PageBM25([plain(t) for t in texts])
    file_bm25 = PageBM25([plain("\n".join(grouped[f]).strip()) for f in file_ids])

    results, seconds = {}, []
    for q in qids:
        started = perf_counter()
        tokens = plain(queries[q]["query"])
        raw = page_bm25.scores(tokens)
        bm25_norm = normalise({u: float(raw[i]) for i, u in enumerate(units)})
        page_base = {u: BM25_WEIGHT * bm25_norm[u] for u in units}
        fraw = file_bm25.scores(tokens)
        direct_norm = normalise({f: float(fraw[i]) for i, f in enumerate(file_ids)})
        pooled = defaultdict(float)
        for u, s in page_base.items():
            pooled[page_to_file[u]] = max(pooled[page_to_file[u]], s)
        pool_norm = normalise({f: pooled.get(f, 0.0) for f in file_ids})
        file_scores = {f: FILE_DIRECT * direct_norm[f] + (1 - FILE_DIRECT) * pool_norm[f] for f in file_ids}
        selected = {f for f, _ in sort_scores(file_scores)[:FILE_K]}
        parent_norm = normalise(file_scores)
        page_scores = {u: (1 - PARENT_WEIGHT) * page_base[u] + PARENT_WEIGHT * parent_norm[page_to_file[u]]
                       for u in units if page_to_file[u] in selected}
        ranked = [u for u, _ in sort_scores(page_scores)[:PAGE_K]]
        results[q] = {"selected": sorted(selected), "ranked": ranked}
        seconds.append(perf_counter() - started)

    metrics = {}
    for group, members in groups(qids, queries).items():
        their_file, their_r10, their_r20, their_n10, their_n20 = [], [], [], [], []
        for q in members:
            gold_files = {doc_of(u) for u in gold[q]}
            their_file.append(len(set(results[q]["selected"]) & gold_files) / len(gold_files))
            top = results[q]["ranked"]
            their_r10.append(len(set(top[:10]) & set(gold[q])) / len(gold[q]))
            their_r20.append(len(set(top[:20]) & set(gold[q])) / len(gold[q]))
            their_n10.append(their_ndcg(top, gold[q], 10))
            their_n20.append(their_ndcg(top, gold[q], 20))
        sub = {q: as_run(results[q]["ranked"]) for q in members}
        n10, r10, _ = evaluate(sub, gold, 10)
        n20, r20, _ = evaluate(sub, gold, 20)
        metrics[group] = {
            "n": len(members),
            "their_file_recall@3": round(100 * float(np.mean(their_file)), 2),
            "their_ndcg@10": round(100 * float(np.mean(their_n10)), 2),
            "their_recall@10": round(100 * float(np.mean(their_r10)), 2),
            "their_ndcg@20": round(100 * float(np.mean(their_n20)), 2),
            "their_recall@20": round(100 * float(np.mean(their_r20)), 2),
            "our_file_recall@20": round(100 * float(np.mean([file_hit(results[q]["ranked"], gold[q]) for q in members])), 2),
            "our_ndcg@10": round(n10, 2), "our_recall@10": round(r10, 2),
            "our_ndcg@20": round(n20, 2), "our_recall@20": round(r20, 2)}

    payload = {"variant": "hierarchical_filek3_bm25_k20", "pages_source": pages_name,
               "file_k": FILE_K, "page_k": PAGE_K, "bundle_fingerprint": fingerprint(bench),
               "n_queries": len(qids), "n_corpus_pages": len(units), "n_files": len(file_ids),
               "metrics": metrics, "query_seconds_mean": round(float(np.mean(seconds)), 4)}
    out = work("results", tag, bench=bench)
    (out / "metrics.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for group, m in metrics.items():
        print(f"{group:22s} n={m['n']:3d} | THEIR file@3 {m['their_file_recall@3']:6.2f} ndcg@10 {m['their_ndcg@10']:6.2f} "
              f"r@10 {m['their_recall@10']:6.2f} ndcg@20 {m['their_ndcg@20']:6.2f} r@20 {m['their_recall@20']:6.2f} "
              f"| OURS file@20 {m['our_file_recall@20']:6.2f} ndcg@10 {m['our_ndcg@10']:6.2f} r@20 {m['our_recall@20']:6.2f}")
    return payload


if __name__ == "__main__":
    run()
