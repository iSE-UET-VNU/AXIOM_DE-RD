import json
from time import perf_counter

import numpy as np

from .arms import alpha_fuse, maxp
from .bench import BENCH, fingerprint, gold as load_gold, queries as load_queries, work, write_jsonl
from .evaluate import evaluate, groups
from .text import ChunkBM25

ALPHA, TOP_K = 0.7, 100


def run(bench=BENCH):
    folder = work("embeddings", "kdl", bench=bench)
    owners = np.array(json.loads((folder / "chunk_owners.json").read_text()))
    texts = json.loads((folder / "chunk_texts.json").read_text())
    vectors = np.load(folder / "chunk_vectors.npy")
    vectors = vectors / np.clip(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12, None)
    qvec_all = np.load(folder / "query_vectors.npy")
    qvec_all = qvec_all / np.clip(np.linalg.norm(qvec_all, axis=-1, keepdims=True), 1e-12, None)
    qvec = {q: qvec_all[i] for i, q in enumerate(json.loads((folder / "query_ids.json").read_text()))}

    gold, queries = load_gold(bench), load_queries(bench)
    qids = sorted(gold)
    bm25 = ChunkBM25(texts)
    run_rows, seconds = {}, []
    for q in qids:
        started = perf_counter()
        lexical = maxp(bm25.search(queries[q]["query"], len(texts)), owners)
        sims = vectors @ qvec[q]
        order = np.argsort(-sims)[:TOP_K * 10]
        dense = maxp([(int(j), float(sims[j])) for j in order], owners)
        run_rows[q] = alpha_fuse(lexical, dense, ALPHA, TOP_K)
        seconds.append(perf_counter() - started)

    metrics = {}
    for group, members in groups(qids, queries).items():
        sub = {q: run_rows[q] for q in members}
        n10, r10, _ = evaluate(sub, gold, 10)
        n20, r20, _ = evaluate(sub, gold, 20)
        metrics[group] = {"n": len(members), "ndcg@10": round(n10, 2), "recall@10": round(r10, 2),
                          "ndcg@20": round(n20, 2), "recall@20": round(r20, 2)}

    result = {"variant": "kdl_chunk512_te3small_hybrid0.7_fullcorpus", "alpha_dense": ALPHA, "depth": TOP_K,
              "bundle_fingerprint": fingerprint(bench), "n_queries": len(qids), "n_chunks": len(texts),
              "n_pages": int(len(set(owners.tolist()))), "metrics": metrics,
              "query_seconds_mean": round(float(np.mean(seconds)), 4),
              "query_seconds_total": round(float(np.sum(seconds)), 2)}
    out = work("results", "legacy_fullcorpus", bench=bench)
    (out / "metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_jsonl(out / "per_query.jsonl", [
        {"query_id": q, "source": queries[q]["source"], "query": queries[q]["query"], "answers": queries[q]["answers"],
         "gold": [{"unit": u, "relevance": g} for u, g in gold[q].items()],
         "arms": {"legacy": {"top": [{"unit": u, "rank": i + 1, "score": round(float(v), 6)}
                                     for i, (u, v) in enumerate(sorted(run_rows[q].items(), key=lambda x: -x[1]))]}}}
        for q in qids])
    for group, m in metrics.items():
        print(f"{group:22s} n={m['n']:3d} ndcg@10 {m['ndcg@10']:6.2f} r@10 {m['recall@10']:6.2f} "
              f"ndcg@20 {m['ndcg@20']:6.2f} r@20 {m['recall@20']:6.2f}")
    print(f"{result['n_chunks']} chunks over {result['n_pages']} pages, "
          f"{result['query_seconds_total']}s total, {result['query_seconds_mean']}s/query")
    return result


if __name__ == "__main__":
    run()
