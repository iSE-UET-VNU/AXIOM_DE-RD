import json
from time import perf_counter

import numpy as np
import tiktoken
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .bench import BENCH, fingerprint, gold as load_gold, load_jsonl, queries as load_queries, work
from .evaluate import as_run, evaluate, file_hit, groups, page_recall
from .openrouter import embed
from .text import PageBM25, enfr, real_text
from .timing import Timings

QWEN = "qwen/qwen3-embedding-8b"
QWEN_PREFIX = "Instruct: Given a question, retrieve document pages that answer the question\nQuery: "
ALPHA, K, K_CONFIDENT, CONFIDENCE, POOL = 0.2, 20, 10, 0.7, 200


def minmax_top100(x):
    top = np.sort(x)[-100:]
    return np.clip((x - top[0]) / (top[-1] - top[0] + 1e-9), 0, None)


def top_n(scores, n=POOL):
    idx = np.argpartition(-scores, n)[:n]
    return idx[np.argsort(-scores[idx], kind="stable")]


def dense_vectors(rows, qids, queries, bench, tag=""):
    folder = work("light_v2", bench=bench)
    stem = f"dense_qwen8b{tag}"
    pages, qvec, meta = folder / f"{stem}_pages.npy", folder / f"{stem}_queries.npy", folder / f"{stem}_meta.json"
    units = [r["page_id"] for r in rows]
    if not pages.exists():
        enc = tiktoken.get_encoding("cl100k_base")
        texts = [real_text(r["text"]) for r in rows]
        texts = [enc.decode(enc.encode(t)[:6000]) for t in texts]
        keep = [i for i, t in enumerate(texts) if t.strip()]
        started = perf_counter()
        V = np.zeros((len(rows), 4096), dtype=np.float32)
        V[keep] = embed([texts[i] for i in keep], QWEN, work("embedding_cache", "qwen8b", bench=bench))
        page_seconds = perf_counter() - started
        started = perf_counter()
        Q = embed([QWEN_PREFIX + queries[q]["query"] for q in qids], QWEN, work("embedding_cache", "qwen8b", bench=bench))
        np.save(pages, V)
        np.save(qvec, Q)
        meta.write_text(json.dumps({"model": QWEN, "page_seconds": page_seconds,
                                    "query_seconds_each": (perf_counter() - started) / len(qids), "qids": qids, "units": units}))
    m = json.loads(meta.read_text())
    if m["units"] != units or m["qids"] != qids:
        raise ValueError("Qwen vectors are not in the order of the current pages/queries")
    V, Q = np.load(pages), np.load(qvec)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    return Q @ V.T


def build(bench=BENCH, name="gate_k20_best.json", pages_name="pages_ocr_sparse.jsonl", tag=""):
    out = work("light_prep", bench=bench)
    rows = load_jsonl(out / pages_name)
    gold, queries = load_gold(bench), load_queries(bench)
    qids = sorted(gold)
    units = [r["page_id"] for r in rows]
    position = {u: i for i, u in enumerate(units)}
    bm25 = PageBM25([enfr(real_text(r["text"])) for r in rows])
    sims = dense_vectors(rows, qids, queries, bench, tag)
    dense = {q: sims[i] for i, q in enumerate(qids)}
    timings = Timings(out / "gate_timings.jsonl", stage_group="gate")
    lexical, fused, ranked = {}, {}, {}
    for q in qids:
        started = perf_counter()
        lexical[q] = bm25.scores(enfr(queries[q]["query"]))
        fused[q] = ALPHA * minmax_top100(lexical[q]) + (1 - ALPHA) * minmax_top100(dense[q])
        ranked[q] = [units[i] for i in top_n(fused[q])]
        timings.record("light_query", "query", 1, perf_counter() - started, qid=q)

    def top_set(x, k):
        return set(np.argsort(-x)[:k])

    features, target = [], []
    for q in qids:
        b, d = lexical[q], dense[q]
        ds, bs = np.sort(d)[::-1], np.sort(b)[::-1]
        features.append([len(top_set(b, 10) & top_set(d, 10)) / 10, ds[0], ds[0] - ds[9], ds[0] - ds[1],
                         bs[0] / (bs[:100].mean() + 1e-9), (bs[0] - bs[9]) / (bs[0] + 1e-9), len(queries[q]["query"].split())])
        target.append(page_recall(ranked[q][:10], gold[q]) == 1.0)
    confidence = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.5)), np.array(features),
                                   np.array(target), cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                   method="predict_proba")[:, 1]
    adaptive = {q: (K_CONFIDENT if confidence[i] >= CONFIDENCE else K) for i, q in enumerate(qids)}

    gate = {q: ranked[q][:K] for q in qids}
    metrics = {}
    for group, members in groups(qids, queries).items():
        sub = {q: as_run(ranked[q][:100]) for q in members}
        n10, r10, _ = evaluate(sub, gold, 10)
        n20, r20, _ = evaluate(sub, gold, 20)
        metrics[group] = {"n": len(members), "light_ndcg@10": round(n10, 2), "light_recall@10": round(r10, 2),
                         "light_ndcg@20": round(n20, 2), "light_recall@20": round(r20, 2),
                         "gate_recall@20": round(100 * float(np.mean([page_recall(gate[q], gold[q]) for q in members])), 2),
                         "file_recall@20": round(100 * float(np.mean([file_hit(gate[q], gold[q]) for q in members])), 2)}
    union = sorted(set().union(*gate.values()))
    payload = {"gate_k": K, "gate_query_lang": "english", "variant": "bm25stem0.2_qwen3emb8b0.8", "pages_source": pages_name,
               "bundle_fingerprint": fingerprint(bench), "n_queries": len(qids), "n_corpus_pages": len(units),
               "n_union_pages": len(union), "metrics": metrics, "gate": gate,
               "gate_scores": {q: [round(float(fused[q][position[u]]), 6) for u in gate[q]] for q in qids},
               "union": union, "adaptive_k": adaptive,
               "adaptive_rule": f"k={K_CONFIDENT} if CV confidence >= {CONFIDENCE} else {K}",
               "gold_rank": {q: {u: (ranked[q].index(u) + 1 if u in ranked[q] else None) for u in gold[q]} for q in qids},
               "light_run_top100": {q: ranked[q][:100] for q in qids}}
    path = out / name
    path.write_text(json.dumps(payload))
    return path
