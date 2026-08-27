"""Nemotron Rerank VL 1B V2 (free, OpenRouter) on the KDL production pool.

Fills the gap ledger §21 left open -- Voyage rerank was only ever run on
`vidore_page`, never on the pool the pipeline ships (KDL -> fixed_512/128 ->
MaxP -> alpha0.7). `nvidia/llama-nemotron-rerank-vl-1b-v2:free` is a 1.7B
multimodal cross-encoder, free via OpenRouter's /rerank endpoint, so it is not
subject to the Voyage free-tier 3 RPM / 1200-char clip.

Text-only for now (the model also takes page images -- a natural follow-up that
would fold the ColQwen2 signal into the same call). Depth 20 to match how Voyage
was measured. Checkpointed per query, resumable.

Arms scored on the identical candidate pool (paired):
  baseline KDL alpha0.7
  + Nemotron rerank
  + SEP -> Nemotron rerank
  + SEP + ColQwen2 fusion -> Nemotron rerank
  + Nemotron rerank -> SEP + ColQwen2 fusion

    python research/experiments/physics_rerank_nemotron.py --depth 20
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytrec_eval
import requests

from src.utils.env import load_dotenv_file

load_dotenv_file(ROOT)

from research.experiments.physics_sep import minmax, propagate
from research.experiments.physics_sep_test import BETA, GAMMA, TOPM, W, permutation
from src.chunking_embedding.chunkers.builtin import fixed_overlap
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks
from src.evaluation.retrieval import alpha_fuse
from src.retrieval.sparse import BM25Index

SUB, LANG, ALPHA, POOL = "physics", "french", 0.7, 100
SEP_LAMBDA, WV = 0.5, 0.7
MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
URL = "https://openrouter.ai/api/v1/rerank"
RESULTS = ROOT / "data/benchmark/vidore_v3/results"
COLQWEN = ROOT / "data/work/vidore_physics_colqwen"
CKPT = RESULTS / "physics_rerank_nemotron_scores.json"
TEXTS = RESULTS / "physics_kdl_page_texts.json"
norm = lambda m: m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)
MAX_CHARS = 4000


def kdl_prod_pool():
    """KDL -> fixed_512/128 -> MaxP -> alpha0.7, with scores. All from cache."""
    run = next((ROOT / f"data_vidore_parsed_physics/output/benchmarks/vidore-v3-{SUB}-kdl").iterdir())
    pages = {}
    for d in documents(run):
        doc = canonical_doc(d.get("document", {}).get("file_name"))
        for pg, bl in page_blocks(d).items():
            pages[unit_id(SUB, doc, pg)] = "\n".join(b["text"] for b in bl if (b.get("text") or "").strip())

    bench = load("vidore_v3", subset=SUB, language=LANG)
    qrels = bench.qrels()
    questions = [q for q in bench.questions() if qrels.get(q.qid)]
    emb = OpenRouterEmbedder(cache_dir=ROOT / "data/work/vidore_physics_kdl_chunk_emb", batch_size=64)
    qv = norm(np.asarray(emb.embed([q.query for q in questions]), dtype=np.float32))

    recs, owner = [], []
    for u, t in pages.items():
        if not t.strip():
            continue
        for sp in fixed_overlap(t, n_words=512, overlap=128):
            seg = t[sp[0]:sp[1]]
            if seg.strip():
                recs.append(seg)
                owner.append(u)
    bm = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": f"c{i}", "doc_id": owner[i], "text": recs[i]} for i in range(len(recs))])
    M = norm(np.asarray(emb.embed(recs), dtype=np.float32))
    owner = np.array(owner)

    def maxp(scored):
        best = defaultdict(lambda: -1e9)
        for pos, s in scored:
            best[owner[pos]] = max(best[owner[pos]], s)
        return sorted(best.items(), key=lambda kv: -kv[1])

    pool = {}
    for q, v in zip(questions, qv):
        lex = bm.search(q.query, 1000)
        ds = M @ v
        top = np.argpartition(-ds, min(1000, len(ds) - 1))[:1000]
        dn = sorted(((int(i), float(ds[i])) for i in top), key=lambda p: -p[1])
        lp, dp = maxp(lex), maxp(dn)
        uids = list(dict.fromkeys([u for u, _ in lp] + [u for u, _ in dp]))
        idx = {u: i for i, u in enumerate(uids)}
        pool[q.qid] = {uids[p]: float(s) for p, s in alpha_fuse(
            [(idx[u], s) for u, s in lp], [(idx[u], s) for u, s in dp], ALPHA, POOL)}
    return qrels, {q.qid: q.query for q in questions}, pool


def rerank_call(key, query, docs, depth, retries=5):
    for attempt in range(retries):
        r = requests.post(URL, headers={"Authorization": f"Bearer {key}"},
                          json={"model": MODEL, "query": query,
                                "documents": [d[:MAX_CHARS] for d in docs], "top_n": len(docs)},
                          timeout=90)
        if r.status_code == 200:
            return {item["index"]: float(item["relevance_score"]) for item in r.json()["results"]}
        if r.status_code in (429, 502, 503):
            wait = 2 ** attempt * 3
            print(f"    {r.status_code}, wait {wait}s", flush=True)
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError("exhausted retries")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--pace", type=float, default=1.4, help="seconds between calls")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    qrels, qtext, pool = kdl_prod_pool()
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_10"})
    page_texts = json.loads(TEXTS.read_text())

    def rep(run):
        s = ev.evaluate(run)
        n = len(s)
        return (100 * sum(v["ndcg_cut_10"] for v in s.values()) / n,
                100 * sum(v["recall_10"] for v in s.values()) / n,
                {q: 100 * v["ndcg_cut_10"] for q, v in s.items()})

    nb, rb, pqb = rep({q: dict(s) for q, s in pool.items()})
    print(f"KDL alpha0.7 baseline: NDCG@10 {nb:.2f}  R@10 {rb:.2f}   (302 q, depth {args.depth})")
    if args.dry:
        return

    key = os.environ["OPENROUTER_API_KEY"]
    scores = json.loads(CKPT.read_text()) if CKPT.exists() else {}
    print(f"resuming {len(scores)}/{len(pool)}" if scores else "fresh run")
    for i, (qid, cand_scores) in enumerate(pool.items(), 1):
        if qid in scores:
            continue
        head = sorted(cand_scores, key=lambda c: -cand_scores[c])[:args.depth]
        docs = [page_texts.get(c, "") for c in head]
        t = time.time()
        ranked = rerank_call(key, qtext[qid], docs, args.depth)
        scores[qid] = {head[idx]: sc for idx, sc in ranked.items()}
        CKPT.write_text(json.dumps(scores))
        if i % 20 == 0:
            print(f"  {i}/{len(pool)}  ({time.time() - t:.1f}s last)", flush=True)
        time.sleep(args.pace)

    # ---- score every arm on the identical pool ----------------------------
    matrix = np.load(COLQWEN / "physics_colqwen_scores.npy")
    ckeys = json.loads((COLQWEN / "physics_colqwen_keys.json").read_text())
    cqids = json.loads((COLQWEN / "physics_colqwen_qids.json").read_text())
    kidx = {k: idx for idx, k in enumerate(ckeys)}
    qrow = {q: idx for idx, q in enumerate(cqids)}
    vis = {q: {c: float(matrix[qrow[q], kidx[c]]) for c in pool[q] if c in kidx}
           for q in pool if q in qrow}

    def fuse(text, v, w):
        t, vv = minmax(text), minmax(v)
        out = {u: (1 - w) * s + w * vv.get(u, 0.0) for u, s in t.items()}
        for u, s in vv.items():
            out.setdefault(u, w * s)
        return out

    def apply_rerank(order_scores):
        """dict cand->score -> reranked order using cached Nemotron scores on the top-`depth`."""
        out = {}
        for q, sc in order_scores.items():
            order = sorted(sc, key=lambda c: -sc[c])
            rr = scores.get(q, {})
            head = sorted((c for c in order[:args.depth] if c in rr), key=lambda c: -rr[c])
            head += [c for c in order[:args.depth] if c not in rr]
            tail = order[args.depth:]
            out[q] = {c: float(len(order) - i) for i, c in enumerate(head + tail)}
        return out

    rows = []

    def emit(name, run, ref_pq, ref):
        n, r, pq = rep(run)
        d, p, b, w, t = permutation(ref_pq, pq)
        print(f"{name:34s} {n:7.2f} {r:7.2f} {d:+7.2f} {p:9.4f} vs {ref}{'' if p < 0.05 else '  n.s.'}")
        rows.append(dict(arm=name, ndcg10=round(n, 2), recall10=round(r, 2),
                         delta=round(d, 3), p=round(p, 5), vs=ref))
        return pq

    print(f"\n{'arm':34s} {'NDCG@10':>7s} {'R@10':>7s} {'Δ':>7s} {'p':>9s}")
    print(f"{'baseline KDL α=0.7':34s} {nb:7.2f} {rb:7.2f}")

    sep = {q: propagate(dict(pool[q]), SEP_LAMBDA, W, GAMMA, BETA, TOPM) for q in pool}
    sepvis = {q: fuse(sep[q], vis[q], WV) for q in pool if q in vis}

    nem_pq = emit("+ Nemotron rerank", apply_rerank({q: dict(pool[q]) for q in pool}), pqb, "baseline")
    emit("+ SEP → Nemotron rerank", apply_rerank(sep), pqb, "baseline")
    emit("+ SEP+ColQwen2 → Nemotron", apply_rerank(sepvis), pqb, "baseline")
    # rerank first, then SEP+ColQwen2 on the reranked order (rank-proxy scores)
    rr_scores = apply_rerank({q: dict(pool[q]) for q in pool})
    rr_sep = {q: propagate(rr_scores[q], SEP_LAMBDA, W, GAMMA, BETA, TOPM) for q in pool}
    rr_sepvis = {q: fuse(rr_sep[q], vis[q], WV) for q in pool if q in vis}
    emit("+ Nemotron → SEP+ColQwen2", rr_sepvis, nem_pq, "Nemotron")

    (RESULTS / "physics_rerank_nemotron.json").write_text(json.dumps(
        dict(model=MODEL, depth=args.depth, baseline=round(nb, 2), rows=rows), indent=2))
    print(f"\n-> {RESULTS / 'physics_rerank_nemotron.json'}")


if __name__ == "__main__":
    main()
