"""Re-measure the verified retrieval levers on the pools our pipeline actually
ships: KDL and light-preparation, both chunked fixed_512/128 -> MaxP to pages ->
hybrid alpha=0.7 (the CSV 'Baseline Legacy' recipe).

The ledger's Voyage (+5.08) and ColQwen2 (+3.22) rows were measured on ViDoRe
V3's *own* supplied page text (`vidore_page`), which uses none of our parsing.
This script puts SEP and ColQwen2 on the KDL / light-prep pools so the results
table has like-for-like rows. Everything is cached -- no API, no GPU:

  - KDL parse        data_vidore_parsed_physics/output/benchmarks/vidore-v3-physics-kdl
  - light prep       data/work/discovery_physics_lightprep
  - chunk embeddings data/work/vidore_physics_{kdl_chunk,lightprep}_emb  (OpenRouter cache)
  - ColQwen2 matrix  data/work/vidore_physics_colqwen/*.npy

Voyage rerank on these pools needs a fresh ~100 min API run (free-tier capped)
and is NOT covered here -- dump `pool` to JSON and feed physics_rerank_voyage.py
--pool/--texts when budget allows.

    python research/experiments/physics_kdl_arms.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytrec_eval

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
SEP_LAMBDA = 0.5
WEIGHTS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
COLQWEN = ROOT / "data/work/vidore_physics_colqwen"
RESULTS = ROOT / "data/benchmark/vidore_v3/results"
norm = lambda m: m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)


def kdl_pages() -> dict[str, str]:
    run = next((ROOT / f"data_vidore_parsed_physics/output/benchmarks/vidore-v3-{SUB}-kdl").iterdir())
    pages = {}
    for document in documents(run):
        doc = canonical_doc(document.get("document", {}).get("file_name"))
        for page, blocks in page_blocks(document).items():
            pages[unit_id(SUB, doc, page)] = "\n".join(
                b["text"] for b in blocks if (b.get("text") or "").strip())
    return pages


def lightprep_pages() -> dict[str, str]:
    from research.data_discovery.pipeline import PageIndex
    index = PageIndex.load(ROOT / "data/work/discovery_physics_lightprep")
    return {unit_id(SUB, canonical_doc(Path(p.file_path).name), p.page_number - 1): p.text
            for p in index.pages}


def build_pool(pages, questions, qv, ids, bm25, matrix):
    """fixed_512/128 chunks already folded into ids/bm25/matrix; alpha-fuse per query."""
    owner = np.array([o for o, _ in ids])
    seg_matrix = matrix

    def maxp(scored):
        best = defaultdict(lambda: -1e9)
        for pos, s in scored:
            best[owner[pos]] = max(best[owner[pos]], s)
        return sorted(best.items(), key=lambda kv: -kv[1])

    pool = {}
    for q, v in zip(questions, qv):
        lex = bm25.search(q.query, 1000)
        ds = seg_matrix @ v
        top = np.argpartition(-ds, min(1000, len(ds) - 1))[:1000]
        dn = sorted(((int(i), float(ds[i])) for i in top), key=lambda p: -p[1])
        lp, dp = maxp(lex), maxp(dn)
        uids = list(dict.fromkeys([u for u, _ in lp] + [u for u, _ in dp]))
        idx = {u: i for i, u in enumerate(uids)}
        fl = [(idx[u], s) for u, s in lp]
        fd = [(idx[u], s) for u, s in dp]
        pool[q.qid] = {uids[p]: float(s) for p, s in alpha_fuse(fl, fd, ALPHA, POOL)}
    return pool


def chunk_index(pages, cache):
    embedder = OpenRouterEmbedder(cache_dir=ROOT / f"data/work/{cache}", batch_size=64)
    recs, owner = [], []
    for u, t in pages.items():
        if not t.strip():
            continue
        for sp in fixed_overlap(t, n_words=512, overlap=128):
            seg = t[sp[0]:sp[1]]
            if seg.strip():
                recs.append(seg)
                owner.append(u)
    bm25 = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": f"c{i}", "doc_id": owner[i], "text": recs[i]} for i in range(len(recs))])
    matrix = norm(np.asarray(embedder.embed(recs), dtype=np.float32))
    return list(zip(owner, range(len(owner)))), bm25, matrix, len(pages), len(recs)


def main() -> None:
    bench = load("vidore_v3", subset=SUB, language=LANG)
    qrels = bench.qrels()
    questions = [q for q in bench.questions() if qrels.get(q.qid)]
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_10"})

    matrix = np.load(COLQWEN / "physics_colqwen_scores.npy")
    ckeys = json.loads((COLQWEN / "physics_colqwen_keys.json").read_text())
    cqids = json.loads((COLQWEN / "physics_colqwen_qids.json").read_text())
    kidx = {k: i for i, k in enumerate(ckeys)}
    qrow = {q: i for i, q in enumerate(cqids)}

    def rep(run):
        s = ev.evaluate(run)
        n = len(s)
        return (100 * sum(v["ndcg_cut_10"] for v in s.values()) / n,
                100 * sum(v["recall_10"] for v in s.values()) / n,
                {q: 100 * v["ndcg_cut_10"] for q, v in s.items()})

    def fuse(text, vis, w):
        t, v = minmax(text), minmax(vis)
        out = {u: (1 - w) * s + w * v.get(u, 0.0) for u, s in t.items()}
        for u, s in v.items():
            out.setdefault(u, w * s)
        return out

    report = {}
    for arm, loader, cache in (("KDL", kdl_pages, "vidore_physics_kdl_chunk_emb"),
                               ("light-prep", lightprep_pages, "vidore_physics_lightprep_emb")):
        pages = loader()
        embedder = OpenRouterEmbedder(cache_dir=ROOT / f"data/work/{cache}", batch_size=64)
        qv = norm(np.asarray(embedder.embed([q.query for q in questions]), dtype=np.float32))
        ids, bm25, seg_matrix, npages, nchunks = chunk_index(pages, cache)
        pool = build_pool(pages, questions, qv, ids, bm25, seg_matrix)

        nb, rb, pqb = rep({q: dict(s) for q, s in pool.items()})
        vis = {q: {c: float(matrix[qrow[q], kidx[c]]) for c in pool[q] if c in kidx}
               for q in pool if q in qrow}

        print(f"\n=== {arm}  ({npages} pages, {nchunks} chunks, {nchunks / npages:.2f}/page) ===")
        print(f"{'arm':32s} {'NDCG@10':>8s} {'R@10':>7s} {'Δ':>7s} {'p':>9s} vs")
        print(f"{'baseline α=0.7':32s} {nb:8.2f} {rb:7.2f}")
        rows = {"baseline": {"ndcg@10": round(nb, 2), "recall@10": round(rb, 2)}}

        def emit(name, run, ref_pq, ref):
            n, r, pq = rep(run)
            d, p, b, w, t = permutation(ref_pq, pq)
            print(f"{name:32s} {n:8.2f} {r:7.2f} {d:+7.2f} {p:9.4f} {ref}"
                  f"{'' if p < 0.05 else '  n.s.'}")
            rows[name] = {"ndcg@10": round(n, 2), "recall@10": round(r, 2),
                          "delta": round(d, 3), "p": round(p, 5), "vs": ref,
                          "better": b, "worse": w, "tied": t}
            return pq

        sep_scores = {q: propagate(dict(pool[q]), SEP_LAMBDA, W, GAMMA, BETA, TOPM) for q in pool}
        sep_pq = emit(f"+ SEP (λ={SEP_LAMBDA})", sep_scores, pqb, "baseline")

        for w in WEIGHTS:
            run = {q: fuse(dict(pool[q]), vis[q], w) for q in pool if q in vis}
            emit(f"+ ColQwen2 fusion w={w}", run, pqb, "baseline")
        for w in WEIGHTS:
            run = {q: fuse(sep_scores[q], vis[q], w) for q in sep_scores if q in vis}
            emit(f"+ SEP + ColQwen2 w={w}", run, sep_pq, "SEP")

        report[arm] = {"pages": npages, "chunks": nchunks, "rows": rows}
        qtext = {q.qid: q.query for q in questions}
        dump = {"index": f"{arm}_page", "arm": "alpha0.7", "depth": POOL, "queries": {
            q: {"query": qtext[q],
                "candidates": sorted(s, key=lambda c: -s[c])[:POOL]}
            for q, s in pool.items()}}
        (RESULTS / f"physics_{arm.replace('-', '_')}_pool.json").write_text(
            json.dumps(dump), encoding="utf-8")

    (RESULTS / "physics_kdl_arms.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {RESULTS / 'physics_kdl_arms.json'}")


if __name__ == "__main__":
    main()
