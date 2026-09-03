"""LambdaMART learned re-ranking of the KDL pool -- the offline form of Metarank.

Metarank (github.com/metarank/metarank) is a LightGBM LambdaMART ranker plus a
feature store and an HTTP serving layer. With no click stream, the benchmark
reduction is the model itself: train LGBMRanker on per-(query, page) features of
the KDL Baseline Legacy pool and see whether a learned per-query blend of the
text / structural / visual signals beats the fixed-weight stack (SEP + ColVec =
52.91) and closes any of the +8.60 oracle-alpha headroom (ledger S12).

All features are cached: KDL chunk embeddings, SEP propagate(), the ColVec
MaxSim matrix. Evaluation is 5-fold GroupKFold by query -- only held-out folds
are scored, so the number is not fit on its own test set.

    python research/experiments/physics_ltr.py
    python research/experiments/physics_ltr.py --visual-name colvec --folds 5
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # lightgbm libomp vs mkl libiomp5 on macOS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytrec_eval
from sklearn.model_selection import GroupKFold

from src.utils.env import load_dotenv_file

load_dotenv_file(ROOT)

from research.experiments.physics_sep import minmax, propagate, split
from research.experiments.physics_sep_test import BETA, GAMMA, TOPM, W
from src.chunking_embedding.chunkers.builtin import fixed_overlap
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks
from src.evaluation.retrieval import alpha_fuse
from src.retrieval.sparse import BM25Index

SUB, LANG, ALPHA, POOL = "physics", "french", 0.7, 100
LAM, SEP_W = 0.5, 2
RESULTS = ROOT / "data/benchmark/vidore_v3/results"
PARSE_RUNS = {"pdf-inspector": "vidore-v3-physics-kdl-pdf-inspector", "kdl": "vidore-v3-physics-kdl"}
norm = lambda m: m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)


def kdl_pool_scored(parse: str = "pdf-inspector"):
    """KDL alpha0.7 pool, keeping the bm25 / dense / fused component per candidate."""
    run = next((ROOT / f"data_vidore_parsed_physics/output/benchmarks/{PARSE_RUNS[parse]}").iterdir())
    pages = {}
    for d in documents(run):
        doc = canonical_doc(d.get("document", {}).get("file_name"))
        for pg, bl in page_blocks(d).items():
            pages[unit_id(SUB, doc, pg)] = "\n".join(b["text"] for b in bl if (b.get("text") or "").strip())

    bench = load("vidore_v3", subset=SUB, language=LANG)
    qrels = bench.qrels()
    qs = [q for q in bench.questions() if qrels.get(q.qid)]
    emb = OpenRouterEmbedder(
        cache_dir=ROOT / f"data/work/vidore_physics_{parse.replace('-', '_')}_chunk_emb", batch_size=64)
    qv = norm(np.asarray(emb.embed([q.query for q in qs]), dtype=np.float32))

    recs, owner = [], []
    for u, t in pages.items():
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
        for p, s in scored:
            best[owner[p]] = max(best[owner[p]], s)
        return best

    pool = {}
    for q, v in zip(qs, qv):
        lex_pairs = bm.search(q.query, 1000)
        ds = M @ v
        top = np.argpartition(-ds, min(1000, len(ds) - 1))[:1000]
        dn = [(int(i), float(ds[i])) for i in top]
        lp, dp = maxp(lex_pairs), maxp(dn)
        uids = list(dict.fromkeys(list(lp) + list(dp)))
        idx = {u: i for i, u in enumerate(uids)}
        fused = dict(alpha_fuse([(idx[u], s) for u, s in lp.items()],
                                [(idx[u], s) for u, s in dp.items()], ALPHA, POOL))
        cand = {}
        for i, u in enumerate(uids):
            if i in fused:
                cand[u] = {"fused": float(fused[i]),
                           "bm25": float(lp.get(u, 0.0)),
                           "dense": float(dp.get(u, -1.0))}
        pool[q.qid] = {"query": q.query, "cand": cand}
    return qrels, pool


def features(qrels, pool, vis):
    X, y, groups, meta = [], [], [], []
    for qid, entry in pool.items():
        cand = entry["cand"]
        if not cand:
            continue
        fused = {u: c["fused"] for u, c in cand.items()}
        bm25 = minmax({u: c["bm25"] for u, c in cand.items()})
        dense = minmax({u: c["dense"] for u, c in cand.items()})
        fz = minmax(fused)
        sep = propagate(dict(fused), LAM, W, GAMMA, BETA, TOPM)
        sepz = minmax(sep)
        vv = minmax({u: vis[qid][u] for u in cand if qid in vis and u in vis.get(qid, {})})
        stack = {u: 0.2 * sepz.get(u, 0.0) + 0.8 * vv.get(u, 0.0) for u in cand}  # SEP+ColVec wv=0.8
        order_f = sorted(cand, key=lambda u: -fused[u])
        rank_f = {u: i for i, u in enumerate(order_f)}
        order_v = sorted(vv, key=lambda u: -vv[u]) if vv else []
        rank_v = {u: i for i, u in enumerate(order_v)}
        rank_s = {u: i for i, u in enumerate(sorted(cand, key=lambda u: -stack[u]))}
        top_fused = max(fused.values())

        by_file = defaultdict(dict)
        for u in cand:
            p = split(u)
            if p:
                by_file[p[0]][p[1]] = fz[u]
        file_count = {f: len(pg) for f, pg in by_file.items()}
        file_agg = {}
        neigh = {}
        for u in cand:
            p = split(u)
            if p is None:
                file_agg[u] = neigh[u] = 0.0
                continue
            pgs = by_file[p[0]]
            tops = sorted(pgs.values(), reverse=True)[:3]
            file_agg[u] = sum(tops) / len(tops)
            neigh[u] = sum((GAMMA ** abs(d)) * pgs[p[1] + d]
                           for d in range(-SEP_W, SEP_W + 1) if d and (p[1] + d) in pgs)
        neigh = minmax(neigh)
        qwords = len(entry["query"].split())

        for u, c in cand.items():
            p = split(u)
            X.append([
                stack[u], rank_s[u],
                fz[u], bm25.get(u, 0.0), dense.get(u, 0.0),
                sepz[u], vv.get(u, 0.0),
                vv.get(u, 0.0) - fz[u],
                rank_f[u], rank_v.get(u, len(cand)),
                top_fused - fused[u],
                file_agg[u], neigh[u], file_count.get(p[0] if p else "", 1),
                p[1] if p else 0, qwords, len(cand),
            ])
            y.append(qrels.get(qid, {}).get(u, 0))
            groups.append(qid)
            meta.append((qid, u))
    return np.array(X, dtype=np.float32), np.array(y), np.array(groups), meta


FEAT_NAMES = ["stack", "rank_stack", "fused", "bm25", "dense", "sep", "colvec",
              "colvec_minus_fused", "rank_fused", "rank_colvec", "fused_gap_to_top",
              "file_agg", "neighbour", "file_count", "page_num", "query_words", "pool_size"]


def ndcg10(run, qrels):
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in run}, {"ndcg_cut_10", "recall_10"})
    s = ev.evaluate(run)
    n = len(s)
    return (100 * sum(v["ndcg_cut_10"] for v in s.values()) / n,
            100 * sum(v["recall_10"] for v in s.values()) / n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visual-dir", default="data/work/vidore_physics_colvec")
    ap.add_argument("--visual-name", default="colvec")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parse", default="pdf-inspector", choices=list(PARSE_RUNS))
    args = ap.parse_args()

    qrels, pool = kdl_pool_scored(args.parse)

    vdir = ROOT / args.visual_dir
    prefix = f"physics_{args.visual_name}"
    matrix = np.load(vdir / f"{prefix}_scores.npy")
    ck = json.loads((vdir / f"{prefix}_keys.json").read_text())
    cq = json.loads((vdir / f"{prefix}_qids.json").read_text())
    ki = {k: i for i, k in enumerate(ck)}
    ri = {q: i for i, q in enumerate(cq)}
    vis = {q: {c: float(matrix[ri[q], ki[c]]) for c in pool[q]["cand"] if c in ki}
           for q in pool if q in ri}

    X, y, groups, meta = features(qrels, pool, vis)
    uniq = sorted(set(groups))
    print(f"{len(uniq)} queries, {len(X)} (query,page) rows, {X.shape[1]} features\n")

    base_run = {q: {u: pool[q]["cand"][u]["fused"] for u in pool[q]["cand"]} for q in uniq}
    print(f"baseline fused α=0.7       {ndcg10(base_run, qrels)[0]:6.2f}  {ndcg10(base_run, qrels)[1]:.2f}")

    import lightgbm as lgb

    gkf = GroupKFold(n_splits=args.folds)
    oof = {}
    importances = np.zeros(X.shape[1])
    for tr, te in gkf.split(X, y, groups):
        gtr = groups[tr]
        order = np.argsort(gtr, kind="stable")
        tr = tr[order]
        _, cnt = np.unique(groups[tr], return_counts=True)
        model = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", n_estimators=300, learning_rate=0.05,
            num_leaves=15, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            label_gain=[0, 1, 3], random_state=args.seed, n_jobs=-1, verbosity=-1)
        model.fit(X[tr], y[tr], group=cnt)
        importances += model.feature_importances_
        pred = model.predict(X[te])
        for (qid, u), s in zip((meta[i] for i in te), pred):
            oof.setdefault(qid, {})[u] = float(s)

    n, r = ndcg10(oof, qrels)
    print(f"LambdaMART (5-fold OOF)    {n:6.2f}  {r:.2f}")
    print(f"  ref: SEP+ColVec fixed     {'53.15' if args.parse == 'pdf-inspector' else '52.91'}  (physics_kdl_slate)\n")

    imp = sorted(zip(FEAT_NAMES, importances / args.folds), key=lambda t: -t[1])
    print("feature importance (mean split gain count):")
    for name, val in imp:
        print(f"  {name:20s} {val:8.1f}")

    out = RESULTS / f"physics_ltr_{args.visual_name}.json"
    out.write_text(json.dumps({
        "baseline": round(ndcg10(base_run, qrels)[0], 2),
        "lambdamart_oof_ndcg10": round(n, 2), "lambdamart_oof_recall10": round(r, 2),
        "folds": args.folds, "features": FEAT_NAMES,
        "importance": {k: round(v, 1) for k, v in imp},
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
