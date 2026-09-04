"""Full slate of KDL-pool arms for the results CSV -- every combination of
SEP / visual fusion / Nemotron rerank on the 'Baseline Legacy' pool, all from
cache (no API, no GPU: Nemotron scores are the checkpoint from
physics_rerank_nemotron.py; the visual matrix is exported from a Colab notebook).

    python research/experiments/physics_kdl_slate.py                       # ColQwen2
    python research/experiments/physics_kdl_slate.py \\
        --visual-dir data/work/vidore_physics_colvec --visual-name colvec  # webAI-ColVec
"""
import argparse
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
LAM, WV, DEPTH = 0.5, 0.7, 20
RESULTS = ROOT / "data/benchmark/vidore_v3/results"
NEM = RESULTS / "physics_rerank_nemotron_scores.json"
norm = lambda m: m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)

# "Baseline Legacy" in the results CSV = pdf-inspector + KDL layout. "kdl" is the
# plain-KDL parse §21-§24 were measured on; kept so those numbers stay reproducible.
PARSE_RUNS = {
    "pdf-inspector": "vidore-v3-physics-kdl-pdf-inspector",
    "kdl": "vidore-v3-physics-kdl",
}


def kdl_pool(parse: str = "pdf-inspector"):
    run = next((ROOT / f"data_vidore_parsed_physics/output/benchmarks/{PARSE_RUNS[parse]}").iterdir())
    pages = {}
    for d in documents(run):
        doc = canonical_doc(d.get("document", {}).get("file_name"))
        for pg, bl in page_blocks(d).items():
            pages[unit_id(SUB, doc, pg)] = "\n".join(b["text"] for b in bl if (b.get("text") or "").strip())
    bench = load("vidore_v3", subset=SUB, language=LANG)
    qrels = bench.qrels()
    qs = [q for q in bench.questions() if qrels.get(q.qid)]
    emb = OpenRouterEmbedder(cache_dir=ROOT / f"data/work/vidore_physics_{parse.replace('-', '_')}_chunk_emb", batch_size=64)
    qv = norm(np.asarray(emb.embed([q.query for q in qs]), dtype=np.float32))
    recs, owner = [], []
    for u, t in pages.items():
        if not t.strip():
            continue
        for sp in fixed_overlap(t, n_words=512, overlap=128):
            seg = t[sp[0]:sp[1]]
            if seg.strip():
                recs.append(seg); owner.append(u)
    bm = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": f"c{i}", "doc_id": owner[i], "text": recs[i]} for i in range(len(recs))])
    M = norm(np.asarray(emb.embed(recs), dtype=np.float32))
    owner = np.array(owner)

    def maxp(scored):
        best = defaultdict(lambda: -1e9)
        for p, s in scored:
            best[owner[p]] = max(best[owner[p]], s)
        return sorted(best.items(), key=lambda kv: -kv[1])

    pool = {}
    for q, v in zip(qs, qv):
        lex = bm.search(q.query, 1000)
        ds = M @ v
        top = np.argpartition(-ds, min(1000, len(ds) - 1))[:1000]
        dn = sorted(((int(i), float(ds[i])) for i in top), key=lambda p: -p[1])
        lp, dp = maxp(lex), maxp(dn)
        uids = list(dict.fromkeys([u for u, _ in lp] + [u for u, _ in dp]))
        idx = {u: i for i, u in enumerate(uids)}
        pool[q.qid] = {uids[p]: float(s) for p, s in alpha_fuse(
            [(idx[u], s) for u, s in lp], [(idx[u], s) for u, s in dp], ALPHA, POOL)}
    return qrels, pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visual-dir", default="data/work/vidore_physics_colqwen")
    ap.add_argument("--visual-name", default="colqwen", help="label + file prefix: physics_<name>_{scores.npy,keys.json,qids.json}")
    ap.add_argument("--wv", type=float, default=WV, help="visual fusion weight")
    ap.add_argument("--parse", default="pdf-inspector", choices=list(PARSE_RUNS),
                    help="parse run: pdf-inspector = 'Baseline Legacy' (default); kdl = the §21-§24 numbers")
    args = ap.parse_args()

    qrels, pool = kdl_pool(args.parse)
    # ndcg_cut_5 is the ViDoRe V3 leaderboard metric; _10 is this ladder's internal
    # working number. Report both -- see ledger §26.
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_5", "ndcg_cut_10", "recall_10"})
    nem = json.loads(NEM.read_text())

    vdir = ROOT / args.visual_dir
    prefix = f"physics_{args.visual_name}" if args.visual_name != "colqwen" else "physics_colqwen"
    matrix = np.load(vdir / f"{prefix}_scores.npy")
    ck = json.loads((vdir / f"{prefix}_keys.json").read_text())
    cq = json.loads((vdir / f"{prefix}_qids.json").read_text())
    ki = {k: i for i, k in enumerate(ck)}
    ri = {q: i for i, q in enumerate(cq)}
    vis = {q: {c: float(matrix[ri[q], ki[c]]) for c in pool[q] if c in ki} for q in pool if q in ri}
    VNAME = args.visual_name
    wv = args.wv

    def rep(run):
        s = ev.evaluate(run)
        n = len(s)
        return (100 * sum(v["ndcg_cut_10"] for v in s.values()) / n,
                100 * sum(v["recall_10"] for v in s.values()) / n,
                {q: 100 * v["ndcg_cut_10"] for q, v in s.items()},
                100 * sum(v["ndcg_cut_5"] for v in s.values()) / n)

    def as_run(scores):
        return {q: dict(s) for q, s in scores.items()}

    def fuse(text, w):
        out = {}
        for q, ts in text.items():
            if q not in vis:
                out[q] = dict(ts)
                continue
            t, vv = minmax(ts), minmax(vis[q])
            o = {u: (1 - w) * s + w * vv.get(u, 0.0) for u, s in t.items()}
            for u, s in vv.items():
                o.setdefault(u, w * s)
            out[q] = o
        return out

    def sep(text):
        return {q: propagate(dict(s), LAM, W, GAMMA, BETA, TOPM) for q, s in text.items()}

    def rerank(text):
        out = {}
        for q, s in text.items():
            order = sorted(s, key=lambda c: -s[c])
            rr = nem.get(q, {})
            head = sorted((c for c in order[:DEPTH] if c in rr), key=lambda c: -rr[c])
            head += [c for c in order[:DEPTH] if c not in rr]
            full = head + order[DEPTH:]
            out[q] = {c: float(len(full) - i) for i, c in enumerate(full)}
        return out

    V = VNAME  # visual arm label in the printed/saved rows
    base = as_run(pool)
    nb, rb, pqb, nb5 = rep(base)
    nem_pq = rep(rerank(pool))[2]

    arms = [
        ("baseline KDL α=0.7", base, "baseline"),
        ("+ SEP", sep(pool), "baseline"),
        (f"+ {V} fusion", fuse(pool, wv), "baseline"),
        (f"+ {V} only (pool-reranked)", {q: dict(vis[q]) for q in vis}, "baseline"),
        ("+ Nemotron rerank", rerank(pool), "baseline"),
        (f"+ SEP + {V}", fuse(sep(pool), wv), "baseline"),
        (f"+ {V} → Nemotron", rerank(fuse(pool, wv)), "baseline"),
        (f"+ Nemotron → {V}", fuse(rerank(pool), wv), "Nemotron"),
        ("+ SEP → Nemotron", rerank(sep(pool)), "baseline"),
        (f"+ SEP+{V} → Nemotron", rerank(fuse(sep(pool), wv)), "baseline"),
        (f"+ Nemotron → SEP+{V}", fuse(sep(rerank(pool)), wv), "Nemotron"),
    ]

    print(f"parse: {args.parse}   visual arm: {V}  (fusion weight wv={wv})")
    print("NDCG@5 = ViDoRe V3 leaderboard metric; @10 = internal working number (ledger §26)\n")
    print(f"{'arm':32s} {'NDCG@5':>7s} {'NDCG@10':>8s} {'R@10':>7s} {'Δ@10':>7s} {'p':>9s}  vs")
    print(f"{'baseline KDL α=0.7':32s} {nb5:7.2f} {nb:8.2f} {rb:7.2f}")
    out = [{"arm": "baseline", "ndcg5": round(nb5, 2), "ndcg10": round(nb, 2), "recall10": round(rb, 2)}]
    for name, run, ref in arms[1:]:
        n, r, pq, n5 = rep(run)
        refpq = pqb if ref == "baseline" else nem_pq
        d, p, b, w, t = permutation(refpq, pq)
        flag = "" if p < 0.05 else "  n.s."
        print(f"{name:32s} {n5:7.2f} {n:8.2f} {r:7.2f} {d:+7.2f} {p:9.4f}  {ref}{flag}")
        out.append({"arm": name, "ndcg5": round(n5, 2), "ndcg10": round(n, 2), "recall10": round(r, 2),
                    "delta": round(d, 3), "p": round(p, 5), "vs": ref})
    tag = ("" if V == "colqwen" else f"_{V}") + ("" if args.parse == "pdf-inspector" else f"_{args.parse}")
    (RESULTS / f"physics_kdl_slate{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {RESULTS / f'physics_kdl_slate{tag}.json'}")


if __name__ == "__main__":
    main()
