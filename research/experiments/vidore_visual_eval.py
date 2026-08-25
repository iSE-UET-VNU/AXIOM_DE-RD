"""Score the visual arm alone, and fused with the text arm.

The question is not whether CLIP beats text -- at 224px on French physics pages
it will not. It is whether the visual signal is COMPLEMENTARY: does adding it to
the text fusion recover anything text misses? A weak-but-complementary signal
justifies a document-VLM; a weak-and-redundant one does not.
"""
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytrec_eval

from src.utils.env import load_dotenv_file

load_dotenv_file(ROOT)

from research.experiments.physics_sep import propagate
from research.experiments.physics_sep_test import BETA, GAMMA, TOPM, W, permutation
from src.chunking_embedding.chunkers.builtin import fixed_overlap
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks
from src.evaluation.retrieval import alpha_fuse
from src.retrieval.sparse import BM25Index

POOL = 100
norm = lambda m: m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)

data = np.load(ROOT / "data/work/vidore_physics_clip.npz", allow_pickle=True)
img, qv_img = data["image_vectors"], data["query_vectors"]
keys, qids = list(data["keys"]), list(data["qids"])

bench = load("vidore_v3", subset="physics", language="french")
qrels = bench.qrels()
questions = [q for q in bench.questions() if qrels.get(q.qid)]
assert [q.qid for q in questions] == qids, "query order drifted"
ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_10"})


def report(run):
    s = ev.evaluate(run)
    return (100 * sum(v["ndcg_cut_10"] for v in s.values()) / len(s),
            100 * sum(v["recall_10"] for v in s.values()) / len(s),
            {k: 100 * v["ndcg_cut_10"] for k, v in s.items()})


# --- text arm (production config) ---
run_dir = next((ROOT / "data_vidore_parsed_physics/output/benchmarks/vidore-v3-physics-kdl").iterdir())
pages = {}
for document in documents(run_dir):
    doc = canonical_doc(document.get("document", {}).get("file_name"))
    for page, blocks in page_blocks(document).items():
        pages[unit_id("physics", doc, page)] = "\n".join(
            b["text"] for b in blocks if (b.get("text") or "").strip())
emb = OpenRouterEmbedder(cache_dir=ROOT / "data/work/vidore_physics_kdl_chunk_emb", batch_size=64)
qv_txt = norm(np.asarray(emb.embed([q.query for q in questions]), dtype=np.float32))
recs, owner = [], []
for u, t in pages.items():
    if not t.strip():
        continue
    for sp in fixed_overlap(t, n_words=512, overlap=128):
        seg = t[sp[0]:sp[1]]
        if seg.strip():
            recs.append(seg); owner.append(u)
M = norm(np.asarray(emb.embed(recs), dtype=np.float32))
owner = np.array(owner)
bm = BM25Index(analyzer_name="plain").build(
    [{"chunk_id": f"c{i}", "doc_id": owner[i], "text": recs[i]} for i in range(len(recs))])


def maxp(scored):
    best = defaultdict(lambda: -1e9)
    for p, s in scored:
        best[owner[p]] = max(best[owner[p]], s)
    return sorted(best.items(), key=lambda kv: -kv[1])


text_pool, vis_only, fused_pool = {}, {}, {}
for i, (q, vt) in enumerate(zip(questions, qv_txt)):
    lex = bm.search(q.query, 1000)
    ds = M @ vt
    top = np.argpartition(-ds, min(1000, len(ds) - 1))[:1000]
    dn = sorted(((int(j), float(ds[j])) for j in top), key=lambda p: -p[1])
    lp, dp = maxp(list(lex)), maxp(dn)
    uids = list(dict.fromkeys([u for u, _ in lp] + [u for u, _ in dp]))
    idx = {u: k for k, u in enumerate(uids)}
    text_pool[q.qid] = {uids[p]: float(s) for p, s in alpha_fuse(
        [(idx[u], s) for u, s in lp], [(idx[u], s) for u, s in dp], 0.7, POOL)}
    vs = img @ qv_img[i]
    order = np.argsort(-vs)[:POOL]
    vis_only[q.qid] = {keys[j]: float(vs[j]) for j in order}

nt, rt, tpq = report(text_pool)
nv, rv, vpq = report(vis_only)
print(f"{'arm':38s} {'NDCG@10':>8s} {'R@10':>7s}")
print(f"{'text alpha=0.7 (production)':38s} {nt:8.2f} {rt:7.2f}")
print(f"{'visual only (CLIP ViT-B/32, 224px)':38s} {nv:8.2f} {rv:7.2f}")

# complementarity: does visual find gold the text arm misses?
gold_of = lambda qid: {u for u, v in qrels[qid].items() if v > 0}
only_v = only_t = 0
for q in questions:
    g = gold_of(q.qid)
    t10 = set(list(text_pool[q.qid])[:10])
    v10 = set(list(vis_only[q.qid])[:10])
    only_v += len(g & (v10 - t10)) / len(g)
    only_t += len(g & (t10 - v10)) / len(g)
n = len(questions)
print(f"\ngold@10 found ONLY by visual: {100*only_v/n:.2f}%   ONLY by text: {100*only_t/n:.2f}%")

print(f"\n{'text + visual fusion':38s} {'NDCG@10':>8s} {'vs text':>8s} {'p':>8s}")
for wv in (0.1, 0.2, 0.3, 0.5):
    out = {}
    for i, q in enumerate(questions):
        vs = img @ qv_img[i]
        vmap = {keys[j]: float(vs[j]) for j in range(len(keys))}
        tp = text_pool[q.qid]
        lo, hi = min(tp.values()), max(tp.values())
        span = max(hi - lo, 1e-12)
        vv = [vmap.get(u, 0.0) for u in tp]
        vlo, vhi = min(vv), max(vv)
        vspan = max(vhi - vlo, 1e-12)
        out[q.qid] = {u: (1 - wv) * ((s - lo) / span)
                          + wv * ((vmap.get(u, 0.0) - vlo) / vspan) for u, s in tp.items()}
    nf, rf, fpq = report(out)
    d, p, *_ = permutation(tpq, fpq, 10000)
    print(f"{'  w_visual=' + str(wv):38s} {nf:8.2f} {nf-nt:+8.2f} {p:8.4f}"
          f"{'' if p < 0.05 else '  n.s.'}")
