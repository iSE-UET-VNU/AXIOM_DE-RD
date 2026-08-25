"""Fill out the results table: SEP standalone vs SEP on the light-preparation path.

Two preparation arms over the same 42 physics PDFs and the same gold:

  KDL          the accurate parse (GPU, vLLM) -- what the pipeline ships
  light prep   pdf-inspector native text only (CPU, ~140 pages/s), the
               "light preparation" stage of the on-demand branch

Each is scored with bm25 / dense / alpha=0.7, then with SEP applied to the
alpha=0.7 pool. SEP config is unchanged from physics (w=2, gamma=0.5,
beta=0.75, top-m=3); lambda is reported as a curve rather than tuned.

Note pdf-inspector page numbers are 1-based while KDL and the ViDoRe gold are
0-based; the -1 offset here is verified by exact unit-id overlap with the KDL
parse (1674/1674), not assumed.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytrec_eval

from src.utils.env import load_dotenv_file

load_dotenv_file(ROOT)

from research.data_discovery.pipeline import PageIndex
from research.experiments.physics_sep import ndcg, propagate
from research.experiments.physics_sep_test import BETA, GAMMA, TOPM, W, permutation
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks
from src.evaluation.retrieval import alpha_fuse
from src.retrieval.sparse import BM25Index

SUB, LANG, ALPHA, POOL = "physics", "french", 0.7, 100
LAMBDAS = (0.5, 0.6, 0.7)
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
    index = PageIndex.load(ROOT / "data/work/discovery_physics_lightprep")
    return {unit_id(SUB, canonical_doc(Path(p.file_path).name), p.page_number - 1): p.text
            for p in index.pages}


def main() -> None:
    bench = load("vidore_v3", subset=SUB, language=LANG)
    qrels = bench.qrels()
    questions = [q for q in bench.questions() if qrels.get(q.qid)]
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_10"})

    report = {}
    for arm, loader, cache in (("KDL", kdl_pages, "vidore_physics_kdl_emb"),
                               ("light prep", lightprep_pages, "vidore_physics_lightprep_emb")):
        pages = loader()
        embedder = OpenRouterEmbedder(cache_dir=ROOT / f"data/work/{cache}", batch_size=64)
        qv = norm(np.asarray(embedder.embed([q.query for q in questions]), dtype=np.float32))
        ids = [u for u, t in pages.items() if t.strip()]
        bm25 = BM25Index(analyzer_name="plain").build(
            [{"chunk_id": u, "doc_id": u, "text": pages[u]} for u in ids])
        matrix = norm(np.asarray(embedder.embed([pages[u] for u in ids]), dtype=np.float32))

        legs = {"bm25": {}, "dense": {}, "alpha0.7": {}}
        for question, vector in zip(questions, qv):
            lexical = bm25.search(question.query, POOL)
            dense_scores = matrix @ vector
            top = np.argpartition(-dense_scores, min(POOL, len(dense_scores) - 1))[:POOL]
            dense = sorted(((int(i), float(dense_scores[i])) for i in top), key=lambda p: -p[1])
            legs["bm25"][question.qid] = {ids[p]: float(s) for p, s in lexical[:POOL]}
            legs["dense"][question.qid] = {ids[p]: float(s) for p, s in dense}
            legs["alpha0.7"][question.qid] = {ids[p]: float(s)
                                              for p, s in alpha_fuse(lexical, dense, ALPHA, POOL)}

        def full(run):
            scored = evaluator.evaluate(run)
            nd = 100 * sum(v["ndcg_cut_10"] for v in scored.values()) / len(scored)
            rc = 100 * sum(v["recall_10"] for v in scored.values()) / len(scored)
            return nd, rc, {q: 100 * v["ndcg_cut_10"] for q, v in scored.items()}

        rows = {}
        chars = [len(t) for t in pages.values() if t.strip()]
        print(f"\n=== {arm}  ({len(ids)} pages, mean {sum(chars)//len(chars)} chars/page) ===")
        print(f"{'arm':22s} {'NDCG@10':>8s} {'R@10':>7s} {'delta':>7s} {'p':>8s}")
        for name, run in legs.items():
            nd, rc, _ = full(run)
            rows[name] = {"ndcg@10": round(nd, 2), "recall@10": round(rc, 2)}
            print(f"{name:22s} {nd:8.2f} {rc:7.2f}")
        _, _, base_pq = full(legs["alpha0.7"])
        base_nd = rows["alpha0.7"]["ndcg@10"]
        for lam in LAMBDAS:
            rescored = {q: propagate(s, lam, W, GAMMA, BETA, TOPM)
                        for q, s in legs["alpha0.7"].items()}
            nd, rc, pq = full(rescored)
            delta, p, better, worse, tied = permutation(base_pq, pq, 10000)
            rows[f"alpha0.7+SEP(l={lam})"] = {
                "ndcg@10": round(nd, 2), "recall@10": round(rc, 2),
                "delta": round(nd - base_nd, 2), "p": round(p, 5),
                "better": better, "worse": worse, "tied": tied}
            print(f"{'alpha0.7 + SEP l=' + str(lam):22s} {nd:8.2f} {rc:7.2f} "
                  f"{nd - base_nd:+7.2f} {p:8.4f}{'' if p < 0.05 else '  n.s.'}")
        report[arm] = {"pages": len(ids), "mean_chars": sum(chars) // len(chars), "arms": rows}

    out = ROOT / "data/benchmark/vidore_v3/results/sep_results_table.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
