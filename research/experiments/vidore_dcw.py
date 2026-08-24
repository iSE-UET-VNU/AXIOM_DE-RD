"""DCW -- factorising page relevance into between-document and within-document terms.

The diagnostic says 69% of physics failures are "right file, wrong pages within
it". A bi-encoder is structurally bad at that: every page of a document shares
its topical vocabulary, so each page embedding carries a large document-topic
component that is near-identical across the file. That component is what makes
the file findable and simultaneously what makes its pages indistinguishable. One
cosine conflates two questions.

DCW splits them. With mu_f the topical centroid of file f:

    e_hat_p = e_p - kappa * mu_f                 page residual
    score   = a * cos(q, mu_f) + b * cos(q, e_hat_p)
              between-document      within-document

SEP only ever estimated the first term, and crudely, from pooled scores. The
second is the one the diagnostic says is broken and which nothing in the
pipeline computes.

Prediction, registered before running: DCW should behave *opposite* to SEP.
Estimating mu_f and having pages to disambiguate both improve with file size, so
DCW should get stronger as documents get larger -- exactly where SEP dies
(industrial, 194 pages/file). If that holds the two cover complementary regimes.

Costs nothing: pure numpy over embeddings already cached.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytrec_eval

from src.utils.env import load_dotenv_file

load_dotenv_file(ROOT)

from src.evaluation.benchmarks import load
from src.evaluation.retrieval import alpha_fuse
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.retrieval.sparse import BM25Index
from research.experiments.physics_sep_test import permutation

POOL, ALPHA = 100, 0.7
UNIT = re.compile(r"^[^:]+::(?P<file>.+)#page=(?P<page>\d+)$")


def file_of(unit_id: str) -> str | None:
    m = UNIT.match(unit_id)
    return m["file"] if m else None


def unit_norm(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="physics")
    parser.add_argument("--language", default="french")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    bench = load("vidore_v3", subset=args.subset, language=args.language)
    qrels = bench.qrels()
    questions = [q for q in bench.questions() if qrels.get(q.qid)]
    pages = {d.doc_id: (d.text or "") for d in bench.corpus()}

    cache = (ROOT / "data/work/vidore_physics_emb" if args.subset == "physics"
             else ROOT / f"data/work/vidore_{args.subset}_emb")
    embedder = OpenRouterEmbedder(cache_dir=cache, batch_size=64)

    qv = unit_norm(np.asarray(embedder.embed([q.query for q in questions]), dtype=np.float32))
    ids = [u for u, text in pages.items() if text.strip()]
    emb = unit_norm(np.asarray(embedder.embed([pages[u] for u in ids]), dtype=np.float32))
    bm25 = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": u, "doc_id": u, "text": pages[u]} for u in ids])

    # per-file topical centroid
    rows_by_file: dict[str, list[int]] = defaultdict(list)
    for i, unit in enumerate(ids):
        if (f := file_of(unit)):
            rows_by_file[f].append(i)
    centroid = np.zeros_like(emb)
    file_index = {}
    mus = []
    for f, rows in rows_by_file.items():
        mu = emb[rows].mean(axis=0)
        file_index[f] = len(mus)
        mus.append(mu)
        centroid[rows] = mu
    mus = unit_norm(np.asarray(mus, dtype=np.float32))
    file_row = np.array([file_index[file_of(u)] for u in ids])
    sizes = np.array([len(rows_by_file[file_of(u)]) for u in ids])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})

    def score_pool(dense_scores_fn):
        """Fuse BM25 with a caller-supplied dense scorer, over the same pool."""
        run = {}
        for question, vector in zip(questions, qv):
            lexical = bm25.search(question.query, POOL)
            dense_all = dense_scores_fn(vector)
            top = np.argpartition(-dense_all, min(POOL, len(dense_all) - 1))[:POOL]
            dense = sorted(((int(i), float(dense_all[i])) for i in top), key=lambda p: -p[1])
            fused = alpha_fuse(lexical, dense, ALPHA, POOL)
            run[question.qid] = {ids[p]: float(s) for p, s in fused}
        scored = evaluator.evaluate(run)
        per_q = {q: 100 * v["ndcg_cut_10"] for q, v in scored.items()}
        return sum(per_q.values()) / len(per_q), per_q

    baseline, base_per_q = score_pool(lambda v: emb @ v)
    mean_pages = len(ids) / len(rows_by_file)
    print(f"\n=== vidore_v3/{args.subset}/{args.language} "
          f"({len(questions)} queries, {len(rows_by_file)} files, "
          f"{mean_pages:.1f} pages/file) ===")
    print(f"baseline alpha0.7 (plain dense)   NDCG@10 = {baseline:.2f}")
    print(f"\n{'kappa':>6s} {'a':>5s} {'b':>5s} {'ndcg':>7s} {'delta':>7s} {'p':>8s}   b/w/t")

    rows = []
    for kappa in (0.25, 0.5, 0.75, 1.0):
        residual = unit_norm(emb - kappa * centroid)
        for a, b in ((0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25)):
            def dense_fn(v, a=a, b=b, residual=residual):
                return a * (mus @ v)[file_row] + b * (residual @ v)
            overall, per_q = score_pool(dense_fn)
            delta, p, better, worse, tied = permutation(base_per_q, per_q, 2000)
            flag = "" if p < 0.05 else "  n.s."
            print(f"{kappa:6.2f} {a:5.2f} {b:5.2f} {overall:7.2f} {delta:+7.2f} {p:8.4f}   "
                  f"{better:3d}/{worse:3d}/{tied:3d}{flag}")
            rows.append({"kappa": kappa, "a": a, "b": b, "ndcg@10": round(overall, 2),
                         "delta": round(delta, 3), "p": round(p, 5),
                         "better": better, "worse": worse, "tied": tied,
                         "per_question": {q: round(x, 4) for q, x in per_q.items()}})

    out = Path(args.out) if args.out else (
        ROOT / f"data/benchmark/vidore_v3/results/dcw_{args.subset}_{args.language}.json")
    out.write_text(json.dumps({
        "subset": args.subset, "language": args.language, "queries": len(questions),
        "files": len(rows_by_file), "pages_per_file": round(mean_pages, 1),
        "baseline": round(baseline, 2), "grid": rows,
        "baseline_per_question": {q: round(x, 4) for q, x in base_per_q.items()},
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
