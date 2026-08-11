"""ViDoRe physics, French: end-to-end retrieval -> generation -> judge.

Three contexts against Table 3's physics column: gold pages (Oracle/Text, 71.2),
our alpha0.7 top-10 over ViDoRe page text, and the same over the chandra2 parse.
Retrieval is identical to physics_ladder.py; only the context source differs.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytrec_eval

from src.utils.env import load_dotenv_file

load_dotenv_file(Path(__file__).resolve().parents[2])

from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.benchmarks.vidore_v3_judge import (
    ANSWER_PROMPT, ViDoreVerdict, judge_answer, render_documents, score,
)
from src.evaluation.llm import complete
from src.evaluation.model_guard import assert_real
from src.evaluation.pipeline_pages import BOILERPLATE, canonical_doc, documents, page_blocks
from src.evaluation.retrieval import alpha_fuse
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.retrieval.sparse import BM25Index

RUN = (str(Path(__file__).resolve().parents[2] / "data_vidore_parsed_physics/output/benchmarks/vidore-v3-physics-chandra2/c6e049e4fd9c4b9b"))
SUBSET, LANG = "physics", "french"
GENERATOR, JUDGE = "deepseek/deepseek-v4-flash", "openai/gpt-4o"
TOP_K, DEPTH, ALPHA, WORKERS = 10, 100, 0.7, 12
OUT = Path(__file__).resolve().parents[2] / "data/benchmark/vidore_v3/results" / "physics_e2e"
PUBLISHED = {"oracle": 71.2, "best_nonoracle": 69.2, "colembed": 64.9}

OUT.mkdir(exist_ok=True)
resolved = assert_real([GENERATOR, JUDGE])
for alias, r in resolved.items():
    print(f"  {alias:20s} -> {r.provider}/{r.upstream_model_id}")
assert resolved[GENERATOR].upstream_model_id != resolved[JUDGE].upstream_model_id

bench = load("vidore_v3", subset=SUBSET, language=LANG)
qrels = bench.qrels()
questions = [q for q in bench.questions() if qrels.get(q.qid)]
vidore_pages = {d.doc_id: (d.text or "") for d in bench.corpus()}

chandra_pages = {}
for document in documents(RUN):
    doc = canonical_doc(document.get("document", {}).get("file_name"))
    for page, blocks in page_blocks(document).items():
        chandra_pages[unit_id(SUBSET, doc, page)] = "\n".join(
            b["text"] for b in blocks if b["text"].strip())

embedder = OpenRouterEmbedder(cache_dir=Path(__file__).resolve().parents[2] / "data/work/vidore_physics_emb", batch_size=64)
query_vectors = np.asarray(embedder.embed([q.query for q in questions]), dtype=np.float32)
query_vectors /= np.clip(np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12, None)
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})


def retrieve(pages):
    """alpha0.7 over BM25 + te3-small, exactly as in the ladder."""
    units = [u for u, t in pages.items() if t.strip()]
    texts = [pages[u] for u in units]
    bm25 = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": u, "doc_id": u, "text": t} for u, t in zip(units, texts)])
    matrix = np.asarray(embedder.embed(texts), dtype=np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

    ranked, run = {}, {}
    for question, vector in zip(questions, query_vectors):
        lexical = bm25.search(question.query, DEPTH)
        scores = matrix @ vector
        top = np.argpartition(-scores, DEPTH)[:DEPTH]
        dense = sorted(((int(i), float(scores[i])) for i in top), key=lambda p: -p[1])
        fused = alpha_fuse(lexical, dense, ALPHA, DEPTH)
        ranked[question.qid] = [units[p] for p, _ in fused[:TOP_K]]
        run[question.qid] = {units[p]: s for p, s in fused}
    result = evaluator.evaluate(run)
    ndcg = 100 * sum(v["ndcg_cut_10"] for v in result.values()) / len(result)
    return ranked, ndcg


gold_ranked = {q.qid: sorted(qrels[q.qid]) for q in questions}
CONTEXTS = {}
# Same gold pages, each parser's own rendering: the parser under perfect retrieval.
CONTEXTS["oracle"] = (gold_ranked, None, vidore_pages)
CONTEXTS["oracle_chandra2"] = (gold_ranked, None, chandra_pages)
for name, pages in (("retrieved_vidore", vidore_pages), ("retrieved_chandra2", chandra_pages)):
    ranked, ndcg = retrieve(pages)
    print(f"  {name:20s} NDCG@10 = {ndcg:.2f}")
    CONTEXTS[name] = (ranked, ndcg, pages)


def generate_one(item):
    qid, query, context = item
    try:
        return qid, complete(GENERATOR, ANSWER_PROMPT.format(documents=context, query=query),
                             temperature=0.0, max_output_tokens=512).strip()
    except Exception as exc:  # noqa: BLE001
        return qid, f"__ERROR__ {exc}"


def judge_one(item):
    qid, query, gold, test = item
    if test.startswith("__ERROR__"):
        return ViDoreVerdict(qid, "Incorrect", error=test)
    try:
        return judge_answer(qid, query, gold, test, model=JUDGE, generator_model=GENERATOR)
    except Exception as exc:  # noqa: BLE001
        return ViDoreVerdict(qid, "Incorrect", error=str(exc))


summary = []
for name, (ranked, ndcg, pages) in CONTEXTS.items():
    path = OUT / f"{name}.json"
    if path.exists():
        rows = json.loads(path.read_text())
    else:
        jobs, meta = [], {}
        for question in questions:
            units = ranked[question.qid]
            context = render_documents([pages[u] for u in units if pages.get(u, "").strip()])
            jobs.append((question.qid, question.query, context))
            meta[question.qid] = {"query": question.query, "gold_answer": question.answer,
                                  "n_context_pages": len(units), "context_chars": len(context),
                                  "gold_hit": len(set(units) & set(qrels[question.qid]))}
        with ThreadPoolExecutor(WORKERS) as pool:
            gens = dict(pool.map(generate_one, jobs))
        with ThreadPoolExecutor(WORKERS) as pool:
            verdicts = list(pool.map(judge_one, [
                (q, meta[q]["query"], meta[q]["gold_answer"], gens[q]) for q in gens]))
        rows = [{"arm": name, "qid": v.qid, "answer": gens[v.qid], "judgment": v.judgment,
                 "error": v.error, **meta[v.qid]} for v in verdicts]
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    s = score([ViDoreVerdict(r["qid"], r["judgment"]) for r in rows])
    summary.append({"arm": name, "ndcg@10": ndcg, "n": s["n"],
                    "correct_only": round(100 * s["correct_only"], 1),
                    "correct_plus_partial": round(100 * s["correct_plus_partial"], 1),
                    "ctx_pages": round(sum(r["n_context_pages"] for r in rows) / len(rows), 1),
                    "ctx_chars": round(sum(r["context_chars"] for r in rows) / len(rows)),
                    "gold_in_ctx": round(sum(r["gold_hit"] for r in rows) / len(rows), 2)})

(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\n{'arm':20s} {'NDCG@10':>8s} {'n':>5s} {'correct':>9s} {'+partial':>9s} "
      f"{'pages':>6s} {'chars':>7s} {'gold in ctx':>12s}")
for r in summary:
    print(f"{r['arm']:20s} {r['ndcg@10'] or 0:8.1f} {r['n']:5d} {r['correct_only']:8.1f}% "
          f"{r['correct_plus_partial']:8.1f}% {r['ctx_pages']:6.1f} {r['ctx_chars']:7d} "
          f"{r['gold_in_ctx']:12.2f}")
print(f"\npaper Table 3, physics: Oracle/Text {PUBLISHED['oracle']}  "
      f"best non-oracle {PUBLISHED['best_nonoracle']}  ColEmbed-3B-v2/Text {PUBLISHED['colembed']}")
