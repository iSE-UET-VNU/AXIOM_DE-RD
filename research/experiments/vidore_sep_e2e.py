"""End-to-end QA for the SEP arm on ViDoRe V3 physics (KDL production config).

Mirrors physics_e2e_hybrid.py -- French questions, page-level retrieval over the
KDL parse with fixed_512/128 + MaxP, alpha=0.7, top-10 full-page context,
DeepSeek V4 Flash generation, GPT-4o judging -- and adds the SEP rescoring pass
so the QA columns can be filled for the solution row.

Runs both arms (baseline and +SEP) over the same questions so the QA delta is
paired and attributable. Resumable.
"""
import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytrec_eval

from src.utils.env import load_dotenv_file

load_dotenv_file(ROOT)

from research.experiments.physics_sep import propagate
from research.experiments.physics_sep_test import BETA, GAMMA, TOPM, W
from src.chunking_embedding.chunkers.builtin import fixed_overlap
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.benchmarks.vidore_v3_judge import (
    ANSWER_PROMPT, judge_answer, render_documents)
from src.evaluation.llm import complete
from src.evaluation.model_guard import assert_real
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks
from src.evaluation.retrieval import alpha_fuse
from src.retrieval.sparse import BM25Index

SUB, LANG = "physics", "french"
GENERATOR, JUDGE = "deepseek/deepseek-v4-flash", "openai/gpt-4o"
TOP_K, POOL, ALPHA, LAM = 10, 100, 0.7, 0.5
RESULTS = ROOT / "data/benchmark/vidore_v3/results"
norm = lambda m: m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    assert_real([GENERATOR, JUDGE])

    run = next((ROOT / f"data_vidore_parsed_physics/output/benchmarks/vidore-v3-{SUB}-kdl").iterdir())
    pages = {}
    for document in documents(run):
        doc = canonical_doc(document.get("document", {}).get("file_name"))
        for page, blocks in page_blocks(document).items():
            pages[unit_id(SUB, doc, page)] = "\n".join(
                b["text"] for b in blocks if (b.get("text") or "").strip())

    bench = load("vidore_v3", subset=SUB, language=LANG)
    qrels = bench.qrels()
    questions = [q for q in bench.questions() if qrels.get(q.qid)]
    if args.limit:
        questions = questions[:args.limit]
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_10"})

    embedder = OpenRouterEmbedder(cache_dir=ROOT / "data/work/vidore_physics_kdl_chunk_emb",
                                  batch_size=64)
    qv = norm(np.asarray(embedder.embed([q.query for q in questions]), dtype=np.float32))
    recs, owner = [], []
    for unit, text in pages.items():
        if not text.strip():
            continue
        for span in fixed_overlap(text, n_words=512, overlap=128):
            seg = text[span[0]:span[1]]
            if seg.strip():
                recs.append(seg)
                owner.append(unit)
    matrix = norm(np.asarray(embedder.embed(recs), dtype=np.float32))
    owner = np.array(owner)
    bm25 = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": f"c{i}", "doc_id": owner[i], "text": recs[i]} for i in range(len(recs))])

    def maxp(scored):
        best = defaultdict(lambda: -1e9)
        for pos, score in scored:
            best[owner[pos]] = max(best[owner[pos]], score)
        return sorted(best.items(), key=lambda kv: -kv[1])

    started = perf_counter()
    arms = {"baseline": {}, "sep": {}}
    for question, vector in zip(questions, qv):
        lexical = bm25.search(question.query, 1000)
        dense_scores = matrix @ vector
        top = np.argpartition(-dense_scores, min(1000, len(dense_scores) - 1))[:1000]
        dense = sorted(((int(i), float(dense_scores[i])) for i in top), key=lambda p: -p[1])
        lp, dp = maxp(list(lexical)), maxp(dense)
        uids = list(dict.fromkeys([u for u, _ in lp] + [u for u, _ in dp]))
        idx = {u: i for i, u in enumerate(uids)}
        pool = {uids[p]: float(s) for p, s in alpha_fuse(
            [(idx[u], s) for u, s in lp], [(idx[u], s) for u, s in dp], ALPHA, POOL)}
        arms["baseline"][question.qid] = pool
        arms["sep"][question.qid] = propagate(pool, LAM, W, GAMMA, BETA, TOPM)
    retrieval_seconds = perf_counter() - started

    report = {}
    for arm, pools in arms.items():
        scored = evaluator.evaluate(pools)
        ndcg = 100 * sum(v["ndcg_cut_10"] for v in scored.values()) / len(scored)
        recall = 100 * sum(v["recall_10"] for v in scored.values()) / len(scored)
        ranked = {q: [u for u, _ in sorted(p.items(), key=lambda kv: -kv[1])][:TOP_K]
                  for q, p in pools.items()}

        checkpoint = RESULTS / f"sep_e2e_{arm}.json"
        rows = {r["qid"]: r for r in json.loads(checkpoint.read_text())} if checkpoint.exists() else {}
        todo = [q for q in questions if q.qid not in rows or not rows[q.qid].get("answer")]
        print(f"\n[{arm}] NDCG@10 {ndcg:.2f}  R@10 {recall:.2f} | generating {len(todo)} answers...")

        def work(question):
            context = render_documents([pages.get(u, "") for u in ranked[question.qid]])
            try:
                answer = complete(GENERATOR, ANSWER_PROMPT.format(
                    documents=context, query=question.query),
                    temperature=0.0, max_output_tokens=512).strip()
            except Exception as error:
                return question.qid, {"qid": question.qid, "answer": "", "error": str(error)}
            verdict = judge_answer(question.qid, question.query, question.answer, answer,
                                   model=JUDGE, generator_model=GENERATOR)
            return question.qid, {"qid": question.qid, "query": question.query,
                                  "answer": answer, "label": verdict.judgment,
                                  "error": verdict.error,
                                  "correct": verdict.judgment == "Correct",
                                  "credited": verdict.judgment in ("Correct", "Partially Correct")}

        gen_started = perf_counter()
        if todo:
            with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
                futures = [pool_exec.submit(work, q) for q in todo]
                for n, future in enumerate(as_completed(futures), 1):
                    qid, row = future.result()
                    rows[qid] = row
                    if n % 50 == 0:
                        checkpoint.write_text(json.dumps(list(rows.values()), ensure_ascii=False))
                        print(f"   {n}/{len(todo)}  {perf_counter()-gen_started:.0f}s", flush=True)
            checkpoint.write_text(json.dumps(list(rows.values()), ensure_ascii=False))
        answered = [rows[q.qid] for q in questions if rows.get(q.qid, {}).get("label")]
        correct = 100 * sum(r["correct"] for r in answered) / len(answered)
        credited = 100 * sum(r["credited"] for r in answered) / len(answered)
        print(f"[{arm}] correct_only {correct:.2f}   correct+partial {credited:.2f}"
              f"   (n={len(answered)})")
        report[arm] = {"ndcg@10": round(ndcg, 2), "recall@10": round(recall, 2),
                       "correct_only": round(correct, 2), "correct_plus_partial": round(credited, 2),
                       "answered": len(answered),
                       "retrieval_seconds_all_queries": round(retrieval_seconds, 2),
                       "retrieval_ms_per_query": round(1000 * retrieval_seconds / len(questions), 1)}

    (RESULTS / "sep_e2e_summary.json").write_text(json.dumps(
        {"generator": GENERATOR, "judge": JUDGE, "top_k": TOP_K, "alpha": ALPHA,
         "sep_lambda": LAM, "queries": len(questions), "arms": report}, indent=1))
    print("\n=== summary ===")
    for arm, r in report.items():
        print(f"  {arm:9s} NDCG@10 {r['ndcg@10']:.2f}  R@10 {r['recall@10']:.2f}  "
              f"correct {r['correct_only']:.2f}  correct+partial {r['correct_plus_partial']:.2f}")


if __name__ == "__main__":
    main()
