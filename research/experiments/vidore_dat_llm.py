"""DAT with a real LLM judge (arXiv 2503.23013), measured on ViDoRe V3 physics.

Per query, one LLM call scores how well each leg's top-1 page answers the
question. The two effectiveness scores normalise into a per-query alpha, which
replaces the fixed 0.7 in the hybrid fusion.

The ceiling test (vidore_dat_ceiling.py) put a perfect judge at 47.13 alone and
48.87 with SEP, against a 43.86 production baseline. This measures how much of
that an actual judge recovers.

Resumable: judgements are checkpointed per query.
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from src.evaluation.llm import complete
from src.evaluation.model_guard import assert_real
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks
from src.evaluation.retrieval import alpha_fuse
from src.retrieval.sparse import BM25Index

SUB, LANG, POOL = "physics", "french", 100
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MAX_CHARS = 1500
RESULTS = ROOT / "data/benchmark/vidore_v3/results"

PROMPT_GRADED = """You are judging which of two retrieval systems returned a better first result.

Question:
{query}

Result A (from the keyword retriever):
{doc_a}

Result B (from the semantic retriever):
{doc_b}

Score how well each result helps answer the question, on a 0-5 scale:
0 = irrelevant, 1-2 = related topic but does not answer, 3-4 = partially answers,
5 = directly answers.

Reply with exactly two integers separated by a space: the score for A, then B.
No other text."""

# The ceiling used ABSOLUTE relevance (is this page gold?), which yields alpha in
# {0, 0.5, 1}. A graded 0-5 judge hedges both documents into the mid-range and
# alpha collapses to 0.5, recovering almost none of the headroom. This asks the
# same binary question the ceiling answered.
PROMPT_BINARY = """Question:
{query}

Passage A:
{doc_a}

Passage B:
{doc_b}

For EACH passage independently, decide whether it contains information that
helps answer the question. Be strict: a passage merely on the same broad topic
does NOT count -- it must contain relevant specifics.

Reply with exactly two words separated by a space, each YES or NO: the verdict
for A, then B. No other text."""

norm = lambda m: m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)


def parse_scores(reply: str, mode: str) -> tuple[float, float] | None:
    if mode == "binary":
        verdicts = re.findall(r"\b(YES|NO)\b", (reply or "").upper())
        if len(verdicts) < 2:
            return None
        return float(verdicts[0] == "YES"), float(verdicts[1] == "YES")
    nums = re.findall(r"\d+", reply or "")
    if len(nums) < 2:
        return None
    a, b = float(nums[0]), float(nums[1])
    if not (0 <= a <= 5 and 0 <= b <= 5):
        return None
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default="openai/gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mode", default="graded", choices=["graded", "binary"])
    args = parser.parse_args()
    assert_real([args.judge])

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

    legs = {}
    for question, vector in zip(questions, qv):
        lexical = bm25.search(question.query, 1000)
        dense_scores = matrix @ vector
        top = np.argpartition(-dense_scores, min(1000, len(dense_scores) - 1))[:1000]
        dense = sorted(((int(i), float(dense_scores[i])) for i in top), key=lambda p: -p[1])
        legs[question.qid] = (maxp(list(lexical)), maxp(dense))

    def fuse(qid, alpha):
        lp, dp = legs[qid]
        uids = list(dict.fromkeys([u for u, _ in lp] + [u for u, _ in dp]))
        idx = {u: i for i, u in enumerate(uids)}
        fl = [(idx[u], s) for u, s in lp]
        fd = [(idx[u], s) for u, s in dp]
        return {uids[p]: float(s) for p, s in alpha_fuse(fl, fd, alpha, POOL)}

    # --- the judge ---
    # Checkpoint must be keyed by the JUDGE as well as the mode -- keying on mode
    # alone silently reloads another model's verdicts and reports them as this
    # model's result.
    tag = f"{args.mode}_{args.judge.replace('/', '-')}"
    checkpoint = RESULTS / f"dat_llm_judgements_{tag}.json"
    scores = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    todo = [q for q in questions if q.qid not in scores]
    print(f"judging {len(todo)} queries with {args.judge} ({len(scores)} cached)...")

    def judge(question):
        lp, dp = legs[question.qid]
        a = pages.get(lp[0][0], "")[:MAX_CHARS] if lp else ""
        b = pages.get(dp[0][0], "")[:MAX_CHARS] if dp else ""
        template = PROMPT_BINARY if args.mode == "binary" else PROMPT_GRADED
        reply = complete(args.judge, template.format(query=question.query, doc_a=a, doc_b=b),
                         temperature=0.0, max_output_tokens=16)
        return question.qid, parse_scores(reply, args.mode), reply

    started = time.time()
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(judge, q) for q in todo]
            for n, future in enumerate(as_completed(futures), 1):
                qid, parsed, reply = future.result()
                scores[qid] = parsed if parsed else None
                if n % 25 == 0:
                    checkpoint.write_text(json.dumps(scores))
                    print(f"  {n}/{len(todo)}  {time.time()-started:.0f}s", flush=True)
        checkpoint.write_text(json.dumps(scores))
    judge_seconds = time.time() - started
    unparsed = sum(1 for q in questions if not scores.get(q.qid))
    print(f"judge done in {judge_seconds:.0f}s; unparsable {unparsed}/{len(questions)}")

    snap = lambda x: min(ALPHAS, key=lambda a: abs(a - x))

    def dat_alpha(qid):
        pair = scores.get(qid)
        if not pair:
            return 0.5
        sa, sb = pair
        return 0.5 if (sa + sb) == 0 else sb / (sa + sb)

    def evaluate(run_):
        s = evaluator.evaluate(run_)
        return (100 * sum(v["ndcg_cut_10"] for v in s.values()) / len(s),
                100 * sum(v["recall_10"] for v in s.values()) / len(s),
                {k: 100 * v["ndcg_cut_10"] for k, v in s.items()})

    arms = {}
    arms["production alpha=0.7"] = {q.qid: fuse(q.qid, 0.7) for q in questions}
    arms["DAT (LLM judge)"] = {q.qid: fuse(q.qid, snap(dat_alpha(q.qid))) for q in questions}
    arms["SEP only"] = {q.qid: propagate(fuse(q.qid, 0.7), 0.5, W, GAMMA, BETA, TOPM)
                        for q in questions}
    arms["DAT + SEP"] = {q.qid: propagate(fuse(q.qid, snap(dat_alpha(q.qid))), 0.5,
                                          W, GAMMA, BETA, TOPM) for q in questions}

    print(f"\n{'arm':24s} {'NDCG@10':>8s} {'R@10':>7s} {'delta':>7s} {'p':>8s}")
    base_nd, base_rc, base_pq = evaluate(arms["production alpha=0.7"])
    out = {}
    for name, run_ in arms.items():
        nd, rc, pq = evaluate(run_)
        if name == "production alpha=0.7":
            print(f"{name:24s} {nd:8.2f} {rc:7.2f}")
            out[name] = {"ndcg@10": round(nd, 2), "recall@10": round(rc, 2)}
            continue
        delta, p, better, worse, tied = permutation(base_pq, pq, 10000)
        print(f"{name:24s} {nd:8.2f} {rc:7.2f} {nd-base_nd:+7.2f} {p:8.4f}"
              f"{'' if p < 0.05 else '  n.s.'}   {better}/{worse}/{tied}")
        out[name] = {"ndcg@10": round(nd, 2), "recall@10": round(rc, 2),
                     "delta": round(nd - base_nd, 2), "p": round(p, 5)}

    alphas_used = [snap(dat_alpha(q.qid)) for q in questions]
    dist = {f"{a:.1f}": alphas_used.count(a) for a in ALPHAS if alphas_used.count(a)}
    print(f"\nalpha distribution chosen by the judge: {dist}")
    print(f"judge latency: {judge_seconds:.0f}s total for {len(todo) or len(questions)} queries"
          f" ({judge_seconds/max(len(todo),1):.2f}s/query at {args.workers} workers)")
    (RESULTS / f"dat_llm_results_{tag}.json").write_text(json.dumps(
        {"judge": args.judge, "mode": args.mode, "arms": out, "alpha_distribution": dist,
         "judge_seconds": round(judge_seconds, 1), "queries": len(questions),
         "unparsable": unparsed}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
