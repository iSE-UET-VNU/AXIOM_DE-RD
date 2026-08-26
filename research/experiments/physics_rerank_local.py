"""E1b: local cross-encoder rerank of the served pool -- no API, no GPU.

Voyage's free tier caps rerank depth at 20 (3 RPM / 10K TPM -- see the
axiom-rerank-env-limits memory), so the +5.08 rerank result was never measured
past that depth. Xenova ships an ONNX export of bge-reranker-base
(multilingual, matches these French queries) that runs on CPU via ONNX
Runtime, the same workaround vidore_visual_arm.py uses for CLIP: it sidesteps
the torch 2.2.2 / transformers>=2.4 requirement entirely, no local torch
install needed. This measures the "trained ranker" lever
(docs/phan_tich_first_principles.md Part 4, item 2) at full depth-100 instead
of the API-capped depth-20, at zero cost.

    python research/experiments/physics_rerank_local.py --dry            # baseline + timing probe only
    python research/experiments/physics_rerank_local.py --depth 20       # sanity check vs Voyage's 49.23
    python research/experiments/physics_rerank_local.py --depth 100      # the actual extended-depth measurement
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import pytrec_eval
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from src.evaluation.benchmarks import load
from research.experiments.physics_sep_test import permutation

RESULTS = ROOT / "data/benchmark/vidore_v3/results"
POOL = RESULTS / "physics_served_pool.json"


def ndcg10(run: dict[str, dict[str, float]], qrels: dict) -> tuple[float, dict]:
    scored = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"}).evaluate(run)
    per_q = {q: 100 * v["ndcg_cut_10"] for q, v in scored.items()}
    return sum(per_q.values()) / len(per_q), per_q


def as_run(order: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    return {qid: {c: float(len(ids) - i) for i, c in enumerate(ids)} for qid, ids in order.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Xenova/bge-reranker-base")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pool", default=str(POOL))
    parser.add_argument("--tag", default="")
    parser.add_argument("--dry", action="store_true", help="Baseline + timing probe only, no full run.")
    args = parser.parse_args()

    tag = args.tag or f"_d{args.depth}"
    checkpoint = RESULTS / f"physics_rerank_local_scores{tag}.json"
    out = RESULTS / f"physics_rerank_local{tag}.json"

    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))["queries"]
    bench = load("vidore_v3", subset="physics", language="french")
    qrels = {q: g for q, g in bench.qrels().items() if q in pool}
    pages = {d.doc_id: (d.text or "") for d in bench.corpus()}

    served = {qid: entry["candidates"] for qid, entry in pool.items()}
    baseline, base_per_q = ndcg10(as_run(served), qrels)
    print(f"queries={len(served)}  served alpha0.7 NDCG@10 = {baseline:.2f}")

    print(f"loading {args.repo} (ONNX, CPU)...", flush=True)
    model_path = hf_hub_download(args.repo, "onnx/model.onnx")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(args.repo)

    def score_batch(query: str, texts: list[str]) -> list[float]:
        enc = tokenizer([query] * len(texts), texts, padding=True, truncation=True,
                        max_length=args.max_length, return_tensors="np")
        logits = session.run(["logits"], {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        return logits[:, 0].tolist()

    def score_query(query: str, ids: list[str], texts: list[str]) -> dict[str, float]:
        # Batches pad to the longest sequence in the batch; sorting by length
        # first keeps padding waste local instead of paying max-length on
        # every batch because of one long outlier candidate.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vals = [0.0] * len(texts)
        sorted_texts = [texts[i] for i in order]
        pos = 0
        for start in range(0, len(sorted_texts), args.batch):
            batch = sorted_texts[start:start + args.batch]
            for offset, score in enumerate(score_batch(query, batch)):
                vals[order[pos + offset]] = score
            pos += len(batch)
        return dict(zip(ids, vals))

    # Timing probe: one query at the requested depth, to project total wall time.
    sample_qid = next(iter(served))
    sample_head = served[sample_qid][:args.depth]
    sample_texts = [pages.get(c, "") for c in sample_head]
    t0 = time.time()
    score_query(pool[sample_qid]["query"], sample_head, sample_texts)
    per_query_s = time.time() - t0
    print(f"timing probe: {per_query_s:.2f}s/query at depth={args.depth} "
          f"-> ~{per_query_s * len(served) / 60:.1f} min for {len(served)} queries")

    if args.dry:
        return

    scores: dict[str, dict[str, float]] = {}
    if checkpoint.exists():
        scores = json.loads(checkpoint.read_text(encoding="utf-8"))
        print(f"resuming: {len(scores)}/{len(served)} already scored")

    for done, (qid, candidates) in enumerate(served.items(), start=1):
        if qid in scores:
            continue
        head = candidates[:args.depth]
        texts = [pages.get(c, "") for c in head]
        scores[qid] = score_query(pool[qid]["query"], head, texts)
        if done % 20 == 0:
            checkpoint.write_text(json.dumps(scores), encoding="utf-8")
            print(f"  {done}/{len(served)}", flush=True)
    checkpoint.write_text(json.dumps(scores), encoding="utf-8")

    reranked = {}
    for qid, candidates in served.items():
        head = sorted(scores[qid], key=lambda c: -scores[qid][c])
        reranked[qid] = head + [c for c in candidates if c not in scores[qid]]

    final, final_per_q = ndcg10(as_run(reranked), qrels)
    delta, p, better, worse, tied = permutation(base_per_q, final_per_q)
    flag = "" if p < 0.05 else "   n.s."
    print(f"\nserved alpha0.7                NDCG@10 = {baseline:.2f}")
    print(f"+ {args.repo} top-{args.depth}  NDCG@10 = {final:.2f}   "
          f"({delta:+.2f}, p={p:.4f}{flag})   {better}/{worse}/{tied} better/worse/tied")

    out.write_text(json.dumps({
        "arm": f"alpha0.7+{args.repo}", "depth": args.depth, "repo": args.repo,
        "baseline_ndcg10": round(baseline, 2), "reranked_ndcg10": round(final, 2),
        "delta": round(delta, 3), "p": round(p, 5),
        "better": better, "worse": worse, "tied": tied,
        "run": {qid: ids[:100] for qid, ids in reranked.items()},
    }, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
