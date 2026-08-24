"""E1: cross-encoder rerank of the served alpha0.7 pool, measured on physics.

Reads the candidates from ``physics_served_pool.json`` rather than re-retrieving,
so the reranked arm reorders the exact ranking the 44.15 baseline was scored on
and the delta is attributable to reranking alone.

Depth is 20 because that is ``RETRIEVAL_K3``, the depth production already serves
-- not a value tuned on these 302 queries. The free Voyage tier (3 RPM / 10K TPM)
also cannot fit a depth-50 request, so the plan's depth ablation needs a paid tier.

    python research/experiments/physics_rerank_voyage.py --dry   # baseline, no API
    python research/experiments/physics_rerank_voyage.py         # ~100 min, resumable
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytrec_eval

from src.utils.env import load_dotenv_file

load_dotenv_file(ROOT)

from src.evaluation.benchmarks import load
from src.evaluation.voyage_rerank import MAX_DOC_CHARS, VoyageReranker

DEPTH = 20
MODEL = "rerank-2.5"
RESULTS = ROOT / "data/benchmark/vidore_v3/results"
POOL = RESULTS / "physics_served_pool.json"
CHECKPOINT = RESULTS / "physics_rerank_voyage_scores.json"
OUT = RESULTS / "physics_rerank_voyage.json"


def ndcg10(run: dict[str, dict[str, float]], qrels: dict) -> float:
    scored = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"}).evaluate(run)
    return 100 * sum(v["ndcg_cut_10"] for v in scored.values()) / len(scored)


def as_run(order: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    return {qid: {c: float(len(ids) - i) for i, c in enumerate(ids)} for qid, ids in order.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true", help="Baseline and cost only; no API calls.")
    parser.add_argument("--depth", type=int, default=DEPTH)
    # Raising --depth without raising these is a crash, not a slow run: a single
    # request over the token cap can never satisfy _throttle, which then takes
    # min() of an empty history. Free tier is 3 / 10000.
    parser.add_argument("--rpm", type=int, default=3)
    parser.add_argument("--tpm", type=int, default=10_000)
    parser.add_argument("--pool", default=str(POOL))
    parser.add_argument("--texts", default=None,
                        help="Page texts as JSON, for pools the benchmark corpus does not cover.")
    parser.add_argument("--tag", default="", help="Suffix for the checkpoint and output files.")
    args = parser.parse_args()

    checkpoint = CHECKPOINT.with_name(CHECKPOINT.stem + args.tag + CHECKPOINT.suffix)
    out = OUT.with_name(OUT.stem + args.tag + OUT.suffix)

    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))["queries"]
    bench = load("vidore_v3", subset="physics", language="french")
    qrels = {q: g for q, g in bench.qrels().items() if q in pool}
    pages = (json.loads(Path(args.texts).read_text(encoding="utf-8")) if args.texts
             else {d.doc_id: (d.text or "") for d in bench.corpus()})

    served = {qid: entry["candidates"] for qid, entry in pool.items()}
    baseline = ndcg10(as_run(served), qrels)
    print(f"queries={len(served)}  served alpha0.7 NDCG@10 = {baseline:.2f}")

    # A reranker fed truncated pages is not comparable to the paper's; measure it.
    lengths = np.array([len(pages.get(c, "")) for ids in served.values() for c in ids[:args.depth]])
    over = float((lengths > MAX_DOC_CHARS).mean())
    print(f"passages: mean={lengths.mean():.0f} median={np.median(lengths):.0f} chars, "
          f"{100 * over:.1f}% clipped at {MAX_DOC_CHARS}")
    print(f"per-request estimate ~{(min(lengths.max(), MAX_DOC_CHARS) * args.depth) // 3} tokens (cap 10000)")
    if args.dry:
        return

    scores: dict[str, dict[str, float]] = {}
    if checkpoint.exists():
        scores = json.loads(checkpoint.read_text(encoding="utf-8"))
        print(f"resuming: {len(scores)}/{len(served)} already scored")

    reranker = VoyageReranker(model=MODEL, depth=args.depth, rpm=args.rpm, tpm=args.tpm)
    for done, (qid, candidates) in enumerate(served.items(), start=1):
        if qid in scores:
            continue
        head = candidates[:args.depth]
        texts = [pages.get(c, "") for c in head]
        ranked = reranker.rerank(pool[qid]["query"], list(enumerate([0.0] * len(head))), texts)
        scores[qid] = {head[index]: score for index, score in ranked}
        checkpoint.write_text(json.dumps(scores), encoding="utf-8")
        if done % 10 == 0:
            print(f"  {done}/{len(served)}  {reranker.stats}", flush=True)

    reranked = {}
    for qid, candidates in served.items():
        head = sorted(scores[qid], key=lambda c: -scores[qid][c])
        reranked[qid] = head + [c for c in candidates if c not in scores[qid]]

    final = ndcg10(as_run(reranked), qrels)
    print(f"\nserved alpha0.7            NDCG@10 = {baseline:.2f}")
    print(f"+ {MODEL} top-{args.depth}    NDCG@10 = {final:.2f}   ({final - baseline:+.2f})")

    out.write_text(json.dumps({
        "arm": f"alpha0.7+{MODEL}", "depth": args.depth, "model": MODEL,
        "baseline_ndcg10": baseline, "reranked_ndcg10": final, "delta": final - baseline,
        "clipped_fraction": over, "stats": reranker.stats,
        "run": {qid: ids[:100] for qid, ids in reranked.items()},
    }, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
