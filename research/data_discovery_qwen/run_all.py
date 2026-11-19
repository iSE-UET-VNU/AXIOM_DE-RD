"""Run the data-discovery stage end to end: light preparation -> light retrieval -> adaptive retrieval.

    python run_all.py           # replay the chosen result (Qwen3-32B on DeepInfra, 214 questions) from cache/, no API call
    python run_all.py --live    # allow OpenRouter calls for anything missing from cache/ (needs OPENROUTER_API_KEY)

Needs: numpy, scipy, scikit-learn, xgboost, PyStemmer, requests (tiktoken only to re-embed pages with --live).
Writes outputs/ (see README.md) and prints a per-stage summary with timings.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_discovery import adaptive_retrieval as AR  # noqa: E402
from data_discovery import common as C  # noqa: E402
from data_discovery import light_preparation as LP  # noqa: E402
from data_discovery import light_retrieval as LR  # noqa: E402

# The chosen result (2026-11-14): Qwen3-32B pinned to DeepInfra, reasoning off; 214 questions (220 minus the 6 of
# inputs/excluded_questions.json). outputs/ always holds exactly this result.
PROVIDER = "DeepInfra"
EXPECTED = {
    "light_retrieval": {"R@10": 67.36, "R@20": 78.11, "R@100": 93.16, "nDCG@10": 62.82, "nDCG@20": 66.28, "file_recall@5": 97.66},
    "adaptive": {"mean_k": 58.86, "prefix_recall": 90.82, "prefix_precision": 6.79, "prefix_ndcg": 71.03},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true", help="allow API calls for cache misses")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--pages", type=Path, default=None, help="pages_ppocrv5.jsonl (5,091 pages); default: repo cache path")
    ap.add_argument("--page-vectors", type=Path, default=None, help="dense_qwen8b_ppocr_pages.npy; default: repo cache path")
    args = ap.parse_args()
    if args.pages:
        C.PAGES_PATH = args.pages
    if args.page_vectors:
        C.PAGE_VECTORS = args.page_vectors
    excluded = json.loads(C.EXCLUDED.read_text(encoding="utf-8")) if C.EXCLUDED.exists() else {"qids": []}
    EX = frozenset(excluded["qids"])
    timings = {"environment": {"python": platform.python_version(), "machine": platform.machine(), "mode": "live" if args.live else "replay"}}
    t_all = time.perf_counter()

    # ---------------- stage 1: light preparation
    t = time.perf_counter(); corpus = LP.prepare_data(args.live); timings["1a_data_preparation"] = {**corpus.timing, "wall_s": round(time.perf_counter() - t, 3)}
    t = time.perf_counter(); prepared, qvecs, qt = LP.prepare_queries(args.live, PROVIDER, args.workers)
    prepared = [r for r in prepared if r["qid"] not in EX]          # official set: 214 questions
    timings["1b_query_preparation"] = {**qt, "wall_s": round(time.perf_counter() - t, 3)}
    C.write_jsonl(C.OUTPUTS / "query_preparation.jsonl", prepared)
    qrels = {q: g for q, g in C.load_qrels().items() if q not in EX}
    n_q = len(prepared)
    assert n_q == 220 - len(EX) and set(qrels) == {r["qid"] for r in prepared}, f"expected {220 - len(EX)} questions with qrels"

    # ---------------- stage 2: light retrieval
    t = time.perf_counter(); light, lmetrics, lt = LR.run(corpus, prepared, qvecs, qrels, ablations=not args.no_ablations)
    timings["2_light_retrieval"] = {**lt, "wall_s_including_ablations": round(time.perf_counter() - t, 3)}
    C.write_jsonl(C.OUTPUTS / "light_retrieval_top100.jsonl", light)

    # ---------------- stage 3: adaptive retrieval
    t = time.perf_counter(); arows, ametrics, at, final_model, deploy = AR.run(light, qrels)
    timings["3_adaptive_retrieval"] = {**at, "wall_s": round(time.perf_counter() - t, 3)}
    C.write_jsonl(C.OUTPUTS / "adaptive_k.jsonl", arows)
    final_model.save_model(str(C.OUTPUTS / "adaptive_model.json"))
    C.write_json(C.OUTPUTS / "adaptive_calibration.json", deploy)
    timings["total_wall_s"] = round(time.perf_counter() - t_all, 2)

    # ---------------- handoff file for accurate retrieval
    P = {r["qid"]: r for r in prepared}; A = {r["qid"]: r for r in arows}
    out = []
    for r in light:
        q = r["qid"]
        out.append({"qid": q, "source": q.split("::")[0], "query": P[q]["query"],
                    "query_preparation": {k: P[q][k] for k in ("scope", "topic", "anchor", "discard", "bm25_text")},
                    "selected_files": r["selected_files"], "k": A[q]["k"], "k_final_model": A[q]["k_final_model"],
                    "selected_pages": r["ranked_pages"][:A[q]["k"]],
                    "top100": [{"page_id": p, "score": s} for p, s in zip(r["ranked_pages"], r["scores"])]})
    C.write_jsonl(C.OUTPUTS / "data_discovery_output.jsonl", out)

    # ---------------- check against the reported numbers
    lr_all = lmetrics["chosen: BM25 concepts + dense full question + file score + top-5 files"]["all"]
    ad_all = ametrics["adaptive K (nested OOF)"]["all"]
    exp = EXPECTED
    checks = {} if exp is None else {f"light_retrieval {k}": (lr_all[k], v) for k, v in exp["light_retrieval"].items()}
    if exp:
        checks.update({f"adaptive {k}": (ad_all[k], v) for k, v in exp["adaptive"].items()})
    check_rows = {k: {"got": g, "expected": e, "ok": abs(g - e) <= 0.011} for k, (g, e) in checks.items()}
    C.write_json(C.OUTPUTS / "metrics.json", {"light_retrieval": lmetrics, "adaptive_retrieval": ametrics,
                                              "query_preparation": {
                                                  "questions": len(prepared), "excluded": sorted(EX),
                                                  "bm25_fallback_to_question": sum(r["bm25_fallback_to_question"] for r in prepared),
                                                  "llm_failed": sum(r["llm_failed"] for r in prepared),
                                                  "with_scope": sum(bool(r["scope"]) for r in prepared),
                                                  "with_topic": sum(bool(r["topic"]) for r in prepared),
                                                  "with_anchor": sum(bool(r["anchor"]) for r in prepared)},
                                              "reproduction_check": check_rows})
    C.write_json(C.OUTPUTS / "timings.json", timings)

    print("\n== 1. light preparation")
    print("   data   :", timings["1a_data_preparation"])
    print("   queries:", timings["1b_query_preparation"])
    print("== 2. light retrieval (chosen)")
    for s, m in lmetrics["chosen: BM25 concepts + dense full question + file score + top-5 files"].items():
        print(f"   {s:17s}", " ".join(f"{k}={v}" for k, v in m.items()))
    print("  ", timings["2_light_retrieval"])
    print("== 3. adaptive retrieval (nested OOF)")
    for s, m in ametrics["adaptive K (nested OOF)"].items():
        print(f"   {s:17s}", " ".join(f"{k}={v}" for k, v in m.items()))
    print("  ", timings["3_adaptive_retrieval"])
    bad = [k for k, v in check_rows.items() if not v["ok"]]
    run_name = f"{PROVIDER}, {n_q} questions"
    if not check_rows:
        print(f"\nreproduction check: no expected numbers for decomposition run '{run_name}'")
    else:
        print(f"\nreproduction check ({run_name}): {'OK, all ' + str(len(check_rows)) + ' numbers match' if not bad else 'DIFFERENT: ' + ', '.join(bad)}")
    print("total wall", timings["total_wall_s"], "s | outputs in", C.OUTPUTS)


if __name__ == "__main__":
    main()
