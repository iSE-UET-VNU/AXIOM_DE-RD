"""Compare text rerankers between frozen PP-OCRv5 discovery and EVIE.

No adaptive K: 100 candidates -> rerank -> fixed 50 -> cached EVIE top 10.
Qrels and answers are loaded only by the evaluation/QA stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/adaptive/llm/scripts"))
import run_adaptive_accurate_qa as qa  # noqa: E402

OUT = ROOT / "data/work/frozen_top100_reranker_evie_20261026"
GATE = qa.GATE_PATH
PAGES = qa.HANDOFF / "light_prep/pages_ppocrv5.jsonl"
QA_ROWS = qa.QA_RECORDS_PATH
MODEL_IDS = {
    "cohere4pro": "cohere/rerank-4-pro",
    "qwen3_8b": "qwen/qwen3-reranker-8b",
    "voyage25": "voyageai/rerank-2.5",
    "bge_m3_local": "BAAI/bge-reranker-v2-m3",
}
WORD_LIMIT = 256
DOCS_IN = 100
DOCS_TO_EVIE = 50
EVIE_OUT = 10


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_inputs() -> tuple[dict, dict, dict]:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    pages = {row["page_id"]: " ".join(str(row.get("text") or "").split()[:WORD_LIMIT]) or " " for row in load_jsonl(PAGES)}
    queries = {row["query_id"]: row for row in load_jsonl(QA_ROWS)}
    ranked = gate["light_run_top100"]
    assert len(ranked) == len(queries) == 220
    assert len(pages) == 5091
    assert all(len(v) == DOCS_IN and len(set(v)) == DOCS_IN and all(p in pages for p in v) for v in ranked.values())
    assert set(ranked) == set(queries) == set(gate["gold_rank"])
    return gate, pages, queries


def api_key() -> str:
    return qa.load_api_key()


def api_rerank(model: str, query: str, docs: list[str]) -> dict:
    started = perf_counter()
    for attempt in range(4):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/rerank",
                headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
                json={"model": model, "query": query, "documents": docs, "top_n": len(docs), "return_documents": False},
                timeout=240,
            )
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(min(30, 2 ** attempt * 2))
                continue
            response.raise_for_status()
            body = response.json()
            results = body.get("results") or body.get("data") or []
            scores = [float("nan")] * len(docs)
            for item in results:
                idx = int(item["index"])
                scores[idx] = float(item["relevance_score"])
            if len(results) != len(docs) or any(not math.isfinite(s) for s in scores):
                raise ValueError(f"Incomplete score response: {len(results)}/{len(docs)}")
            return {"scores": scores, "latency_seconds": perf_counter() - started,
                    "usage": body.get("usage", {}), "provider_model": body.get("model"),
                    "response_id": body.get("id"), "response_result_count": len(results)}
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt * 2)
    raise AssertionError("unreachable")


def load_bge():
    # The execution environment currently has CPU PyTorch; keep the model cache
    # within the workspace, and avoid silently changing the scoring model.
    os.environ.setdefault("HF_HOME", str(OUT / "hf_cache"))
    import torch
    from sentence_transformers import CrossEncoder
    snapshots = sorted((OUT / "hf_cache/hub/models--BAAI--bge-reranker-v2-m3/snapshots").glob("*"))
    if len(snapshots) != 1 or not (snapshots[0] / "model.safetensors").exists():
        raise FileNotFoundError("Download the public BGE checkpoint to the experiment HF cache first")
    use_cuda = torch.cuda.is_available()
    print(f"[bge_m3_local] device={'cuda' if use_cuda else 'cpu'}; dtype={'float16' if use_cuda else 'float32'}", flush=True)
    return CrossEncoder(str(snapshots[0]), max_length=512, device="cuda" if use_cuda else "cpu",
                        local_files_only=True, model_kwargs={"dtype": torch.float16} if use_cuda else None)


def run_arm(arm: str, qids: list[str], gate: dict, pages: dict, queries: dict, workers: int) -> None:
    model_id = MODEL_IDS[arm]
    folder = OUT / "rerank" / arm
    folder.mkdir(parents=True, exist_ok=True)
    bge = load_bge() if arm == "bge_m3_local" else None

    def one(qid: str) -> tuple[str, str, float]:
        candidates = gate["light_run_top100"][qid]
        query = queries[qid]["query"]
        docs = [pages[p] for p in candidates]
        backend = "api" if bge is None else f"local-{bge.device}-{next(bge.model.parameters()).dtype}"
        input_hash = digest([model_id, backend, WORD_LIMIT, query, candidates, docs])
        path = folder / (hashlib.sha256(qid.encode()).hexdigest() + ".json")
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("input_hash") == input_hash and len(old.get("scores", [])) == DOCS_IN:
                return qid, "cached", 0.0
        if bge is None:
            result = api_rerank(model_id, query, docs)
        else:
            started = perf_counter()
            scores = bge.predict([(query, doc) for doc in docs], batch_size=16, show_progress_bar=False)
            result = {"scores": np.asarray(scores).reshape(-1).astype(float).tolist(),
                      "latency_seconds": perf_counter() - started, "usage": {},
                      "provider_model": model_id, "response_result_count": len(docs)}
        if len(result["scores"]) != DOCS_IN or not all(math.isfinite(s) for s in result["scores"]):
            raise ValueError(f"Invalid rerank scores for {qid}")
        write_json(path, {"qid": qid, "arm": arm, "model": model_id, "backend": backend, "input_hash": input_hash,
                          "candidate_page_ids": candidates, **result})
        return qid, "live", float(result["latency_seconds"])

    started = perf_counter()
    if bge is not None:
        workers = 1
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, qid): qid for qid in qids}
        for count, future in enumerate(as_completed(futures), 1):
            qid = futures[future]
            try:
                _, status, seconds = future.result()
                print(f"[{arm}] {count}/{len(qids)} {status} {seconds:.2f}s {qid}", flush=True)
            except Exception as exc:
                errors.append((qid, f"{type(exc).__name__}: {exc}"))
                print(f"[{arm}] {count}/{len(qids)} ERROR {qid}: {type(exc).__name__}: {exc}", flush=True)
    print(f"[{arm}] wall {perf_counter()-started:.1f}s", flush=True)
    if errors:
        write_json(OUT / f"{arm}_errors.json", errors)
        raise RuntimeError(f"{arm}: {len(errors)}/{len(qids)} rerank requests failed; see {arm}_errors.json")


def ndcg(page_ids: list[str], relevant: set[str]) -> float:
    dcg = sum(1 / math.log2(rank + 2) for rank, page in enumerate(page_ids) if page in relevant)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), len(page_ids))))
    return dcg / ideal if ideal else 0.0


def evaluate(arms: list[str], qids: list[str], gate: dict, queries: dict) -> list[dict]:
    q_index = {qid: i for i, qid in enumerate(json.loads((qa.COLVEC_DIR / "colvec_qids.json").read_text(encoding="utf-8")))}
    p_index = {pid: i for i, pid in enumerate(json.loads((qa.COLVEC_DIR / "colvec_keys.json").read_text(encoding="utf-8")))}
    scores = np.load(qa.EVIE_DIR / "evie_scores_2048.npy", mmap_mode="r")
    assert scores.shape == (220, 5091) and len(q_index) == 220 and len(p_index) == 5091
    rows = []
    for arm in ["baseline"] + arms:
        for qid in qids:
            candidates = gate["light_run_top100"][qid]
            record = None
            if arm == "baseline":
                ranked = candidates
            else:
                path = OUT / "rerank" / arm / (hashlib.sha256(qid.encode()).hexdigest() + ".json")
                if not path.exists():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["candidate_page_ids"] != candidates:
                    raise ValueError(f"Candidate mismatch {arm} {qid}")
                ranked = [candidates[i] for i in sorted(range(DOCS_IN), key=lambda i: (-record["scores"][i], i))]
            prefix = ranked[:DOCS_TO_EVIE]
            accurate = sorted(prefix, key=lambda page: -float(scores[q_index[qid], p_index[page]]))[:EVIE_OUT]
            gold = set(gate["gold_rank"][qid])
            rows.append({"arm": arm, "qid": qid, "source": queries[qid]["source"],
                         "candidate_k": DOCS_IN, "to_evie_k": DOCS_TO_EVIE,
                         "rerank_top100": ranked, "to_evie_pages": prefix, "accurate_top_pages": accurate,
                         "gold_count": len(gold), "rerank_recall_at_10": len(set(ranked[:10]) & gold) / len(gold),
                         "prefix_recall_at_50": len(set(prefix) & gold) / len(gold),
                         "evie_recall_at_10": len(set(accurate) & gold) / len(gold),
                         "rerank_ndcg_at_10": ndcg(ranked[:10], gold),
                         "evie_ndcg_at_10": ndcg(accurate, gold),
                         "rerank_latency_seconds": record.get("latency_seconds") if record else 0.0,
                         "rerank_usage": record.get("usage") if record else {}})
    (OUT / "per_query_retrieval.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    report = {"protocol": {"candidates": 100, "rerank_text_words": WORD_LIMIT, "evie_input": 50,
                           "qa_input": 10, "evie_scores": "cached: inference latency excluded"}, "arms": {}}
    for arm in ["baseline"] + arms:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        def agg(part):
            return {"n": len(part), **{field: round(100 * float(np.mean([r[field] for r in part])), 3)
                                          for field in ("rerank_recall_at_10", "prefix_recall_at_50", "evie_recall_at_10",
                                                        "rerank_ndcg_at_10", "evie_ndcg_at_10")}}
        report["arms"][arm] = {"all": agg(sub), "by_source": {source: agg([r for r in sub if r["source"] == source])
                                                            for source in sorted({r["source"] for r in sub})},
                                "rerank_wall_sum_seconds": sum(float(r["rerank_latency_seconds"] or 0) for r in sub),
                                "rerank_cost_usd": sum(float((r["rerank_usage"] or {}).get("cost") or 0) for r in sub)}
    write_json(OUT / "retrieval_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return rows


def evaluate_cached_qa(rows: list[dict]) -> None:
    """Reuse only exact page-order matches from completed QA arms; never impute."""
    cache = {}
    sources = sorted((ROOT / "research/adaptive/llm/results").glob("*accurate*qa*/per_query.jsonl"))
    sources += sorted((ROOT / "research/adaptive/llm/results").glob("fixed_k58*/per_query.jsonl"))
    for source in sources:
        for row in load_jsonl(source):
            pages = tuple(row.get("accurate_top_pages") or [])
            if row.get("judgment") in qa.LABELS and pages:
                cache.setdefault((row["qid"], pages), {"answer": row.get("answer"),
                                                        "judgment": row["judgment"], "source": str(source)})
    matched = []
    for row in rows:
        found = cache.get((row["qid"], tuple(row["accurate_top_pages"])))
        if found:
            matched.append({"arm": row["arm"], "qid": row["qid"], "source": row["source"], **found})
    (OUT / "exact_cached_qa.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in matched) + "\n", encoding="utf-8")
    report = {}
    for arm in sorted({r["arm"] for r in rows}):
        subset = [r for r in matched if r["arm"] == arm]
        total = sum(r["arm"] == arm for r in rows)
        report[arm] = {"exact_cache_matches": len(subset), "total": total,
                       "correct_among_matched": sum(r["judgment"] == "Correct" for r in subset),
                       "partial_among_matched": sum(r["judgment"] == "Partially Correct" for r in subset),
                       "warning": "Not a QA benchmark unless exact_cache_matches == total"}
    write_json(OUT / "exact_cached_qa_report.json", report)
    print("Exact cached QA coverage:", json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["all", *MODEL_IDS], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pilot-per-source", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    gate, pages, queries = load_inputs()
    qids = sorted(queries)[:args.limit or None]
    if args.pilot_per_source:
        qids = []
        for source in sorted({row["source"] for row in queries.values()}):
            qids.extend([qid for qid in sorted(queries) if queries[qid]["source"] == source][:args.pilot_per_source])
    arms = list(MODEL_IDS) if args.arm == "all" else [args.arm]
    write_json(OUT / "manifest.json", {"model_ids": MODEL_IDS, "candidate_source": str(GATE),
                                       "text_source": str(PAGES), "query_source": str(QA_ROWS),
                                       "queries": len(qids), "corpus_pages": len(pages), "word_limit": WORD_LIMIT,
                                       "candidate_count": DOCS_IN, "evie_input_count": DOCS_TO_EVIE,
                                       "qa_input_count": EVIE_OUT})
    print(f"Preflight OK: {len(qids)} queries, {len(pages)} pages, 100 candidates/query", flush=True)
    if args.preflight:
        return
    if not args.evaluate_only:
        for arm in arms:
            run_arm(arm, qids, gate, pages, queries, args.workers)
    rows = evaluate(arms, qids, gate, queries)
    evaluate_cached_qa(rows)


if __name__ == "__main__":
    main()
