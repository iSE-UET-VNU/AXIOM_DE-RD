import json
from collections import Counter, defaultdict
from time import perf_counter

import numpy as np

from ..bench import BENCH, fingerprint, gold as load_gold, load_jsonl, queries as load_queries, work, write_jsonl
from ..evaluate import evaluate, groups
from . import chunk_rank as _chunk_rank, kpolicy as _kpolicy, router as _router  # noqa: F401
from .branches import enrich, light, visual  # noqa: F401
from .chunk_rank import ChunkRank
from .embedding import Embedder
from .registry import build, names
from .subdata import SubData


def load_gate(name, bench=BENCH):
    path = work("light_prep", bench=bench) / name
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `python -m ondemand light-retrieval` first")
    gate = json.loads(path.read_text())
    if gate["bundle_fingerprint"] != fingerprint(bench):
        raise SystemExit(f"{name} is from a different bundle")
    return gate


def run(tag, gate_name, k="fixed:20", router="fixed:visual", branch_args=None, refine_k="fixed:10",
        chunker="fixed:512:128", ranker="hybrid", top_chunks=10, pages_name="pages_ocr_ppocr.jsonl",
        cache_name="te3s_ppocr", store=None, allow_api=False, limit=None, bench=BENCH, with_qa=False):
    gate = load_gate(gate_name, bench)
    gold, queries = load_gold(bench), load_queries(bench)
    qids = sorted(gate["gate"])
    if limit:
        qids = qids[:limit]

    embedder = Embedder(bench=bench, cache_name=cache_name, allow_api=allow_api, store=store,
                        query_ids={q["query"]: q["query_id"] for q in queries.values()})
    context = {"bench": bench, "pages_name": pages_name, "refine_k": refine_k, "embedder": embedder,
               "cache_name": cache_name, "allow_api": allow_api}
    k_policy = build("k", k, **context)
    route = build("router", router, **context)
    chunk_rank = ChunkRank(build("chunker", chunker, **context), build("ranker", ranker, **context), top_chunks)
    branch_args = branch_args or {}
    built = {}

    records, costs, chosen = [], defaultdict(list), Counter()
    for qid in qids:
        query = queries[qid]["query"]
        started = perf_counter()
        pages = gate["gate"][qid]
        scores = dict(zip(pages, gate.get("gate_scores", {}).get(qid, [])))
        subdata = SubData(qid, query, gate.get("light_run_top100", {}).get(qid, pages), scores, len(pages))
        subdata.k = k_policy.choose(subdata)
        name, route_notes = route.route(subdata)
        if name not in built:
            built[name] = build("branch", branch_args.get(name, name) if branch_args.get(name) else name, **context)
        branch = built[name]
        output = branch.run(subdata)
        top, chunk_notes = chunk_rank.run(query, output.passages)
        seconds = perf_counter() - started
        chosen[name] += 1
        costs[name].append({**output.cost, "total_seconds": seconds, "chunk_seconds": chunk_notes["seconds"]})
        records.append({
            "query_id": qid, "source": queries[qid]["source"], "query": query,
            "answers": queries[qid]["answers"],
            "gold": [{"unit": u, "relevance": g} for u, g in gold[qid].items()],
            "k": subdata.k, "branch": name, "route": route_notes,
            "subdata": [{"unit": u, "rank": i + 1} for i, u in enumerate(subdata.top())],
            "branch_pages": [{"unit": p.page_id, "rank": p.rank} for p in output.passages],
            "branch_notes": output.notes, "cost": costs[name][-1],
            "chunks": [{"page_id": c.page_id, "page_rank": c.page_rank, "text": c.text} for c in top],
            "chunk_notes": {kk: vv for kk, vv in chunk_notes.items() if kk != "seconds"},
            "arms": {"adaptive": {"top": [{"unit": p.page_id, "rank": p.rank} for p in output.passages]}}})

    out = work("results", tag, bench=bench)
    run_rows = {r["query_id"]: {p["unit"]: float(len(r["branch_pages"]) - i)
                                for i, p in enumerate(r["branch_pages"])} for r in records}
    metrics = {}
    for group, members in groups(qids, queries).items():
        sub = {q: run_rows[q] for q in members if q in run_rows}
        n10, r10, _ = evaluate(sub, gold, 10)
        n20, r20, _ = evaluate(sub, gold, 20)
        by_qid = {r["query_id"]: r for r in records}
        page_hits = [bool({c["page_id"] for c in by_qid[q]["chunks"]} & {u for u, g in gold[q].items() if g > 0})
                     for q in members]
        metrics[group] = {"n": len(members), "ndcg@10": round(n10, 2), "recall@10": round(r10, 2),
                          "ndcg@20": round(n20, 2), "recall@20": round(r20, 2),
                          "chunk_page_hit": round(100 * float(np.mean(page_hits)), 2)}

    summary = {
        "tag": tag, "gate": gate_name, "gate_variant": gate["variant"], "pages_source": pages_name,
        "bundle_fingerprint": fingerprint(bench), "n_queries": len(qids),
        "settings": {"k": k_policy.describe(), "router": route.describe(), "refine_k": refine_k,
                     "chunker": chunk_rank.chunker.describe(), "ranker": chunk_rank.ranker.describe(),
                     "top_chunks": top_chunks, "branches": {n: b.describe() for n, b in built.items()},
                     "embedding_cache": cache_name, "embedding_store": store, "allow_api": allow_api},
        "branch_counts": dict(chosen), "embedding_api_calls": embedder.api_calls,
        "embedding_store_hits": embedder.store_hits, "embedding_cache_hits": embedder.cache_hits,
        "mean_k": round(float(np.mean([r["k"] for r in records])), 2),
        "cost": {n: {"queries": len(v),
                     "mean_seconds": round(float(np.mean([c["total_seconds"] for c in v])), 4),
                     "mean_pages": round(float(np.mean([c["pages"] for c in v])), 2),
                     "mean_parsed_pages": round(float(np.mean([c["parsed_pages"] for c in v])), 2)}
                 for n, v in costs.items()},
        "metrics": metrics}
    (out / "metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write_jsonl(out / "per_query.jsonl", records)

    if with_qa:
        from .qa import run as run_qa
        summary["qa"] = run_qa(tag, bench=bench)
        (out / "metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    for group, m in metrics.items():
        print(f"{group:22s} n={m['n']:3d} ndcg@10 {m['ndcg@10']:6.2f} r@10 {m['recall@10']:6.2f} "
              f"r@20 {m['recall@20']:6.2f} chunk_page_hit {m['chunk_page_hit']:6.2f}")
    print(json.dumps({"branches": summary["branch_counts"], "mean_k": summary["mean_k"],
                      "embedding_api_calls": summary["embedding_api_calls"]}))
    print(out)
    return summary


def available():
    return {kind: names(kind) for kind in ("k", "router", "branch", "scorer", "parser", "enricher",
                                           "chunker", "ranker")}


def artifacts(bench=BENCH, pages_name="pages_ocr_ppocr.jsonl"):
    wanted = {
        "light prep pages": work("light_prep", bench=bench) / pages_name,
        "ColVec scores": work("colvec", bench=bench) / "colvec_scores.npy",
        "KDL pages": work("kdl", bench=bench) / "kdl_pages.jsonl",
        "te3-small cache": work("embedding_cache", "te3s_ppocr", bench=bench),
        "enriched pages": work("enrich", bench=bench) / "enriched_pages.jsonl"}
    gates = sorted(p.name for p in work("light_prep", bench=bench).glob("gate_*.json"))
    return {"bundle_fingerprint": fingerprint(bench),
            "present": {k: str(v) for k, v in wanted.items() if v.exists()},
            "missing": {k: str(v) for k, v in wanted.items() if not v.exists()},
            "gates": gates}
