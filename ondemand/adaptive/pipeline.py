import hashlib
import json
from collections import Counter, defaultdict
from time import perf_counter

import numpy as np

from ..bench import BENCH, FLAWED_QIDS, fingerprint, gold as load_gold, load_jsonl, queries as load_queries, work, write_jsonl
from ..evaluate import evaluate, groups
from . import chunk_rank as _chunk_rank, kpolicy as _kpolicy, router as _router  # noqa: F401
from .branches import enrich, light, visual  # noqa: F401
from ..light_retrieval import LiveRetriever
from .chunk_rank import ChunkRank
from . import fallback
from .config import get_config
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
        cache_name="te3s_ppocr", store=None, allow_api=False, limit=None, bench=BENCH, with_qa=False,
        exclude_flawed=False, chandra_endpoint=None, colpali_endpoint=None):
    fallback.reset()
    gate = load_gate(gate_name, bench)
    gold, queries = load_gold(bench), load_queries(bench)
    qids = sorted(gate["gate"])
    if exclude_flawed:
        qids = [q for q in qids if q not in FLAWED_QIDS]
        print(f"[adaptive] excluded {len(FLAWED_QIDS)} flawed questions; evaluating on {len(qids)} clean queries.")
    if limit:
        qids = qids[:limit]

    cfg = get_config()
    chandra_ep = chandra_endpoint or cfg.chandra_endpoint
    colpali_ep = colpali_endpoint or cfg.colpali_endpoint

    embedder = Embedder(bench=bench, cache_name=cache_name, allow_api=allow_api, store=store,
                        query_ids={q["query"]: q["query_id"] for q in queries.values()})
    context = {"bench": bench, "pages_name": pages_name, "refine_k": refine_k, "embedder": embedder,
               "cache_name": cache_name, "allow_api": allow_api, "gate": gate,
               "chandra_endpoint": chandra_ep, "colpali_endpoint": colpali_ep}
    k_policy = build("k", k, **context)
    route = build("router", router, **context)
    chunk_rank = ChunkRank(build("chunker", chunker, **context), build("ranker", ranker, **context), top_chunks)
    branch_args = branch_args or {}
    built = {}

    records, costs, chosen = [], defaultdict(list), Counter()
    print(f"[adaptive] running pipeline on {len(qids)} queries (router: {router}, ranker: {ranker})...")
    for i, qid in enumerate(qids, 1):
        query = queries[qid]["query"]
        started = perf_counter()
        before = fallback.snapshot()
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
            "k": subdata.k, "branch": name, "route": route_notes, "fallbacks": fallback.since(before),
            "subdata": [{"unit": u, "rank": i + 1} for i, u in enumerate(subdata.top())],
            "branch_pages": [{"unit": p.page_id, "rank": p.rank} for p in output.passages],
            "branch_notes": output.notes, "cost": costs[name][-1],
            "chunks": [{"page_id": c.page_id, "page_rank": c.page_rank, "text": c.text} for c in top],
            "chunk_notes": {kk: vv for kk, vv in chunk_notes.items() if kk != "seconds"},
            "arms": {"adaptive": {"top": [{"unit": p.page_id, "rank": p.rank} for p in output.passages]}}})
        if i % 10 == 0 or i == len(qids):
            print(f"\r[adaptive] processed {i}/{len(qids)} queries (latest branch: {name})...", end="", flush=True)
    print("\r" + " " * 65 + "\r", end="", flush=True)

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

    used = defaultdict(dict)
    for r in records:
        for kind, n in r["fallbacks"].items():
            used[kind][r["query_id"]] = n
    summary = {
        "tag": tag, "gate": gate_name, "gate_variant": gate["variant"], "pages_source": pages_name,
        "bundle_fingerprint": fingerprint(bench), "n_queries": len(qids),
        "settings": {"k": k_policy.describe(), "router": route.describe(), "refine_k": refine_k,
                     "chunker": chunk_rank.chunker.describe(), "ranker": chunk_rank.ranker.describe(),
                     "top_chunks": top_chunks, "branches": {n: b.describe() for n, b in built.items()},
                     "embedding_cache": cache_name, "embedding_store": store, "allow_api": allow_api},
        "fallbacks": {kind: {"queries": len(q), "calls": sum(q.values())} for kind, q in sorted(used.items())},
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
        summary["qa"] = run_qa(tag, bench=bench, allow_api=allow_api)
        (out / "metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    for group, m in metrics.items():
        print(f"{group:22s} n={m['n']:3d} ndcg@10 {m['ndcg@10']:6.2f} r@10 {m['recall@10']:6.2f} "
              f"r@20 {m['recall@20']:6.2f} chunk_page_hit {m['chunk_page_hit']:6.2f}")
    fallback.banner(used, len(records))
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


class AdaptivePipeline:
    """High-level Python tool-calling interface for teammates.

    Example:
        >>> from ondemand.adaptive import AdaptivePipeline
        >>> pipe = AdaptivePipeline(gate_name="gate_ppocrv5_all.json", router="rule")
        >>> result = pipe.query("ohrbench::file_65c13bc2ecc7d7ea#page=4")
        >>> print(result["branch"], [c["text"][:50] for c in result["chunks"]])
    """

    def __init__(self, gate_name="gate_ppocrv5_all.json", k="fixed:20", router="rule",
                 pages_name="pages_ppocrv5.jsonl", ranker="hybrid", top_chunks=10, store="kdl",
                 chandra_endpoint=None, colpali_endpoint=None, bench=BENCH, allow_api=False):
        self.bench = bench
        self.pages_name = pages_name
        self.allow_api = allow_api
        self.live = None
        self.gate = load_gate(gate_name, bench)
        self.queries = load_queries(bench)
        self.embedder = Embedder(bench=bench, allow_api=allow_api, store=store,
                                query_ids={q["query"]: q["query_id"] for q in self.queries.values()})
        cfg = get_config()
        chandra_ep = chandra_endpoint or cfg.chandra_endpoint
        colpali_ep = colpali_endpoint or cfg.colpali_endpoint

        self.context = {
            "bench": bench, "pages_name": pages_name, "embedder": self.embedder,
            "gate": self.gate, "chandra_endpoint": chandra_ep,
            "colpali_endpoint": colpali_ep, "allow_api": allow_api
        }
        self.k_policy = build("k", k, **self.context)
        self.router = build("router", router, **self.context)
        self.chunk_rank = ChunkRank(build("chunker", "fixed:512:128", **self.context),
                                    build("ranker", ranker, **self.context), top_chunks)
        self.branches = {}

    def get_branch(self, name):
        if name not in self.branches:
            self.branches[name] = build("branch", name, **self.context)
        return self.branches[name]

    def query(self, qid_or_text, candidate_pages=None):
        """Execute on-demand retrieval for a benchmark query ID or raw query text."""
        fallback.reset()
        if qid_or_text in self.queries:
            qid = qid_or_text
            query = self.queries[qid]["query"]
            pages = candidate_pages or self.gate["gate"].get(qid, [])
            scores = dict(zip(pages, self.gate.get("gate_scores", {}).get(qid, [])))
        else:
            qid = "custom::" + hashlib.sha1(qid_or_text.encode("utf-8")).hexdigest()[:10]
            query = qid_or_text
            if candidate_pages:
                pages, scores = candidate_pages, {}
            else:
                if self.live is None:
                    self.live = LiveRetriever(bench=self.bench, pages_name=self.pages_name, allow_api=self.allow_api)
                ranked, ranked_scores, _ = self.live.retrieve(query)
                pages = ranked
                scores = {p: ranked_scores[p] for p in ranked[:20]}

        subdata = SubData(qid, query, pages, scores, len(pages))
        subdata.k = self.k_policy.choose(subdata)
        branch_name, route_notes = self.router.route(subdata)
        branch = self.get_branch(branch_name)
        output = branch.run(subdata)
        top_chunks, chunk_notes = self.chunk_rank.run(query, output.passages)
        return {
            "qid": qid,
            "query": query,
            "branch": branch_name,
            "route_notes": route_notes,
            "k": subdata.k,
            "passages": output.passages,
            "chunks": [{"page_id": c.page_id, "page_rank": c.page_rank, "text": c.text} for c in top_chunks],
            "cost": output.cost,
            "fallbacks": fallback.since({}),
        }

