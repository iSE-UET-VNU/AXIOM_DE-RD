import hashlib
import json
import logging
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from time import perf_counter

import numpy as np

from ..bench import BENCH, FLAWED_QIDS, ROOT, fingerprint, gold as load_gold, load_jsonl, queries as load_queries, work
from ..evaluate import evaluate, groups
from . import chunk_rank as _chunk_rank, kpolicy as _kpolicy, router as _router  # noqa: F401
from .branches import enrich, light, visual  # noqa: F401
from ..light_retrieval import LiveRetriever
from .chunk_rank import ChunkRank
from .escalate import jev_check, route_action
from . import fallback, services
from .config import MODES, get_config
from .embedding import Embedder
from .registry import build, names
from .services.base import attach, console, detach, gpu_info
from .subdata import SubData

STAGE_ORDER = ("visual", "enrich", "light")
LIVE_FALLBACKS = {f"{service}_{mode}" for service, modes in MODES.items() for mode in modes if mode != "cached"}
log = logging.getLogger("ondemand.run")


def load_gate(name, bench=BENCH):
    path = work("light_prep", bench=bench) / name
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `python -m ondemand.adaptive prep` first")
    gate = json.loads(path.read_text())
    if gate["bundle_fingerprint"] != fingerprint(bench):
        raise SystemExit(f"{name} is from a different bundle")
    return gate


def append(path, row):
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        sink.flush()
        os.fsync(sink.fileno())


def git_state():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return {"sha": sha, "uncommitted_files": len(dirty.splitlines())}
    except OSError:
        return {}


def subdata_for(gate, qid, query):
    pages = gate["gate"][qid]
    # Continuous Adaptive-K experiments use a cap of 200.  Prefer the deepest
    # explicitly materialised ranking, while retaining old top-100 gates.
    ranked = gate.get("light_run_top200", {}).get(qid)
    ranked_scores = gate.get("light_run_scores_top200", {}).get(qid)
    if not ranked:
        ranked = gate.get("light_run_top100", {}).get(qid, pages)
        ranked_scores = gate.get("light_run_scores_top100", {}).get(qid)
    if ranked_scores is None:
        # Legacy gates only stored scores for the small initial gate.
        ranked_scores = gate.get("gate_scores", {}).get(qid, [])
    scores = dict(zip(ranked, ranked_scores))
    return SubData(qid, query, ranked, scores, len(pages))


def run(tag, gate_name, k="fixed:20", router="fixed:visual", escalate="none", branch_args=None, refine_k="fixed:10",
        chunker="fixed:512:128", ranker="hybrid", top_chunks=10, pages_name="pages_ppocrv5.jsonl",
        cache_name="te3s_ppocr", store=None, allow_api=False, limit=None, bench=BENCH, with_qa=False,
        exclude_flawed=False, resume=False, keep_loaded=False, **service_overrides):
    out = work("results", tag, bench=bench)
    records_path = out / "per_query.jsonl"
    if records_path.exists() and not resume:
        raise SystemExit(f"{out} already holds results; pass --resume to continue it, or use a new --tag")
    console()
    handler = attach(out / "run.log")
    try:
        return _run(tag, out, records_path, gate_name, k, router, escalate, branch_args or {}, refine_k, chunker,
                    ranker, top_chunks, pages_name, cache_name, store, allow_api, limit, bench, with_qa,
                    exclude_flawed, keep_loaded, service_overrides)
    except BaseException as error:
        log.error("run %s stopped: %s: %s; rerun with --resume to continue", tag, type(error).__name__, error)
        raise
    finally:
        services.close_all()
        detach(handler)


def _run(tag, out, records_path, gate_name, k, router, escalate, branch_args, refine_k, chunker, ranker, top_chunks,
         pages_name, cache_name, store, allow_api, limit, bench, with_qa, exclude_flawed, keep_loaded, overrides):
    fallback.reset()
    services.close_all()
    services.keep_loaded(keep_loaded)
    cfg = get_config().apply(**overrides)
    live = [name for name, info in cfg.services().items() if info["mode"] != "cached" and name != "colvec"]
    if live and ranker.startswith("hybrid") and not allow_api:
        raise SystemExit(f"{', '.join(live)} will produce page text that has no cached embedding, and --ranker hybrid "
                         f"embeds every chunk; pass --allow-api (te3-small) or use --ranker page_order")
    gate = load_gate(gate_name, bench)
    gold, queries = load_gold(bench), load_queries(bench)
    qids = sorted(gate["gate"])
    if exclude_flawed:
        qids = [q for q in qids if q not in FLAWED_QIDS]
    if limit:
        qids = qids[:limit]

    settings = {"gate": gate_name, "gate_variant": gate["variant"], "pages_source": pages_name, "k": k,
                "router": router, "escalate": escalate, "branch_args": branch_args, "refine_k": refine_k,
                "chunker": chunker, "ranker": ranker, "top_chunks": top_chunks, "embedding_cache": cache_name,
                "embedding_store": store, "exclude_flawed": exclude_flawed, "limit": limit, "services": cfg.public()}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["settings"] != settings:
            changed = sorted(key for key in settings if previous["settings"].get(key) != settings[key])
            raise SystemExit(f"--resume with different settings ({', '.join(changed)}); use a new --tag")
    manifest = {"tag": tag, "bundle_fingerprint": fingerprint(bench), "settings": settings, "git": git_state(),
                "gpu": gpu_info(), "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    done = {r["query_id"]: r for r in load_jsonl(records_path)} if records_path.exists() else {}
    degraded = [q for q, r in done.items() if set(r["fallbacks"]) & LIVE_FALLBACKS]
    for qid in degraded:
        del done[qid]
    todo = [q for q in qids if q not in done]
    log.info("run %s: %d queries, %d already done, %d to go (%d retried after a live-model failure) | services %s",
             tag, len(qids), len(done), len(todo), len(degraded), json.dumps(cfg.services()))

    embedder = Embedder(bench=bench, cache_name=cache_name, allow_api=allow_api, store=store,
                        query_ids={q["query"]: q["query_id"] for q in queries.values()})
    context = {"bench": bench, "pages_name": pages_name, "refine_k": refine_k, "embedder": embedder,
               "cache_name": cache_name, "allow_api": allow_api, "gate": gate}
    k_policy = build("k", k, **context)
    route = build("router", router, **context)
    chunk_rank = ChunkRank(build("chunker", chunker, **context), build("ranker", ranker, **context), top_chunks)
    built = {}

    def branch_of(name):
        if name not in built:
            built[name] = build("branch", branch_args.get(name) or name, **context)
        return built[name]

    plan = []
    for qid in todo:
        started, before = perf_counter(), fallback.snapshot()
        subdata = subdata_for(gate, qid, queries[qid]["query"])
        subdata.k = k_policy.choose(subdata)
        name, route_notes = route.route(subdata)
        plan.append({"qid": qid, "subdata": subdata, "branch": name, "route": route_notes,
                     "route_seconds": perf_counter() - started, "fallbacks": fallback.since(before)})
    counts = Counter(p["branch"] for p in plan)
    log.info("routed %d queries: %s", len(plan), dict(counts))

    for name in sorted(counts, key=lambda n: STAGE_ORDER.index(n) if n in STAGE_ORDER else len(STAGE_ORDER)):
        branch = branch_of(name)
        if hasattr(branch, "prepare"):
            started = perf_counter()
            log.info("stage %s: preparing %d queries with %s", name, counts[name], branch.describe())
            branch.prepare([p["subdata"] for p in plan if p["branch"] == name])
            log.info("stage %s: prepared in %.1fs", name, perf_counter() - started)

    started_all = perf_counter()
    for i, item in enumerate(plan, 1):
        qid, subdata, name = item["qid"], item["subdata"], item["branch"]
        query = subdata.query
        started, before = perf_counter(), fallback.snapshot()
        output = branch_of(name).run(subdata)
        top, chunk_notes = chunk_rank.run(query, output.passages)
        escalation = None
        if escalate == "jev" and name == "light":
            verdict = jev_check(query, top, allow_api, bench=bench)
            action = route_action(verdict)
            escalation = {"checked": True, "action": action, "from": name}
            if action != "complete":
                visual = branch_of("visual")
                if hasattr(visual, "prepare"):
                    visual.prepare([subdata])
                output = visual.run(subdata)
                top, chunk_notes = chunk_rank.run(query, output.passages)
                name = "visual"
                escalation["to"] = name
        used = fallback.since(before)
        for kind, n in item["fallbacks"].items():
            used[kind] = used.get(kind, 0) + n
        cost = {**output.cost, "route_seconds": round(item["route_seconds"], 6),
                "total_seconds": round(item["route_seconds"] + perf_counter() - started, 6),
                "chunk_seconds": round(chunk_notes["seconds"], 6)}
        record = {
            "query_id": qid, "source": queries[qid]["source"], "query": query, "answers": queries[qid]["answers"],
            "gold": [{"unit": u, "relevance": g} for u, g in gold[qid].items()],
            "k": subdata.k, "branch": name, "route": item["route"], "escalation": escalation, "fallbacks": used,
            "subdata": [{"unit": u, "rank": r + 1} for r, u in enumerate(subdata.top())],
            "branch_pages": [{"unit": p.page_id, "rank": p.rank} for p in output.passages],
            "branch_notes": output.notes, "cost": cost,
            "chunks": [{"page_id": c.page_id, "page_rank": c.page_rank, "text": c.text} for c in top],
            "chunk_notes": {kk: vv for kk, vv in chunk_notes.items() if kk != "seconds"},
            "arms": {"adaptive": {"top": [{"unit": p.page_id, "rank": p.rank} for p in output.passages]}}}
        append(records_path, record)
        done[qid] = record
        if used:
            log.warning("%s used fallbacks: %s", qid, used)
        if i % 10 == 0 or i == len(plan):
            rate = (perf_counter() - started_all) / i
            log.info("assembled %d/%d queries (%.2fs/query, ETA %.1f min)", i, len(plan), rate,
                     (len(plan) - i) * rate / 60)

    records = [done[q] for q in qids if q in done]
    tmp = records_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    tmp.replace(records_path)
    summary = summarize(tag, gate, gate_name, pages_name, records, qids, queries, gold, settings, embedder, bench)
    summary["services"] = {n: b.describe() for n, b in built.items()}
    (out / "metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if with_qa:
        from .qa import run as run_qa
        log.info("QA on %d queries", len(records))
        summary["qa"] = run_qa(tag, bench=bench, allow_api=allow_api)
        (out / "metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for group, m in summary["metrics"].items():
        log.info("%-22s n=%3d ndcg@10 %6.2f r@10 %6.2f r@20 %6.2f chunk_page_hit %6.2f", group, m["n"],
                 m["ndcg@10"], m["recall@10"], m["recall@20"], m["chunk_page_hit"])
    used = defaultdict(dict)
    for r in records:
        for kind, n in r["fallbacks"].items():
            used[kind][r["query_id"]] = n
    fallback.banner(used, len(records))
    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log.info("results in %s", out)
    return summary


def summarize(tag, gate, gate_name, pages_name, records, qids, queries, gold, settings, embedder, bench):
    run_rows = {r["query_id"]: {p["unit"]: float(len(r["branch_pages"]) - i) for i, p in enumerate(r["branch_pages"])}
                for r in records}
    by_qid = {r["query_id"]: r for r in records}
    metrics = {}
    for group, members in groups(qids, queries).items():
        members = [q for q in members if q in by_qid]
        if not members:
            continue
        sub = {q: run_rows[q] for q in members}
        n10, r10, _ = evaluate(sub, gold, 10)
        n20, r20, _ = evaluate(sub, gold, 20)
        page_hits = [bool({c["page_id"] for c in by_qid[q]["chunks"]} & {u for u, g in gold[q].items() if g > 0})
                     for q in members]
        metrics[group] = {"n": len(members), "ndcg@10": round(n10, 2), "recall@10": round(r10, 2),
                          "ndcg@20": round(n20, 2), "recall@20": round(r20, 2),
                          "chunk_page_hit": round(100 * float(np.mean(page_hits)), 2)}
    used = defaultdict(dict)
    for r in records:
        for kind, n in r["fallbacks"].items():
            used[kind][r["query_id"]] = n
    costs = defaultdict(list)
    for r in records:
        costs[r["branch"]].append(r["cost"])
    model_keys = sorted({key for r in records for key in r["cost"] if key.endswith("_model_seconds")})
    return {
        "tag": tag, "gate": gate_name, "gate_variant": gate["variant"], "pages_source": pages_name,
        "bundle_fingerprint": fingerprint(bench), "n_queries": len(records), "settings": settings,
        "fallbacks": {kind: {"queries": len(q), "calls": sum(q.values())} for kind, q in sorted(used.items())},
        "branch_counts": dict(Counter(r["branch"] for r in records)),
        "embedding_api_calls": embedder.api_calls, "embedding_store_hits": embedder.store_hits,
        "embedding_cache_hits": embedder.cache_hits,
        "mean_k": round(float(np.mean([r["k"] for r in records])), 2) if records else None,
        "cost": {n: {"queries": len(v), "mean_seconds": round(float(np.mean([c["total_seconds"] for c in v])), 4),
                     "mean_pages": round(float(np.mean([c["pages"] for c in v])), 2),
                     "mean_parsed_pages": round(float(np.mean([c["parsed_pages"] for c in v])), 2),
                     **{f"mean_{key}": round(float(np.mean([c.get(key, 0.0) for c in v])), 4) for key in model_keys}}
                 for n, v in costs.items()},
        "metrics": metrics}


def available():
    return {kind: names(kind) for kind in ("k", "router", "branch", "scorer", "parser", "enricher",
                                           "chunker", "ranker")}


def artifacts(bench=BENCH, pages_name="pages_ppocrv5.jsonl"):
    wanted = {
        "light prep pages": work("light_prep", bench=bench) / pages_name,
        "ColVec scores": work("colvec", bench=bench) / "colvec_scores.npy",
        "KDL pages": work("kdl", bench=bench) / "kdl_pages.jsonl",
        "te3-small cache": work("embedding_cache", "te3s_ppocr", bench=bench),
        "enriched pages": work("enrich", bench=bench) / "enriched_pages.jsonl"}
    gates = sorted(p.name for p in work("light_prep", bench=bench).glob("gate_*.json"))
    live = {name: sorted(str(p.parent.name) for p in work(name, "live", bench=bench).glob("*/settings.json"))
            for name in services.SERVICES}
    return {"bundle_fingerprint": fingerprint(bench),
            "present": {k: str(v) for k, v in wanted.items() if v.exists()},
            "missing": {k: str(v) for k, v in wanted.items() if not v.exists()},
            "gates": gates, "live_caches": live}


class AdaptivePipeline:
    def __init__(self, gate_name="gate_ppocrv5_all.json", k="fixed:20", router="rule",
                 pages_name="pages_ppocrv5.jsonl", ranker="hybrid", top_chunks=10, store="kdl", refine_k="fixed:10",
                 chunker="fixed:512:128", bench=BENCH, allow_api=False, keep_loaded=False, **service_overrides):
        console()
        services.close_all()
        services.keep_loaded(keep_loaded)
        get_config().apply(**service_overrides)
        self.bench, self.pages_name, self.allow_api = bench, pages_name, allow_api
        self.live = None
        self.gate = load_gate(gate_name, bench)
        self.queries = load_queries(bench)
        self.embedder = Embedder(bench=bench, allow_api=allow_api, store=store,
                                 query_ids={q["query"]: q["query_id"] for q in self.queries.values()})
        self.context = {"bench": bench, "pages_name": pages_name, "embedder": self.embedder, "gate": self.gate,
                        "allow_api": allow_api, "refine_k": refine_k}
        self.k_policy = build("k", k, **self.context)
        self.router = build("router", router, **self.context)
        self.chunk_rank = ChunkRank(build("chunker", chunker, **self.context),
                                    build("ranker", ranker, **self.context), top_chunks)
        self.branches = {}

    def get_branch(self, name):
        if name not in self.branches:
            self.branches[name] = build("branch", name, **self.context)
        return self.branches[name]

    def query(self, qid_or_text, candidate_pages=None, branch=None):
        fallback.reset()
        if qid_or_text in self.queries:
            qid = qid_or_text
            query = self.queries[qid]["query"]
            subdata = subdata_for(self.gate, qid, query)
            if candidate_pages:
                subdata = SubData(qid, query, candidate_pages, {}, len(candidate_pages))
        else:
            qid = "custom::" + hashlib.sha1(qid_or_text.encode("utf-8")).hexdigest()[:10]
            query = qid_or_text
            if candidate_pages:
                subdata = SubData(qid, query, candidate_pages, {}, len(candidate_pages))
            else:
                if self.live is None:
                    self.live = LiveRetriever(bench=self.bench, pages_name=self.pages_name, allow_api=self.allow_api)
                ranked, ranked_scores, _ = self.live.retrieve(query)
                subdata = SubData(qid, query, ranked, {p: ranked_scores[p] for p in ranked}, len(ranked))
        subdata.k = self.k_policy.choose(subdata)
        branch_name, route_notes = (branch, {"reason": "forced"}) if branch else self.router.route(subdata)
        chosen = self.get_branch(branch_name)
        if hasattr(chosen, "prepare"):
            chosen.prepare([subdata])
        output = chosen.run(subdata)
        top_chunks, chunk_notes = self.chunk_rank.run(query, output.passages)
        return {"qid": qid, "query": query, "branch": branch_name, "route_notes": route_notes, "k": subdata.k,
                "passages": output.passages, "branch_notes": output.notes,
                "chunks": [{"page_id": c.page_id, "page_rank": c.page_rank, "text": c.text} for c in top_chunks],
                "cost": output.cost, "fallbacks": fallback.since({})}

    def close(self):
        services.close_all()
