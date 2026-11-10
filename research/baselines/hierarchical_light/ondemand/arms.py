import json
from collections import defaultdict
from time import perf_counter

import numpy as np

from .bench import BENCH, fingerprint, gold as load_gold, queries as load_queries, work, write_jsonl
from .evaluate import summarize
from .text import ChunkBM25
from .timing import Timings

ALPHA, DEPTH, VISUAL_WEIGHT = 0.7, 100, 0.8


class ColVec:
    def __init__(self, bench=BENCH):
        folder = work("colvec", bench=bench)
        self.meta = json.loads((folder / "colvec_meta.json").read_text())
        if self.meta["bundle_fingerprint"] != fingerprint(bench):
            raise SystemExit("ColVec scores are from a different bundle")
        self.matrix = np.load(folder / "colvec_scores.npy")
        self.col = {k: i for i, k in enumerate(json.loads((folder / "colvec_keys.json").read_text()))}
        self.row = {q: i for i, q in enumerate(json.loads((folder / "colvec_qids.json").read_text()))}

    def score(self, qid, unit):
        return float(self.matrix[self.row[qid], self.col[unit]])


def minmax(values):
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high - low <= 1e-12:
        return {k: 1.0 for k in values}
    return {k: (v - low) / (high - low) for k, v in values.items()}


def minmax_run(run):
    if not run:
        return {}
    values = [s for _, s in run]
    low, high = min(values), max(values)
    if high - low <= 0:
        return {p: 1.0 for p, _ in run}
    return {p: (s - low) / (high - low) for p, s in run}


def alpha_fuse(lexical, dense, alpha, top_k):
    units = list(dict.fromkeys(list(lexical) + list(dense)))
    index = {u: i for i, u in enumerate(units)}
    left = minmax_run([(index[u], s) for u, s in lexical.items()])
    right = minmax_run([(index[u], s) for u, s in dense.items()])
    fused = defaultdict(float)
    for p, s in left.items():
        fused[p] += (1.0 - alpha) * s
    for p, s in right.items():
        fused[p] += alpha * s
    return {units[p]: float(s) for p, s in sorted(fused.items(), key=lambda item: -item[1])[:top_k]}


def visual_fuse(text, visual, weight):
    t, v = minmax(text), minmax(visual)
    fused = {u: (1 - weight) * s + weight * v.get(u, 0.0) for u, s in t.items()}
    for u, s in v.items():
        fused.setdefault(u, weight * s)
    return fused


def maxp(scored, owners):
    best = {}
    for position, score in scored:
        unit = owners[position]
        if score > best.get(unit, -1e30):
            best[unit] = float(score)
    return best


def arm_a(gate, colvec, qids):
    return {q: {u: colvec.score(q, u) for u in gate["gate"][q]} for q in qids}


def arm_b(gate, colvec, qids, queries, timings, bench=BENCH):
    folder = work("embeddings", "kdl", bench=bench)
    owners_all = np.array(json.loads((folder / "chunk_owners.json").read_text()))
    texts_all = json.loads((folder / "chunk_texts.json").read_text())
    vectors_all = np.load(folder / "chunk_vectors.npy")
    vectors_all = vectors_all / np.clip(np.linalg.norm(vectors_all, axis=-1, keepdims=True), 1e-12, None)
    qvec_all = np.load(folder / "query_vectors.npy")
    qvec_all = qvec_all / np.clip(np.linalg.norm(qvec_all, axis=-1, keepdims=True), 1e-12, None)
    qvec = {q: qvec_all[i] for i, q in enumerate(json.loads((folder / "query_ids.json").read_text()))}
    union = set(gate["union"])
    keep = np.array([i for i, u in enumerate(owners_all) if u in union], dtype=int)
    owners, vectors = owners_all[keep], vectors_all[keep]
    bm25 = ChunkBM25([texts_all[i] for i in keep])
    positions = defaultdict(list)
    for j, u in enumerate(owners):
        positions[u].append(j)
    text_run, fused_run = {}, {}
    for q in qids:
        started = perf_counter()
        allowed_units = [u for u in gate["gate"][q] if u in positions]
        if not allowed_units:
            text_run[q], fused_run[q] = {}, {}
            continue
        allowed = {j for u in allowed_units for j in positions[u]}
        lexical = maxp(bm25.search(queries[q]["query"], DEPTH, allowed), owners)
        idx = np.fromiter(allowed, dtype=int, count=len(allowed))
        sims = vectors[idx] @ qvec[q]
        dense = maxp(sorted(zip(idx.tolist(), sims.tolist()), key=lambda x: -x[1])[:DEPTH], owners)
        base = alpha_fuse(lexical, dense, ALPHA, len(allowed_units))
        text_run[q] = base
        fused_run[q] = visual_fuse(base, {u: colvec.score(q, u) for u in allowed_units}, VISUAL_WEIGHT)
        timings.record("arm_b_query", "query", 1, perf_counter() - started, qid=q, candidates=len(allowed_units))
    return text_run, fused_run


def top(run_row, gold_row, k=20):
    ranked = sorted(run_row, key=lambda u: -run_row[u])
    rank = {u: i + 1 for i, u in enumerate(ranked)}
    return {"top": [{"unit": u, "rank": i + 1, "score": round(float(run_row[u]), 6)} for i, u in enumerate(ranked[:k])],
            "gold_rank": {u: rank.get(u) for u in gold_row}}


def run(tag, gate_name="gate_k20_best.json", bench=BENCH):
    gate = json.loads((work("light_prep", bench=bench) / gate_name).read_text())
    if gate["bundle_fingerprint"] != fingerprint(bench):
        raise SystemExit("gate is from a different bundle")
    gold, queries = load_gold(bench), load_queries(bench)
    qids = sorted(gate["gate"])
    out = work("results", tag, bench=bench)
    timings = Timings(out / "timings.jsonl", stage_group="arms", gate=gate["variant"])
    colvec = ColVec(bench)
    runs = {"A": arm_a(gate, colvec, qids)}
    runs["B_text"], runs["B"] = arm_b(gate, colvec, qids, queries, timings, bench)
    metrics = {"gate": {k: gate[k] for k in ("variant", "gate_k", "bundle_fingerprint", "n_queries", "n_union_pages", "metrics")},
               "arms": {}, "alpha": ALPHA, "depth": DEPTH, "visual_weight": VISUAL_WEIGHT}
    per_query = {}
    for name, r in runs.items():
        metrics["arms"][name], per_query[name] = summarize(r, gold, queries, qids)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    records = []
    for q in qids:
        query = queries[q]
        rec = {"query_id": q, "source": query["source"], "visual_class": query["metadata"].get("visual_class"),
               "query": query["query"], "answers": query["answers"],
               "gold": [{"unit": u, "relevance": g} for u, g in gold[q].items()],
               "gate": {"variant": gate["variant"], "units": gate["gate"][q], "gold_rank": gate["gold_rank"][q]}, "arms": {}}
        for name, r in runs.items():
            rec["arms"][name] = {**top(r[q], gold[q]), **per_query[name].get(q, {})}
        records.append(rec)
    write_jsonl(out / "per_query.jsonl", records)
    return metrics
