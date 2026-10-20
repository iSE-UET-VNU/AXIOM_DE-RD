"""Residual novelty fusion screen over the cached mixed-text index."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.raw_hierarchical import RawPage, read_jsonl  # noqa: E402
from research.experiments.run_benchmark_adaptive_k import adaptive_k  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DATASET = ROOT / "data/raw/BENCHMARK"
OUTPUT = CACHE / "mixed_residual_fusion"


def metric(qids, rankings, qrels, choices=None):
    values, r10, r20, ks = [], [], [], []
    for qid in qids:
        gold = qrels[qid]
        ranking = rankings[qid]
        k = int(choices[qid]) if choices is not None else 12
        pages = [pid for pid, _score in ranking[:k]]
        denom = max(len(gold), 1)
        values.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
        ks.append(len(pages))
    return {"recall_at_returned_k": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean(ks))}


def main() -> int:
    pages_all = [RawPage(**row) for row in read_jsonl(CACHE / "parsing/pages.jsonl")]
    pages = [p for p in pages_all if p.text_source in {"pdf_inspector", "tesseract"} and p.text.strip()]
    queries = read_jsonl(DATASET / "queries.jsonl")
    qids = [str(q["query_id"]) for q in queries]
    qrels = defaultdict(set)
    for row in read_jsonl(CACHE / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    index = BM25Index(analyzer_name="auto").build({"chunk_id": p.page_id, "doc_id": p.doc_id, "text": p.text} for p in pages)
    all_ids = [p.page_id for p in pages_all]
    pos = {pid: i for i, pid in enumerate(all_ids)}
    emb = CACHE / "embedding_adaptive_openrouter_te3s_fulltext/embeddings"
    page_vecs = np.load(emb / "page_embeddings.npy", mmap_mode="r")[[pos[p.page_id] for p in pages]]
    query_vecs = np.load(emb / "query_embeddings.npy", mmap_mode="r")
    page_ids = [p.page_id for p in pages]
    page_norm = np.linalg.norm(page_vecs, axis=1)
    ranking_names = ["hybrid_a0.10", "rrf10", "rrf20"] + [f"residual_novelty_{weight:.2f}" for weight in (0.10, 0.20, 0.40, 0.80)]
    rankings = {name: {} for name in ranking_names}
    for qi, query in enumerate(queries):
        qid = str(query["query_id"])
        bm_hits = index.search(str(query["query"]), 100)
        bm = [(index.chunk_ids[p], float(s)) for p, s in bm_hits]
        bm_rank = {pid: rank for rank, (pid, _s) in enumerate(bm, start=1)}
        bm_max = max((s for _pid, s in bm), default=1.0)
        bm_norm = {pid: s / bm_max for pid, s in bm}
        qv = query_vecs[qi]
        ds = (page_vecs @ qv) / (page_norm * np.linalg.norm(qv) + 1e-9)
        dense = sorted(zip(page_ids, ds), key=lambda x: (-float(x[1]), x[0]))[:100]
        dense_rank = {pid: rank for rank, (pid, _s) in enumerate(dense, start=1)}
        dense_by_id = {pid: float(s) for pid, s in dense}
        dense_max = max(dense_by_id.values(), default=1.0)
        dense_norm = {pid: s / dense_max for pid, s in dense_by_id.items()}
        hybrid = {pid: 0.10 * bm_norm.get(pid, 0.0) + 0.90 * dense_norm.get(pid, 0.0) for pid in set(bm_norm) | set(dense_norm)}
        rankings["hybrid_a0.10"][qid] = sorted(hybrid.items(), key=lambda x: (-x[1], x[0]))[:100]
        for constant in (10, 20):
            scores = {pid: 1.0 / (constant + bm_rank.get(pid, 1000)) + 1.0 / (constant + dense_rank.get(pid, 1000)) for pid in set(bm_rank) | set(dense_rank)}
            rankings[f"rrf{constant}"][qid] = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:100]
        bm_head = set(pid for pid, _s in bm[:20])
        for weight in (0.10, 0.20, 0.40, 0.80):
            scores = {pid: bm_norm.get(pid, 0.0) + weight * dense_norm.get(pid, 0.0) * (0.0 if pid in bm_head else 1.0) for pid in set(bm_norm) | set(dense_norm)}
            rankings[f"residual_novelty_{weight:.2f}"][qid] = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:100]

    policies = {}
    rows = []
    for name in ranking_names:
        policies[f"{name}_fixed_k12"] = metric(qids, rankings[name], qrels)
        for offset in (5, 6, 7, 8, 9):
            raw_choices = {}
            for qid in qids:
                decision = adaptive_k([s for _pid, s in rankings[name][qid]], ignore_extreme=0, ignore_extreme_tail=0.1, retrieve_more=offset)
                raw_choices[qid] = int(decision["k"])
            total = sum(raw_choices.values())
            # Exact-budget score-only trimming; current offsets are generally
            # at or above 12, so remove lowest-scoring returned tails.
            while total > 12 * len(qids):
                candidates = [qid for qid in qids if raw_choices[qid] > 1]
                qid = min(candidates, key=lambda q: (rankings[name][q][raw_choices[q] - 1][1], q))
                raw_choices[qid] -= 1
                total -= 1
            while total < 12 * len(qids):
                candidates = [qid for qid in qids if raw_choices[qid] < len(rankings[name][qid])]
                qid = max(candidates, key=lambda q: (rankings[name][q][raw_choices[q]][1], q))
                raw_choices[qid] += 1
                total += 1
            policy_name = f"{name}_gap_plus{offset}_exact_mean12"
            policies[policy_name] = metric(qids, rankings[name], qrels, raw_choices)
            rows.append({"policy": policy_name, "metrics": policies[policy_name]})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"indexed_pages": len(pages), "queries": len(qids), "qrels_used_for_ranking": False, "policies": policies, "notes": ["Residual novelty adds dense score only to pages absent from the BM25 top-20 backbone.", "Every adaptive row is calibrated to exactly total K=12*220 using scores only."]}
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Mixed residual novelty fusion", "", "| Policy | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    for name, value in policies.items():
        lines.append(f"| {name} | {value['recall_at_returned_k']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} | {value['mean_k']:.2f} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
