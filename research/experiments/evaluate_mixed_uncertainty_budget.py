"""Query-uncertainty budget policies on the cached mixed-text index."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.raw_hierarchical import RawPage, read_jsonl  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DATASET = ROOT / "data/raw/BENCHMARK"
OUTPUT = CACHE / "mixed_uncertainty_budget"


def _metric(qids: list[str], pages: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict[str, float]:
    values, r10, r20 = [], [], []
    for qid in qids:
        gold = qrels[qid]
        selected = pages[qid]
        denom = max(len(gold), 1)
        values.append(len(set(selected) & gold) / denom)
        r10.append(len(set(selected[:10]) & gold) / denom)
        r20.append(len(set(selected[:20]) & gold) / denom)
    return {"recall_at_returned_k": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean([len(pages[qid]) for qid in qids]))}


def main() -> int:
    pages_all = [RawPage(**row) for row in read_jsonl(CACHE / "parsing/pages.jsonl")]
    pages = [page for page in pages_all if page.text_source in {"pdf_inspector", "tesseract"} and page.text.strip()]
    queries = read_jsonl(DATASET / "queries.jsonl")
    qids = [str(query["query_id"]) for query in queries]
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(CACHE / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    index = BM25Index(analyzer_name="auto").build({"chunk_id": p.page_id, "doc_id": p.doc_id, "text": p.text} for p in pages)
    all_ids = [p.page_id for p in pages_all]
    positions = {page_id: i for i, page_id in enumerate(all_ids)}
    embedding_dir = CACHE / "embedding_adaptive_openrouter_te3s_fulltext/embeddings"
    page_vectors = np.load(embedding_dir / "page_embeddings.npy", mmap_mode="r")[[positions[p.page_id] for p in pages]]
    query_vectors = np.load(embedding_dir / "query_embeddings.npy", mmap_mode="r")
    page_ids = [p.page_id for p in pages]
    page_norms = np.linalg.norm(page_vectors, axis=1)
    hybrid_pages: dict[str, list[str]] = {}
    file_pages: dict[str, list[str]] = {}
    features: dict[str, dict[str, float]] = {}
    for i, query in enumerate(queries):
        qid = str(query["query_id"])
        bm_hits = index.search(str(query["query"]), 100)
        bm = [(index.chunk_ids[pos], float(score)) for pos, score in bm_hits]
        bm_rank = {pid: rank for rank, (pid, _s) in enumerate(bm, start=1)}
        bm_max = max((score for _pid, score in bm), default=1.0)
        bm_norm = {pid: score / bm_max for pid, score in bm}
        qv = query_vectors[i]
        dense_scores = (page_vectors @ qv) / (page_norms * np.linalg.norm(qv) + 1e-9)
        dense = sorted(zip(page_ids, dense_scores), key=lambda item: (-float(item[1]), item[0]))[:100]
        dense_rank = {pid: rank for rank, (pid, _s) in enumerate(dense, start=1)}
        dense_by_id = {pid: float(score) for pid, score in dense}
        dense_max = max(dense_by_id.values(), default=1.0)
        dense_norm = {pid: score / dense_max for pid, score in dense_by_id.items()}
        hybrid_scores = {pid: 0.10 * bm_norm.get(pid, 0.0) + 0.90 * dense_norm.get(pid, 0.0) for pid in set(bm_norm) | set(dense_norm)}
        hybrid = sorted(hybrid_scores.items(), key=lambda item: (-item[1], item[0]))[:100]
        hybrid_pages[qid] = [pid for pid, _s in hybrid]
        base_max = max((score for _pid, score in hybrid), default=1.0)
        base_norm = {pid: score / base_max for pid, score in hybrid}
        by_file: dict[str, list[float]] = defaultdict(list)
        for pid, score in hybrid:
            by_file[pid.split("#page=", 1)[0]].append(score)
        file_scores = {f: sum(sorted(v, reverse=True)[:2]) for f, v in by_file.items()}
        file_max = max(file_scores.values(), default=1.0)
        file_scores = {f: v / file_max for f, v in file_scores.items()}
        ranked_file = sorted(((pid, 0.70 * base_norm[pid] + 0.30 * file_scores.get(pid.split("#page=", 1)[0], 0.0)) for pid in base_norm), key=lambda item: (-item[1], item[0]))
        file_pages[qid] = [pid for pid, _s in ranked_file]
        top10_bm = set(bm_rank) & set(list(bm_rank)[:10])
        top10_dense = {pid for pid, _s in dense[:10]}
        overlap = len(set(pid for pid, _s in bm[:10]) & top10_dense)
        scores = np.asarray([score for _pid, score in hybrid[:20]], dtype=float)
        probabilities = scores / (scores.sum() + 1e-9)
        entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-9)))
        gap = float(scores[11] - scores[12]) if len(scores) > 12 else 0.0
        distinct_files = len({pid.split("#page=", 1)[0] for pid, _s in hybrid[:12]})
        features[qid] = {"disagreement": float(10 - overlap), "entropy": entropy, "flatness": -gap, "file_diversity": float(distinct_files), "query_length": float(len(str(query["query"]).split()))}

    policies: dict[str, dict[str, float]] = {"hybrid_fixed_k12": _metric(qids, {qid: hybrid_pages[qid][:12] for qid in qids}, qrels), "file_fixed_k12": _metric(qids, {qid: file_pages[qid][:12] for qid in qids}, qrels)}
    rows: list[dict[str, Any]] = []
    for source_name, ranking in (("hybrid", hybrid_pages), ("file", file_pages)):
        for feature_name in ("disagreement", "entropy", "flatness", "file_diversity", "query_length"):
            ordered = sorted(qids, key=lambda qid: (features[qid][feature_name], qid))
            choices = {qid: 8 if position < len(qids) // 2 else 16 for position, qid in enumerate(ordered)}
            selected = {qid: ranking[qid][:choices[qid]] for qid in qids}
            name = f"{source_name}_uncertainty_{feature_name}_k8or16"
            policies[name] = _metric(qids, selected, qrels)
            rows.append({"policy": name, "feature": feature_name, "choices": {"k8": len(qids) // 2, "k16": len(qids) - len(qids) // 2}, "metrics": policies[name]})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"indexed_pages": len(pages), "queries": len(qids), "policies": policies, "details": rows, "notes": ["Each policy assigns K=8 to half the queries and K=16 to half, so mean K is exactly 12.", "Feature ranking is qrel-free; no parameter is fitted on qrels."]}
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Mixed-text uncertainty budget", "", "| Policy | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    for name, value in policies.items():
        lines.append(f"| {name} | {value['recall_at_returned_k']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} | {value['mean_k']:.2f} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
