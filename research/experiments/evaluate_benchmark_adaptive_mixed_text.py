"""Evaluate adaptive/file-aware retrieval on cached PDF-inspector + Tesseract text."""

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
from research.experiments.run_benchmark_adaptive_k import adaptive_k  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DATASET = ROOT / "data/raw/BENCHMARK"
OUTPUT = CACHE / "adaptive_mixed_text"


def metrics(rows: list[dict[str, Any]], qrels: dict[str, set[str]]) -> dict[str, float]:
    values, r10, r20 = [], [], []
    for row in rows:
        gold = qrels[row["query_id"]]
        pages = row["pages"]
        denom = max(len(gold), 1)
        values.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
    return {"recall_at_returned_k": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean([len(row["pages"]) for row in rows]))}


def main() -> int:
    pages_all = [RawPage(**row) for row in read_jsonl(CACHE / "parsing/pages.jsonl")]
    pages = [page for page in pages_all if page.text_source in {"pdf_inspector", "tesseract"} and page.text.strip()]
    queries = read_jsonl(DATASET / "queries.jsonl")
    qids = [str(query["query_id"]) for query in queries]
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(CACHE / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    index = BM25Index(analyzer_name="auto").build({"chunk_id": page.page_id, "doc_id": page.doc_id, "text": page.text} for page in pages)
    all_ids = [page.page_id for page in pages_all]
    position_by_id = {page_id: position for position, page_id in enumerate(all_ids)}
    positions = np.asarray([position_by_id[page.page_id] for page in pages], dtype=np.int64)
    embedding_dir = CACHE / "embedding_adaptive_openrouter_te3s_fulltext/embeddings"
    page_vectors = np.load(embedding_dir / "page_embeddings.npy", mmap_mode="r")[positions]
    query_vectors = np.load(embedding_dir / "query_embeddings.npy", mmap_mode="r")
    page_ids = [page.page_id for page in pages]
    page_norms = np.linalg.norm(page_vectors, axis=1)
    rankings: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for q_index, query in enumerate(queries):
        qid = str(query["query_id"])
        text = str(query["query"])
        bm_hits = index.search(text, 100)
        bm = [(index.chunk_ids[pos], float(score)) for pos, score in bm_hits]
        bm_max = max((score for _pid, score in bm), default=1.0)
        bm_norm = {pid: score / bm_max for pid, score in bm}
        qv = query_vectors[q_index]
        dense_scores = (page_vectors @ qv) / (page_norms * np.linalg.norm(qv) + 1e-9)
        dense_by_id = {pid: float(score) for pid, score in zip(page_ids, dense_scores)}
        dense_max = max(dense_by_id.values(), default=1.0)
        dense_norm = {pid: score / dense_max for pid, score in dense_by_id.items()}
        hybrid_scores = {pid: 0.10 * bm_norm.get(pid, 0.0) + 0.90 * dense_norm[pid] for pid in page_ids}
        hybrid = sorted(hybrid_scores.items(), key=lambda item: (-item[1], item[0]))[:100]
        page_max = max((score for _pid, score in hybrid), default=1.0)
        page_norm = {pid: score / page_max for pid, score in hybrid}
        by_file: dict[str, list[float]] = defaultdict(list)
        for pid, score in hybrid:
            by_file[pid.split("#page=", 1)[0]].append(score)
        file_scores = {file_id: sum(sorted(scores, reverse=True)[:2]) for file_id, scores in by_file.items()}
        file_max = max(file_scores.values(), default=1.0)
        file_norm = {file_id: score / file_max for file_id, score in file_scores.items()}
        file_scores_pages = {pid: 0.70 * page_norm[pid] + 0.30 * file_norm.get(pid.split("#page=", 1)[0], 0.0) for pid in page_norm}
        file_ranked = sorted(file_scores_pages.items(), key=lambda item: (-item[1], item[0]))
        rankings[qid] = {"bm25": bm, "hybrid": hybrid, "file": file_ranked}

    policies: dict[str, dict[str, float]] = {}
    all_rows: list[dict[str, Any]] = []
    for name in ("bm25", "hybrid", "file"):
        fixed = [{"query_id": qid, "pages": [pid for pid, _score in rankings[qid][name][:12]], "scores": [float(score) for _pid, score in rankings[qid][name][:12]]} for qid in qids]
        policy_name = f"{name}_fixed_k12"
        policies[policy_name] = metrics(fixed, qrels)
        all_rows.extend({**row, "policy": policy_name} for row in fixed)
        if name == "file":
            for offset in range(5, 16):
                adaptive_rows = []
                for qid in qids:
                    ranking = rankings[qid][name]
                    decision = adaptive_k([score for _pid, score in ranking], ignore_extreme=0, ignore_extreme_tail=0.1, retrieve_more=offset)
                    k = int(decision["k"])
                    adaptive_rows.append({"query_id": qid, "pages": [pid for pid, _score in ranking[:k]], "scores": [float(score) for _pid, score in ranking[:k]], "k": k})
                policy_name = f"file_largest_gap_plus{offset}"
                policies[policy_name] = metrics(adaptive_rows, qrels)
                all_rows.extend({**row, "policy": policy_name} for row in adaptive_rows)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"pages_all": len(pages_all), "indexed_pages": len(pages), "text_sources": ["pdf_inspector", "tesseract"], "queries": len(qids), "qrels_used_for_ranking": False, "policies": policies}
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "all_policy_rows.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in all_rows) + "\n", encoding="utf-8")
    lines = ["# Adaptive mixed-text retrieval", "", "Cached PDF-inspector + Tesseract pages; empty pages excluded.", "", "| Policy | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    for name, value in policies.items():
        lines.append(f"| {name} | {value['recall_at_returned_k']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} | {value['mean_k']:.2f} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
