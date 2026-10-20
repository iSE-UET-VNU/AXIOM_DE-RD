"""Global retrieval control for cached KDL chunks (no light/Cohere candidate gate)."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.retrieval import alpha_fuse  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402

BUNDLE = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803"
QUERIES = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/baseline_bm25.jsonl"
OLD_QRELS = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval/input/qrels_page_level.jsonl"
OHR_QRELS = ROOT / "data/work/ohr_replacement_20260920/inputs/qrels.jsonl"
FILE_LANGUAGES = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval/reports/language_routed_bm25/file_language_detection.jsonl"
OUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/kdl_global_chunk_retrieval"


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source(qid: str) -> str:
    return qid.split("::", 1)[0]


def file_id_from_page(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def minmax(scores: np.ndarray) -> np.ndarray:
    lower, upper = float(scores.min()), float(scores.max())
    if upper <= lower:
        return np.zeros_like(scores)
    return (scores - lower) / (upper - lower)


def score_metrics(run: dict[str, dict[str, float]], gold: dict[str, set[str]], qids: list[str], k: int) -> dict[str, float]:
    recalls, ndcgs = [], []
    for qid in qids:
        ranked = sorted(run[qid], key=lambda page: (-run[qid][page], page))[:k]
        hits = [page in gold[qid] for page in ranked]
        recalls.append(sum(hits) / len(gold[qid]))
        dcg = sum(1 / math.log2(rank + 1) for rank, hit in enumerate(hits, 1) if hit)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(gold[qid])) + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    return {f"recall@{k}": 100 * float(np.mean(recalls)), f"ndcg@{k}": 100 * float(np.mean(ndcgs))}


def main() -> None:
    emb = BUNDLE / "embeddings/kdl"
    owners = np.asarray(json.loads((emb / "chunk_owners.json").read_text(encoding="utf-8")))
    texts = json.loads((emb / "chunk_texts.json").read_text(encoding="utf-8"))
    vectors = np.load(emb / "chunk_vectors.npy").astype(np.float32)
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    qids = json.loads((emb / "query_ids.json").read_text(encoding="utf-8"))
    qvectors = np.load(emb / "query_vectors.npy").astype(np.float32)
    qvectors /= np.clip(np.linalg.norm(qvectors, axis=1, keepdims=True), 1e-12, None)
    query_text = {str(row["query_id"]): str(row["query"]) for row in read(QUERIES)}
    if set(qids) != set(query_text):
        raise RuntimeError("query ids differ")
    gold: dict[str, set[str]] = defaultdict(set)
    for row in read(OLD_QRELS):
        qid = str(row["query_id"])
        if qid.startswith(("vidore_physics::", "vidore_industrial::")):
            gold[qid].add(str(row["page_id"]))
    for row in read(OHR_QRELS):
        gold[str(row["query_id"])].add(f"{row['doc_id']}#page={int(row['page_index'])}")
    index = BM25Index(analyzer_name="plain").build([{"chunk_id": f"c{i}", "doc_id": str(owners[i]), "text": texts[i]} for i in range(len(owners))])
    detected_languages = {str(row["file_id"]): str(row["detected_language"]) for row in read(FILE_LANGUAGES)}
    # The replacement OHR corpus is English.  For the older sources, use the
    # same detection artifact as the official language-routed light hybrid.
    owner_languages = np.asarray([
        "en" if str(page).startswith("ohrbench::") else detected_languages.get(file_id_from_page(str(page)), "unknown")
        for page in owners
    ])
    missing_language_files = sorted({file_id_from_page(str(page)) for page, language in zip(owners, owner_languages) if language == "unknown"})
    if missing_language_files:
        raise RuntimeError(f"missing detected language for {len(missing_language_files)} KDL files: {missing_language_files[:5]}")
    same_language_chunk = owner_languages == "en"  # Cached query texts are English.
    runs: dict[str, dict[str, dict[str, float]]] = {"kdl_dense_only": {}, "kdl_hybrid_dense07": {}, "kdl_hybrid_dense03": {}, "kdl_language_routed": {}}
    detail = []
    for qpos, qid in enumerate(qids):
        dense_score = vectors @ qvectors[qpos]
        dense_pos = np.argsort(-dense_score)[:100]
        dense = [(int(pos), float(dense_score[pos])) for pos in dense_pos]
        lexical_all = [(int(pos), float(score)) for pos, score in index.search(query_text[qid], len(owners))]
        lexical = lexical_all[:100]
        # Match the official arm's direction: English query vs non-English
        # document is dense-only; English document uses 0.70 BM25 + 0.30 dense.
        bm25_raw = np.zeros(len(owners), dtype=np.float32)
        for pos, score in lexical_all:
            bm25_raw[pos] = score
        routed_scores = minmax(dense_score)
        hybrid_scores = 0.70 * minmax(bm25_raw) + 0.30 * routed_scores
        routed_scores[same_language_chunk] = hybrid_scores[same_language_chunk]
        routed_pos = np.argsort(-routed_scores)[:100]
        routed = [(int(pos), float(routed_scores[pos])) for pos in routed_pos]
        chunk_runs = {"kdl_dense_only": dense, "kdl_hybrid_dense07": alpha_fuse(lexical, dense, alpha=0.7, top_k=100), "kdl_hybrid_dense03": alpha_fuse(lexical, dense, alpha=0.3, top_k=100), "kdl_language_routed": routed}
        row = {"query_id": qid, "source": source(qid), "gold_pages": sorted(gold[qid]), "runs": {}}
        for name, chunks in chunk_runs.items():
            pages: dict[str, float] = {}
            for pos, value in chunks:
                page = str(owners[pos])
                pages[page] = max(pages.get(page, -float("inf")), float(value))
            runs[name][qid] = pages
            row["runs"][name] = [{"page_id": page, "score": score} for page, score in sorted(pages.items(), key=lambda item: (-item[1], item[0]))[:20]]
        detail.append(row)
    report: dict[str, Any] = {"protocol": {"corpus": "all 5,873 cached KDL chunks", "chunking": "512 words / overlap 128", "page_score": "max fused chunk score", "chunk_depth": 100, "candidate_gate": "none", "language_routed": "English query + non-English file: dense-only; English query + English file: 0.70 normalized BM25 + 0.30 normalized dense", "language_source": str(FILE_LANGUAGES.relative_to(ROOT)), "language_counts_by_chunk": {language: int((owner_languages == language).sum()) for language in sorted(set(owner_languages))}}, "metrics": {}}
    for name, run in runs.items():
        report["metrics"][name] = {}
        for group in ("all", "vidore_physics", "vidore_industrial", "ohrbench"):
            group_qids = qids if group == "all" else [qid for qid in qids if source(qid) == group]
            report["metrics"][name][group] = {"queries": len(group_qids), **score_metrics(run, gold, group_qids, 10), **score_metrics(run, gold, group_qids, 20)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "per_query.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in detail), encoding="utf-8")
    lines = ["# Global cached KDL chunk retrieval", "", "| Arm | Recall@10 | nDCG@10 | Recall@20 |", "|---|---:|---:|---:|"]
    for name, values in report["metrics"].items():
        all_values = values["all"]
        lines.append(f"| {name} | {all_values['recall@10']:.2f}% | {all_values['ndcg@10']:.2f} | {all_values['recall@20']:.2f}% |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
