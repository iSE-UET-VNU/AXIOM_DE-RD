"""Dense text retrieval on the full embedded-image OCR cache.

The input is the 5,334-page cache where Tesseract text from 9,695 embedded
image occurrences was inserted in page reading order. A fixed short-token
filter removes isolated one/two-letter OCR noise while retaining common
function words and alphanumeric technical tokens. Embeddings are produced with
the OpenRouter text-embedding-3-small API and cached on disk.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.analyze_benchmark_embedding_adaptive_k import _encode_cached  # noqa: E402


BUNDLE = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20"
FULL_OCR_PAGES = BUNDLE / "embedded_image_ocr_eng_w48/pages_with_embedded_ocr.jsonl"
DATASET = ROOT / "data/raw/BENCHMARK"
OUTPUT = ROOT / "data/work/benchmark_hierarchical_local/embedding_full_embedded_ocr_openrouter_te3s_clean_eng_w48"
FILE_K = 3
PAGE_KS = (10, 20)
MODEL = "openai/text-embedding-3-small"

# Keep short function words so the filter is not simply a destructive length
# cutoff. Technical tokens containing digits, such as 25MM or F-16, are kept.
KEEP_SHORT = {
    "a", "i", "an", "as", "at", "be", "by", "do", "if", "in", "is", "it", "no", "of", "on", "or", "to", "up", "us", "we",
    "au", "de", "du", "en", "et", "la", "le", "les", "un", "une",
}
SHORT_ALPHA = re.compile(r"^[A-Za-z]{1,2}$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_short_ocr_noise(text: str) -> tuple[str, int]:
    removed = 0
    output: list[str] = []
    for token in str(text).split():
        stripped = token.strip(".,;:!?()[]{}<>\"'`|~+=*/\\")
        if SHORT_ALPHA.fullmatch(stripped) and stripped.casefold() not in KEEP_SHORT:
            removed += 1
            continue
        output.append(token)
    return " ".join(output), removed


def normalise(values: Mapping[str, float]) -> dict[str, float]:
    maximum = max((float(value) for value in values.values()), default=0.0)
    if maximum <= 0:
        return {key: 0.0 for key in values}
    return {key: max(0.0, float(value)) / maximum for key, value in values.items()}


def file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def ndcg(ranked: Sequence[str], gold: Mapping[str, int], k: int) -> float:
    gains = {str(page_id): int(value) for page_id, value in gold.items()}
    dcg = sum(
        (2.0 ** gains[page_id] - 1.0) / np.log2(rank + 2)
        for rank, page_id in enumerate(ranked[:k])
        if page_id in gains
    )
    ideal_values = sorted(gains.values(), reverse=True)[:k]
    ideal = sum((2.0 ** value - 1.0) / np.log2(rank + 2) for rank, value in enumerate(ideal_values))
    return float(dcg / ideal) if ideal else 0.0


def evaluate(rows: Sequence[Mapping[str, Any]], qrels: Mapping[str, Mapping[str, int]], k: int) -> dict[str, float]:
    recalls: list[float] = []
    hits: list[float] = []
    ndcgs: list[float] = []
    files: list[float] = []
    for row in rows:
        gold = set(qrels[row["query_id"]])
        ranked = list(row["returned_pages"][:k])
        denominator = max(len(gold), 1)
        recalls.append(len(set(ranked) & gold) / denominator)
        hits.append(float(bool(set(ranked) & gold)))
        ndcgs.append(ndcg(ranked, qrels[row["query_id"]], k))
        gold_files = {file_id(page) for page in gold}
        files.append(len(set(row["selected_files"]) & gold_files) / max(len(gold_files), 1))
    mean = lambda values: float(np.mean(values)) if values else 0.0
    return {
        f"page_recall@{k}": mean(recalls),
        f"page_hit@{k}": mean(hits),
        f"ndcg@{k}": 100.0 * mean(ndcgs),
        "file_recall@3": mean(files),
    }


def main() -> int:
    started = time.perf_counter()
    pages = read_jsonl(FULL_OCR_PAGES)
    queries = read_jsonl(DATASET / "queries.jsonl")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_jsonl(BUNDLE / "input/qrels_page_level.jsonl"):
        qrels[str(row["query_id"])][str(row["page_id"])] = int(row.get("relevance", 1))
    queries = [query for query in queries if str(query["query_id"]) in qrels]

    cleaned_pages: list[dict[str, Any]] = []
    removed_total = 0
    for page in pages:
        cleaned, removed = clean_short_ocr_noise(str(page.get("text") or ""))
        removed_total += removed
        cleaned_pages.append({**page, "text": cleaned})
    page_ids = [str(page["page_id"]) for page in cleaned_pages]
    page_docs = [str(page["doc_id"]) for page in cleaned_pages]
    page_texts = [str(page["text"]) for page in cleaned_pages]
    query_texts = [str(query["query"]) for query in queries]

    embedding_dir = OUTPUT / "embeddings"
    embed_started = time.perf_counter()
    page_vectors, query_vectors, embedding_meta = _encode_cached(
        model_name=MODEL,
        page_texts=page_texts,
        queries=query_texts,
        output_dir=embedding_dir,
        batch_size=256,
        device="cpu",
        chunk_chars=6000,
        provider="openrouter",
        force=False,
    )
    embedding_seconds = time.perf_counter() - embed_started

    positions_by_file: dict[str, list[int]] = defaultdict(list)
    for position, doc_id in enumerate(page_docs):
        positions_by_file[doc_id].append(position)
    page_norms = np.linalg.norm(page_vectors, axis=1)
    policies = {"global": [], "hierarchical": []}
    query_debug: list[dict[str, Any]] = []
    retrieval_started = time.perf_counter()
    for query, query_vector in zip(queries, query_vectors):
        qid = str(query["query_id"])
        scores = np.asarray(page_vectors @ query_vector, dtype=np.float32)
        scores = scores / (page_norms * (float(np.linalg.norm(query_vector)) or 1.0) + 1e-9)
        global_order = np.argsort(-scores, kind="stable")
        file_scores = {
            doc_id: max(float(scores[position]) for position in positions)
            for doc_id, positions in positions_by_file.items()
        }
        ranked_files = sorted(file_scores.items(), key=lambda item: (-item[1], item[0]))
        selected = [doc_id for doc_id, _score in ranked_files[:FILE_K]]
        selected_set = set(selected)
        hier_order = sorted(
            (position for position, doc_id in enumerate(page_docs) if doc_id in selected_set),
            key=lambda position: (-float(scores[position]), position),
        )
        for name, order, chosen_files in (
            ("global", global_order, [doc_id for doc_id, _ in ranked_files]),
            ("hierarchical", hier_order, selected),
        ):
            returned = [page_ids[int(position)] for position in order[:20]]
            policies[name].append({"query_id": qid, "selected_files": chosen_files[:FILE_K], "returned_pages": returned})
        if qid in {"vidore_industrial::111", "vidore_industrial::112", "vidore_industrial::239", "vidore_industrial::102"}:
            policies_top = policies["hierarchical"][-1]
            query_debug.append({"query_id": qid, "query": query["query"], "selected_files": selected, "top20": policies_top["returned_pages"]})
    retrieval_seconds = time.perf_counter() - retrieval_started

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.save(embedding_dir / "page_embeddings.npy", page_vectors)
    np.save(embedding_dir / "query_embeddings.npy", query_vectors)
    (OUTPUT / "page_metadata.json").write_text(
        json.dumps({"page_ids": page_ids, "page_sources": [page.get("source") for page in pages], "cleaning": "remove isolated 1-2 alphabetic characters except fixed function-word whitelist", "removed_short_tokens": removed_total}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "input_pages": str(FULL_OCR_PAGES.resolve()),
        "model": embedding_meta,
        "pages": len(pages), "files": len(positions_by_file), "queries": len(queries),
        "ocr_input": {"image_occurrences": 9695, "nonempty_ocr_occurrences": 5919, "language": "eng", "workers": 48},
        "cleaning": {"removed_short_tokens": removed_total, "policy": "remove isolated 1-2 alphabetic characters except fixed English/French function-word whitelist"},
        "metrics": {name: {str(k): evaluate(rows, qrels, k) for k in PAGE_KS} for name, rows in policies.items()},
        "timing_seconds": {"embedding": embedding_seconds, "retrieval": retrieval_seconds, "total": time.perf_counter() - started},
        "qrels_used_for_ranking": False,
        "query_debug": query_debug,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, rows in policies.items():
        (OUTPUT / f"{name}_top20.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8")
    lines = ["# Full embedded-image OCR + text embeddings", "", "| Policy | Recall@10 | nDCG@10 | File recall@3 | Recall@20 | nDCG@20 |", "|---|---:|---:|---:|---:|---:|"]
    for name, rows in policies.items():
        m10 = evaluate(rows, qrels, 10); m20 = evaluate(rows, qrels, 20)
        lines.append(f"| {name} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m10['file_recall@3']:.2%} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} |")
    lines.extend(["", f"Embedding seconds: `{embedding_seconds:.2f}`; retrieval seconds: `{retrieval_seconds:.2f}`; removed short tokens: `{removed_total}`.", "", "Input is full embedded-image OCR in page reading order; qrels are used only for evaluation."])
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Output: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
