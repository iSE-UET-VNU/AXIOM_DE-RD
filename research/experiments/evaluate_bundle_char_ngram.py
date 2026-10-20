"""Quick character n-gram BM25 ablation for the cached benchmark bundle.

This is an offline diagnostic only. It reuses the existing preparation cache,
does not parse PDFs, call an API, or use qrels during ranking. The experiment
compares the current word analyzer with character trigrams and a word+trigram
analyzer under the same hierarchical Kf=3/Kp=10/20 protocol.
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

from src.chunking_embedding.lexical import analyze  # noqa: E402
from src.retrieval import sparse as sparse_module  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


BUNDLE = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20"
OUTPUT = BUNDLE / "char_ngram_ablation"
FILE_K = 3
PAGE_KS = (10, 20)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def char3(text: str) -> list[str]:
    """Emit word-local character trigrams.

    Trigrams stay inside each lexical token, so we do not create grams that
    cross genuine word boundaries. This lets ``cor rosion`` share ``cor``,
    ``ros``, ``osi`` and ``ion`` with ``corrosion`` without indexing the whole
    page as one whitespace-free string.
    """
    output: list[str] = []
    for token in analyze(text):
        if len(token) < 3:
            output.append(token)
        else:
            output.extend(token[i : i + 3] for i in range(len(token) - 2))
    return output


def word_char3(text: str) -> list[str]:
    return analyze(text) + char3(text)


ANALYZERS = {
    "word": analyze,
    "char3": char3,
    "word_char3": word_char3,
}


def normalise(values: Mapping[str, float]) -> dict[str, float]:
    maximum = max((float(value) for value in values.values() if float(value) > 0), default=0.0)
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


def metrics(rows: list[dict[str, Any]], qrels: Mapping[str, Mapping[str, int]], k: int) -> dict[str, float]:
    recalls: list[float] = []
    hits: list[float] = []
    ndcgs: list[float] = []
    file_recalls: list[float] = []
    for row in rows:
        gold = set(qrels[row["query_id"]])
        ranked = row["pages"][:k]
        denominator = max(len(gold), 1)
        recalls.append(len(set(ranked) & gold) / denominator)
        hits.append(float(bool(set(ranked) & gold)))
        ndcgs.append(ndcg(ranked, qrels[row["query_id"]], k))
        gold_files = {file_id(page) for page in gold}
        selected = set(row["selected_files"])
        file_recalls.append(len(selected & gold_files) / max(len(gold_files), 1))
    return {
        f"page_recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"page_hit@{k}": float(np.mean(hits)) if hits else 0.0,
        f"ndcg@{k}": 100.0 * float(np.mean(ndcgs)) if ndcgs else 0.0,
        "file_recall@3": float(np.mean(file_recalls)) if file_recalls else 0.0,
    }


def run_arm(
    name: str,
    analyzer,
    pages: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    qrels: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    page_ids = [str(page["page_id"]) for page in pages]
    page_to_file = {str(page["page_id"]): str(page["doc_id"]) for page in pages}
    file_text: dict[str, list[str]] = defaultdict(list)
    for page in pages:
        file_text[str(page["doc_id"])].append(str(page.get("text") or ""))
    file_ids = sorted(file_text)

    page_index = BM25Index(analyzer_name=name).build(
        {"chunk_id": str(page["page_id"]), "doc_id": str(page["doc_id"]), "text": str(page.get("text") or "")}
        for page in pages
    )
    file_index = BM25Index(analyzer_name=name).build(
        {"chunk_id": doc_id, "doc_id": doc_id, "text": "\n".join(file_text[doc_id])}
        for doc_id in file_ids
    )
    counts = defaultdict(int)
    for page in pages:
        counts[str(page["doc_id"])] += 1
    rows: list[dict[str, Any]] = []
    for query in queries:
        qid = str(query["query_id"])
        text = str(query.get("query") or "")
        page_hits = page_index.search(text, len(page_ids))
        raw_page = {page_index.chunk_ids[pos]: float(score) for pos, score in page_hits}
        page_norm = normalise(raw_page)
        direct_hits = file_index.search(text, len(file_ids))
        direct = {file_index.chunk_ids[pos]: float(score) for pos, score in direct_hits}
        page_pool = {
            doc_id: max((page_norm.get(page_id, 0.0) for page_id in page_ids if page_to_file[page_id] == doc_id), default=0.0)
            for doc_id in file_ids
        }
        file_scores = {
            doc_id: 0.5 * normalise(direct).get(doc_id, 0.0) + 0.5 * normalise(page_pool).get(doc_id, 0.0)
            for doc_id in file_ids
        }
        selected = [doc_id for doc_id, _ in sorted(file_scores.items(), key=lambda item: (-item[1], item[0]))[:FILE_K]]
        parent = normalise(file_scores)
        allowed = set(selected)
        final = {
            page_id: 0.85 * page_norm.get(page_id, 0.0) + 0.15 * parent.get(page_to_file[page_id], 0.0)
            for page_id in page_ids
            if page_to_file[page_id] in allowed
        }
        ranked = [page_id for page_id, _ in sorted(final.items(), key=lambda item: (-item[1], item[0]))]
        rows.append({"query_id": qid, "pages": ranked, "selected_files": selected})
    report = {
        "analyzer": name,
        "pages": len(pages),
        "files": len(file_ids),
        "queries": len(queries),
        "metrics": {str(k): metrics(rows, qrels, k) for k in PAGE_KS},
        "q111": next(row for row in rows if row["query_id"] == "vidore_industrial::111"),
    }
    return report, rows


def main() -> int:
    started = time.perf_counter()
    sparse_module.ANALYZERS.update(ANALYZERS)
    pages = read_jsonl(BUNDLE / "preparation/pages.jsonl")
    queries = read_jsonl(BUNDLE / "input/queries.jsonl")
    qrels_rows = read_jsonl(BUNDLE / "input/qrels_page_level.jsonl")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrels_rows:
        qrels[str(row["query_id"])][str(row["page_id"])] = int(row.get("relevance", 1))
    queries = [query for query in queries if str(query["query_id"]) in qrels]
    reports: dict[str, Any] = {}
    per_query: dict[str, Any] = {}
    for name, analyzer in ANALYZERS.items():
        report, rows = run_arm(name, analyzer, pages, queries, qrels)
        reports[name] = report
        per_query[name] = {row["query_id"]: row for row in rows}
    result = {
        "protocol": {"file_k": FILE_K, "page_k": list(PAGE_KS), "analyzers": list(ANALYZERS), "qrels_used_only_for_evaluation": True},
        "bundle": str(BUNDLE),
        "reports": reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "per_query.json").write_text(json.dumps(per_query, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Character n-gram BM25 quick ablation", "", "Same cached preparation; hierarchical Kf=3; qrels used only for evaluation.", "", "| Analyzer | Recall@10 | nDCG@10 | File recall@3 | Recall@20 | nDCG@20 |", "|---|---:|---:|---:|---:|---:|"]
    for name, report in reports.items():
        m10 = report["metrics"]["10"]
        m20 = report["metrics"]["20"]
        lines.append(f"| {name} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m10['file_recall@3']:.2%} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} |")
    lines.extend(["", f"Elapsed: `{result['elapsed_seconds']:.2f}s`.", "", "The char3 analyzer uses word-local character trigrams; it does not remove all whitespace from a page."])
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Output: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
