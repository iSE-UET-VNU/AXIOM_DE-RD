"""Evaluate the replacement benchmark on one consistent 220-query corpus.

The corpus is 80 Physics + 75 Industrial + 65 OHR queries.  Physics and
Industrial preparation/embeddings are reused from the debug bundle; OHR
preparation and the merged embedding cache are reused from
``data/work/ohr_replacement_20260920``.

This runner recomputes the BM25 and PyMuPDF arms over the merged 5,091-page
corpus, reuses the cached merged hybrid scores, and optionally reranks the
merged hybrid top-200 candidate pools with Cohere through OpenRouter.  It does
not use qrels for ranking.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUNDLE = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval"
OHR_ROOT = ROOT / "data/work/ohr_replacement_20260920"
OUTPUT_DEFAULT = OHR_ROOT / "retrieval/merged_220_all_arms"

OLD_BASE_PAGES = BUNDLE / "preparation/baseline_pdf_inspector_tesseract/pages.jsonl"
OLD_OCR_PAGES = BUNDLE / "preparation/embedded_image_ocr/pages_with_embedded_ocr.jsonl"
OLD_PYMUPDF_PAGES = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/pymupdf_ablation/pymupdf_first/pages.jsonl"
OLD_DOCS = BUNDLE / "input/documents.jsonl"
OLD_QUERIES = BUNDLE / "input/queries.jsonl"
OLD_QRELS = BUNDLE / "input/qrels_page_level.jsonl"
OHR_BASE_PAGES = OHR_ROOT / "ohr_preparation/parsing/pages.jsonl"
OHR_OCR_PAGES = OHR_ROOT / "embedded_ocr_filtered/size_filter_only/pages.jsonl"
OHR_QUERIES = OHR_ROOT / "inputs/queries.jsonl"
OHR_QRELS = OHR_ROOT / "inputs/qrels.jsonl"
MERGED_HYBRID = OHR_ROOT / "retrieval/combined_220_base/combined_top20.jsonl"
MERGED_HYBRID_META = OHR_ROOT / "retrieval/combined_220_base/metadata.json"
MERGED_DENSE = OHR_ROOT / "retrieval/combined_220_base/query_page_dense.npy"
OLD_LANGUAGE = BUNDLE / "reports/language_routed_bm25/file_language_detection.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def normalise(values: np.ndarray | Mapping[str, float]) -> np.ndarray | dict[str, float]:
    if isinstance(values, Mapping):
        maximum = max((float(value) for value in values.values()), default=0.0)
        return {key: float(value) / maximum if maximum > 0 else 0.0 for key, value in values.items()}
    values = np.asarray(values, dtype=np.float64)
    maximum = float(values.max()) if values.size else 0.0
    return values / maximum if maximum > 0 else np.zeros_like(values)


def ndcg(ranked: Sequence[str], gold: set[str], depth: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 2.0) for rank, page in enumerate(ranked[:depth]) if page in gold)
    ideal = sum(1.0 / math.log2(rank + 2.0) for rank in range(min(len(gold), depth)))
    return dcg / ideal if ideal else 0.0


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, str]]:
    old_queries = [row for row in read_jsonl(OLD_QUERIES) if str(row["query_id"]).startswith(("vidore_physics::", "vidore_industrial::"))]
    ohr_queries = read_jsonl(OHR_QUERIES)
    queries = old_queries + ohr_queries

    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(OLD_QRELS):
        qid = str(row["query_id"])
        if qid.startswith(("vidore_physics::", "vidore_industrial::")):
            qrels[qid].add(str(row["page_id"]))
    for row in read_jsonl(OHR_QRELS):
        qid = str(row["query_id"])
        qrels[qid].add(f"{row['doc_id']}#page={int(row['page_index'])}")

    language = {}
    if OLD_LANGUAGE.is_file():
        for row in read_jsonl(OLD_LANGUAGE):
            fid = str(row.get("file_id") or row.get("doc_id") or "")
            if fid:
                language[fid] = str(row.get("detected_language") or "en")
    for row in read_jsonl(OHR_BASE_PAGES):
        language[file_id(str(row["page_id"]))] = "en"

    missing = [str(row["query_id"]) for row in queries if str(row["query_id"]) not in qrels]
    if missing:
        raise RuntimeError(f"Missing qrels for {len(missing)} queries")
    if len(queries) != 220 or len(qrels) != 220:
        raise RuntimeError(f"Expected 220 queries/qrels, got {len(queries)}/{len(qrels)}")
    return queries, dict(qrels), language


def filter_old(rows: list[dict[str, Any]], source_by_doc: Mapping[str, str]) -> list[dict[str, Any]]:
    return [row for row in rows if source_by_doc.get(str(row.get("doc_id") or file_id(str(row["page_id"])))) in {"vidore_physics", "vidore_industrial"}]


def load_page_corpora() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    docs = read_jsonl(OLD_DOCS)
    source_by_doc = {str(row["doc_id"]): str(row.get("source") or "") for row in docs}
    old_base = filter_old(read_jsonl(OLD_BASE_PAGES), source_by_doc)
    old_ocr = filter_old(read_jsonl(OLD_OCR_PAGES), source_by_doc)
    old_py = filter_old(read_jsonl(OLD_PYMUPDF_PAGES), source_by_doc)
    ohr_base = read_jsonl(OHR_BASE_PAGES)
    ohr_ocr = read_jsonl(OHR_OCR_PAGES)

    def by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        result = {}
        for row in rows:
            page = str(row["page_id"])
            if page in result:
                raise RuntimeError(f"Duplicate page ID: {page}")
            result[page] = dict(row)
        return result

    corpora = {
        "baseline_bm25": {**by_id(old_base), **by_id(ohr_base)},
        "embedded_image_ocr_bm25": {**by_id(old_ocr), **by_id(ohr_ocr)},
        "pymupdf": {**by_id(old_py), **by_id(ohr_base)},
    }

    # For OHR, replace the cached page/Tesseract fallback with native PyMuPDF
    # text wherever native extraction is non-empty.  The old PyMuPDF pages are
    # already cached by the dedicated ablation run.
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # type: ignore
    native_cache: dict[str, str] = {}
    path_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ohr_base:
        path_groups[str(row["source_path"])].append(row)
    for path_text, rows in path_groups.items():
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        with fitz.open(path) as document:
            for row in rows:
                index = int(row["page_index"])
                native = document[index].get_text("text").strip()
                page = str(row["page_id"])
                if native:
                    native_cache[page] = native
    for page, row in by_id(ohr_base).items():
        native = native_cache.get(page, "")
        corpora["pymupdf"][page] = {**row, "text": native or str(row.get("text") or ""), "text_source": "pymupdf" if native else str(row.get("text_source") or "")}

    expected = {"baseline_bm25": 5091, "embedded_image_ocr_bm25": 5091, "pymupdf": 5091}
    for name, corpus in corpora.items():
        if len(corpus) != expected[name]:
            raise RuntimeError(f"{name}: expected {expected[name]} pages, got {len(corpus)}")
    return {name: {page: str(row.get("text") or "") for page, row in corpus.items()} for name, corpus in corpora.items()}, corpora


def build_bm25_arm(page_texts: Mapping[str, str], queries: Sequence[Mapping[str, Any]], file_languages: Mapping[str, str]) -> list[dict[str, Any]]:
    from src.retrieval.sparse import BM25Index

    page_ids = list(page_texts)
    file_ids = list(dict.fromkeys(file_id(page) for page in page_ids))
    page_files = [file_id(page) for page in page_ids]
    page_index = BM25Index().build([{"chunk_id": page, "doc_id": file_id(page), "text": page_texts[page]} for page in page_ids])
    file_text = {fid: "\n".join(page_texts[page] for page in page_ids if file_id(page) == fid) for fid in file_ids}
    file_index = BM25Index().build([{"chunk_id": fid, "doc_id": fid, "text": file_text[fid]} for fid in file_ids])
    rows = []
    for query in queries:
        query_text = str(query["query"])
        page_raw = np.zeros(len(page_ids), dtype=np.float64)
        file_raw = np.zeros(len(file_ids), dtype=np.float64)
        for position, score in page_index.search(query_text, len(page_ids)):
            page_raw[int(position)] = float(score)
        for position, score in file_index.search(query_text, len(file_ids)):
            file_raw[int(position)] = float(score)
        page_norm = normalise(page_raw)
        file_norm = normalise(file_raw)
        page_base = 0.70 * page_norm
        file_max = {fid: float(max((page_base[i] for i, pf in enumerate(page_files) if pf == fid), default=0.0)) for fid in file_ids}
        max_norm = normalise(np.asarray([file_max[fid] for fid in file_ids], dtype=np.float64))
        file_score = {fid: 0.50 * float(file_norm[i]) + 0.50 * float(max_norm[i]) for i, fid in enumerate(file_ids)}
        selected = sorted(file_ids, key=lambda fid: (-file_score[fid], fid))[:3]
        parent = normalise(file_score)
        selected_set = set(selected)
        scored = [(page, 0.85 * float(page_base[i]) + 0.15 * float(parent.get(page_files[i], 0.0))) for i, page in enumerate(page_ids) if page_files[i] in selected_set]
        scored.sort(key=lambda item: (-item[1], item[0]))
        rows.append({"query_id": str(query["query_id"]), "query": query_text, "selected_files": selected, "candidate_pages": [page for page, _ in scored[:200]], "returned_pages": [page for page, _ in scored[:20]], "returned_scores": [float(score) for _, score in scored[:20]]})
    return rows


def build_dense_arm(page_ids: Sequence[str], dense_scores: np.ndarray, queries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    position = {page: i for i, page in enumerate(page_ids)}
    rows = []
    for qi, query in enumerate(queries):
        scores = np.asarray(dense_scores[qi], dtype=np.float64)
        file_best: dict[str, float] = {}
        for i, page in enumerate(page_ids):
            fid = file_id(page)
            file_best[fid] = max(file_best.get(fid, -np.inf), float(scores[i]))
        selected = sorted(file_best, key=lambda fid: (-file_best[fid], fid))[:3]
        selected_set = set(selected)
        order = sorted((i for i, page in enumerate(page_ids) if file_id(page) in selected_set), key=lambda i: (-float(scores[i]), i))
        rows.append({"query_id": str(query["query_id"]), "query": str(query["query"]), "selected_files": selected, "candidate_pages": [page_ids[i] for i in order[:200]], "returned_pages": [page_ids[i] for i in order[:20]], "returned_scores": [float(scores[i]) for i in order[:20]]})
    return rows


def load_hybrid_rows() -> list[dict[str, Any]]:
    rows = read_jsonl(MERGED_HYBRID)
    if len(rows) != 220:
        raise RuntimeError(f"Merged hybrid run must have 220 rows, got {len(rows)}")
    return rows


def metric_rows(rows: Sequence[Mapping[str, Any]], qrels: Mapping[str, set[str]], depth: int) -> dict[str, Any]:
    recalls, hits, ndcgs, files, ceilings = [], [], [], [], []
    for row in rows:
        qid = str(row["query_id"])
        gold = qrels[qid]
        ranked = list(row.get("returned_pages") or row.get("reranked_pages") or [])
        top = set(ranked[:depth])
        recalls.append(len(top & gold) / max(len(gold), 1))
        hits.append(float(bool(top & gold)))
        ndcgs.append(ndcg(ranked, gold, depth))
        gold_files = {file_id(page) for page in gold}
        files.append(len(set(row.get("selected_files", [])) & gold_files) / max(len(gold_files), 1))
        ceilings.append(len(set(row.get("candidate_pages", [])) & gold) / max(len(gold), 1))
    return {"queries": len(rows), f"page_recall@{depth}": float(np.mean(recalls)), f"page_hit@{depth}": float(np.mean(hits)), f"ndcg@{depth}": 100.0 * float(np.mean(ndcgs)), "file_recall@3": float(np.mean(files)), "candidate_pool_recall@200": float(np.mean(ceilings))}


def load_key() -> str:
    from src.utils.env import load_dotenv_file
    load_dotenv_file(ROOT)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; offline arms were completed")
    return key


def run_cohere(rows: Sequence[Mapping[str, Any]], page_texts: Mapping[str, str], queries: Mapping[str, str], output_dir: Path, workers: int) -> list[dict[str, Any]]:
    key = load_key()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "cohere_cache.jsonl"
    cached = {str(row["query_id"]): row for row in read_jsonl(cache_path)} if cache_path.is_file() else {}
    pending = [row for row in rows if cached.get(str(row["query_id"]), {}).get("candidate_pages") != row["candidate_pages"]]

    def one(row: Mapping[str, Any]) -> dict[str, Any]:
        page_ids = list(row["candidate_pages"])
        payload = {"model": "cohere/rerank-v3.5", "query": queries[str(row["query_id"])], "documents": [page_texts[page][:12000] for page in page_ids], "top_n": len(page_ids), "return_documents": False}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "AXIOM merged 220 rerank"}
        last = ""
        for attempt in range(3):
            try:
                response = requests.post("https://openrouter.ai/api/v1/rerank", headers=headers, json=payload, timeout=180)
                if response.status_code == 429 or response.status_code >= 500:
                    last = f"HTTP {response.status_code}: {response.text[:300]}"
                    time.sleep(min(2 ** attempt, 8))
                    continue
                response.raise_for_status()
                result = response.json().get("results")
                if not isinstance(result, list) or len(result) != len(page_ids):
                    raise RuntimeError(f"Expected {len(page_ids)} results")
                ranked = sorted(((page_ids[int(item["index"])], float(item["relevance_score"])) for item in result), key=lambda item: (-item[1], item[0]))
                return {"query_id": str(row["query_id"]), "candidate_pages": page_ids, "reranked_pages": [page for page, _ in ranked], "rerank_scores": [score for _, score in ranked]}
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Cohere failed for {row['query_id']}: {last}")

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(one, row): str(row["query_id"]) for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                value = future.result()
                cached[str(value["query_id"])] = value
                write_jsonl(cache_path, [cached[qid] for qid in queries if qid in cached])
                if index % 10 == 0 or index == len(pending):
                    print(f"Cohere merged rerank {index}/{len(pending)}", flush=True)
    result = []
    for row in rows:
        item = cached[str(row["query_id"])]
        result.append({**row, "returned_pages": item["reranked_pages"][:20], "reranked_pages": item["reranked_pages"], "rerank_scores": item["rerank_scores"]})
    write_jsonl(output_dir / "cohere_top20.jsonl", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--skip-cohere", action="store_true")
    parser.add_argument("--cohere-workers", type=int, default=32)
    args = parser.parse_args()
    started = time.perf_counter()
    queries, qrels, languages = load_inputs()
    query_map = {str(row["query_id"]): str(row["query"]) for row in queries}
    page_texts, page_rows = load_page_corpora()
    expected_pages = 5091
    if len(page_texts["baseline_bm25"]) != expected_pages:
        raise RuntimeError("Merged page count mismatch")

    page_ids = json.loads(MERGED_HYBRID_META.read_text(encoding="utf-8"))["page_ids"]
    dense = np.load(MERGED_DENSE, mmap_mode="r")
    meta = json.loads(MERGED_HYBRID_META.read_text(encoding="utf-8"))
    if list(meta["query_ids"]) != [str(row["query_id"]) for row in queries]:
        raise RuntimeError("Merged dense query order does not match inputs")
    if dense.shape != (220, 5091):
        raise RuntimeError(f"Unexpected merged dense shape: {dense.shape}")

    runs: dict[str, list[dict[str, Any]]] = {}
    expected_query_ids = {str(row["query_id"]) for row in queries}

    def load_complete_run(path: Path) -> list[dict[str, Any]] | None:
        if not path.is_file():
            return None
        rows = read_jsonl(path)
        if len(rows) != len(queries):
            return None
        if {str(row.get("query_id")) for row in rows} != expected_query_ids:
            return None
        return rows

    for name in ("baseline_bm25", "embedded_image_ocr_bm25", "pymupdf"):
        run_path = args.output_dir / "runs" / f"{name}.jsonl"
        cached_run = load_complete_run(run_path)
        if cached_run is not None:
            print(f"Reusing {name} run: {run_path}", flush=True)
            runs[name] = cached_run
        else:
            print(f"Building {name}...", flush=True)
            runs[name] = build_bm25_arm(page_texts[name], queries, languages)
            write_jsonl(run_path, runs[name])

    page_position = {page: i for i, page in enumerate(page_ids)}
    dense_path = args.output_dir / "runs/text_embeddings.jsonl"
    dense_rows = load_complete_run(dense_path)
    if dense_rows is None:
        dense_rows = build_dense_arm(page_ids, np.asarray(dense), queries)
        write_jsonl(dense_path, dense_rows)
    else:
        print(f"Reusing text_embeddings run: {dense_path}", flush=True)
    runs["text_embeddings"] = dense_rows

    hybrid_path = args.output_dir / "runs/hybrid_bm25_embeddings.jsonl"
    hybrid = load_complete_run(hybrid_path)
    if hybrid is None:
        hybrid = load_hybrid_rows()
        write_jsonl(hybrid_path, hybrid)
    else:
        print(f"Reusing hybrid run: {hybrid_path}", flush=True)
    runs["hybrid_bm25_embeddings"] = hybrid

    if args.skip_cohere:
        print("Cohere skipped by flag", flush=True)
    else:
        cohere_rows = run_cohere(hybrid, page_texts["embedded_image_ocr_bm25"], query_map, args.output_dir / "cohere", args.cohere_workers)
        runs["cohere_reranker"] = cohere_rows

    reports = {}
    for name, rows in runs.items():
        reports[name] = {"@10": metric_rows(rows, qrels, 10), "@20": metric_rows(rows, qrels, 20)}
    report = {
        "contract_version": "ohr-replacement-merged-220-all-arms-v1",
        "protocol": {"queries": "80 Physics + 75 Industrial + 65 OHR", "files": 103, "pages": 5091, "file_k": 3, "candidate_depth": 200, "saved_depth": 20, "qrels_used_for_ranking": False, "cohere_workers": args.cohere_workers},
        "arms": reports,
        "timing_seconds": {"total": time.perf_counter() - started},
        "sources": {"ohr_preparation": str((OHR_ROOT / "ohr_preparation").resolve()), "merged_hybrid": str(MERGED_HYBRID.resolve()), "merged_dense": str(MERGED_DENSE.resolve())},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# OHR replacement: merged 220-query all-arm comparison", "", "| Arm | Recall@10 | Recall@20 | File recall@3 | nDCG@10 | nDCG@20 |", "|---|---:|---:|---:|---:|---:|"]
    labels = {"baseline_bm25": "Baseline BM25", "embedded_image_ocr_bm25": "Embedded-image OCR + BM25", "text_embeddings": "Text embeddings", "hybrid_bm25_embeddings": "Hybrid BM25 + embeddings", "cohere_reranker": "Hybrid + Cohere reranker", "pymupdf": "PyMuPDF + BM25"}
    for name, values in reports.items():
        m10, m20 = values["@10"], values["@20"]
        lines.append(f"| {labels[name]} | {m10['page_recall@10']:.2%} | {m20['page_recall@20']:.2%} | {m10['file_recall@3']:.2%} | {m10['ndcg@10']:.2f} | {m20['ndcg@20']:.2f} |")
    if args.skip_cohere:
        lines.extend(["", "Cohere was skipped; rerun without `--skip-cohere` to complete that arm."])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Output: {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
