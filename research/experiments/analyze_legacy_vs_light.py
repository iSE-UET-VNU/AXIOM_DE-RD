"""Offline investigation of the legacy KDL retriever versus light retrieval.

This script deliberately keeps three axes separate:

* parsing/page representation (KDL versus light text and embedded-image OCR),
* retrieval unit/protocol (legacy chunks/files versus light pages/files), and
* ranking quality (candidate pool versus final page ordering).

It consumes existing caches only.  It does not render PDFs, call an API, or
use qrels to build a ranking.  The KDL-derived structured page control is an
explicit upper bound: it is useful for attribution, but is not a claim that
the light pipeline already has those fields.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.fusion import alpha_fuse
from src.retrieval.sparse import BM25Index


PAGE_RE = re.compile(r"(?im)(?:^|\n)\s*\*{0,2}Page\s+(\d+)\*{0,2}")
PAGE_CITATION_RE = re.compile(r"/page/(\d+)")
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
MOJIBAKE_RE = re.compile(r"[�]|Ã.|Â.|â[€\x80-\xBF]|ð[\x80-\xBF]")
IMAGE_MARKER_RE = re.compile(r"(?i)(?:\[image\]|<image>|image\s+placeholder|\bfigure\b)")

PRIMARY_SOURCES = {"vidore_physics", "vidore_industrial"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(text_value(item) for item in value)
    if isinstance(value, dict):
        return " ".join(text_value(item) for item in value.values())
    return str(value)


def normalize_text(value: Any) -> str:
    text = html.unescape(text_value(value))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[`*_#|]", " ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def tokens(value: Any) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(normalize_text(value))]


def token_set(value: Any) -> set[str]:
    return set(tokens(value))


def meaningful_tokens(value: Any) -> set[str]:
    return {token for token in token_set(value) if len(token) >= 3}


def coverage(source: Any, target: Any) -> tuple[float, list[str]]:
    source_terms = meaningful_tokens(source)
    target_terms = token_set(target)
    if not source_terms:
        return 1.0, []
    missing = sorted(term for term in source_terms if term not in target_terms)
    return (len(source_terms) - len(missing)) / len(source_terms), missing[:40]


def source_from_doc_id(doc_id: str) -> str:
    if "::" in doc_id:
        return doc_id.split("::", 1)[0]
    return "unknown"


def file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def page_index(page_id: str) -> int | None:
    match = re.search(r"#page=(\d+)$", page_id)
    return int(match.group(1)) if match else None


def canonical_doc_from_filename(file_name: str) -> str | None:
    stem = Path(file_name).name
    if stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    for prefix in ("vidore_physics__", "vidore_industrial__", "mpdocvqa__"):
        if stem.startswith(prefix):
            return prefix[:-2] + "::" + stem[len(prefix):]
    return None


def citation_pages(value: Any) -> list[int]:
    pages: list[int] = []
    values = value if isinstance(value, list) else [value]
    for item in values:
        for match in PAGE_CITATION_RE.finditer(text_value(item)):
            pages.append(int(match.group(1)))
    return sorted(set(pages))


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def page_recall(ranked: list[str], gold: set[str], depth: int) -> float:
    return len(set(ranked[:depth]) & gold) / len(gold) if gold else 0.0


def page_hit(ranked: list[str], gold: set[str], depth: int) -> float:
    return float(bool(set(ranked[:depth]) & gold))


def file_recall(ranked: list[str], gold: set[str], depth: int) -> float:
    docs = {file_id(item) for item in ranked[:depth]}
    return len(docs & gold) / len(gold) if gold else 0.0


def ndcg(ranked: list[str], gold: set[str], depth: int) -> float:
    if not gold:
        return 0.0
    gain = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked[:depth], 1) if item in gold)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(depth, len(gold)) + 1))
    return gain / ideal if ideal else 0.0


def first_rank(ranked: list[str], targets: set[str]) -> int | None:
    target = set(targets)
    for rank, item in enumerate(ranked, 1):
        if item in target:
            return rank
    return None


def dedupe(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def selected_files_from_row(row: dict[str, Any]) -> list[str]:
    values = row.get("selected_files") or []
    if values and isinstance(values[0], str):
        return dedupe(str(value) for value in values)
    return dedupe(str(value.get("file_id")) for value in values if isinstance(value, dict) and value.get("file_id"))


def ranked_pages_from_row(row: dict[str, Any]) -> list[str]:
    for key in ("returned_pages", "reranked_pages", "ranked_pages", "pages"):
        values = row.get(key)
        if isinstance(values, list) and values and isinstance(values[0], str):
            return dedupe(str(value) for value in values)
    chunks = row.get("chunks") or []
    if isinstance(chunks, list):
        return dedupe(str(value.get("page_id")) for value in chunks if isinstance(value, dict) and value.get("page_id"))
    return []


def candidate_pages_from_row(row: dict[str, Any], pages_by_file: dict[str, list[str]]) -> list[str]:
    values = row.get("candidate_pages")
    if isinstance(values, list) and values and isinstance(values[0], str):
        return dedupe(str(value) for value in values)
    selected = selected_files_from_row(row)
    if selected:
        return [page for doc in selected for page in pages_by_file.get(doc, [])]
    return ranked_pages_from_row(row)


def build_gold(qrels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = defaultdict(lambda: {"pages": set(), "files": set()})
    for row in qrels:
        if float(row.get("relevance", 1)) <= 0:
            continue
        page = str(row["page_id"])
        result[str(row["query_id"])]["pages"].add(page)
        result[str(row["query_id"])]["files"].add(file_id(page))
    return dict(result)


def load_light_pages(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["page_id"]): row for row in read_jsonl(path)}


def load_kdl_pages(legacy_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pages: dict[str, dict[str, Any]] = {}
    documents = 0
    parse_runs: list[str] = []
    parse_root = legacy_root / "vidore-v3-physics-drive-kdl-pdf-inspector-parse-only"
    for run_dir in sorted(parse_root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "documents").is_dir():
            continue
        parse_runs.append(run_dir.name)
        for path in sorted((run_dir / "documents").glob("*.json")):
            obj = read_json(path)
            source_meta = ((obj.get("source") or {}).get("metadata") or {})
            canonical_doc = canonical_doc_from_filename(str(source_meta.get("file_name", "")))
            if not canonical_doc:
                continue
            documents += 1
            parsed = obj.get("parsed") or {}
            rows = parsed.get("rows") or []
            if not rows:
                continue
            row = rows[0]
            blocks = row.get("source_blocks") or []
            order = {str(component): index for index, component in enumerate(row.get("reading_order") or [])}
            by_page: dict[int, dict[str, Any]] = defaultdict(lambda: {
                "blocks": [], "block_types": Counter(), "text_parts": [], "bbox_count": 0,
                "equation_parts": [],
            })
            for block in blocks:
                try:
                    page = int(block.get("page", 0))
                except (TypeError, ValueError):
                    continue
                entry = by_page[page]
                entry["blocks"].append(block)
                block_type = str(block.get("type") or block.get("raw_label") or "unknown")
                entry["block_types"][block_type] += 1
                block_text = text_value(block.get("text"))
                if block_text:
                    entry["text_parts"].append(block_text)
                if block.get("bbox"):
                    entry["bbox_count"] += 1
                if block_type.casefold() in {"equationblock", "formula", "equation"} and block_text:
                    entry["equation_parts"].append(block_text)

            def add_extraction(items: Any, kind: str, text_key: str, citation_keys: tuple[str, ...]) -> None:
                for item in items or []:
                    if not isinstance(item, dict):
                        continue
                    pages_for_item: set[int] = set()
                    for key in citation_keys:
                        pages_for_item.update(citation_pages(item.get(key)))
                    if not pages_for_item:
                        pages_for_item.add(0)
                    content = text_value(item.get(text_key))
                    caption = text_value(item.get("caption"))
                    for page in pages_for_item:
                        entry = by_page[page]
                        entry.setdefault(kind, []).append({"text": content, "caption": caption})

            extraction = row.get("extraction") or {}
            add_extraction(extraction.get("figures"), "figures", "description", ("description_citations", "caption_citations"))
            add_extraction(extraction.get("tables"), "tables", "content", ("content_citations", "caption_citations"))
            add_extraction(extraction.get("formulas"), "formulas", "formula", ("formula_citations", "citations"))
            # Some KDL outputs expose formulas only as EquationBlock source blocks.
            page_count = int((parsed.get("metadata") or {}).get("page_count") or max(by_page.keys(), default=-1) + 1)
            for page in range(page_count):
                entry = by_page[page]
                block_list = sorted(entry["blocks"], key=lambda item: (order.get(str(item.get("component_id")), 10**9), int(item.get("block_index", 0) or 0)))
                body = " ".join(text_value(item.get("text")) for item in block_list if text_value(item.get("text")))
                figures = entry.get("figures", [])
                tables = entry.get("tables", [])
                formulas = entry.get("formulas", [])
                equation_parts = entry.get("equation_parts", [])
                figure_text = " ".join(item["text"] + " " + item["caption"] for item in figures)
                table_text = " ".join(item["text"] + " " + item["caption"] for item in tables)
                formula_text = " ".join(item["text"] + " " + item["caption"] for item in formulas) + " " + " ".join(equation_parts)
                structured = " ".join(part for part in (body, figure_text, table_text, formula_text) if part).strip()
                type_counts = Counter(entry["block_types"])
                if figures:
                    type_counts["FigureDescription"] += len(figures)
                if tables:
                    type_counts["TableContent"] += len(tables)
                if formulas or equation_parts:
                    type_counts["Equation"] += len(formulas) + len(equation_parts)
                modalities = []
                if figures:
                    modalities.append("figure")
                if tables:
                    modalities.append("table")
                if formulas or equation_parts or type_counts.get("EquationBlock", 0):
                    modalities.append("formula")
                kind = "mixed" if len(modalities) > 1 else modalities[0] if modalities else "text_only"
                page_id = f"{canonical_doc}#page={page}"
                pages[page_id] = {
                    "page_id": page_id,
                    "doc_id": canonical_doc,
                    "source": source_from_doc_id(canonical_doc),
                    "page_index": page,
                    "text": body,
                    "structured_text": structured,
                    "figure_text": figure_text,
                    "table_text": table_text,
                    "formula_text": formula_text,
                    "figure_count": len(figures),
                    "table_count": len(tables),
                    "formula_count": len(formulas) + len(equation_parts),
                    "block_count": len(block_list),
                    "block_types": dict(type_counts),
                    "bbox_count": entry["bbox_count"],
                    "modality": kind,
                    "parse_run": run_dir.name,
                    "source_document": str(path),
                    "legacy_doc_id": str(obj.get("document_id") or ""),
                }
    meta = {"document_count_seen": documents, "parse_runs": parse_runs, "page_count": len(pages)}
    return pages, meta


def page_alignment(kdl_pages: dict[str, dict[str, Any]], base_pages: dict[str, dict[str, Any]], ocr_pages: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page_id in sorted(set(kdl_pages) | set(base_pages) | set(ocr_pages)):
        kdl = kdl_pages.get(page_id, {})
        base = base_pages.get(page_id, {})
        ocr = ocr_pages.get(page_id, {})
        kdl_modality_text = " ".join((kdl.get("figure_text", ""), kdl.get("table_text", ""), kdl.get("formula_text", "")))
        base_text = text_value(base.get("text"))
        ocr_text = text_value(ocr.get("text"))
        base_cov, base_missing = coverage(kdl_modality_text, base_text)
        ocr_cov, ocr_missing = coverage(kdl_modality_text, ocr_text)
        modalities = kdl.get("modality", "text_only")
        row = {
            "page_id": page_id,
            "doc_id": kdl.get("doc_id") or base.get("doc_id"),
            "source": kdl.get("source") or base.get("source"),
            "page_index": kdl.get("page_index", base.get("page_index")),
            "kdl_text_chars": len(text_value(kdl.get("text"))),
            "kdl_structured_chars": len(text_value(kdl.get("structured_text"))),
            "light_base_chars": len(base_text),
            "light_ocr_chars": len(ocr_text),
            "kdl_token_count": len(tokens(kdl.get("structured_text"))),
            "light_base_token_count": len(tokens(base_text)),
            "light_ocr_token_count": len(tokens(ocr_text)),
            "kdl_figure_count": int(kdl.get("figure_count", 0)),
            "kdl_table_count": int(kdl.get("table_count", 0)),
            "kdl_formula_count": int(kdl.get("formula_count", 0)),
            "kdl_block_count": int(kdl.get("block_count", 0)),
            "kdl_block_types": kdl.get("block_types", {}),
            "kdl_modality": modalities,
            "kdl_modality_term_count": len(meaningful_tokens(kdl_modality_text)),
            "light_base_modality_coverage": base_cov,
            "light_ocr_modality_coverage": ocr_cov,
            "light_base_missing_modality_terms": base_missing,
            "light_ocr_missing_modality_terms": ocr_missing,
            "light_visual_only": bool(base.get("visual_only", False)),
            "light_needs_ocr": bool(base.get("needs_ocr", False)),
            "light_ocr_applied": bool(base.get("ocr_applied", False)),
            "light_ocr_error": base.get("ocr_error"),
            "light_parse_status": base.get("parse_status"),
            "light_text_source": base.get("text_source"),
            "light_base_empty": not bool(normalize_text(base_text)),
            "light_embedded_ocr_inserted": bool(ocr.get("embedded_image_ocr_inserted", False)),
            "light_image_marker": bool(IMAGE_MARKER_RE.search(base_text)),
            "light_mojibake": bool(MOJIBAKE_RE.search(base_text)),
            "ocr_mojibake": bool(MOJIBAKE_RE.search(ocr_text)),
            "kdl_source_document": kdl.get("source_document"),
        }
        rows.append(row)
    return rows


def load_runs(light_root: Path, pages_by_file: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    base = light_root / "bundle_debug_light-retrieval"
    candidates = {
        "light_hier_bm25": base / "runs/baseline_hierarchical_bm25/retrieval/hierarchical_kf3_top20.jsonl",
        "light_embedded_ocr_bm25": base / "runs/bm25_embedded_image_ocr/retrieval/runs/hierarchical_bm25_kf3_top20.jsonl",
        "light_dense_ocr": base / "runs/dense_embedded_image_ocr/hierarchical_top20_scored.jsonl",
        "light_dense_no_ocr": base / "runs/dense_no_embedded_image_ocr/hierarchical_top20_scored.jsonl",
        "light_hybrid_ocr_bm25_70_dense_30": base / "reports/hybrid_search_cached/embedded_image_ocr__hybrid_bm25_0.70_dense_0.30_hierarchical.jsonl",
        "light_cohere_r100": base / "reports/cohere_rerank_top3_pages/run_top20.jsonl",
        "light_cohere_r200": base / "reports/cohere_rerank_top3_pages_r200/run_top20.jsonl",
    }
    result: dict[str, dict[str, Any]] = {}
    for name, path in candidates.items():
        if path.is_file():
            rows = read_jsonl(path)
            result[name] = {"path": str(path), "rows": {str(row["query_id"]): row for row in rows}, "row_count": len(rows)}
    return result


def summarize_retrieval(ranked: list[str], candidate: list[str], selected: list[str], gold: dict[str, Any]) -> dict[str, Any]:
    gold_pages: set[str] = gold["pages"]
    gold_files: set[str] = gold["files"]
    return {
        "page_recall@10": page_recall(ranked, gold_pages, 10),
        "page_recall@20": page_recall(ranked, gold_pages, 20),
        "page_hit@10": page_hit(ranked, gold_pages, 10),
        "page_hit@20": page_hit(ranked, gold_pages, 20),
        "ndcg@10": ndcg(ranked, gold_pages, 10),
        "ndcg@20": ndcg(ranked, gold_pages, 20),
        "file_recall@1": file_recall(selected, gold_files, 1),
        "file_recall@3": file_recall(selected, gold_files, 3),
        "file_recall@10": file_recall(selected, gold_files, 10),
        "file_hit@3": float(bool(set(selected[:3]) & gold_files)),
        "candidate_pool_recall": page_recall(candidate, gold_pages, len(candidate) if candidate else 0),
        "candidate_pool_hit": page_hit(candidate, gold_pages, len(candidate) if candidate else 0),
        "first_gold_page_rank": first_rank(ranked, gold_pages),
        "first_gold_file_rank": first_rank(selected, gold_files),
        "gold_pages_outside_selected_files": sorted(gold_pages - {page for doc in selected for page in []}) if False else sorted(page for page in gold_pages if file_id(page) not in set(selected)),
        "gold_pages_outside_candidate_pool": sorted(gold_pages - set(candidate)),
    }


def aggregate_metrics(per_query: list[dict[str, Any]], source: str | None = None) -> dict[str, float | int]:
    rows = [row for row in per_query if source is None or row.get("source") == source]
    if not rows:
        return {"query_count": 0}
    keys = [
        "page_recall@10", "page_recall@20", "page_hit@10", "page_hit@20", "ndcg@10", "ndcg@20",
        "file_recall@1", "file_recall@3", "file_recall@10", "file_hit@3", "candidate_pool_recall", "candidate_pool_hit",
    ]
    return {"query_count": len(rows), **{key: safe_mean([float(row[key]) for row in rows]) for key in keys}}


def evaluate_run(run: dict[str, Any], qids: list[str], gold: dict[str, Any], pages_by_file: dict[str, list[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for qid in qids:
        row = run["rows"].get(qid)
        if not row or qid not in gold:
            continue
        ranked = ranked_pages_from_row(row)
        selected = selected_files_from_row(row)
        candidate = candidate_pages_from_row(row, pages_by_file)
        summary = summarize_retrieval(ranked, candidate, selected, gold[qid])
        summary.update({"query_id": qid, "source": source_from_doc_id(qid), "ranked_pages": ranked, "candidate_pages": candidate, "selected_files": selected})
        summaries.append(summary)
        details.append({"query_id": qid, "ranked": ranked, "candidate": candidate, "selected": selected, "summary": summary})
    return summaries, details


def project_legacy_pages(chunk_text: str, default_page: int = 0) -> list[int]:
    matches = [int(match.group(1)) - 1 for match in PAGE_RE.finditer(chunk_text)]
    return [page for page in matches if page >= 0] or [default_page]


def load_legacy(legacy_root: Path, queries: list[dict[str, Any]], gold: dict[str, Any], kdl_pages: dict[str, dict[str, Any]], depths: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index_root = legacy_root / "alpha0.7-6f03a77aa90d"
    manifest = read_json(index_root / "manifest.json")
    chunks = read_jsonl(index_root / "chunks.jsonl")
    chunk_by_id = {str(row["chunk_id"]): row for row in chunks}
    index = BM25Index.load(index_root / "bm25.json")
    vectors = np.load(index_root / "vectors.npy", mmap_mode="r")
    query_vectors_path = ROOT / "data/work/benchmark_hierarchical_local/embedding_adaptive_openrouter_te3s_fulltext/embeddings/query_embeddings.npy"
    query_vectors = np.load(query_vectors_path)
    if len(query_vectors) != len(queries):
        raise ValueError(f"legacy query vector count {len(query_vectors)} != query count {len(queries)}")
    if vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError(f"legacy dim mismatch: vectors={vectors.shape}, queries={query_vectors.shape}")
    doc_to_canonical: dict[str, str] = {}
    for page_id in kdl_pages:
        legacy_doc_id = str(kdl_pages[page_id].get("legacy_doc_id") or "")
        if legacy_doc_id:
            doc_to_canonical.setdefault(legacy_doc_id, page_id.split("#page=", 1)[0])
        doc_to_canonical.setdefault(page_id.split("#page=", 1)[0].split("::", 1)[1], page_id.split("#page=", 1)[0])
    # The legacy chunk doc_id is opaque.  Its metadata carries the canonical filename in the chunk manifest when available.
    for chunk in chunks:
        raw = str(chunk.get("doc_id", ""))
        metadata = chunk.get("metadata") or {}
        candidate = metadata.get("canonical_doc_id") or metadata.get("source_doc_id") or metadata.get("file_id")
        if candidate and "::" in str(candidate):
            doc_to_canonical[raw] = str(candidate)
    # Fall back to the hash embedded in the source filename from the KDL pages.
    canonical_by_hash = {doc.split("::", 1)[1]: doc for doc in {file_id(page_id) for page_id in kdl_pages}}
    for chunk in chunks:
        raw = str(chunk.get("doc_id", ""))
        if raw in doc_to_canonical:
            continue
        raw_lower = raw.casefold()
        for hash_part, canonical in canonical_by_hash.items():
            if raw_lower == hash_part.casefold() or hash_part.casefold() in raw_lower:
                doc_to_canonical[raw] = canonical
                break

    rows: list[dict[str, Any]] = []
    query_index = {str(row["query_id"]): index for index, row in enumerate(queries)}
    for query in queries:
        qid = str(query["query_id"])
        if qid not in gold:
            continue
        qvec = np.asarray(query_vectors[query_index[qid]], dtype=np.float32)
        dense_scores = vectors @ qvec
        for depth in depths:
            dense_positions = np.argsort(-dense_scores)[:depth]
            dense_hits = [(index.chunk_ids[int(position)], float(dense_scores[int(position)])) for position in dense_positions]
            sparse_hits = [(index.chunk_ids[position], score) for position, score in index.search(str(query.get("query", "")), depth)]
            dense_by_chunk = {chunk_id: score for chunk_id, score in dense_hits}
            sparse_by_chunk = {chunk_id: score for chunk_id, score in sparse_hits}
            fused = alpha_fuse(dense_hits, sparse_hits, alpha=0.7, top_k=depth)
            top_chunks = fused[:depth]
            ranked_docs: list[str] = []
            ranked_pages: list[str] = []
            chunk_rows: list[dict[str, Any]] = []
            for rank, (chunk_id, score) in enumerate(top_chunks, 1):
                chunk = chunk_by_id.get(chunk_id, {})
                raw_doc = str(chunk.get("doc_id", ""))
                canonical = doc_to_canonical.get(raw_doc, raw_doc)
                if canonical not in ranked_docs:
                    ranked_docs.append(canonical)
                pages = project_legacy_pages(text_value(chunk.get("text")), int((chunk.get("metadata") or {}).get("page_index", 0) or 0))
                for page in pages:
                    page_id = f"{canonical}#page={page}" if "::" in canonical else ""
                    if page_id and page_id not in ranked_pages:
                        ranked_pages.append(page_id)
                chunk_rows.append({
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "raw_doc_id": raw_doc,
                    "canonical_doc_id": canonical,
                    "dense_score": dense_by_chunk.get(chunk_id),
                    "bm25_score": sparse_by_chunk.get(chunk_id),
                    "score": score,
                    "metadata": chunk.get("metadata", {}),
                    "page_markers": pages,
                    "text_preview": text_value(chunk.get("text"))[:500],
                })
            gold_pages = gold[qid]["pages"]
            gold_files = gold[qid]["files"]
            rows.append({
                "query_id": qid,
                "source": str(query.get("source") or source_from_doc_id(next(iter(gold_pages)))),
                "query": query.get("query", ""),
                "depth": depth,
                "metric_unit": "legacy_chunk_projected_to_page_is_approximate",
                "ranked_docs": ranked_docs,
                "top_files_at_k": {str(k): ranked_docs[:k] for k in (1, 3, 5, 10)},
                "ranked_pages_approx": ranked_pages,
                "gold_files": sorted(gold_files),
                "gold_pages": sorted(gold_pages),
                "doc_recall@1": file_recall(ranked_docs, gold_files, 1),
                "doc_recall@3": file_recall(ranked_docs, gold_files, 3),
                "doc_recall@5": file_recall(ranked_docs, gold_files, 5),
                "doc_recall@10": file_recall(ranked_docs, gold_files, 10),
                "doc_hit@10": float(bool(set(ranked_docs[:10]) & gold_files)),
                "page_recall@10_approx": page_recall(ranked_pages, gold_pages, 10),
                "page_recall@20_approx": page_recall(ranked_pages, gold_pages, 20),
                "first_gold_file_rank": first_rank(ranked_docs, gold_files),
                "first_gold_page_rank_approx": first_rank(ranked_pages, gold_pages),
                "chunks": chunk_rows,
            })
    metadata = {
        "manifest": manifest,
        "index_root": str(index_root),
        "vectors_shape": list(vectors.shape),
        "query_vectors": str(query_vectors_path),
        "query_vector_shape": list(query_vectors.shape),
        "doc_mapping_count": len(doc_to_canonical),
        "unmapped_legacy_docs": sorted({str(chunk.get("doc_id")) for chunk in chunks if str(chunk.get("doc_id")) not in doc_to_canonical})[:50],
    }
    return rows, metadata


def control_page_runs(output_dir: Path, queries: list[dict[str, Any]], gold: dict[str, Any], alignment: list[dict[str, Any]], limit: int = 20) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    page_rows = {row["page_id"]: row for row in alignment}
    ordered_pages = sorted(page_rows)
    texts = {
        "light_page_bm25_control": [text_value(page_rows[page_id].get("light_base_text", "")) for page_id in ordered_pages],
        "light_embedded_ocr_page_bm25_control": [text_value(page_rows[page_id].get("light_ocr_text", "")) for page_id in ordered_pages],
        "kdl_page_bm25_control": [text_value(page_rows[page_id].get("kdl_text", "")) for page_id in ordered_pages],
            "light_plus_kdl_structured_control": [text_value(page_rows[page_id].get("light_base_text", "")) + " " + text_value(page_rows[page_id].get("kdl_modality_text", "")) for page_id in ordered_pages],
    }
    # Alignment JSON intentionally stores compact stats, but the control needs the actual text.
    # Rehydrate it from fields attached by main before this function is called.
    runs: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] = {}
    for name, documents in texts.items():
        index = BM25Index(analyzer_name="plain").build([{"chunk_id": page_id, "doc_id": file_id(page_id), "text": text} for page_id, text in zip(ordered_pages, documents)])
        rows: list[dict[str, Any]] = []
        for query in queries:
            qid = str(query["query_id"])
            if qid not in gold:
                continue
            hits = index.search(str(query.get("query", "")), limit)
            ranked = [index.chunk_ids[position] for position, _ in hits]
            rows.append({"query_id": qid, "ranked_pages": ranked, "scores": [score for _, score in hits]})
        write_jsonl(output_dir / "page_controls" / f"{name}.jsonl", rows)
        per_query = []
        for row in rows:
            qgold = gold[row["query_id"]]
            per_query.append({"query_id": row["query_id"], "source": source_from_doc_id(row["query_id"]), **summarize_retrieval(row["ranked_pages"], row["ranked_pages"], [file_id(page) for page in row["ranked_pages"]], qgold)})
        metrics[name] = aggregate_metrics(per_query)
        runs[name] = {"rows": {row["query_id"]: row for row in rows}, "metrics": metrics[name]}

    # A small field-weighted control keeps body/table/figure/formula evidence
    # separate.  Scores are min-max normalized within each field and combined;
    # this is intentionally exploratory, not tuned on a held-out split.
    field_texts = {
        "body": [text_value(page_rows[page_id].get("kdl_body_text", "")) for page_id in ordered_pages],
        "figure": [text_value(page_rows[page_id].get("kdl_figure_text", "")) for page_id in ordered_pages],
        "table": [text_value(page_rows[page_id].get("kdl_table_text", "")) for page_id in ordered_pages],
        "formula": [text_value(page_rows[page_id].get("kdl_formula_text", "")) for page_id in ordered_pages],
    }
    field_indexes = {
        field: BM25Index(analyzer_name="plain").build([{"chunk_id": page_id, "doc_id": file_id(page_id), "text": text} for page_id, text in zip(ordered_pages, documents)])
        for field, documents in field_texts.items()
    }
    field_weights = {"body": 1.0, "figure": 1.5, "table": 1.5, "formula": 2.0}
    rows = []
    for query in queries:
        qid = str(query["query_id"])
        if qid not in gold:
            continue
        combined: dict[str, float] = defaultdict(float)
        for field, index in field_indexes.items():
            hits = index.search(str(query.get("query", "")), len(ordered_pages))
            if not hits:
                continue
            scores = [score for _, score in hits]
            low, high = min(scores), max(scores)
            for position, score in hits:
                normalized = 1.0 if high - low < 1e-12 else (score - low) / (high - low)
                combined[index.chunk_ids[position]] += field_weights[field] * normalized
        ranked = [page_id for page_id, _ in sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:limit]]
        rows.append({"query_id": qid, "ranked_pages": ranked, "scores": [combined[page_id] for page_id in ranked]})
    name = "kdl_field_weighted_control"
    write_jsonl(output_dir / "page_controls" / f"{name}.jsonl", rows)
    per_query = []
    for row in rows:
        qgold = gold[row["query_id"]]
        per_query.append({"query_id": row["query_id"], "source": source_from_doc_id(row["query_id"]), **summarize_retrieval(row["ranked_pages"], row["ranked_pages"], [file_id(page) for page in row["ranked_pages"]], qgold)})
    metrics[name] = aggregate_metrics(per_query)
    runs[name] = {"rows": {row["query_id"]: row for row in rows}, "metrics": metrics[name]}
    return runs, metrics


def cause_labels(qid: str, gold: dict[str, Any], alignment_by_page: dict[str, dict[str, Any]], run_summaries: dict[str, dict[str, Any]]) -> list[str]:
    labels: set[str] = set()
    qgold = gold[qid]
    for page in qgold["pages"]:
        prep = alignment_by_page.get(page, {})
        if prep.get("kdl_modality_term_count", 0) > 0:
            if float(prep.get("light_base_modality_coverage", 0.0)) < 0.35:
                labels.add("kdl_only_signal")
            if float(prep.get("light_ocr_modality_coverage", 0.0)) > float(prep.get("light_base_modality_coverage", 0.0)) + 0.10:
                labels.add("ocr_recoverable")
        if prep.get("light_mojibake") or prep.get("ocr_mojibake"):
            labels.add("parser_garbling")
        for name, summary in run_summaries.items():
            selected = set(summary.get("selected_files", []))
            candidate = set(summary.get("candidate_pages", []))
            ranked = set(summary.get("ranked_pages", []))
            if file_id(page) not in selected and selected:
                labels.add("file_selection_miss")
            elif candidate and page not in candidate:
                labels.add("candidate_pool_miss")
            elif candidate and page not in ranked:
                labels.add("page_ranking_miss")
        cohere = run_summaries.get("light_cohere_r100")
        if cohere:
            candidate = set(cohere.get("candidate_pages", []))
            reranked = set(cohere.get("ranked_pages", []))
            if page in candidate and page not in reranked:
                labels.add("reranker_ordering_only")
    labels.add("chunking_or_unit_mismatch")
    return sorted(labels)


def page_cause_labels(page: str, prep: dict[str, Any], run_summaries: dict[str, dict[str, Any]]) -> list[str]:
    """Assign observable labels to one query/gold-page pair.

    The baseline stage labels use the hierarchical BM25 arm.  Candidate versus
    final labels use Cohere r100 when available.  This avoids declaring a page
    a generic failure merely because another, unrelated arm ranked it lower.
    """
    labels: set[str] = set()
    if prep.get("kdl_modality_term_count", 0) > 0:
        base_cov = float(prep.get("light_base_modality_coverage", 0.0))
        ocr_cov = float(prep.get("light_ocr_modality_coverage", 0.0))
        if base_cov < 0.35:
            labels.add("kdl_only_signal")
        if ocr_cov > base_cov + 0.10:
            labels.add("ocr_recoverable")
    if prep.get("light_mojibake") or prep.get("ocr_mojibake"):
        labels.add("parser_garbling")

    baseline = run_summaries.get("light_hier_bm25")
    if baseline:
        selected = set(baseline.get("selected_files", []))
        candidate = set(baseline.get("candidate_pages", []))
        ranked = set(baseline.get("ranked_pages", []))
        if file_id(page) not in selected:
            labels.add("file_selection_miss")
        elif page not in candidate:
            labels.add("candidate_pool_miss")
        elif page not in ranked:
            labels.add("page_ranking_miss")

    cohere = run_summaries.get("light_cohere_r100")
    if cohere:
        candidate_list = list(cohere.get("candidate_pages", []))
        ranked_list = list(cohere.get("ranked_pages", []))
        candidate_set = set(candidate_list)
        ranked_set = set(ranked_list)
        if file_id(page) not in set(cohere.get("selected_files", [])):
            labels.add("file_selection_miss")
        elif page not in candidate_set:
            labels.add("candidate_pool_miss")
        elif page not in ranked_set:
            labels.add("page_ranking_miss")
        else:
            candidate_rank = candidate_list.index(page) + 1
            final_rank = ranked_list.index(page) + 1
            if candidate_rank != final_rank:
                labels.add("reranker_ordering_only")
    return sorted(labels)


def format_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(rows[0]))]
    lines = ["| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(rows[0])) + " |"]
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    lines.extend("| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |" for row in rows[1:])
    return "\n".join(lines)


def write_visual_samples(output_dir: Path, alignment: list[dict[str, Any]], base_pages: dict[str, dict[str, Any]], per_source: int = 5) -> list[dict[str, Any]]:
    """Write a small manual-review manifest without rendering the whole corpus."""
    samples: list[dict[str, Any]] = []
    categories = {
        "figure_low_coverage": lambda row: int(row.get("kdl_figure_count", 0)) > 0,
        "table_low_coverage": lambda row: int(row.get("kdl_table_count", 0)) > 0,
        "formula_low_coverage": lambda row: int(row.get("kdl_formula_count", 0)) > 0,
        "mixed_low_coverage": lambda row: row.get("kdl_modality") == "mixed",
        "light_empty_or_ocr_error": lambda row: bool(row.get("light_base_empty")) or bool(row.get("light_ocr_error")),
    }
    for source in sorted(PRIMARY_SOURCES):
        source_rows = [row for row in alignment if row.get("source") == source]
        for category, predicate in categories.items():
            candidates = [row for row in source_rows if predicate(row)]
            candidates.sort(key=lambda row: (float(row.get("light_base_modality_coverage", 1.0)), -int(row.get("kdl_modality_term_count", 0)), row.get("page_id", "")))
            for row in candidates[:per_source]:
                page_id = row["page_id"]
                base = base_pages.get(page_id, {})
                samples.append({
                    "source": source,
                    "category": category,
                    "page_id": page_id,
                    "page_index": row.get("page_index"),
                    "page_number": base.get("page_number"),
                    "relative_path": base.get("relative_path"),
                    "source_path": base.get("source_path"),
                    "kdl_modality": row.get("kdl_modality"),
                    "kdl_figure_count": row.get("kdl_figure_count"),
                    "kdl_table_count": row.get("kdl_table_count"),
                    "kdl_formula_count": row.get("kdl_formula_count"),
                    "base_modality_coverage": row.get("light_base_modality_coverage"),
                    "ocr_modality_coverage": row.get("light_ocr_modality_coverage"),
                    "kdl_figure_text": row.get("kdl_figure_text", "")[:1200],
                    "kdl_table_text": row.get("kdl_table_text", "")[:1200],
                    "kdl_formula_text": row.get("kdl_formula_text", "")[:1200],
                    "light_base_preview": row.get("light_base_preview", "")[:1200],
                    "light_ocr_preview": row.get("light_ocr_text", "")[:1200],
                })
    write_jsonl(output_dir / "visual_review_samples.jsonl", samples)
    return samples


def write_report(output_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Legacy KDL vs Light Preparation / Light Retrieval",
        "",
        "## Scope and metric warning",
        "",
        "Primary conclusion is restricted to the 80 Physics and 75 Industrial queries (155 total). MPDocVQA is retained only as a diagnostic control.",
        "Legacy retrieval is chunk/document-level; light retrieval is page-level. Legacy page numbers are projected from `Page N` markers and are approximate. Therefore a legacy document recall must not be compared directly with light page recall.",
        "The user-provided 82.5% is recorded as an unresolved provenance claim. The repository evaluator defines legacy `recall@10` as document-level set recall, while the reconstructed alpha=0.7 values are reported separately below.",
        "",
        "## Validation",
        "",
        f"- aligned pages: `{report['validation']['aligned_pages']}` / `{report['validation']['light_pages']}`; missing KDL: `{report['validation']['missing_kdl_pages']}`; missing light: `{report['validation']['missing_light_pages']}`",
        f"- queries: `{report['validation']['query_count']}` total; Physics `{report['validation']['source_query_counts'].get('vidore_physics', 0)}`; Industrial `{report['validation']['source_query_counts'].get('vidore_industrial', 0)}`; MPDocVQA `{report['validation']['source_query_counts'].get('mpdocvqa', 0)}`",
        f"- qrel pages: `{report['validation']['qrel_count']}`; qrel pages outside light corpus: `{report['validation']['qrel_pages_missing']}`",
        "",
        "## Parsing comparison",
        "",
        format_table([
            ["Source", "Pages", "KDL figure", "KDL table", "KDL formula", "KDL modality pages", "Light base empty", "Base modality coverage", "OCR modality coverage"],
            *[[source, str(item["page_count"]), str(item["kdl_figures"]), str(item["kdl_tables"]), str(item["kdl_formulas"]), str(item["modality_pages"]), str(item["light_empty"]), pct(item["mean_base_modality_coverage"]), pct(item["mean_ocr_modality_coverage"])] for source, item in report["parsing_summary"].items()],
        ]),
        "",
        "Coverage is token coverage of KDL figure/table/equation text in light text; it is a diagnostic, not a semantic accuracy score. The structured control injects KDL fields into the page representation and is therefore an attribution control, not a production result.",
        "",
        "A manual-review manifest for representative figure/table/formula/mixed/empty pages is in `visual_review_samples.jsonl`; it contains canonical page IDs and source PDF paths so selected pages can be rendered on demand.",
        "",
        "## Retrieval comparison",
        "",
    ]
    for source in ("vidore_physics", "vidore_industrial", "all"):
        lines += [f"### {source}", "", format_table([
            ["Arm", "N", "Page R@10", "Page R@20", "Hit@10", "nDCG@10", "File R@3", "Pool R"],
            *[[name, str(values.get("query_count", 0)), pct(values.get("page_recall@10")), pct(values.get("page_recall@20")), pct(values.get("page_hit@10")), f"{100.0 * float(values.get('ndcg@10', 0.0)):.2f}", pct(values.get("file_recall@3")), pct(values.get("candidate_pool_recall"))] for name, metrics in report["retrieval_metrics"].items() for values in [metrics.get(source, {"query_count": 0})]],
        ]), ""]
    lines += [
        "## Legacy reconstruction",
        "",
        format_table([
            ["Depth", "Source", "N", "Doc R@1", "Doc R@3", "Doc R@10", "Doc hit@10", "Projected page R@10"],
            *[[str(depth), source, str(values.get("query_count", 0)), pct(values.get("doc_recall@1")), pct(values.get("doc_recall@3")), pct(values.get("doc_recall@10")), pct(values.get("doc_hit@10")), pct(values.get("page_recall@10_approx"))] for depth, group in report["legacy_metrics"].items() for source, values in group.items()],
        ]),
        "",
        "## Cause labels",
        "",
        format_table([["Label", "Physics", "Industrial", "Primary + MPDoc diagnostic"]] + [[label, str(report["cause_counts"].get(label, {}).get("vidore_physics", 0)), str(report["cause_counts"].get(label, {}).get("vidore_industrial", 0)), str(sum(report["cause_counts"].get(label, {}).values()))] for label in sorted(report["cause_counts"])]),
        "",
        "Labels overlap and are evidence-based diagnostics, not qrel-trained causal proof. Inspect `per_query_comparison.jsonl` and `gold_page_diagnostics.jsonl` for exact query/page evidence.",
        "",
        "## Offline controls",
        "",
        format_table([
            ["Control", "N", "Page R@10", "Page R@20", "Hit@10", "nDCG@10"],
            *[[name, str(values.get("query_count", 0)), pct(values.get("page_recall@10")), pct(values.get("page_recall@20")), pct(values.get("page_hit@10")), f"{100.0 * float(values.get('ndcg@10', 0.0)):.2f}"] for name, values in report["control_metrics"].items()],
        ]),
        "",
        "### Offline controls by primary source",
        "",
        format_table([
            ["Control", "Source", "N", "Page R@10", "Page R@20", "Hit@10", "nDCG@10"],
            *[[name, source, str(values.get("query_count", 0)), pct(values.get("page_recall@10")), pct(values.get("page_recall@20")), pct(values.get("page_hit@10")), f"{100.0 * float(values.get('ndcg@10', 0.0)):.2f}"] for name, metric_group in report["control_metrics_by_source"].items() for source in ("vidore_physics", "vidore_industrial") for values in [metric_group.get(source, {"query_count": 0})]],
        ]),
        "",
        "Interpretation: if KDL-page BM25 materially beats light-page BM25 at the same page unit, parser representation is implicated. The structured control isolates the value of KDL figure/table/equation terms, but can lose under plain BM25 because it also changes document length and term weighting.",
        "",
        "## Reproducibility",
        "",
        "```powershell",
        "python research/experiments/analyze_legacy_vs_light.py",
        "```",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", type=Path, default=ROOT / "data/work/baseline_legacy-full-old-benchmark")
    parser.add_argument("--light-root", type=Path, default=ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/work/baseline_legacy-full-old-benchmark/analysis")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_root = args.light_root / "bundle_debug_light-retrieval"
    queries = read_jsonl(input_root / "input/queries.jsonl")
    qrels = read_jsonl(input_root / "input/qrels_page_level.jsonl")
    base_pages = load_light_pages(input_root / "preparation/baseline_pdf_inspector_tesseract/pages.jsonl")
    ocr_pages = load_light_pages(input_root / "preparation/embedded_image_ocr/pages_with_embedded_ocr.jsonl")
    kdl_pages, kdl_meta = load_kdl_pages(args.legacy_root)
    pages_by_file: dict[str, list[str]] = defaultdict(list)
    for page_id in sorted(base_pages):
        pages_by_file[file_id(page_id)].append(page_id)
    gold = build_gold(qrels)
    qids = [str(query["query_id"]) for query in queries if str(query["query_id"]) in gold]
    source_query_counts = Counter(str(query.get("source") or source_from_doc_id(next(iter(gold[str(query["query_id"])] ["pages"])))) for query in queries if str(query["query_id"]) in gold)

    compact_alignment = page_alignment(kdl_pages, base_pages, ocr_pages)
    # Attach texts only in memory for same-unit controls; the persisted alignment stays compact.
    for row in compact_alignment:
        page_id = row["page_id"]
        kdl = kdl_pages.get(page_id, {})
        base = base_pages.get(page_id, {})
        row["kdl_text"] = kdl.get("structured_text", "")
        row["kdl_body_text"] = kdl.get("text", "")
        row["kdl_figure_text"] = kdl.get("figure_text", "")
        row["kdl_table_text"] = kdl.get("table_text", "")
        row["kdl_formula_text"] = kdl.get("formula_text", "")
        row["kdl_modality_text"] = " ".join((kdl.get("figure_text", ""), kdl.get("table_text", ""), kdl.get("formula_text", "")))
        row["light_base_text"] = base.get("text", "")
        row["light_ocr_text"] = ocr_pages.get(page_id, {}).get("text", "")
    persisted_alignment = []
    for row in compact_alignment:
        persisted = dict(row)
        for key in ("kdl_text", "kdl_modality_text", "light_base_text"):
            persisted.pop(key, None)
        persisted["kdl_structured_preview"] = text_value(row.get("kdl_text"))[:600]
        persisted["kdl_figure_text"] = text_value(row.get("kdl_figure_text"))
        persisted["kdl_table_text"] = text_value(row.get("kdl_table_text"))
        persisted["kdl_formula_text"] = text_value(row.get("kdl_formula_text"))
        persisted["light_base_text"] = text_value(row.get("light_base_text"))
        persisted["light_ocr_text"] = text_value(row.get("light_ocr_text"))
        persisted["light_base_preview"] = text_value(row.get("light_base_text"))[:600]
        persisted_alignment.append(persisted)
    write_jsonl(output_dir / "page_alignment.jsonl", persisted_alignment)

    runs = load_runs(args.light_root, dict(pages_by_file))
    run_metrics: dict[str, dict[str, Any]] = {}
    run_details: dict[str, dict[str, dict[str, Any]]] = {}
    for name, run in runs.items():
        summaries, details = evaluate_run(run, qids, gold, dict(pages_by_file))
        run_metrics[name] = {source: aggregate_metrics(summaries, source if source != "all" else None) for source in ("vidore_physics", "vidore_industrial", "mpdocvqa", "all")}
        run_details[name] = {row["query_id"]: row for row in details}

    legacy_rows, legacy_meta = load_legacy(args.legacy_root, queries, gold, kdl_pages, [100, 200])
    legacy_metrics: dict[str, dict[str, Any]] = {}
    for depth in (100, 200):
        selected = [row for row in legacy_rows if int(row["depth"]) == depth]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[row["source"]].append(row)
        legacy_metrics[str(depth)] = {}
        for source, rows in grouped.items():
            legacy_metrics[str(depth)][source] = {"query_count": len(rows), **{key: safe_mean([float(row[key]) for row in rows]) for key in ("doc_recall@1", "doc_recall@3", "doc_recall@5", "doc_recall@10", "doc_hit@10", "page_recall@10_approx", "page_recall@20_approx")}}
        legacy_metrics[str(depth)]["all"] = {"query_count": len(selected), **{key: safe_mean([float(row[key]) for row in selected]) for key in ("doc_recall@1", "doc_recall@3", "doc_recall@5", "doc_recall@10", "doc_hit@10", "page_recall@10_approx", "page_recall@20_approx")}}
    write_jsonl(output_dir / "legacy_retrieval.jsonl", legacy_rows)

    controls, control_metrics = control_page_runs(output_dir, queries, gold, compact_alignment)
    control_metrics_by_source: dict[str, dict[str, Any]] = {}
    for name, control in controls.items():
        summaries = []
        for row in control["rows"].values():
            qid = row["query_id"]
            if qid not in gold:
                continue
            ranked = list(row.get("ranked_pages", []))
            summaries.append({"query_id": qid, "source": source_from_doc_id(qid), **summarize_retrieval(ranked, ranked, [file_id(page) for page in ranked], gold[qid])})
        control_metrics_by_source[name] = {source: aggregate_metrics(summaries, source if source != "all" else None) for source in ("vidore_physics", "vidore_industrial", "mpdocvqa", "all")}
    alignment_by_page = {row["page_id"]: row for row in compact_alignment}
    legacy_by_query_depth = {(row["query_id"], int(row["depth"])): row for row in legacy_rows}
    per_query: list[dict[str, Any]] = []
    gold_page_diagnostics: list[dict[str, Any]] = []
    for query in queries:
        qid = str(query["query_id"])
        if qid not in gold:
            continue
        source = str(query.get("source") or source_from_doc_id(next(iter(gold[qid]["pages"]))))
        arm_summaries: dict[str, dict[str, Any]] = {}
        for name, details in run_details.items():
            detail = details.get(qid)
            if detail:
                arm_summaries[name] = detail["summary"]
        control_summaries: dict[str, dict[str, Any]] = {}
        for name, control in controls.items():
            row = control["rows"].get(qid)
            if row:
                ranked = list(row.get("ranked_pages", []))
                control_summaries[name] = summarize_retrieval(ranked, ranked, [file_id(page) for page in ranked], gold[qid])
                control_summaries[name].update({"query_id": qid, "source": source, "ranked_pages": ranked, "candidate_pages": ranked, "selected_files": dedupe(file_id(page) for page in ranked)})
        query_row = {
            "query_id": qid,
            "source": source,
            "primary_scope": source in PRIMARY_SOURCES,
            "query": query.get("query", ""),
            "answers": query.get("answers", []),
            "gold_pages": sorted(gold[qid]["pages"]),
            "gold_files": sorted(gold[qid]["files"]),
            "legacy": {str(depth): legacy_by_query_depth.get((qid, depth)) for depth in (100, 200)},
            "light_arms": arm_summaries,
            "control_arms": control_summaries,
            "cause_labels": [],
        }
        query_labels: set[str] = set()
        for page in sorted(gold[qid]["pages"]):
            prep = alignment_by_page.get(page, {})
            page_labels = page_cause_labels(page, prep, arm_summaries)
            query_labels.update(page_labels)
            page_row = {"query_id": qid, "source": source, "page_id": page, "page_preparation": prep, "cause_labels": page_labels}
            for name, summary in arm_summaries.items():
                page_row.setdefault("retrieval", {})[name] = {
                    "selected_file": file_id(page) in set(summary.get("selected_files", [])),
                    "in_candidate_pool": page in set(summary.get("candidate_pages", [])),
                    "in_ranked_pages": page in set(summary.get("ranked_pages", [])),
                    "rank": (summary.get("ranked_pages", []).index(page) + 1) if page in summary.get("ranked_pages", []) else None,
                }
            for name, summary in control_summaries.items():
                page_row.setdefault("retrieval", {})[name] = {
                    "selected_file": file_id(page) in set(summary.get("selected_files", [])),
                    "in_candidate_pool": page in set(summary.get("candidate_pages", [])),
                    "in_ranked_pages": page in set(summary.get("ranked_pages", [])),
                    "rank": (summary.get("ranked_pages", []).index(page) + 1) if page in summary.get("ranked_pages", []) else None,
                }
            gold_page_diagnostics.append(page_row)
        query_row["cause_labels"] = sorted(query_labels)
        per_query.append(query_row)
    write_jsonl(output_dir / "per_query_comparison.jsonl", per_query)
    write_jsonl(output_dir / "gold_page_diagnostics.jsonl", gold_page_diagnostics)
    visual_samples = write_visual_samples(output_dir, compact_alignment, base_pages)

    parsing_summary: dict[str, Any] = {}
    for source in sorted({row["source"] for row in compact_alignment}):
        rows = [row for row in compact_alignment if row["source"] == source]
        parsing_summary[source] = {
            "page_count": len(rows),
            "kdl_figures": sum(int(row["kdl_figure_count"]) for row in rows),
            "kdl_tables": sum(int(row["kdl_table_count"]) for row in rows),
            "kdl_formulas": sum(int(row["kdl_formula_count"]) for row in rows),
            "modality_pages": sum(row["kdl_modality"] != "text_only" for row in rows),
            "light_empty": sum(bool(row["light_base_empty"]) for row in rows),
            "mean_base_modality_coverage": safe_mean([float(row["light_base_modality_coverage"]) for row in rows if row["kdl_modality_term_count"]]),
            "mean_ocr_modality_coverage": safe_mean([float(row["light_ocr_modality_coverage"]) for row in rows if row["kdl_modality_term_count"]]),
            "zero_base_modality_coverage": sum(row["kdl_modality_term_count"] > 0 and row["light_base_modality_coverage"] == 0 for row in rows),
            "zero_ocr_modality_coverage": sum(row["kdl_modality_term_count"] > 0 and row["light_ocr_modality_coverage"] == 0 for row in rows),
        }

    cause_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in gold_page_diagnostics:
        for label in row["cause_labels"]:
            cause_counts[label][row["source"]] += 1
    report = {
        "protocol": {
            "primary_scope": ["vidore_physics", "vidore_industrial"],
            "diagnostic_scope": ["mpdocvqa"],
            "legacy_alpha": 0.7,
            "legacy_depths": [100, 200],
            "light_saved_page_depth": 20,
            "cohere_candidate_depths": [100, 200],
            "qrels_used_only_for_evaluation": True,
            "legacy_page_projection": "approximate; Page N markers in chunks",
        },
        "validation": {
            "query_count": len(qids),
            "source_query_counts": dict(source_query_counts),
            "qrel_count": len(qrels),
            "light_pages": len(base_pages),
            "kdl_pages": len(kdl_pages),
            "aligned_pages": sum(page_id in base_pages and page_id in kdl_pages for page_id in set(base_pages) | set(kdl_pages)),
            "missing_kdl_pages": len(set(base_pages) - set(kdl_pages)),
            "missing_light_pages": len(set(kdl_pages) - set(base_pages)),
            "qrel_pages_missing": sum(row["page_id"] not in base_pages for row in qrels),
            "kdl_meta": kdl_meta,
            "legacy_meta": legacy_meta,
        },
        "provenance_82_5_percent": {
            "user_claim": 0.825,
            "status": "unresolved",
            "repository_semantics": "src/evaluation/metrics.py uses document-level set recall; the claim cannot be compared to page recall without its original report/run",
            "reconstructed_legacy_doc_recall_at_10": {depth: legacy_metrics[str(depth)].get("all", {}).get("doc_recall@10") for depth in (100, 200)},
        },
        "parsing_summary": parsing_summary,
        "retrieval_metrics": run_metrics,
        "legacy_metrics": legacy_metrics,
        "control_metrics": control_metrics,
        "control_metrics_by_source": control_metrics_by_source,
        "cause_counts": {label: dict(counter) for label, counter in cause_counts.items()},
        "output_files": {
            "page_alignment": "page_alignment.jsonl",
            "per_query": "per_query_comparison.jsonl",
            "gold_page_diagnostics": "gold_page_diagnostics.jsonl",
            "legacy_retrieval": "legacy_retrieval.jsonl",
            "page_controls": "page_controls/",
            "visual_review_samples": "visual_review_samples.jsonl",
        },
    }
    write_json(output_dir / "report.json", report)
    write_report(output_dir, report)
    # Remove bulky in-memory text from any accidental future report serialization; current outputs stay compact.
    print(json.dumps({"output_dir": str(output_dir), "query_count": len(qids), "aligned_pages": report["validation"]["aligned_pages"], "runs": sorted(runs), "control_metrics": control_metrics}, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
