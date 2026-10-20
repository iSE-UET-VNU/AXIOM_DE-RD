"""Exact page-level evaluation of the persisted legacy chunk hybrid index.

The index stores chunk text/vectors but not page ids.  This script recreates
the *same* field-routed chunks from the persisted KDL extraction, verifies each
chunk id against ``chunks.jsonl``, and maps original chunk spans to pages before
retrieval.  It then applies the persisted hybrid (dense .7 + BM25 .3) at chunk
level and pools fused chunk scores to pages with max pooling.

No parser, embedding API, or network call is made.  A result is refused unless
all indexed chunk ids can be reconstructed and mapped to canonical pages.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunking_embedding.fields import FieldContext, route_document  # noqa: E402
from src.retrieval.fusion import alpha_fuse  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402

PAGE_MARKER = re.compile(r"\*\*Page\s+(\d+)\*\*")
PAGE_CITATION = re.compile(r"/page/(\d+)/")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_doc(file_name: str) -> str | None:
    stem = Path(file_name).name.removesuffix(".pdf")
    for prefix in ("vidore_physics__", "vidore_industrial__", "mpdocvqa__"):
        if stem.startswith(prefix):
            return prefix[:-2] + "::" + stem[len(prefix):]
    return None


def source_of(value: str) -> str:
    return value.split("::", 1)[0]


def page_intervals(main_text: str, known_page_ids: list[str]) -> list[tuple[int, int, str]]:
    """Return page intervals in original main_text coordinates, not chunk text."""
    by_number = {int(page.rsplit("=", 1)[1]): page for page in known_page_ids}
    markers = [(m.start(), int(m.group(1)) - 1) for m in PAGE_MARKER.finditer(main_text)]
    if not markers:
        return [(0, len(main_text), known_page_ids[0])] if known_page_ids else []
    result: list[tuple[int, int, str]] = []
    if markers[0][0] and 0 in by_number:
        result.append((0, markers[0][0], by_number[0]))
    for index, (start, page_number) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(main_text)
        page = by_number.get(page_number)
        if page is not None and end > start:
            result.append((start, end, page))
    return result


def pages_for_span(start: int, end: int, intervals: Iterable[tuple[int, int, str]]) -> list[str]:
    pages = [page for left, right, page in intervals if max(start, left) < min(end, right)]
    return list(dict.fromkeys(pages))


def citation_page_ids(value: Any, file_id: str, existing: set[str]) -> list[str]:
    values = value if isinstance(value, list) else [value]
    page_ids: list[str] = []
    for item in values:
        for match in PAGE_CITATION.finditer(str(item)):
            page = f"{file_id}#page={int(match.group(1))}"
            if page in existing:
                page_ids.append(page)
    return list(dict.fromkeys(page_ids))


def chunk_pages(chunk: Any, extraction: dict[str, Any], file_id: str, existing: set[str], intervals: list[tuple[int, int, str]]) -> list[str]:
    component = str(chunk.metadata.get("component_type", ""))
    if component == "main_text":
        pages = pages_for_span(chunk.start, chunk.end, intervals)
    elif component == "table":
        item = extraction.get("tables", [])[int(chunk.field_path.removeprefix("tables[").removesuffix("]"))]
        pages = citation_page_ids(item.get("content_citations") or item.get("caption_citations"), file_id, existing)
    elif component == "figure":
        item = extraction.get("figures", [])[int(chunk.field_path.removeprefix("figures[").removesuffix("]"))]
        pages = citation_page_ids(item.get("description_citations") or item.get("caption_citations") or item.get("content_citations"), file_id, existing)
    elif component == "formula":
        item = extraction.get("formulas", [])[int(chunk.field_path.removeprefix("formulas[").removesuffix("]"))]
        formula_index = int(chunk.field_path.removeprefix("formulas[").removesuffix("]"))
        citations = (
            (item.get("citations") or item.get("content_citations"))
            if isinstance(item, dict)
            else (extraction.get("formulas_citations") or [])[formula_index:formula_index + 1]
        )
        pages = citation_page_ids(citations, file_id, existing)
    else:
        pages = []
    if not pages and chunk.metadata.get("page") is not None:
        candidate = f"{file_id}#page={int(chunk.metadata['page'])}"
        if candidate in existing:
            pages = [candidate]
    if not pages:
        raise RuntimeError(f"No exact page mapping for {chunk.chunk_id} ({component}, {chunk.field_path})")
    return pages


def recall_at(ranked: list[str], gold: set[str], depth: int) -> float:
    return len(set(ranked[:depth]).intersection(gold)) / len(gold) if gold else 0.0


def hit_at(ranked: list[str], gold: set[str], depth: int) -> float:
    return float(bool(set(ranked[:depth]).intersection(gold)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", type=Path, default=ROOT / "data/work/baseline_legacy-full-old-benchmark")
    parser.add_argument("--input-root", type=Path, default=ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval/input")
    parser.add_argument("--query-vectors", type=Path, default=ROOT / "data/work/benchmark_hierarchical_local/embedding_adaptive_openrouter_te3s_fulltext/embeddings/query_embeddings.npy")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/work/baseline_legacy-full-old-benchmark/analysis/exact_chunk_page")
    parser.add_argument("--depth", type=int, nargs="+", default=[100, 200])
    args = parser.parse_args()

    index_root = next(args.legacy_root.glob("alpha0.7-*"))
    indexed = read_jsonl(index_root / "chunks.jsonl")
    indexed_by_id = {row["chunk_id"]: row for row in indexed}
    pages = read_jsonl(args.input_root.parent / "preparation/baseline_pdf_inspector_tesseract/pages.jsonl")
    # The default input root is a sibling of preparation in the debug bundle.
    if not pages:
        raise RuntimeError("No page corpus")
    existing_by_file: dict[str, set[str]] = defaultdict(set)
    for row in pages:
        existing_by_file[str(row["doc_id"])].add(str(row["page_id"]))

    mapping: dict[str, list[str]] = {}
    mapping_rows: list[dict[str, Any]] = []
    reconstructed_ids: set[str] = set()
    for parsed_dir in sorted(args.legacy_root.glob("vidore-v3-physics-drive-kdl-pdf-inspector-parse-only/*/documents")):
        for path in sorted(parsed_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            extraction = payload["parsed"]["rows"][0]["extraction"]
            raw_doc = str(payload["document_id"])
            file_id = canonical_doc(str((payload.get("source", {}).get("metadata", {}) or {}).get("file_name", "")))
            if not file_id or file_id not in existing_by_file:
                continue
            existing = existing_by_file[file_id]
            ordered_pages = sorted(existing, key=lambda value: int(value.rsplit("=", 1)[1]))
            intervals = page_intervals(str(extraction.get("main_text") or ""), ordered_pages)
            chunks = route_document(extraction, FieldContext(raw_doc, "fixed_overlap", {"n_words": 512, "overlap": 128}, 20, None))
            for chunk in chunks:
                reconstructed_ids.add(chunk.chunk_id)
                if chunk.chunk_id not in indexed_by_id:
                    continue
                mapped = chunk_pages(chunk, extraction, file_id, existing, intervals)
                mapping[chunk.chunk_id] = mapped
                mapping_rows.append({"chunk_id": chunk.chunk_id, "file_id": file_id, "pages": mapped, "field_path": chunk.field_path, "component_type": chunk.metadata.get("component_type"), "start": chunk.start, "end": chunk.end, "mapping": "original_field_span_or_citation"})

    missing_reconstruction = set(indexed_by_id).difference(reconstructed_ids)
    missing_mapping = set(indexed_by_id).difference(mapping)
    unexpected_reconstruction = reconstructed_ids.difference(indexed_by_id)
    if missing_reconstruction or missing_mapping:
        raise RuntimeError(f"Exact verification failed: missing reconstructed={len(missing_reconstruction)}, unmapped={len(missing_mapping)}")

    queries = read_jsonl(args.input_root / "queries.jsonl")
    qrels = read_jsonl(args.input_root / "qrels_page_level.jsonl")
    gold: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        if source_of(str(row["query_id"])) in {"vidore_physics", "vidore_industrial"}:
            gold[str(row["query_id"])].add(str(row["page_id"]))
    selected_queries = [q for q in queries if str(q["query_id"]) in gold]
    query_index = {str(q["query_id"]): index for index, q in enumerate(queries)}
    qvecs = np.load(args.query_vectors, mmap_mode="r")
    vectors = np.load(index_root / "vectors.npy", mmap_mode="r")
    if qvecs.shape[0] != len(queries) or qvecs.shape[1] != vectors.shape[1]:
        raise RuntimeError(f"Vector/query shape mismatch: chunks={vectors.shape}, queries={qvecs.shape}, query_rows={len(queries)}")
    bm25 = BM25Index.load(index_root / "bm25.json")
    positions = {chunk_id: pos for pos, chunk_id in enumerate(bm25.chunk_ids)}
    if set(positions) != set(mapping):
        raise RuntimeError("BM25 ids and exact mapping ids differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "chunk_page_mapping.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mapping_rows), encoding="utf-8")
    report_path = args.output_dir / "report.json"
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    report.update({"protocol": {"unit": "chunk hybrid -> exact source page", "alpha_dense": 0.7, "page_pool": "max fused score", "chunk_mapping": "reconstructed original field spans/citations", "verification": "all indexed chunk ids reconstructed and mapped"}, "verification": {"indexed_chunks": len(indexed_by_id), "reconstructed_chunks": len(reconstructed_ids), "mapped_chunks": len(mapping), "unexpected_reconstructed_chunks": len(unexpected_reconstruction), "cross_page_chunks": sum(len(value) > 1 for value in mapping.values())}})
    for depth in args.depth:
        rows: list[dict[str, Any]] = []
        for query in selected_queries:
            qid, text = str(query["query_id"]), str(query["query"])
            scores = vectors @ np.asarray(qvecs[query_index[qid]], dtype=np.float32)
            dense_pos = np.argsort(-scores)[:depth]
            dense_hits = [(bm25.chunk_ids[int(pos)], float(scores[int(pos)])) for pos in dense_pos]
            sparse_hits = [(bm25.chunk_ids[pos], float(score)) for pos, score in bm25.search(text, depth)]
            fused = alpha_fuse(dense_hits, sparse_hits, alpha=0.7, top_k=depth)
            # This is intentionally not a page-ranking pool: every page touched
            # by any retrieved chunk is admitted to the page candidate set.
            candidate_pages = list(dict.fromkeys(
                page for chunk_id, _ in fused for page in mapping[chunk_id]
            ))
            page_scores: dict[str, float] = {}
            for chunk_id, score in fused:
                for page in mapping[chunk_id]:
                    page_scores[page] = max(page_scores.get(page, float("-inf")), float(score))
            ranked = [page for page, _ in sorted(page_scores.items(), key=lambda item: (-item[1], item[0]))]
            row = {"query_id": qid, "source": source_of(qid), "query": text, "depth": depth, "gold_pages": sorted(gold[qid]), "candidate_pages_from_top_chunks": candidate_pages, "candidate_page_count": len(candidate_pages), "page_candidate_recall_from_top_chunks": len(set(candidate_pages).intersection(gold[qid])) / len(gold[qid]), "page_candidate_hit_from_top_chunks": float(bool(set(candidate_pages).intersection(gold[qid]))), "ranked_pages": ranked, "page_recall@10": recall_at(ranked, gold[qid], 10), "page_hit@10": hit_at(ranked, gold[qid], 10), "page_recall@20": recall_at(ranked, gold[qid], 20), "page_hit@20": hit_at(ranked, gold[qid], 20), "top_chunks": [{"chunk_id": cid, "score": score, "pages": mapping[cid]} for cid, score in fused]}
            rows.append(row)
        (args.output_dir / f"run_depth{depth}_max.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        by_source: dict[str, Any] = {}
        for source in ("vidore_physics", "vidore_industrial", "all"):
            subset = rows if source == "all" else [row for row in rows if row["source"] == source]
            by_source[source] = {"queries": len(subset), "page_candidate_recall_from_top_chunks": float(np.mean([row["page_candidate_recall_from_top_chunks"] for row in subset])), "page_candidate_hit_from_top_chunks": float(np.mean([row["page_candidate_hit_from_top_chunks"] for row in subset])), "mean_page_candidates_from_top_chunks": float(np.mean([row["candidate_page_count"] for row in subset])), "page_recall@10": float(np.mean([row["page_recall@10"] for row in subset])), "page_hit@10": float(np.mean([row["page_hit@10"] for row in subset])), "page_recall@20": float(np.mean([row["page_recall@20"] for row in subset])), "page_hit@20": float(np.mean([row["page_hit@20"] for row in subset]))}
        report[f"depth_{depth}"] = by_source
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Exact legacy chunk-to-page retrieval", "", "Chunks are reconstructed from the persisted KDL extraction and verified against the index before hybrid retrieval. Page score is the maximum fused score among chunks whose original source span/citation belongs to that page.", "", "| Depth | Source | Queries | Page recall@10 | Page hit@10 | Page recall@20 | Page hit@20 |", "|---:|---|---:|---:|---:|---:|---:|"]
    for depth in sorted(key for key in (int(name.removeprefix("depth_")) for name in report if name.startswith("depth_"))):
        for source, values in report[f"depth_{depth}"].items():
            lines.append(f"| {depth} | {source} | {values['queries']} | {100*values['page_recall@10']:.2f}% | {100*values['page_hit@10']:.2f}% | {100*values['page_recall@20']:.2f}% | {100*values['page_hit@20']:.2f}% |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
