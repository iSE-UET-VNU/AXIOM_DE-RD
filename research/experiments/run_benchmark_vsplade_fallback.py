"""Encode only low-quality OCR/PDF-inspector pages with V-SPLADE.

This is intentionally a small, resumable experiment runner for the merged
benchmark.  It does not replace the BM25 light run and it does not use qrels
for selection.  A page list produced by the OCR-quality audit is the only
input that decides which pages are sent to the visual encoder.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from research.experiments.run_benchmark_hierarchical import (
    VECTOR_DIM,
    _save_csr,
    atomic_json_dump,
    read_jsonl,
    safe_name,
    sha256_file,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pages", type=Path, required=True, help="Fallback page audit JSONL")
    parser.add_argument("--parsing", type=Path, required=True, help="Merged parsing cache JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="naver/v-splade-efficient")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--render-dpi", type=int, default=144)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    return parser.parse_args()


def _write_event(path: Path, event: str, status: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
        "status": status,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_inputs(dataset_root: Path, parsing_path: Path, page_list_path: Path):
    documents = [dict(row) for row in read_jsonl(dataset_root / "documents.jsonl")]
    queries = [dict(row) for row in read_jsonl(dataset_root / "queries.jsonl")]
    pages = [dict(row) for row in read_jsonl(parsing_path)]
    wanted = [dict(row) for row in read_jsonl(page_list_path)]
    wanted_ids = [row["page_id"] for row in wanted]
    wanted_set = set(wanted_ids)
    page_by_id = {row["page_id"]: row for row in pages}
    missing = [page_id for page_id in wanted_ids if page_id not in page_by_id]
    if missing:
        raise RuntimeError(f"Fallback page list contains {len(missing)} pages missing from parsing cache")
    selected = [page_by_id[page_id] for page_id in wanted_ids]
    if len(selected) != len(set(wanted_ids)):
        raise RuntimeError("Fallback page list contains duplicate page IDs")
    doc_by_id = {row["doc_id"]: row for row in documents}
    missing_docs = sorted({row["doc_id"] for row in selected} - set(doc_by_id))
    if missing_docs:
        raise RuntimeError(f"Missing documents for fallback pages: {missing_docs[:5]}")
    return documents, queries, selected, doc_by_id


def _document_path(dataset_root: Path, document: dict[str, Any]) -> Path:
    candidate = dataset_root / str(document["path"])
    if candidate.is_file():
        return candidate
    relative = document.get("relative_path") or document.get("metadata", {}).get("source_path")
    if relative:
        candidate = dataset_root / str(relative)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot resolve document path for {document['doc_id']}")


def _encode_images(model: Any, images: list[Any], batch_size: int) -> sparse.csr_matrix:
    batches: list[sparse.csr_matrix] = []
    for begin in range(0, len(images), batch_size):
        batch = images[begin : begin + batch_size]
        encoded = model.encode_document(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        dense = encoded.to_dense().float().cpu().numpy()
        matrix = sparse.csr_matrix(dense, dtype=np.float32)
        if matrix.shape[1] != VECTOR_DIM:
            raise RuntimeError(f"Unexpected V-SPLADE dimension: {matrix.shape}")
        batches.append(matrix)
    return sparse.vstack(batches, format="csr") if batches else sparse.csr_matrix((0, VECTOR_DIM), dtype=np.float32)


def _encode_document_pages(
    model: Any,
    source_path: Path,
    doc_pages: list[dict[str, Any]],
    render_dpi: int,
    batch_size: int,
) -> sparse.csr_matrix:
    """Render and encode one batch at a time to keep CPU RAM bounded."""
    import fitz
    from PIL import Image

    batches: list[sparse.csr_matrix] = []
    if source_path.suffix.lower() == ".pdf":
        with fitz.open(str(source_path)) as pdf:
            for begin in range(0, len(doc_pages), batch_size):
                images: list[Any] = []
                try:
                    for page in doc_pages[begin : begin + batch_size]:
                        pix = pdf[page["page_index"]].get_pixmap(dpi=render_dpi, alpha=False)
                        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        images.append(image)
                    batches.append(_encode_images(model, images, batch_size=len(images)))
                finally:
                    for image in images:
                        image.close()
    else:
        with Image.open(source_path) as source_image:
            image = source_image.convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        try:
            batches.append(_encode_images(model, [image], batch_size=1))
        finally:
            image.close()
    return sparse.vstack(batches, format="csr") if batches else sparse.csr_matrix((0, VECTOR_DIM), dtype=np.float32)


def main() -> int:
    args = _args()
    if args.batch_size <= 0 or args.query_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    root = args.output_dir
    checkpoint_root = root / "page_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    events = root / "events.jsonl"
    started_total = time.perf_counter()

    documents, queries, selected, doc_by_id = _load_inputs(args.dataset_root, args.parsing, args.pages)
    selected_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in selected:
        selected_by_doc[page["doc_id"]].append(page)
    config = {
        "dataset_root": str(args.dataset_root),
        "pages": str(args.pages),
        "parsing": str(args.parsing),
        "model": args.model,
        "device": args.device,
        "batch_size": args.batch_size,
        "query_batch_size": args.query_batch_size,
        "render_dpi": args.render_dpi,
        "dtype": args.dtype,
        "selected_pages": len(selected),
        "queries": len(queries),
        "vector_dim": VECTOR_DIM,
    }
    atomic_json_dump(root / "config.json", config)
    atomic_json_dump(
        root / "input_summary.json",
        {
            "documents": len(documents),
            "queries": len(queries),
            "selected_pages": len(selected),
            "selected_files": len(selected_by_doc),
            "page_ids": [page["page_id"] for page in selected],
        },
    )
    _write_event(events, "validate_inputs", "complete", pages=len(selected), queries=len(queries), files=len(selected_by_doc))

    try:
        import fitz
        from PIL import Image
        from sentence_transformers import SparseEncoder
    except Exception as exc:
        _write_event(events, "model_import", "error", error_type=type(exc).__name__, error=str(exc))
        raise

    model_started = time.perf_counter()
    _write_event(events, "model_load", "started", model=args.model, device=args.device)
    model_kwargs: dict[str, Any] = {}
    if args.dtype != "auto":
        import torch

        model_kwargs["torch_dtype"] = torch.float16 if args.dtype == "float16" else torch.float32
    model = SparseEncoder(args.model, trust_remote_code=True, device=args.device, model_kwargs=model_kwargs)
    model_seconds = time.perf_counter() - model_started
    _write_event(events, "model_load", "complete", seconds=model_seconds)

    matrices: list[sparse.csr_matrix] = []
    metadata: list[dict[str, Any]] = []
    page_stats: list[dict[str, Any]] = []

    for document in documents:
        doc_pages = selected_by_doc.get(document["doc_id"], [])
        if not doc_pages:
            continue
        source_path = _document_path(args.dataset_root, document)
        input_sha = sha256_file(source_path)
        checkpoint = checkpoint_root / f"{safe_name(document['doc_id'])}.npz"
        checkpoint_meta = checkpoint_root / f"{safe_name(document['doc_id'])}.metadata.json"
        if checkpoint.is_file() and checkpoint_meta.is_file():
            matrix = sparse.load_npz(checkpoint).tocsr()
            doc_meta = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
            if (
                matrix.shape[0] != len(doc_pages)
                or len(doc_meta) != len(doc_pages)
                or any(item.get("input_sha256") != input_sha for item in doc_meta)
                or [item["page_id"] for item in doc_meta] != [page["page_id"] for page in doc_pages]
            ):
                raise RuntimeError(f"Checkpoint mismatch for {document['doc_id']}")
            matrices.append(matrix)
            metadata.extend(doc_meta)
            page_stats.append({"doc_id": document["doc_id"], "pages": len(doc_pages), "seconds": 0.0, "resumed": True})
            _write_event(events, "encode_document", "skipped", doc_id=document["doc_id"], pages=len(doc_pages))
            continue

        started = time.perf_counter()
        _write_event(events, "encode_document", "started", doc_id=document["doc_id"], pages=len(doc_pages))
        matrix = _encode_document_pages(model, source_path, doc_pages, args.render_dpi, args.batch_size)
        if matrix.shape[0] != len(doc_pages):
            raise RuntimeError(f"Encoded page count mismatch for {document['doc_id']}: {matrix.shape}")
        doc_meta = [
            {
                "page_id": page["page_id"],
                "doc_id": page["doc_id"],
                "source": page["source"],
                "relative_path": page["relative_path"],
                "page_index": page["page_index"],
                "page_number": page["page_number"],
                "fallback_reasons": page.get("reasons", []),
                "input_sha256": input_sha,
            }
            for page in doc_pages
        ]
        _save_csr(checkpoint, matrix)
        atomic_json_dump(checkpoint_meta, doc_meta)
        seconds = time.perf_counter() - started
        stat = {"doc_id": document["doc_id"], "pages": len(doc_pages), "seconds": seconds, "nnz": int(matrix.nnz), "resumed": False}
        atomic_json_dump(checkpoint_root / f"{safe_name(document['doc_id'])}.timing.json", stat)
        matrices.append(matrix)
        metadata.extend(doc_meta)
        page_stats.append(stat)
        _write_event(events, "encode_document", "complete", **stat)
        print(f"[vsplade] {document['doc_id']}: {len(doc_pages)} pages in {seconds:.2f}s", flush=True)

    page_matrix = sparse.vstack(matrices, format="csr") if matrices else sparse.csr_matrix((0, VECTOR_DIM), dtype=np.float32)
    expected_page_ids = [page["page_id"] for page in selected]
    if [item["page_id"] for item in metadata] != expected_page_ids:
        raise RuntimeError("V-SPLADE fallback metadata ordering mismatch")
    _save_csr(root / "page_vectors.npz", page_matrix)
    atomic_json_dump(root / "page_metadata.json", metadata)

    query_vector_path = root / "query_vectors.npz"
    query_metadata_path = root / "query_metadata.json"
    query_started = time.perf_counter()
    if query_vector_path.is_file() and query_metadata_path.is_file():
        query_matrix = sparse.load_npz(query_vector_path).tocsr()
        query_meta = json.loads(query_metadata_path.read_text(encoding="utf-8"))
        if query_matrix.shape != (len(queries), VECTOR_DIM) or [row["query_id"] for row in query_meta] != [row["query_id"] for row in queries]:
            raise RuntimeError("Query vector checkpoint mismatch")
        query_resumed = True
    else:
        _write_event(events, "encode_queries", "started", queries=len(queries))
        query_encoded = model.encode_query(
            [row["query"] for row in queries],
            batch_size=args.query_batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        query_matrix = sparse.csr_matrix(query_encoded.to_dense().float().cpu().numpy(), dtype=np.float32)
        if query_matrix.shape != (len(queries), VECTOR_DIM):
            raise RuntimeError(f"Unexpected query vector shape: {query_matrix.shape}")
        _save_csr(query_vector_path, query_matrix)
        atomic_json_dump(query_metadata_path, [{"row": i, "query_id": row["query_id"], "query": row["query"]} for i, row in enumerate(queries)])
        query_resumed = False
    query_seconds = time.perf_counter() - query_started
    _write_event(events, "encode_queries", "skipped" if query_resumed else "complete", seconds=query_seconds, queries=len(queries), resumed=query_resumed)

    score_started = time.perf_counter()
    scores = (query_matrix @ page_matrix.T).toarray().astype(np.float32)
    np.save(root / "query_page_scores.npy", scores)
    atomic_json_dump(
        root / "score_metadata.json",
        {"query_count": len(queries), "page_count": len(selected), "shape": list(scores.shape), "score_type": "V-SPLADE dot product"},
    )
    score_seconds = time.perf_counter() - score_started
    timing = {
        "model_load_seconds": model_seconds,
        "page_documents": page_stats,
        "query_encoding_seconds": query_seconds,
        "score_seconds": score_seconds,
        "page_count": len(selected),
        "file_count": len(selected_by_doc),
        "query_count": len(queries),
        "page_nnz": int(page_matrix.nnz),
        "query_nnz": int(query_matrix.nnz),
        "device": args.device,
        "model": args.model,
        "batch_size": args.batch_size,
        "total_wall_seconds": time.perf_counter() - started_total,
    }
    atomic_json_dump(root / "timing.json", timing)
    atomic_json_dump(root / "manifest.json", {"config": config, "timing": timing, "page_ids": expected_page_ids})
    _write_event(events, "complete", "complete", **timing)
    print(json.dumps(timing, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
