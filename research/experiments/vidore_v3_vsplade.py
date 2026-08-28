"""Run the local V-SPLADE checkpoint on a raw ViDoRe V3 image corpus.

The Industrial release already contains rendered page PNG bytes in the corpus
parquet shards, so this experiment measures image decoding + V-SPLADE page
encoding rather than PDF rendering.  The output is a CSR sparse index, a
retrieval run, and the repository's standard ViDoRe V3 metrics.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_csr(path: Path, matrix: sparse.csr_matrix) -> None:
    np.savez_compressed(
        path,
        indices=matrix.indices.astype(np.int32, copy=False),
        indptr=matrix.indptr.astype(np.int32, copy=False),
        format=np.asarray("csr").astype("S3"),
        shape=np.asarray(matrix.shape, dtype=np.int64),
        data=matrix.data.astype(np.float32, copy=False),
    )


def _load_corpus_shards(corpus_dir: Path) -> list[Path]:
    paths = sorted(corpus_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No corpus parquet shards under {corpus_dir}")
    return paths


def _encode_pages(model: Any, shards: list[Path], output_dir: Path, batch_size: int) -> tuple[sparse.csr_matrix, list[dict[str, Any]], dict[str, float]]:
    metadata: list[dict[str, Any]] = []
    indices_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    indptr = [0]
    total_nnz = 0
    read_seconds = 0.0
    decode_seconds = 0.0
    encode_seconds = 0.0
    page_count = 0

    for shard_no, shard in enumerate(shards, start=1):
        started = time.perf_counter()
        frame = pd.read_parquet(
            shard,
            engine="fastparquet",
            columns=["corpus_id", "doc_id", "page_number_in_doc", "image.bytes", "image.path"],
        ).sort_values("corpus_id")
        read_seconds += time.perf_counter() - started

        rows = list(frame.itertuples(index=False))
        for offset in range(0, len(rows), batch_size):
            batch_rows = rows[offset : offset + batch_size]
            started = time.perf_counter()
            images: list[Image.Image] = []
            batch_meta: list[dict[str, Any]] = []
            for row in batch_rows:
                image = Image.open(io.BytesIO(row[3])).convert("RGB")
                images.append(image)
                batch_meta.append(
                    {
                        "corpus_id": int(row[0]),
                        "doc_id": str(row[1]),
                        "page_number_in_doc": int(row[2]),
                        "image_path": str(row[4]),
                    }
                )
            decode_seconds += time.perf_counter() - started

            started = time.perf_counter()
            encoded = model.encode_document(
                images,
                batch_size=len(images),
                show_progress_bar=False,
                convert_to_tensor=True,
            )
            encode_seconds += time.perf_counter() - started

            # The Sentence Transformers sparse encoder returns sparse COO on
            # CUDA. Batches are deliberately small, so densifying only this
            # batch is safe and lets scipy build the persistent CSR index.
            dense = encoded.to_dense().float().cpu().numpy()
            batch_csr = sparse.csr_matrix(dense, dtype=np.float32)
            indices_parts.append(batch_csr.indices.copy())
            data_parts.append(batch_csr.data.copy())
            total_nnz += int(batch_csr.nnz)
            indptr.extend((batch_csr.indptr[1:] + indptr[-1]).tolist())
            metadata.extend(batch_meta)
            page_count += len(batch_rows)

            if page_count % 100 < len(batch_rows) or page_count == 1:
                elapsed = max(time.perf_counter() - started + encode_seconds, 1e-6)
                print(f"encoded pages={page_count} shard={shard_no}/{len(shards)} nnz={total_nnz}", flush=True)

            for image in images:
                image.close()

    indices = np.concatenate(indices_parts) if indices_parts else np.empty(0, dtype=np.int32)
    data = np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.float32)
    matrix = sparse.csr_matrix(
        (data, indices, np.asarray(indptr, dtype=np.int32)),
        shape=(len(metadata), 50368),
        dtype=np.float32,
    )
    _save_csr(output_dir / "page_vectors.npz", matrix)
    _json_dump(output_dir / "page_metadata.json", metadata)
    return matrix, metadata, {
        "corpus_read_seconds": read_seconds,
        "image_decode_seconds": decode_seconds,
        "page_encoding_seconds": encode_seconds,
        "page_count": len(metadata),
        "page_nnz": int(matrix.nnz),
    }


def _encode_queries(model: Any, query_path: Path, language: str) -> tuple[sparse.csr_matrix, list[dict[str, Any]], float]:
    queries = pd.read_parquet(query_path, engine="fastparquet")
    queries = queries[queries["language"].astype(str).str.lower() == language].sort_values("query_id")
    records = [
        {
            "query_id": int(row.query_id),
            "qid": f"industrial::{int(row.query_id)}",
            "query": str(row.query),
            "language": str(row.language),
            "query_types": list(row.query_types or []),
            "query_format": str(row.query_format),
            "content_type": list(row.content_type or []),
            "source_type": str(row.source_type),
        }
        for row in queries.itertuples(index=False)
    ]
    started = time.perf_counter()
    encoded = model.encode_query(
        [row["query"] for row in records],
        batch_size=128,
        show_progress_bar=False,
        convert_to_tensor=True,
    )
    seconds = time.perf_counter() - started
    matrix = sparse.csr_matrix(encoded.to_dense().float().cpu().numpy(), dtype=np.float32)
    return matrix, records, seconds


def _retrieve(
    queries: sparse.csr_matrix,
    pages: sparse.csr_matrix,
    page_metadata: list[dict[str, Any]],
    query_metadata: list[dict[str, Any]],
    output_path: Path,
    top_k: int,
) -> float:
    started = time.perf_counter()
    score_matrix = (queries @ pages.T).toarray()
    top_k = min(top_k, pages.shape[0])
    top_indices = np.argsort(-score_matrix, axis=1, kind="stable")[:, :top_k]

    with output_path.open("w", encoding="utf-8") as handle:
        for row_no, query in enumerate(query_metadata):
            chunks = []
            for rank, page_index in enumerate(top_indices[row_no], start=1):
                page = page_metadata[int(page_index)]
                doc_id = f"industrial::{page['doc_id']}#page={page['page_number_in_doc']}"
                chunks.append(
                    {
                        "chunk_id": doc_id,
                        "doc_id": doc_id,
                        "text": "",
                        "score": round(float(score_matrix[row_no, page_index]), 6),
                        "rank": rank,
                    }
                )
            handle.write(
                json.dumps(
                    {
                        "qid": query["qid"],
                        "query": query["query"],
                        "retriever_id": "v-splade-efficient",
                        "index_id": "industrial-page-image-vsplade",
                        "params_hash": "batch3-csr-v1",
                        "chunks": chunks,
                        "scope_doc_ids": None,
                        "latency_ms": 0.0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return time.perf_counter() - started


def _evaluate(run_path: Path, dataset_root: Path, subset: str, language: str) -> dict[str, Any]:
    from src.evaluation.benchmarks.vidore_v3 import ViDoreV3
    from src.evaluation.run_retrieval import evaluate
    from src.retrieval.runs import read

    benchmark = ViDoreV3(root=dataset_root, subset=subset, language=language)
    records = read(run_path)
    return evaluate(benchmark, records, k=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/benchmarks/vidore_v3"))
    parser.add_argument("--model", type=str, default="data/output/vsplade/efficient")
    parser.add_argument("--output-dir", type=Path, default=Path("data/output/vsplade/vidore_v3_industrial_english_283q"))
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--subset", type=str, default="industrial")
    args = parser.parse_args()

    from sentence_transformers import SparseEncoder

    dataset_dir = args.dataset_root / f"vidore_v3_{args.subset}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"loading model={args.model}", flush=True)
    model = SparseEncoder(args.model, trust_remote_code=True, device=args.device)
    model_load_seconds = time.perf_counter() - started

    page_vectors, page_metadata, page_timing = _encode_pages(
        model,
        _load_corpus_shards(dataset_dir / "corpus"),
        args.output_dir,
        args.batch_size,
    )
    query_vectors, query_metadata, query_encoding_seconds = _encode_queries(
        model,
        dataset_dir / "queries" / "test-00000-of-00001.parquet",
        args.language,
    )
    _save_csr(args.output_dir / "query_vectors.npz", query_vectors)
    _json_dump(args.output_dir / "query_metadata.json", query_metadata)

    run_path = args.output_dir / "retrieval_english.jsonl"
    retrieval_seconds = _retrieve(
        query_vectors,
        page_vectors,
        page_metadata,
        query_metadata,
        run_path,
        args.top_k,
    )
    metrics = _evaluate(run_path, args.dataset_root, args.subset, args.language)
    metrics.update(
        {
            "model": str(args.model),
            "dataset": str(dataset_dir),
            "language": args.language,
            "retrieval_unit": "one rendered ViDoRe page image",
            "page_count": len(page_metadata),
            "queries": len(query_metadata),
            "batch_size": args.batch_size,
            "top_k_saved": args.top_k,
            "timing": {
                "model_load_seconds": model_load_seconds,
                **page_timing,
                "query_encoding_seconds": query_encoding_seconds,
                "retrieval_seconds": retrieval_seconds,
                "total_seconds": time.perf_counter() - started,
            },
        }
    )
    _json_dump(args.output_dir / "metrics.json", metrics)
    print(json.dumps({k: metrics.get(k) for k in ["recall@10", "ndcg@10", "page_recall@10", "single_evidence_recall@10", "multi_evidence_recall@10", "timing"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
