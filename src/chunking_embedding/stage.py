"""Canonical in-memory stage: enriched data -> retrieval records and vectors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any
import logging
import re
import time

from ..models import RetrievalRecord
from .contracts import ChunkEmbedConfig
from .embedders import create_embedder
from .fields import FieldContext, Resources, route_document
from .lexical import build_lexical_payload, content_text
from .embedders.openrouter import OpenRouterLLM
from .registry import chunker_needs

_RETRIEVAL_TYPE = {"text_chunk": "text_chunk", "table_chunk": "table", "figure_chunk": "image"}
_FIELD_INDEX = re.compile(r"\[(\d+)\]")

logger = logging.getLogger(__name__)


@dataclass
class ChunkingEmbeddingOutput:
    retrieval_records: list[RetrievalRecord] = field(default_factory=list)
    quality_report: dict[str, Any] = field(default_factory=dict)
    vector_records: list[dict[str, Any]] = field(default_factory=list)
    embedding_report: dict[str, Any] = field(default_factory=dict)
    skipped_docs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ChunkEmbedResult:
    """Internal result shared by the in-memory stage and artifact CLI."""

    chunk_records: list[dict[str, Any]] = field(default_factory=list)
    vector_records: list[dict[str, Any]] = field(default_factory=list)
    skipped_docs: list[dict[str, Any]] = field(default_factory=list)
    embedded_doc_ids: list[str] = field(default_factory=list)
    llm_stats: dict[str, Any] = field(default_factory=dict)
    chunk_time: float = 0.0
    embed_time: float = 0.0


def chunk_and_embed(
    enriched_records: list[dict[str, Any]],
    config: ChunkEmbedConfig,
    embedder: Any | None = None,
    skip_keys: set[tuple[str, str]] | None = None,
) -> ChunkEmbedResult:
    """Chunk and embed ``EnrichedData`` records without file-system I/O."""
    embedder = embedder or create_embedder(config.embedder, config.embedder_params)
    llm_client = OpenRouterLLM(**config.llm) if "llm" in chunker_needs(config.chunker) else None
    resources = Resources(embedder=embedder, llm=llm_client)
    config_hash = config.config_hash()
    skip_keys = skip_keys or set()
    result = ChunkEmbedResult()

    # Chunk every document first, then embed the complete chunk stream in
    # request-sized batches.  Embedding per document is correct but produces
    # many tiny remote requests when discovery passes one page per document.
    pending_docs: list[tuple[str, Any, list[Any]]] = []
    for enriched in enriched_records:
        metadata = enriched.get("metadata") or {}
        doc_id = str(enriched.get("source_object_id") or "")
        source_uri = enriched.get("source_uri") or metadata.get("source_uri")
        try:
            if not doc_id:
                raise ValueError("enriched record has no source_object_id")
            if (doc_id, config_hash) in skip_keys:
                continue
            extraction = _extraction_of(enriched)
            ctx = FieldContext(
                doc_id=doc_id,
                chunker_name=config.chunker,
                chunker_params=config.chunker_params,
                max_rows_per_chunk=config.max_rows_per_chunk,
                language=extraction.get("language"),
            )
            started = time.perf_counter()
            chunks = route_document(extraction, ctx, resources)
            result.chunk_time += time.perf_counter() - started
            pending_docs.append((doc_id, source_uri, chunks))
        except Exception as exc:  # one bad document must never crash the batch
            logger.exception("Chunk/embed failed for doc %s (%s)", doc_id or "?", source_uri)
            result.skipped_docs.append(
                {
                    "doc_id": doc_id or None,
                    "source_uri": source_uri,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    pending: list[tuple[str, Any]] = [
        (doc_id, chunk)
        for doc_id, _source_uri, chunks in pending_docs
        for chunk in chunks
    ]
    vectors_by_position: dict[int, list[float]] = {}
    failed_docs: dict[str, str] = {}
    batch_size = max(1, int((config.embedder_params or {}).get("batch_size", 64)))
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        try:
            started = time.perf_counter()
            vectors = embedder.embed([chunk.embedding_text() for _doc_id, chunk in batch])
            result.embed_time += time.perf_counter() - started
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embedder returned {len(vectors)} vectors for {len(batch)} chunks"
                )
        except Exception as exc:  # keep later batches resumable
            reason = f"{type(exc).__name__}: {exc}"
            for doc_id, _chunk in batch:
                failed_docs.setdefault(doc_id, reason)
            logger.exception("Embedding batch failed for %s chunk(s)", len(batch))
            continue
        for offset, vector in enumerate(vectors):
            vectors_by_position[start + offset] = vector

    for doc_id, source_uri, chunks in pending_docs:
        positions = [index for index, (owner, _chunk) in enumerate(pending) if owner == doc_id]
        reason = failed_docs.get(doc_id)
        if reason or any(index not in vectors_by_position for index in positions):
            result.skipped_docs.append(
                {
                    "doc_id": doc_id,
                    "source_uri": source_uri,
                    "reason": reason or "embedding batch did not return all vectors",
                }
            )
            continue
        doc_metadata = {
            "source_uri": source_uri,
            "title": None,
            "document_type": None,
        }
        # The extraction metadata is not needed for batching correctness; keep
        # the existing fields when available on the source record.
        for enriched in enriched_records:
            if str(enriched.get("source_object_id") or "") == doc_id:
                extraction = _extraction_of(enriched)
                doc_metadata.update(
                    title=extraction.get("title"),
                    document_type=extraction.get("document_type"),
                )
                break
        for position, chunk in zip(positions, chunks):
            vector = vectors_by_position[position]
            chunk_record = chunk.to_record(config_hash)
            chunk_record["metadata"].update(doc_metadata)
            result.chunk_records.append(chunk_record)
            result.vector_records.append(
                {
                    "id": chunk.chunk_id,
                    "vector": [round(float(value), 6) for value in vector],
                    "model": embedder.model,
                    "dim": len(vector),
                }
            )
        result.embedded_doc_ids.append(doc_id)
    result.llm_stats = dict(llm_client.stats) if llm_client else {}
    return result


def run(enriched_data: list[dict[str, Any]], config_section: Any) -> ChunkingEmbeddingOutput:
    """Run chunking and embedding on enrichment-stage records.

    The input is intentionally a list of serialized ``EnrichedData`` objects,
    matching the format passed by ``src.pipeline``.  This stage does not read
    parsed or cleaned data and does not own upstream lifecycle decisions.
    """
    config = ChunkEmbedConfig.from_mapping(config_section)
    embedder = create_embedder(config.embedder, config.embedder_params)
    result = chunk_and_embed(enriched_data, config, embedder=embedder)
    config_hash = config.config_hash()

    retrieval_records: list[RetrievalRecord] = []
    vector_records: list[dict[str, Any]] = []
    text_index: dict[str, int] = {}
    for chunk, vector in zip(result.chunk_records, result.vector_records):
        doc_id = chunk["doc_id"]
        retrieval_type = _RETRIEVAL_TYPE.get(chunk["chunk_type"], "text_chunk")
        record_id = chunk["chunk_id"]
        retrieval_records.append(
            RetrievalRecord(
                record_id=record_id,
                retrieval_type=retrieval_type,
                source_object_id=doc_id,
                payload=_payload(retrieval_type, chunk, text_index, config_hash),
                metadata={
                    "component_type": (
                        (chunk.get("metadata") or {}).get("component_type")
                        or chunk["chunk_type"]
                    ),
                    "field_path": chunk["field_path"],
                },
            )
        )
        vector_records.append(
            {
                "record_id": record_id,
                "source_object_id": doc_id,
                "retrieval_type": retrieval_type,
                "embedding": vector["vector"],
                "embedding_model": vector["model"],
                "embedding_dimension": vector["dim"],
                "embedding_status": "embedded",
            }
        )

    counts = Counter(record.retrieval_type for record in retrieval_records)
    retrieved_doc_ids = {record.source_object_id for record in retrieval_records}
    stage_errors = list(result.skipped_docs)
    failed_doc_ids = {str(error.get("doc_id") or "") for error in stage_errors}
    for record in enriched_data:
        normalized = _record_identity(record)
        doc_id = normalized[0]
        if doc_id and doc_id not in retrieved_doc_ids and doc_id not in failed_doc_ids:
            stage_errors.append(
                {
                    "doc_id": doc_id,
                    "source_uri": normalized[1],
                    "reason": "No retrieval records were produced from the enriched extraction.",
                }
            )
    stage_failed = bool(stage_errors)
    embedding_report = {
        "contract_version": "embedding-report-v1",
        "status": "failed" if stage_failed else "passed",
        "provider": "chunking_embedding",
        "model": embedder.model,
        "dimension": embedder.dim,
        "eligible_count": len(retrieval_records),
        "generated_count": len(vector_records),
        "skipped_count": len(stage_errors),
        "warnings": [],
        "errors": stage_errors,
    }
    quality_report = {
        "contract_version": "chunk-embed-quality-v1",
        "status": "failed" if stage_failed else "passed",
        "document_count": len(enriched_data),
        "embedded_document_count": len(retrieved_doc_ids),
        "skipped_document_count": len(stage_errors),
        "record_count": len(retrieval_records),
        "counts_by_retrieval_type": dict(counts),
        "errors": stage_errors,
    }
    return ChunkingEmbeddingOutput(
        retrieval_records=retrieval_records,
        quality_report=quality_report,
        vector_records=vector_records,
        embedding_report=embedding_report,
        skipped_docs=stage_errors,
    )


def _record_identity(record: dict[str, Any]) -> tuple[str, Any]:
    metadata = record.get("metadata") or {}
    return (
        str(record.get("source_object_id") or ""),
        record.get("source_uri") or metadata.get("source_uri"),
    )


def _extraction_of(record: dict[str, Any]) -> dict[str, Any]:
    rows = record.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("enriched record has no rows")
    extraction = rows[0].get("extraction") if isinstance(rows[0], dict) else None
    if not isinstance(extraction, dict):
        raise ValueError("enriched record has no rows[0].extraction")
    return extraction


def _payload(
    retrieval_type: str,
    chunk: dict[str, Any],
    text_index: dict[str, int],
    config_hash: str = "",
) -> dict[str, Any]:
    doc_id = chunk["doc_id"]
    content = chunk["content"]
    caption = (chunk.get("metadata") or {}).get("caption")
    if retrieval_type == "text_chunk":
        position = text_index.get(doc_id, 0)
        text_index[doc_id] = position + 1
        return {
            "document_id": doc_id,
            "chunk_id": chunk["chunk_id"],
            "chunk_index": position,
            "text": content,
            "start_char": chunk["char_start"],
            "end_char": chunk["char_end"],
            "config_hash": config_hash,
            **_lexical(content),
        }
    field_index = _field_index(chunk["field_path"])
    body = {"content": content, "caption": caption}
    if retrieval_type == "table":
        return {
            "document_id": doc_id,
            "table_id": chunk["chunk_id"],
            "table_index": field_index,
            "table": body,
            "config_hash": config_hash,
            **_lexical(content_text(body)),
        }
    return {
        "document_id": doc_id,
        "image_id": chunk["chunk_id"],
        "image_index": field_index,
        "image": body,
        "config_hash": config_hash,
        **_lexical(content_text(body)),
    }


def _lexical(text: str) -> dict[str, Any]:
    payload = build_lexical_payload(text)
    return {"lexical": payload} if payload else {}


def _field_index(field_path: str) -> int:
    match = _FIELD_INDEX.search(field_path or "")
    return int(match.group(1)) if match else 0
