"""Pipeline stage: run chunking + embedding and adapt output to the legacy shapes.

lets ``run_pipeline`` use the
chunking_embedding module in place of ``indexing_cataloging`` without changing the
artifacts/output stage: retrieval records and vectors now come from our chunking.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any
import re

from ..indexing_cataloging.lexical import build_lexical_payload, content_text
from ..models import IndexRecord, MetadataRecord
from .contracts import ChunkEmbedConfig
from .embedders import create_embedder
from .runner import chunk_and_embed

_INDEX_TYPE = {"text_chunk": "text_chunk", "table_chunk": "table", "figure_chunk": "image"}
_FIELD_INDEX = re.compile(r"\[(\d+)\]")


@dataclass
class ChunkingCatalogOutput:
    metadata_records: list[MetadataRecord] = field(default_factory=list)
    index_records: list[IndexRecord] = field(default_factory=list)
    index_quality_report: dict[str, Any] = field(default_factory=dict)
    vector_records: list[dict[str, Any]] = field(default_factory=list)
    embedding_report: dict[str, Any] = field(default_factory=dict)


def run(parsed_records: list[dict[str, Any]], config_section: Any) -> ChunkingCatalogOutput:
    config = ChunkEmbedConfig.from_mapping(config_section)
    embedder = create_embedder(config.embedder, config.embedder_params)
    result = chunk_and_embed(parsed_records, config, embedder=embedder)
    config_hash = config.config_hash()

    index_records: list[IndexRecord] = []
    vector_records: list[dict[str, Any]] = []
    text_index: dict[str, int] = {}
    for chunk, vector in zip(result.chunk_records, result.vector_records):
        doc_id = chunk["doc_id"]
        index_type = _INDEX_TYPE.get(chunk["chunk_type"], "text_chunk")
        record_id = chunk["chunk_id"]
        index_records.append(
            IndexRecord(
                record_id=record_id,
                index_type=index_type,
                source_object_id=doc_id,
                payload=_payload(index_type, chunk, text_index, config_hash),
                metadata={"component_type": chunk["chunk_type"], "field_path": chunk["field_path"]},
            )
        )
        vector_records.append(
            {
                "record_id": record_id,
                "source_object_id": doc_id,
                "index_type": index_type,
                "embedding": vector["vector"],
                "embedding_model": vector["model"],
                "embedding_dimension": vector["dim"],
                "embedding_status": "embedded",
            }
        )

    counts = Counter(record.index_type for record in index_records)
    embedding_report = {
        "status": "failed" if result.skipped_docs else "passed",
        "provider": "chunking_embedding",
        "model": embedder.model,
        "dimension": embedder.dim,
        "generated_count": len(vector_records),
        "skipped_count": len(result.skipped_docs),
    }
    index_quality_report = {
        "status": "passed",
        "record_count": len(index_records),
        "counts_by_index_type": dict(counts),
    }
    return ChunkingCatalogOutput(
        index_records=index_records,
        index_quality_report=index_quality_report,
        vector_records=vector_records,
        embedding_report=embedding_report,
    )


def _payload(
    index_type: str,
    chunk: dict[str, Any],
    text_index: dict[str, int],
    config_hash: str = "",
) -> dict[str, Any]:
    doc_id = chunk["doc_id"]
    content = chunk["content"]
    caption = (chunk.get("metadata") or {}).get("caption")
    if index_type == "text_chunk":
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
            # corpus-service scores BM25 from these, server side; without them
            # its keyword-search silently finds nothing for our documents.
            **_lexical(content),
        }
    field_index = _field_index(chunk["field_path"])
    body = {"content": content, "caption": caption}
    if index_type == "table":
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
