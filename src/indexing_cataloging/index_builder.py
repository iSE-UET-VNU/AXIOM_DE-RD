"""Index record builders for normalized document components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import EnrichedData, IndexRecord, MetadataRecord, make_id

INDEX_CONTRACT_VERSION = "indexing-contract-v1"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150


@dataclass
class TextChunk:
    text: str
    start_char: int
    end_char: int


@dataclass
class NormalizedDocument:
    source_object_id: str
    source_uri: str | None = None
    document_type: str | None = None
    language: str | None = None
    title: str | None = None
    main_text: str = ""
    tables: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    formulas: list[Any] = field(default_factory=list)
    row_count: int = 0


def build_index_records(
    enriched_data: list[EnrichedData],
    metadata_records: list[MetadataRecord],
) -> list[IndexRecord]:
    """Build document, chunk, component, and catalog index records."""
    metadata_by_object = {record.source_object_id: record for record in metadata_records}
    records: list[IndexRecord] = []

    for data in enriched_data:
        document = normalize_document(data)
        chunks = chunk_text(document.main_text)
        metadata = metadata_by_object.get(data.source_object_id)

        records.append(build_document_index_record(document, chunks))
        records.extend(build_text_chunk_index_records(document, chunks))
        records.extend(build_table_index_records(document))
        records.extend(build_figure_index_records(document))
        records.append(build_catalog_index_record(document, metadata))

    return records


def normalize_document(data: EnrichedData) -> NormalizedDocument:
    """Normalize Lift-style extraction rows into a document component view."""
    document = NormalizedDocument(
        source_object_id=data.source_object_id,
        source_uri=_metadata_value(data.metadata, "source_uri"),
        row_count=len(data.rows),
    )
    text_parts: list[str] = []

    for row in data.rows:
        extraction = row.get("extraction") if isinstance(row.get("extraction"), dict) else {}

        document.document_type = document.document_type or _optional_text(extraction.get("document_type"))
        document.language = document.language or _optional_text(extraction.get("language"))
        document.title = document.title or _optional_text(extraction.get("title"))
        document.tables.extend(_component_list(extraction.get("tables")))
        document.figures.extend(_component_list(extraction.get("figures")))
        document.formulas.extend(_array_value(extraction.get("formulas")))

        text = _row_text(row, extraction)
        if text:
            text_parts.append(text)

    document.main_text = "\n\n".join(text_parts)
    return document


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Split text into overlapping character chunks."""
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[TextChunk] = []
    start = 0
    text_length = len(text)
    step = chunk_size - overlap

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(TextChunk(text=text[start:end], start_char=start, end_char=end))
        if end == text_length:
            break
        start += step

    return chunks


def build_document_index_record(document: NormalizedDocument, chunks: list[TextChunk]) -> IndexRecord:
    embedding_text = _join_text_parts(
        document.title,
        document.document_type,
        document.language,
        document.main_text,
    )
    return IndexRecord(
        record_id=make_id(document.source_object_id, "document-index"),
        index_type="document",
        source_object_id=document.source_object_id,
        payload={
            "document_id": document.source_object_id,
            "source_uri": document.source_uri,
            "title": document.title,
            "document_type": document.document_type,
            "language": document.language,
            "text_length": len(document.main_text),
            "component_counts": {
                "rows": document.row_count,
                "text_chunks": len(chunks),
                "tables": len(document.tables),
                "figures": len(document.figures),
                "formulas": len(document.formulas),
            },
            **_embedding_payload(embedding_text),
        },
        metadata=_record_metadata("document"),
    )


def build_text_chunk_index_records(
    document: NormalizedDocument,
    chunks: list[TextChunk],
) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    for index, chunk in enumerate(chunks):
        chunk_id = make_id(document.source_object_id, "text-chunk", index)
        records.append(
            IndexRecord(
                record_id=make_id(document.source_object_id, "text-chunk-index", index),
                index_type="text_chunk",
                source_object_id=document.source_object_id,
                payload={
                    "document_id": document.source_object_id,
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "text": chunk.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    **_embedding_payload(chunk.text),
                },
                metadata=_record_metadata("text_chunk"),
            )
        )
    return records


def build_table_index_records(document: NormalizedDocument) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    for index, table in enumerate(document.tables):
        embedding_text = _join_text_parts(table.get("caption"), table.get("content"))
        table_id = make_id(document.source_object_id, "table", index)
        records.append(
            IndexRecord(
                record_id=make_id(document.source_object_id, "table-index", index),
                index_type="table",
                source_object_id=document.source_object_id,
                payload={
                    "document_id": document.source_object_id,
                    "table_id": table_id,
                    "table_index": index,
                    "table": table,
                    **_embedding_payload(embedding_text),
                },
                metadata=_record_metadata("table"),
            )
        )
    return records


def build_figure_index_records(document: NormalizedDocument) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    for index, figure in enumerate(document.figures):
        embedding_text = _join_text_parts(figure.get("caption"), figure.get("description"))
        figure_id = make_id(document.source_object_id, "figure", index)
        records.append(
            IndexRecord(
                record_id=make_id(document.source_object_id, "figure-index", index),
                index_type="figure",
                source_object_id=document.source_object_id,
                payload={
                    "document_id": document.source_object_id,
                    "figure_id": figure_id,
                    "figure_index": index,
                    "figure": figure,
                    **_embedding_payload(embedding_text),
                },
                metadata=_record_metadata("figure"),
            )
        )
    return records


def build_catalog_index_record(
    document: NormalizedDocument,
    metadata_record: MetadataRecord | None,
) -> IndexRecord:
    title = document.title or (metadata_record.title if metadata_record else None)
    tags = metadata_record.tags if metadata_record else []
    search_text = _join_text_parts(title, document.document_type, document.language)
    return IndexRecord(
        record_id=make_id(document.source_object_id, "catalog-index"),
        index_type="catalog",
        source_object_id=document.source_object_id,
        payload={
            "document_id": document.source_object_id,
            "metadata_record_id": metadata_record.record_id if metadata_record else None,
            "schema_id": metadata_record.schema_id if metadata_record else None,
            "source_uri": document.source_uri,
            "title": title,
            "document_type": document.document_type,
            "language": document.language,
            "tags": tags,
            "search_text": search_text,
            **_embedding_payload(search_text),
        },
        metadata=_record_metadata("catalog"),
    )


def _row_text(row: dict[str, Any], extraction: dict[str, Any]) -> str | None:
    for value in (
        extraction.get("main_text"),
        extraction.get("text"),
        extraction.get("markdown"),
        extraction.get("content"),
        row.get("text"),
    ):
        text = _optional_text(value)
        if text:
            return text
    return None


def _component_list(value: Any) -> list[dict[str, Any]]:
    items = _array_value(value)
    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(dict(item))
        else:
            normalized.append({"content": str(item)})
    return normalized


def _array_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _metadata_value(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None and isinstance(metadata.get("source_metadata"), dict):
        value = metadata["source_metadata"].get(key)
    return _optional_text(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _join_text_parts(*parts: Any) -> str:
    return "\n\n".join(str(part) for part in parts if _optional_text(part))


def _embedding_payload(embedding_text: str) -> dict[str, Any]:
    return {
        "embedding_text": embedding_text,
        "embedding": None,
        "embedding_model": None,
        "embedding_status": "pending",
    }


def _record_metadata(component_type: str) -> dict[str, Any]:
    return {
        "contract_version": INDEX_CONTRACT_VERSION,
        "component_type": component_type,
        "embedding_status": "pending",
    }
