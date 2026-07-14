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
    normalized_texts: list[dict[str, Any]] | None = None,
    normalized_images: list[dict[str, Any]] | None = None,
    normalized_tables: list[dict[str, Any]] | None = None,
    normalized_documents: list[dict[str, Any]] | None = None,
) -> list[IndexRecord]:
    """Build document, chunk, component, and catalog index records."""
    metadata_by_object = {record.source_object_id: record for record in metadata_records}
    normalized_by_object = _normalized_by_object(
        normalized_texts or [],
        normalized_images or [],
        normalized_tables or [],
        normalized_documents or [],
    )
    records: list[IndexRecord] = []

    for data in enriched_data:
        metadata = metadata_by_object.get(data.source_object_id)
        normalized = normalized_by_object.get(data.source_object_id)
        if normalized and _has_normalized_components(normalized):
            document = _document_from_normalized(data, normalized)
            text_records = build_normalized_text_chunk_index_records(data.source_object_id, normalized["texts"])
            records.append(build_normalized_document_index_record(document, text_records, normalized))
            records.extend(text_records)
            records.extend(build_normalized_table_index_records(data.source_object_id, normalized["tables"]))
            records.extend(build_normalized_image_index_records(data.source_object_id, normalized["images"]))
            records.append(build_catalog_index_record(document, metadata))
            continue

        document = normalize_document(data)
        chunks = chunk_text(document.main_text)
        records.append(build_document_index_record(document, chunks))
        records.extend(build_text_chunk_index_records(document, chunks))
        records.extend(build_table_index_records(document))
        records.extend(build_image_index_records(document))
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


def _normalized_by_object(
    texts: list[dict[str, Any]],
    images: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for key, items in (("texts", texts), ("images", images), ("tables", tables), ("documents", documents)):
        for item in items:
            document_id = _optional_text(item.get("document_id")) if isinstance(item, dict) else None
            if not document_id:
                continue
            grouped.setdefault(document_id, {"texts": [], "images": [], "tables": [], "documents": []})[key].append(item)
    return grouped


def _has_normalized_components(normalized: dict[str, list[dict[str, Any]]]) -> bool:
    return bool(normalized.get("texts") or normalized.get("images") or normalized.get("tables"))


def _document_from_normalized(data: EnrichedData, normalized: dict[str, list[dict[str, Any]]]) -> NormalizedDocument:
    document_record = normalized.get("documents", [{}])[0] if normalized.get("documents") else {}
    texts = normalized.get("texts", [])
    tables = normalized.get("tables", [])
    images = normalized.get("images", [])
    main_text = "\n\n".join(_optional_text(item.get("text")) or "" for item in texts).strip()
    return NormalizedDocument(
        source_object_id=data.source_object_id,
        source_uri=_optional_text(document_record.get("source_uri")) or _metadata_value(data.metadata, "source_uri"),
        document_type=_optional_text(document_record.get("document_type")),
        language=_optional_text(document_record.get("language")),
        title=_optional_text(document_record.get("title")),
        main_text=main_text,
        tables=[dict(item) for item in tables],
        figures=[dict(item) for item in images],
        formulas=[],
        row_count=len(data.rows),
    )


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


def build_normalized_document_index_record(
    document: NormalizedDocument,
    text_records: list[IndexRecord],
    normalized: dict[str, list[dict[str, Any]]],
) -> IndexRecord:
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
                "text_chunks": len(text_records),
                "tables": len(normalized.get("tables", [])),
                "images": len(normalized.get("images", [])),
                "formulas": 0,
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


def build_normalized_text_chunk_index_records(
    source_object_id: str,
    texts: list[dict[str, Any]],
) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    chunk_index = 0
    for text_record in texts:
        text = _optional_text(text_record.get("text")) or _optional_text(text_record.get("embedding_text"))
        if not text:
            continue
        for local_index, chunk in enumerate(chunk_text(text)):
            source_text_id = _optional_text(text_record.get("text_id")) or _optional_text(text_record.get("source_block_id")) or str(chunk_index)
            chunk_id = make_id(source_object_id, "text-chunk", source_text_id, local_index)
            records.append(
                IndexRecord(
                    record_id=make_id(source_object_id, "text-chunk-index", source_text_id, local_index),
                    index_type="text_chunk",
                    source_object_id=source_object_id,
                    payload={
                        "document_id": source_object_id,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "source_text_id": source_text_id,
                        "source_block_id": text_record.get("source_block_id"),
                        "page": text_record.get("page"),
                        "role": text_record.get("role"),
                        "section_path": text_record.get("section_path", []),
                        "text": chunk.text,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        **_embedding_payload(chunk.text),
                    },
                    metadata=_record_metadata("text_chunk"),
                )
            )
            chunk_index += 1
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


def build_normalized_table_index_records(
    source_object_id: str,
    tables: list[dict[str, Any]],
) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    for index, table in enumerate(tables):
        embedding_text = _optional_text(table.get("embedding_text")) or _join_text_parts(
            table.get("caption"),
            table.get("markdown"),
        )
        table_id = _optional_text(table.get("table_id")) or make_id(source_object_id, "table", index)
        records.append(
            IndexRecord(
                record_id=make_id(source_object_id, "table-index", table_id),
                index_type="table",
                source_object_id=source_object_id,
                payload={
                    "document_id": source_object_id,
                    "table_id": table_id,
                    "table_index": index,
                    "source_block_id": table.get("source_block_id"),
                    "page": table.get("page"),
                    "caption": table.get("caption", ""),
                    "rows": table.get("rows", []),
                    "markdown": table.get("markdown", ""),
                    "table": table,
                    **_embedding_payload(embedding_text),
                },
                metadata=_record_metadata("table"),
            )
        )
    return records


def build_image_index_records(document: NormalizedDocument) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    for index, figure in enumerate(document.figures):
        embedding_text = _join_text_parts(figure.get("caption"), figure.get("description"))
        image_id = make_id(document.source_object_id, "image", index)
        records.append(
            IndexRecord(
                record_id=make_id(document.source_object_id, "image-index", index),
                index_type="image",
                source_object_id=document.source_object_id,
                payload={
                    "document_id": document.source_object_id,
                    "image_id": image_id,
                    "image_index": index,
                    "image": figure,
                    **_embedding_payload(embedding_text),
                },
                metadata=_record_metadata("image"),
            )
        )
    return records


def build_normalized_image_index_records(
    source_object_id: str,
    images: list[dict[str, Any]],
) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    for index, image in enumerate(images):
        embedding_text = _optional_text(image.get("embedding_text")) or _join_text_parts(
            image.get("visible_caption"),
            image.get("generated_description"),
        )
        image_id = _optional_text(image.get("image_id")) or make_id(source_object_id, "image", index)
        records.append(
            IndexRecord(
                record_id=make_id(source_object_id, "image-index", image_id),
                index_type="image",
                source_object_id=source_object_id,
                payload={
                    "document_id": source_object_id,
                    "image_id": image_id,
                    "image_index": index,
                    "source_block_id": image.get("source_block_id"),
                    "page": image.get("page"),
                    "image_path": image.get("image_path"),
                    "visible_caption": image.get("visible_caption", ""),
                    "generated_description": image.get("generated_description", ""),
                    "image": image,
                    **_embedding_payload(embedding_text),
                },
                metadata=_record_metadata("image"),
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
