"""Build canonical document payloads for downstream consumers."""

from __future__ import annotations

from typing import Any

from ..models import EnrichedData, IndexRecord, make_id
from .index_builder import NormalizedDocument, TextChunk, chunk_text, normalize_document

DOCUMENT_ARTIFACT_VERSION = "documents-v1"


def build_documents_artifact(
    enriched_data: list[EnrichedData],
    index_records: list[IndexRecord],
) -> list[dict[str, Any]]:
    """Create document-centric payloads separate from index/vector artifacts."""
    index_records_by_source: dict[str, list[IndexRecord]] = {}
    for record in index_records:
        index_records_by_source.setdefault(record.source_object_id, []).append(record)

    documents: list[dict[str, Any]] = []
    for data in enriched_data:
        document = normalize_document(data)
        chunks = chunk_text(document.main_text)
        records = index_records_by_source.get(data.source_object_id, [])
        documents.append(_document_payload(document, chunks, records))
    return documents


def _document_payload(
    document: NormalizedDocument,
    chunks: list[TextChunk],
    index_records: list[IndexRecord],
) -> dict[str, Any]:
    return {
        "contract_version": DOCUMENT_ARTIFACT_VERSION,
        "document_id": document.source_object_id,
        "source_uri": document.source_uri,
        "document_type": document.document_type,
        "language": document.language,
        "title": document.title,
        "main_text": document.main_text,
        "text_length": len(document.main_text),
        "elements": {
            "tables": _tables(document),
            "figures": _figures(document),
            "formulas": _formulas(document),
        },
        "text_chunks": _text_chunks(document, chunks),
        "component_counts": {
            "rows": document.row_count,
            "text_chunks": len(chunks),
            "tables": len(document.tables),
            "figures": len(document.figures),
            "formulas": len(document.formulas),
        },
        "index_record_ids": _index_record_ids(index_records),
    }


def _tables(document: NormalizedDocument) -> list[dict[str, Any]]:
    tables = []
    for index, table in enumerate(document.tables):
        tables.append(
            {
                "table_id": make_id(document.source_object_id, "table", index),
                "table_index": index,
                "caption": _text(table.get("caption")),
                "content": _text(table.get("content")),
                "raw": table,
            }
        )
    return tables


def _figures(document: NormalizedDocument) -> list[dict[str, Any]]:
    figures = []
    for index, figure in enumerate(document.figures):
        figures.append(
            {
                "figure_id": make_id(document.source_object_id, "figure", index),
                "figure_index": index,
                "caption": _text(figure.get("caption")),
                "description": _text(figure.get("description")),
                "raw": figure,
            }
        )
    return figures


def _formulas(document: NormalizedDocument) -> list[dict[str, Any]]:
    formulas = []
    for index, formula in enumerate(document.formulas):
        formulas.append(
            {
                "formula_id": make_id(document.source_object_id, "formula", index),
                "formula_index": index,
                "text": _text(formula),
            }
        )
    return formulas


def _text_chunks(document: NormalizedDocument, chunks: list[TextChunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": make_id(document.source_object_id, "text-chunk", index),
            "chunk_index": index,
            "text": chunk.text,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }
        for index, chunk in enumerate(chunks)
    ]


def _index_record_ids(index_records: list[IndexRecord]) -> dict[str, list[str]]:
    record_ids: dict[str, list[str]] = {}
    for record in index_records:
        record_ids.setdefault(record.index_type, []).append(record.record_id)
    return record_ids


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
