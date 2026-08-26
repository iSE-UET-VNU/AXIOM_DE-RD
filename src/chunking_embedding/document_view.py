"""Consumer-facing document view built from enriched parser rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import EnrichedData


@dataclass
class DocumentView:
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


def document_from_enriched_data(data: EnrichedData) -> DocumentView:
    """Build a stable semantic document view directly from enriched rows."""
    document = DocumentView(
        source_object_id=data.source_object_id,
        source_uri=_metadata_value(data.metadata, "source_uri"),
        row_count=len(data.rows),
    )
    text_parts: list[str] = []

    for row in data.rows:
        extraction = row.get("extraction")
        extraction = extraction if isinstance(extraction, dict) else {}

        document.document_type = document.document_type or _optional_text(
            extraction.get("document_type")
        )
        document.language = document.language or _optional_text(
            extraction.get("language")
        )
        document.title = document.title or _optional_text(extraction.get("title"))
        document.tables.extend(_component_list(extraction.get("tables")))
        document.figures.extend(_component_list(extraction.get("figures")))
        document.formulas.extend(_array_value(extraction.get("formulas")))

        text = _row_text(row, extraction)
        if text:
            text_parts.append(text)

    document.main_text = "\n\n".join(text_parts)
    return document


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
    records: list[dict[str, Any]] = []
    for item in _array_value(value):
        if isinstance(item, dict):
            records.append(dict(item))
        else:
            records.append({"content": str(item)})
    return records


def _array_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _metadata_value(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    source_metadata = metadata.get("source_metadata")
    if value is None and isinstance(source_metadata, dict):
        value = source_metadata.get(key)
    return _optional_text(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
