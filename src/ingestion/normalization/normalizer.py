"""Parser-agnostic initial document schema normalization for ingestion."""

from __future__ import annotations

from typing import Any

from ...models import InitialSchema, ParsedData, make_id

INITIAL_SCHEMA_CONTRACT_VERSION = "initial-document-schema-v1"

DOCUMENT_FIELDS = {
    "document.document_id": "string",
    "document.source_uri": "string",
    "document.source_format": "string",
    "document.document_type": "string|null",
    "document.language": "string|null",
    "document.title": "string|null",
    "document.main_text": "string",
}

TABLE_FIELDS = {
    "table.document_id": "string",
    "table.table_id": "string",
    "table.table_index": "integer",
    "table.caption": "string",
    "table.content": "string",
}

FIGURE_FIELDS = {
    "figure.document_id": "string",
    "figure.figure_id": "string",
    "figure.figure_index": "integer",
    "figure.caption": "string",
    "figure.description": "string",
}

FORMULA_FIELDS = {
    "formula.document_id": "string",
    "formula.formula_id": "string",
    "formula.formula_index": "integer",
    "formula.text": "string",
}


def build_initial_schema(parsed: ParsedData) -> InitialSchema:
    """Build a normalized initial document schema from raw parsed data.

    Raw parser outputs can include provider-specific fields such as Lift
    citations and verification metadata. This formatter keeps the normalized
    document entities that downstream stages can rely on.
    """
    normalized = normalize_parsed_document(parsed)
    component_counts = {
        "rows": len(parsed.rows),
        "tables": len(normalized["tables"]),
        "figures": len(normalized["figures"]),
        "formulas": len(normalized["formulas"]),
    }

    return InitialSchema(
        schema_id=make_id(parsed.object_id, "initial-schema"),
        source_object_id=parsed.object_id,
        fields=_normalized_fields(),
        entities=_normalized_entities(component_counts),
        relationships=_normalized_relationships(component_counts),
        metadata={
            "contract_version": INITIAL_SCHEMA_CONTRACT_VERSION,
            "schema_type": "document",
            "schema_source": "format_parsing",
            "parser": parsed.metadata.get("parser"),
            "source_format": parsed.source_format,
            "source_uri": parsed.source_uri,
            "row_count": len(parsed.rows),
            "component_counts": component_counts,
            "observed_document_fields": _observed_document_fields(normalized),
            "raw_parser_output": "parsed_data",
        },
    )


def normalize_parsed_document(parsed: ParsedData) -> dict[str, Any]:
    """Normalize raw parser rows into document components."""
    document: dict[str, Any] = {
        "document_id": parsed.object_id,
        "source_uri": parsed.source_uri,
        "source_format": parsed.source_format,
        "document_type": None,
        "language": None,
        "title": None,
        "main_text": "",
        "tables": [],
        "figures": [],
        "formulas": [],
    }
    text_parts: list[str] = []

    for row in parsed.rows:
        extraction = row.get("extraction") if isinstance(row.get("extraction"), dict) else {}

        document["document_type"] = document["document_type"] or _text_or_none(extraction.get("document_type"))
        document["language"] = document["language"] or _text_or_none(extraction.get("language"))
        document["title"] = document["title"] or _text_or_none(extraction.get("title"))
        document["tables"].extend(_tables(extraction.get("tables")))
        document["figures"].extend(_figures(extraction.get("figures")))
        document["formulas"].extend(_formulas(extraction.get("formulas")))

        text = _first_text(
            extraction.get("main_text"),
            extraction.get("text"),
            extraction.get("markdown"),
            extraction.get("content"),
            row.get("text"),
            parsed.text,
        )
        if text:
            text_parts.append(text)

    document["main_text"] = "\n\n".join(text_parts)
    return document


def _normalized_fields() -> dict[str, str]:
    return {
        **DOCUMENT_FIELDS,
        **TABLE_FIELDS,
        **FIGURE_FIELDS,
        **FORMULA_FIELDS,
    }


def _normalized_entities(component_counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {
            "entity_type": "document",
            "primary_key": "document_id",
            "count": 1,
            "fields": DOCUMENT_FIELDS,
        },
        {
            "entity_type": "table",
            "primary_key": "table_id",
            "count": component_counts["tables"],
            "fields": TABLE_FIELDS,
        },
        {
            "entity_type": "figure",
            "primary_key": "figure_id",
            "count": component_counts["figures"],
            "fields": FIGURE_FIELDS,
        },
        {
            "entity_type": "formula",
            "primary_key": "formula_id",
            "count": component_counts["formulas"],
            "fields": FORMULA_FIELDS,
        },
    ]


def _normalized_relationships(component_counts: dict[str, int]) -> list[dict[str, str]]:
    relationships = []
    for entity_type, relationship_type in (
        ("table", "has_table"),
        ("figure", "has_figure"),
        ("formula", "has_formula"),
    ):
        if component_counts[f"{entity_type}s"] > 0:
            relationships.append(
                {
                    "source_entity": "document",
                    "target_entity": entity_type,
                    "relationship_type": relationship_type,
                }
            )
    return relationships


def _observed_document_fields(document: dict[str, Any]) -> list[str]:
    observed = []
    for field in ("document_type", "language", "title", "main_text"):
        value = document.get(field)
        if value:
            observed.append(f"document.{field}")
    for field in ("tables", "figures", "formulas"):
        if document.get(field):
            observed.append(field)
    return observed


def _tables(value: Any) -> list[dict[str, str]]:
    tables = []
    for item in _array(value):
        if isinstance(item, dict):
            tables.append(
                {
                    "caption": _text_or_empty(item.get("caption")),
                    "content": _text_or_empty(item.get("content")),
                }
            )
        else:
            tables.append({"caption": "", "content": str(item)})
    return tables


def _figures(value: Any) -> list[dict[str, str]]:
    figures = []
    for item in _array(value):
        if isinstance(item, dict):
            figures.append(
                {
                    "caption": _text_or_empty(item.get("caption")),
                    "description": _text_or_empty(item.get("description")),
                }
            )
        else:
            figures.append({"caption": "", "description": str(item)})
    return figures


def _formulas(value: Any) -> list[str]:
    return [_text_or_empty(item) for item in _array(value)]


def _array(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _text_or_none(value)
        if text:
            return text
    return None


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_or_empty(value: Any) -> str:
    return _text_or_none(value) or ""
