"""Validation rules for parsed ingestion results."""

from __future__ import annotations

from typing import Any

from ..models import ParsedData


def validate_parsed_document(parsed: ParsedData) -> list[dict[str, Any]]:
    """Return quarantine reasons for an unusable parsed document."""
    reasons: list[dict[str, Any]] = []
    page_count = parsed.metadata.get("page_count")
    parser = str(parsed.metadata.get("parser") or "")

    if _is_zero(page_count):
        reasons.append(
            {
                "code": "zero_page_count",
                "field": "parsed.metadata.page_count",
                "message": "Parser returned page_count=0.",
            }
        )

    if parser == "lift-api" and not _has_extraction(parsed.rows):
        reasons.append(
            {
                "code": "null_extraction",
                "field": "parsed.rows[].extraction",
                "message": "Lift returned no extraction object.",
            }
        )

    if not _has_meaningful_content(parsed):
        reasons.append(
            {
                "code": "empty_parsed_content",
                "field": "parsed",
                "message": "Parsed document has no text, structured row, or component.",
            }
        )

    return reasons


def _is_zero(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return value == 0
    return str(value).strip() == "0"


def _has_extraction(rows: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(row, dict) and row.get("extraction") is not None
        for row in rows
    )


def _has_meaningful_content(parsed: ParsedData) -> bool:
    if _has_value(parsed.text):
        return True

    for row in parsed.rows:
        if not isinstance(row, dict):
            if _has_value(row):
                return True
            continue

        if _has_value(row.get("text")):
            return True

        extraction = row.get("extraction")
        if isinstance(extraction, dict):
            for field in ("main_text", "text", "markdown", "content"):
                if _has_value(extraction.get(field)):
                    return True
            for field in ("tables", "figures", "formulas"):
                if _has_value(extraction.get(field)):
                    return True

        for field, value in row.items():
            if field not in {"text", "extraction"} and _has_value(value):
                return True

    return False


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True
