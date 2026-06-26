"""Simple field-name standardization."""

from __future__ import annotations

from typing import Any
import re


def standardize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize field names while preserving values."""
    return [{_normalize_field_name(key): value for key, value in row.items()} for row in rows]


def _normalize_field_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "field"
