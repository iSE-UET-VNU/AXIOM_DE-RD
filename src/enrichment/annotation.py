"""Placeholder annotation logic."""

from __future__ import annotations

from typing import Any


def annotate_data(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Create lightweight annotations for downstream cataloging.

    TODO: Add domain annotation, classification, and NLP enrichment.
    """
    fields = sorted({field for row in rows for field in row})
    return {
        "field_count": len(fields),
        "fields": fields,
        "has_text": "text" in fields,
    }
