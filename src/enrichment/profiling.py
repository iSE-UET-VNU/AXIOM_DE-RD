"""Minimal profiling helpers."""

from __future__ import annotations

from typing import Any


def profile_data(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute simple row and field statistics."""
    fields = sorted({field for row in rows for field in row})
    field_profiles = {}
    for field in fields:
        values = [row.get(field) for row in rows]
        missing = sum(value is None or value == "" for value in values)
        field_profiles[field] = {
            "non_null_count": len(values) - missing,
            "missing_count": missing,
        }

    return {
        "row_count": len(rows),
        "field_count": len(fields),
        "fields": field_profiles,
    }
