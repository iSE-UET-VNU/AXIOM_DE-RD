"""Placeholder missing-value imputation."""

from __future__ import annotations

from typing import Any


def impute_missing_values(rows: list[dict[str, Any]], fill_value: Any = None) -> list[dict[str, Any]]:
    """Make row shapes consistent by filling absent fields.

    TODO: Replace the constant fill strategy with field-aware imputation.
    """
    if not rows:
        return rows

    fields = sorted({field for row in rows for field in row})
    return [{field: row.get(field, fill_value) for field in fields} for row in rows]
