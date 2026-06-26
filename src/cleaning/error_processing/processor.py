"""Placeholder error processing."""

from __future__ import annotations

from typing import Any


def process_errors(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return rows plus lightweight issue records.

    TODO: Add validation rules, quarantine behavior, and severity levels.
    """
    issues: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not row:
            issues.append({"row_index": index, "code": "empty_row", "message": "Row has no fields."})
    return rows, issues
