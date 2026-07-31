"""Internal row helpers shared by text and table backends."""

from __future__ import annotations

from typing import Any, Iterable


def normalize_headers(
    values: Iterable[Any],
    *,
    width: int | None = None,
    reserved_names: Iterable[str] = (),
) -> list[str]:
    """Trim, fill, and de-duplicate headers deterministically."""
    raw = ["" if value is None else str(value).strip() for value in values]
    target_width = max(len(raw), width or 0)
    raw.extend([""] * (target_width - len(raw)))

    headers: list[str] = []
    used = {str(name) for name in reserved_names}
    for index, value in enumerate(raw, start=1):
        base = value or f"column_{index}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        headers.append(candidate)
        used.add(candidate)
    return headers


def row_is_empty(row: Iterable[Any]) -> bool:
    return not any(str(value).strip() for value in row if value is not None)
