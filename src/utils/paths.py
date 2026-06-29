"""Path helpers for portable artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def portable_path(value: str | Path, project_root: str | Path | None = None) -> str:
    """Return a stable path string for artifacts.

    Paths inside the project are serialized relative to the project root with
    POSIX separators. External paths stay absolute.
    """
    path = Path(value)
    root = Path(project_root).resolve() if project_root else None

    if root:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            pass

    return path.as_posix()


def portable_path_value(value: Any, project_root: str | Path | None = None) -> Any:
    """Convert path-like strings under project root into portable strings."""
    if isinstance(value, Path):
        return portable_path(value, project_root)
    if isinstance(value, str):
        root = Path(project_root).resolve() if project_root else None
        if root and _looks_like_path(value):
            path = Path(value)
            if path.is_absolute():
                return portable_path(path, root)
        return value
    return value


def _looks_like_path(value: str) -> bool:
    return (
        "\\" in value
        or "/" in value
        or value.startswith(".")
        or (len(value) >= 3 and value[1:3] == ":\\")
        or (len(value) >= 3 and value[1:3] == ":/")
    )
