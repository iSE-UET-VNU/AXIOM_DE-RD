"""Low-level JSON and JSONL artifact writers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from ..utils.paths import portable_path, portable_path_value


class LocalArtifactWriter:
    """Write portable pipeline artifacts beneath one local directory."""

    def __init__(self, root: str | Path, project_root: str | Path | None = None) -> None:
        self.root = Path(root)
        self.project_root = Path(project_root).resolve() if project_root else None
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(
        self,
        name: str,
        payload: Any,
        *,
        sort_keys: bool = True,
    ) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            _to_jsonable(payload, self.project_root),
            indent=2,
            sort_keys=sort_keys,
            ensure_ascii=False,
        )
        _atomic_write_text(path, text)
        return path

    def write_jsonl(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = _to_jsonable(payload, self.project_root)
        if not isinstance(rows, list):
            raise TypeError(f"JSONL payload must be a list, got: {type(rows).__name__}")
        lines = [
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            for row in rows
        ]
        _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
        return path


def artifact_path(path: str | Path, project_root: str | Path | None = None) -> str:
    """Return a portable path for an artifact written by this project."""
    return portable_path(path, project_root)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _to_jsonable(value: Any, project_root: str | Path | None = None) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value), project_root)
    if isinstance(value, Path):
        return portable_path(value, project_root)
    converted = portable_path_value(value, project_root)
    if converted is not value:
        return converted
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item, project_root) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item, project_root) for item in value]
    return value
