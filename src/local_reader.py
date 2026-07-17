"""Discover local raw files for the pipeline input boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils.paths import portable_path

DEFAULT_INCLUDE_EXTENSIONS = [
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".xlsx",
    ".parquet",
]


@dataclass(frozen=True)
class LocalInputConfig:
    """Configuration for discovering files under a local raw path."""

    path: str = "data/raw"
    recursive: bool = True
    include_hidden: bool = False
    include_extensions: tuple[str, ...] = tuple(DEFAULT_INCLUDE_EXTENSIONS)
    max_files: int | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "LocalInputConfig":
        value = value or {}
        extensions = value.get("include_extensions", DEFAULT_INCLUDE_EXTENSIONS)
        if not isinstance(extensions, list):
            raise ValueError("local_input.include_extensions must be a list.")
        max_files_value = value.get("max_files")
        max_files = int(max_files_value) if max_files_value is not None else None
        if max_files is not None and max_files <= 0:
            raise ValueError("local_input.max_files must be greater than zero.")
        return cls(
            path=str(value.get("path", "data/raw")),
            recursive=bool(value.get("recursive", True)),
            include_hidden=bool(value.get("include_hidden", False)),
            include_extensions=tuple(_normalize_extension(item) for item in extensions),
            max_files=max_files,
        )


@dataclass(frozen=True)
class LocalInput:
    """One local file plus stable provenance metadata."""

    path: Path
    source_uri: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LocalInputBatch:
    """A local input root and its deterministically ordered files."""

    source: str
    inputs: list[LocalInput]


def read_local_inputs(
    config_value: dict[str, Any] | None,
    *,
    local_raw: str | Path | None = None,
    project_root: str | Path | None = None,
) -> LocalInputBatch:
    """Resolve a local file or directory and return eligible input files."""
    config = LocalInputConfig.from_mapping(config_value)
    root = _resolve_local_path(local_raw or config.path, project_root)
    if not root.exists():
        raise FileNotFoundError(f"Local raw input does not exist: {root}")

    if root.is_file():
        paths = [root]
        relative_root = root.parent
    elif root.is_dir():
        candidates = root.rglob("*") if config.recursive else root.glob("*")
        paths = sorted(
            (
                path
                for path in candidates
                if path.is_file()
                and (config.include_hidden or not _is_hidden(path, root))
                and _matches_extension(path, config.include_extensions)
            ),
            key=lambda path: path.relative_to(root).as_posix().casefold(),
        )
        relative_root = root
    else:
        raise ValueError(f"Local raw input must be a file or directory: {root}")

    if config.max_files is not None:
        paths = paths[: config.max_files]
    if not paths:
        raise RuntimeError(f"No eligible local raw files found under: {root}")

    inputs = [
        LocalInput(
            path=path,
            source_uri=portable_path(path, project_root),
            metadata={
                "input_provider": "local_raw",
                "file_name": path.relative_to(relative_root).as_posix(),
                "size_bytes": path.stat().st_size,
            },
        )
        for path in paths
    ]
    return LocalInputBatch(
        source=portable_path(root, project_root),
        inputs=inputs,
    )


def _resolve_local_path(value: str | Path, project_root: str | Path | None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if project_root:
        return Path(project_root) / path
    return path


def _normalize_extension(value: Any) -> str:
    extension = str(value).strip().lower()
    if not extension:
        raise ValueError("local_input.include_extensions cannot contain empty values.")
    return extension if extension.startswith(".") else f".{extension}"


def _matches_extension(path: Path, extensions: tuple[str, ...]) -> bool:
    return not extensions or path.suffix.lower() in extensions


def _is_hidden(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)
