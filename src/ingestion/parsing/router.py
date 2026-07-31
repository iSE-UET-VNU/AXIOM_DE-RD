"""Extension-based parser backend resolution."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .contracts import ParserBackend


class ParserRouter:
    """Resolve a source extension to exactly one parser backend."""

    def __init__(self, backends: Iterable[ParserBackend]) -> None:
        self.backends = tuple(backends)
        self._backend_by_extension: dict[str, ParserBackend] = {}
        for backend in self.backends:
            for raw_extension in backend.supported_extensions:
                extension = _normalize_extension(raw_extension)
                existing = self._backend_by_extension.get(extension)
                if existing is not None:
                    raise ValueError(
                        f"Duplicate parser extension {extension!r}: "
                        f"{existing.backend_name!r} and {backend.backend_name!r}"
                    )
                self._backend_by_extension[extension] = backend

    def resolve(self, path: str | Path) -> ParserBackend | None:
        """Return the backend registered for the path extension."""
        return self._backend_by_extension.get(Path(path).suffix.lower())


def _normalize_extension(value: str) -> str:
    extension = str(value).strip().lower()
    if not extension:
        raise ValueError("Parser extensions must not be empty.")
    return extension if extension.startswith(".") else f".{extension}"
