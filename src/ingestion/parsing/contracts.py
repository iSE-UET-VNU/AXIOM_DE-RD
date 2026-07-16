"""Protocols shared by parser routing and document providers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ...models import DataObject, ParsedData, ParseResult


@runtime_checkable
class ParserBackend(Protocol):
    """Parse contract implemented by every routing target."""

    backend_name: str
    supported_extensions: frozenset[str]

    def parse(self, path: str | Path, data_object: DataObject) -> ParseResult:
        """Return the outcome of parsing one source object."""


@runtime_checkable
class DocumentProvider(Protocol):
    """Provider contract used by document parsing implementations."""

    provider_name: str
    supported_extensions: frozenset[str]

    def parse_file(self, path: str | Path, data_object: DataObject) -> ParsedData:
        """Extract provider-neutral data from one document."""
