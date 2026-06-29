"""Vector database mock for tests and local smoke runs."""

from __future__ import annotations

from typing import Any, Iterable


class VectorDBMock:
    def __init__(self, collection_name: str = "mock_text_chunks") -> None:
        self.collection_name = collection_name

    def upsert_vectors(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        records = list(records)
        return {
            "provider": "mock",
            "collection_name": self.collection_name,
            "status": "passed",
            "upserted": len(records),
            "errors": [],
            "warnings": [],
        }
