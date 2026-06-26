"""Local vector database mock.

TODO: Replace with an embedding model and vector database adapter.
"""

from __future__ import annotations

from typing import Any, Iterable


class VectorDBMock:
    def upsert_vectors(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        records = list(records)
        return {"mode": "vector-db-mock", "upserted": len(records)}
