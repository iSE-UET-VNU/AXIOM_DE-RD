"""Local graph database mock.

TODO: Replace with a graph database adapter.
"""

from __future__ import annotations

from typing import Any, Iterable


class GraphDBMock:
    def upsert_relationships(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        records = list(records)
        return {"mode": "graph-db-mock", "upserted": len(records)}
