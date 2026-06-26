"""Local database mock.

TODO: Replace with a real relational database adapter.
"""

from __future__ import annotations

from typing import Any, Iterable


class DatabaseMock:
    def upsert_records(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        records = list(records)
        return {"mode": "database-mock", "upserted": len(records)}
