"""Local database placeholder for future storage development."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DatabaseMock:
    """No-op database adapter used as a future integration placeholder."""

    name: str = "database-mock"

    def upsert_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return {"mode": self.name, "upserted": len(records)}
