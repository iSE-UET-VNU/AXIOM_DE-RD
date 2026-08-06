"""Chunk identity across the storage boundary.


Mirroring another service's private function is fragile by nature: if they change
the derivation, fusion silently degrades to a union again. The durable fix is for
the stored id to be ours, or to be echoed back at index time
"""

from __future__ import annotations

from hashlib import sha256

EMBEDDING_ID_LENGTH = 32


def short_hash(*parts: object, length: int) -> str:
    raw = ":".join(str(part) for part in parts)
    return sha256(raw.encode("utf-8")).hexdigest()[:length]


def embedding_id(run_id: str, item_id: str, item_type: str) -> str:
    """The id corpus-service stores for one of our chunks."""
    return short_hash(run_id, item_id, item_type, length=EMBEDDING_ID_LENGTH)
