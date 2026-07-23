"""Deterministic offline embedder — smoke runs and tests only."""

from __future__ import annotations

from typing import Any
import hashlib

from ..registry import embedder
from ._sanitize import sanitize_text


@embedder("local_hash")
class LocalHashEmbedder:
    name = "local_hash"

    def __init__(self, model: str = "local-hash-embedding-v1", dimension: int = 256, **_ignored: Any) -> None:
        self.model = model
        self.dim = dimension
        self.stats: dict[str, int] = {"tokens": 0, "api_calls": 0, "cache_hits": 0, "retries": 0}

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vector(sanitize_text(t)) for t in texts]

    def _hash_vector(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dim:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1
        values = values[:self.dim]
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]
