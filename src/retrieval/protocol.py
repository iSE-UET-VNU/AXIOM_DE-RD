"""The retrieval interface
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class ChunkRecord:
    
    chunk_id: str
    doc_id: str
    text: str
    page: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    doc_id: str
    score: float
    rank: int
    text: str = ""
    scores: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryEncoding:

    text: str
    dense: Sequence[float] | None = None
    sparse: Mapping[str, float] | None = None
    multi_vector: Sequence[Sequence[float]] | None = None

    def require(self, kind: str) -> Any:
        value = getattr(self, kind, None)
        if value is None:
            raise UnsupportedEncoding(
                f"Index cannot produce a {kind!r} encoding for this query. "
                "A retriever that needs one must not silently fall back to "
                "another leg -- that reports a method it did not run."
            )
        return value


class UnsupportedEncoding(RuntimeError):
    """A retriever asked the index for an encoding it does not provide."""


@runtime_checkable
class Index(Protocol):
    index_id: str

    def encode_query(self, text: str) -> QueryEncoding: ...

    def chunks(self, ids: Sequence[str]) -> Mapping[str, ChunkRecord]: ...

    def doc_ids(self) -> Sequence[str]: ...


@runtime_checkable
class Retriever(Protocol):


    retriever_id: str

    def retrieve(
        self, query: str, k: int, scope: list[str] | None = None
    ) -> list[ScoredChunk]: ...

    def params(self) -> Mapping[str, Any]:
        """Everything that changes results, for the run-record params_hash."""
        ...
