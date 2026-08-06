"""Retrieval orchestration: dense -> sparse -> fuse -> rerank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import logging

from .artifacts import ArtifactSet, Staleness
from .contracts import ContextChunk, RetrievalHit
from .rerank import Reranker
from .settings import Settings
from .sparse import BM25Index
from .upstream import CorpusClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Legs:

    dense: bool
    sparse: bool

    @classmethod
    def from_payload(cls, query: str | None, query_embedding: list[float] | None) -> "Legs":
        raise NotImplementedError

    @property
    def hybrid(self) -> bool:
        raise NotImplementedError


@dataclass
class Degradation:


    reasons: list[str] = field(default_factory=list)

    def add(self, reason: str) -> None:
        raise NotImplementedError

    @property
    def degraded(self) -> bool:
        raise NotImplementedError

    @property
    def reason(self) -> str | None:
        raise NotImplementedError


@dataclass
class ScoredChunk:


    chunk_id: str
    hit: RetrievalHit | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


@dataclass
class SearchOutcome:
    chunks: list[ScoredChunk]
    degradation: Degradation
    timings_ms: dict[str, float]
    legs: Legs


class SearchEngine:


    def __init__(
        self,
        corpus: CorpusClient,
        artifacts: ArtifactSet,
        index: BM25Index,
        reranker: Reranker,
        settings: Settings,
        embeddings_model: str,
        staleness: Staleness,
    ) -> None:
        raise NotImplementedError

    def search(
        self,
        *,
        query: str | None,
        query_embedding: list[float] | None,
        requested_model: str | None,
        top_k: int,
        alpha: float | None = None,
        filters: Mapping[str, Any] | None = None,
        rerank: bool = True,
    ) -> SearchOutcome:
      
        raise NotImplementedError

    def _dense(self, query_embedding: list[float], filters: Mapping[str, Any],
               degradation: Degradation) -> list[RetrievalHit]:
       
        raise NotImplementedError

    def _sparse(self, query: str) -> list[tuple[str, float]]:
        raise NotImplementedError

    def _rerank(self, query: str, fused: list[tuple[str, float]],
                degradation: Degradation) -> tuple[list[tuple[str, float]], dict[str, float]]:
       
        raise NotImplementedError

    def _check_model(self, requested: str | None, degradation: Degradation) -> None:
        
        raise NotImplementedError


def to_hits(outcome: SearchOutcome, expose_scores: bool) -> list[RetrievalHit]:
    
    raise NotImplementedError


def to_chunks(outcome: SearchOutcome, max_chars: int, expose_scores: bool) -> list[ContextChunk]:

    raise NotImplementedError
