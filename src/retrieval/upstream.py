from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import logging

import requests
from fastapi import Request, Response

from .contracts import RetrievalHit
from .settings import CORPUS_MAX_TOP_K, VECTOR_SEARCH_PATH

logger = logging.getLogger(__name__)

HEALTH_PATH = "/api/v1/health"

_SKIP_REQUEST_HEADERS = frozenset({"host", "content-length", "connection"})
_SKIP_RESPONSE_HEADERS = frozenset(
    {"content-length", "content-encoding", "transfer-encoding", "connection"}
)


class CorpusUnavailable(RuntimeError):
    """The dense leg cannot be served.

    """


@dataclass(frozen=True)
class CorpusStats:
    reachable: bool
    document_count: int | None = None
    run_ids: frozenset[str] | None = None


class CorpusClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def vector_search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        embeddings_model: str | None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        raise NotImplementedError

    def health(self) -> bool:
        raise NotImplementedError

    def stats(self) -> CorpusStats:
        raise NotImplementedError

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


def _to_hit(raw: Mapping[str, Any]) -> RetrievalHit | None:
    raise NotImplementedError


async def passthrough(
    request: Request,
    base_url: str,
    timeout_s: float,
    path: str | None = None,
) -> Response:
    raise NotImplementedError
