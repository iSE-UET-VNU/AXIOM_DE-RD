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
        """Dense ANN search. corpus-service owns the vectors; we only read.

        """
        wanted = int(top_k)
        if wanted > CORPUS_MAX_TOP_K:
            logger.warning(
                "top_k %d clamped to corpus-service maximum %d; the dense "
                "candidate pool is shallower than configured.",
                wanted, CORPUS_MAX_TOP_K,
            )
            wanted = CORPUS_MAX_TOP_K

        body: dict[str, Any] = {"query_embedding": list(query_embedding), "top_k": wanted}
        if embeddings_model:
            body["embeddings_model"] = embeddings_model
        # None is dropped, never sent: an explicit null reads as "match null"
        # rather than "do not filter".
        for key, value in (filters or {}).items():
            if value is not None:
                body[key] = value

        payload = self._post(VECTOR_SEARCH_PATH, body)
        results = payload.get("results")
        if not isinstance(results, list):
            raise CorpusUnavailable("vector-search returned no 'results' array")

        hits: list[RetrievalHit] = []
        dropped = 0
        for raw in results:
            hit = _to_hit(raw) if isinstance(raw, Mapping) else None
            if hit is None:
                dropped += 1
            else:
                hits.append(hit)
        if dropped:
            logger.warning("dropped %d/%d malformed hits from corpus-service",
                           dropped, len(results))
        return hits

    def health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}{HEALTH_PATH}", timeout=self.timeout_s)
        except requests.RequestException as error:
            logger.warning("corpus-service unreachable: %s", error)
            return False
        return response.status_code == 200

    def stats(self) -> CorpusStats:
        """What corpus-service holds, for the staleness comparison.

        """
        return CorpusStats(reachable=self.health())

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(
                f"{self.base_url}{path}", json=dict(body), timeout=self.timeout_s
            )
        except requests.RequestException as error:
            raise CorpusUnavailable(f"POST {path}: {error}") from error
        if response.status_code != 200:
            raise CorpusUnavailable(f"POST {path}: HTTP {response.status_code}: {response.text[:200]}")
        try:
            decoded = response.json()
        except ValueError as error:
            raise CorpusUnavailable(f"POST {path}: response is not JSON") from error
        if not isinstance(decoded, dict):
            raise CorpusUnavailable(f"POST {path}: expected a JSON object")
        return decoded


def _to_hit(raw: Mapping[str, Any]) -> RetrievalHit | None:
    try:
        return RetrievalHit.model_validate(dict(raw))
    except Exception as error:  # pydantic ValidationError and anything odd
        logger.debug("malformed hit: %s", error)
        return None


async def passthrough(
    request: Request,
    base_url: str,
    timeout_s: float,
    path: str | None = None,
) -> Response:
    """Forward a request to corpus-service and return its response verbatim.
    """
    target = f"{base_url.rstrip('/')}{path or request.url.path}"
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _SKIP_REQUEST_HEADERS
    }
    body = await request.body()
    try:
        upstream = requests.request(
            request.method,
            target,
            headers=headers,
            params=dict(request.query_params),
            data=body,
            timeout=timeout_s,
        )
    except requests.RequestException as error:
        raise CorpusUnavailable(f"{request.method} {target}: {error}") from error

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers={
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in _SKIP_RESPONSE_HEADERS
        },
        media_type=upstream.headers.get("content-type"),
    )
