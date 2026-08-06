"""HTTP surface.

This service impersonates corpus-service's retrieval API so Methods-Hub can be
re-pointed with one env var and no code change:

    CORPUS_SERVICE_URL=http://retrieval-service:8081

    uvicorn src.retrieval.app:app --port 8081
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
import logging

from fastapi import Depends, FastAPI, Request, Response

from .contracts import (
    ContextResponse,
    HealthResponse,
    KeywordSearchRequest,
    RetrieveContextRequest,
    SearchResponse,
    VectorSearchRequest,
)
from .settings import (
    CONTEXT_PATH,
    INGESTED_DATA_PATH,
    KEYWORD_SEARCH_PATH,
    NEIGHBOR_CHUNKS_PATH,
    VECTOR_SEARCH_PATH,
    Settings,
)

logger = logging.getLogger(__name__)


class ServiceState:
    """Everything loaded once at startup.

    Artifacts are loaded only when at least one path is enhanced. In pure
    passthrough mode the service must start with no artifacts on disk at all —
    otherwise the safe adoption step is gated on the risky prerequisite.
    """

    def __init__(self, settings: Settings) -> None:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError

    @property
    def ready(self) -> bool:
        raise NotImplementedError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup is the last place a misconfiguration is cheap.

    Ambiguous or malformed artifacts abort here rather than degrading a live
    request path, because a service that starts and answers wrongly is harder
    to notice than one that refuses to start.
    """
    raise NotImplementedError


app = FastAPI(title="AXIOM Retrieval", version="1.0.0", lifespan=lifespan)


async def require_token(request: Request) -> None:
    """Verify ``Authorization: Bearer`` when RETRIEVAL_REQUIRE_TOKEN is set.

    corpus-service has no authentication of any kind, and Methods-Hub already
    sends this header whenever CORPUS_SERVICE_TOKEN is set — so this is real
    auth for the cost of an env var, with no change on their side.

    Off by default: being stricter than the service we front is a new way to
    break something that works today. A 401 is not in Methods-Hub's retryable
    set, so it fails fast and the body text reaches the agent.
    """
    raise NotImplementedError


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness, not liveness.

    Downstream compose gates on ``condition: service_started``, which a
    crash-looping container satisfies, so this has to carry enough for an
    operator to distinguish "up" from "usable": artifact staleness, the
    resolved embeddings_model, corpus-service reachability, and which paths are
    actually enhanced.
    """
    raise NotImplementedError


@app.post(VECTOR_SEARCH_PATH, dependencies=[Depends(require_token)])
async def vector_search(request: Request, body: VectorSearchRequest) -> Any:
    """Dense only.

    Methods-Hub strips ``query`` on this path, so there is no text — no sparse
    leg, and no reranking either, since every reranker needs the query. What we
    add here is the ``embeddings_model`` check and honest degradation
    reporting; the ranking itself is corpus-service's.
    """
    raise NotImplementedError


@app.post(KEYWORD_SEARCH_PATH, dependencies=[Depends(require_token)])
async def keyword_search(request: Request, body: KeywordSearchRequest) -> Any:
    """Our BM25 instead of theirs.

    corpus-service tokenizes with ``\\b\\w+\\b`` and computes IDF over the
    filtered candidate set of the current query rather than the corpus. Ours
    uses the same analyzer the pipeline used at index time and corpus-global
    statistics from the artifact manifest.
    """
    raise NotImplementedError


@app.post(CONTEXT_PATH, dependencies=[Depends(require_token)])
async def retrieve_context(request: Request, body: RetrieveContextRequest) -> Any:
    """The full stack: dense + sparse -> fuse -> rerank.

    The only path carrying both ``query`` and ``query_embedding``, so the only
    one where everything is possible. This is where the service earns its keep.
    """
    raise NotImplementedError


@app.post(NEIGHBOR_CHUNKS_PATH)
async def neighbor_chunks(request: Request) -> Response:
    """Passthrough. Positional lookup around a chunk index; we add nothing.

    Never enhanced, so deliberately not behind require_token — we do not want
    an auth misconfiguration to break a path we merely forward.
    """
    raise NotImplementedError


@app.post(INGESTED_DATA_PATH)
async def ingested_data(request: Request) -> Response:
    """Passthrough. Not a retrieval path.

    It reaches us only because Methods-Hub routes all five through one
    CORPUS_SERVICE_URL. Methods-Hub reads ``contents`` and ``chunks`` off this
    response to gzip and base64 them, so it must arrive byte-intact.
    """
    raise NotImplementedError
