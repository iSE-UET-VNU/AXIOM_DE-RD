"""Wire models for both directions.

"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

RetrievalSource = Literal["keyword", "vector", "hybrid"]

MAX_TOP_K = 100


# ---------------------------------------------------------------------------
# Inbound: what Methods-Hub sends us
# ---------------------------------------------------------------------------


class VectorSearchRequest(BaseModel):
    """``POST /api/v1/retrieval/vector-search``.
    """

    query_embedding: list[float] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)
    organization_id: str | None = None
    document_id: str | None = None
    document_ids: list[str] | None = Field(default=None, min_length=1)
    embeddings_model: str | None = None
    content_type: str | None = None
    vector_weight: float = Field(default=1.0, ge=0.0, le=1.0)


class KeywordSearchRequest(BaseModel):
    """``POST /api/v1/retrieval/keyword-search``. Text, no vector."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)
    organization_id: str | None = None
    document_id: str | None = None
    document_ids: list[str] | None = Field(default=None, min_length=1)
    content_type: str | None = None
    case_sensitive: bool = False


class RetrieveContextRequest(BaseModel):
    """``POST /api/v1/retrieval/context``.
    """

    query: str | None = None
    query_embedding: list[float] | None = None
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)
    organization_id: str | None = None
    document_id: str | None = None
    document_ids: list[str] | None = Field(default=None, min_length=1)
    embeddings_model: str | None = None
    content_type: str | None = None
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    max_chars_per_chunk: int = Field(default=1500, ge=100, le=10000)

    @model_validator(mode="after")
    def require_query_or_embedding(self) -> "RetrieveContextRequest":
        if not self.query and not self.query_embedding:
            raise ValueError(
                "context requires 'query', 'query_embedding', or both; got neither."
            )
        return self

    def filters(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for name in ("organization_id", "document_id", "document_ids", "content_type"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out


# ---------------------------------------------------------------------------
# Outbound: what we return
# ---------------------------------------------------------------------------


class IndexedDocument(BaseModel):

    document_id: str
    organization_id: str
    source_uri: str
    bucket: str
    object_key: str
    file_name: str
    current_status: str


class RetrievalHit(BaseModel):

    document: IndexedDocument
    run_id: str
    content_id: str | None = None
    embedding_id: str | None = None
    content_type: str
    position: dict = Field(default_factory=dict)
    content: dict = Field(default_factory=dict)
    text: str | None = None
    score: float
    # Ours: dense, bm25, fused, rerank. Gated by RETRIEVAL_EXPOSE_SCORES.
    scores: dict[str, float] = Field(default_factory=dict)
    retrieval_source: RetrievalSource


class ContextChunk(BaseModel):

    document: IndexedDocument
    run_id: str
    content_id: str | None = None
    embedding_id: str | None = None
    content_type: str
    position: dict = Field(default_factory=dict)
    text: str
    score: float
    scores: dict[str, float] = Field(default_factory=dict)
    retrieval_source: RetrievalSource


class SearchResponse(BaseModel):
  

    results: list[RetrievalHit]
    degraded: bool = False
    degraded_reason: str | None = None


class ContextResponse(BaseModel):
    chunks: list[ContextChunk]
    degraded: bool = False
    degraded_reason: str | None = None


class HealthResponse(BaseModel):


    status: Literal["ok", "degraded", "unavailable"]
    config_hash: str
    artifacts_loaded: bool
    staleness: str
    embeddings_model: str
    corpus_service: Literal["ok", "unreachable"]
    reranker: str
    enhanced_paths: list[str]
