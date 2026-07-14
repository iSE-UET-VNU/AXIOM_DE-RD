"""Embedding generation for index records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import hashlib
import math
import os

from ..models import IndexRecord

DEFAULT_TARGET_INDEX_TYPES = ["text_chunk"]
DEFAULT_LOCAL_HASH_DIMENSION = 128
DEFAULT_BATCH_SIZE = 32
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class EmbeddingProvider(Protocol):
    model: str
    dimension: int

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class EmbeddingConfig:
    enabled: bool = False
    provider: str = "local_hash"
    model: str = "local-hash-embedding-v1"
    dimension: int = DEFAULT_LOCAL_HASH_DIMENSION
    batch_size: int = DEFAULT_BATCH_SIZE
    target_index_types: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_INDEX_TYPES))
    fail_on_error: bool = True
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str | None = None
    app_title: str | None = None
    http_referer_env: str | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "EmbeddingConfig":
        if not isinstance(value, dict):
            return cls(target_index_types=list(DEFAULT_TARGET_INDEX_TYPES))

        provider = str(value.get("provider", "local_hash"))
        model = str(value.get("model") or _default_model(provider))
        default_api_key_env = "OPENROUTER_API_KEY"
        target_index_types = value.get("target_index_types", DEFAULT_TARGET_INDEX_TYPES)
        if not isinstance(target_index_types, list):
            target_index_types = DEFAULT_TARGET_INDEX_TYPES

        return cls(
            enabled=bool(value.get("enabled", False)),
            provider=provider,
            model=model,
            dimension=int(value.get("dimension", _default_dimension(provider))),
            batch_size=max(1, int(value.get("batch_size", DEFAULT_BATCH_SIZE))),
            target_index_types=[str(item) for item in target_index_types],
            fail_on_error=bool(value.get("fail_on_error", True)),
            api_key_env=str(value.get("api_key_env", default_api_key_env)),
            base_url=str(value["base_url"]) if value.get("base_url") else None,
            app_title=str(value["app_title"]) if value.get("app_title") else None,
            http_referer_env=str(value["http_referer_env"]) if value.get("http_referer_env") else None,
        )


class LocalHashEmbeddingProvider:
    """Deterministic local embeddings for workflow tests."""

    def __init__(self, model: str, dimension: int) -> None:
        self.model = model
        self.dimension = dimension

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embedding(text, self.dimension) for text in texts]


class OpenAICompatibleEmbeddingProvider:
    """Embedding provider for OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str,
        dimension: int,
        api_key_env: str,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing openai package. Install it with: pip install openai") from exc

        self.model = model
        self.dimension = dimension
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        self.client = OpenAI(**client_kwargs)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimension,
        )
        return [list(item.embedding) for item in result.data]


class OpenRouterEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """OpenRouter embeddings provider using its OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        dimension: int,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = OPENROUTER_BASE_URL,
        app_title: str | None = None,
        http_referer_env: str | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if app_title:
            headers["X-Title"] = app_title

        referer = os.getenv(http_referer_env) if http_referer_env else None
        if referer:
            headers["HTTP-Referer"] = referer

        super().__init__(
            model=model,
            dimension=dimension,
            api_key_env=api_key_env,
            base_url=base_url,
            default_headers=headers or None,
        )


def build_vector_records(
    index_records: list[IndexRecord],
    embedding_config: EmbeddingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate vector records from eligible index records."""
    if not embedding_config.enabled:
        return [], _embedding_report(embedding_config, status="disabled")

    provider = _provider_from_config(embedding_config)
    document_records = _document_records_by_source(index_records)
    eligible_records = [
        record
        for record in index_records
        if record.index_type in set(embedding_config.target_index_types)
    ]

    vector_records: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    records_to_embed: list[IndexRecord] = []
    texts_to_embed: list[str] = []
    for record in eligible_records:
        text = _embedding_text(record)
        if not text:
            warnings.append(
                {
                    "code": "empty_embedding_text",
                    "record_id": record.record_id,
                    "source_object_id": record.source_object_id,
                    "index_type": record.index_type,
                }
            )
            continue
        records_to_embed.append(record)
        texts_to_embed.append(text)

    try:
        for batch_start in range(0, len(records_to_embed), embedding_config.batch_size):
            batch_records = records_to_embed[batch_start: batch_start + embedding_config.batch_size]
            batch_texts = texts_to_embed[batch_start: batch_start + embedding_config.batch_size]
            embeddings = provider.embed_batch(batch_texts)
            for record, embedding in zip(batch_records, embeddings):
                vector_records.append(_vector_record(record, embedding, provider, document_records.get(record.source_object_id)))
    except Exception as exc:
        errors.append({"code": "embedding_failed", "message": str(exc)})
        if embedding_config.fail_on_error:
            raise

    report = _embedding_report(
        embedding_config,
        status="failed" if errors else "passed",
        eligible_count=len(eligible_records),
        generated_count=len(vector_records),
        skipped_count=len(warnings),
        warnings=warnings,
        errors=errors,
    )
    return vector_records, report


def _provider_from_config(config: EmbeddingConfig) -> EmbeddingProvider:
    if config.provider == "local_hash":
        return LocalHashEmbeddingProvider(config.model, config.dimension)
    if config.provider == "openrouter":
        return OpenRouterEmbeddingProvider(
            model=config.model,
            dimension=config.dimension,
            api_key_env=config.api_key_env,
            base_url=config.base_url or OPENROUTER_BASE_URL,
            app_title=config.app_title,
            http_referer_env=config.http_referer_env,
        )
    raise ValueError(f"Unsupported embedding provider: {config.provider}")


def _vector_record(
    record: IndexRecord,
    embedding: list[float],
    provider: EmbeddingProvider,
    document_record: IndexRecord | None,
) -> dict[str, Any]:
    payload = record.payload
    document_payload = document_record.payload if document_record else {}
    return {
        "vector_id": record.record_id,
        "record_id": record.record_id,
        "source_object_id": record.source_object_id,
        "index_type": record.index_type,
        "document_id": payload.get("document_id"),
        "chunk_id": payload.get("chunk_id"),
        "chunk_index": payload.get("chunk_index"),
        "table_id": payload.get("table_id"),
        "image_id": payload.get("image_id"),
        "source_block_id": payload.get("source_block_id"),
        "page": payload.get("page"),
        "source_uri": document_payload.get("source_uri"),
        "title": document_payload.get("title"),
        "document_type": document_payload.get("document_type"),
        "language": document_payload.get("language"),
        "text": payload.get("text") or payload.get("embedding_text", ""),
        "start_char": payload.get("start_char"),
        "end_char": payload.get("end_char"),
        "embedding": embedding,
        "embedding_model": provider.model,
        "embedding_dimension": len(embedding),
        "embedding_status": "embedded",
        "metadata": {
            "index_record_id": record.record_id,
            "index_contract_version": record.metadata.get("contract_version"),
            "component_type": record.metadata.get("component_type"),
        },
    }


def _document_records_by_source(index_records: list[IndexRecord]) -> dict[str, IndexRecord]:
    return {
        record.source_object_id: record
        for record in index_records
        if record.index_type == "document"
    }


def _embedding_text(record: IndexRecord) -> str:
    value = record.payload.get("embedding_text")
    if value is None:
        return ""
    return str(value).strip()


def _hash_embedding(text: str, dimension: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
        for byte in digest:
            values.append((byte / 127.5) - 1.0)
            if len(values) == dimension:
                break
        counter += 1

    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [round(value / norm, 8) for value in values]


def _embedding_report(
    config: EmbeddingConfig,
    status: str,
    eligible_count: int = 0,
    generated_count: int = 0,
    skipped_count: int = 0,
    warnings: list[dict[str, Any]] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "contract_version": "embedding-report-v1",
        "status": status,
        "provider": config.provider,
        "model": config.model,
        "dimension": config.dimension,
        "target_index_types": config.target_index_types or list(DEFAULT_TARGET_INDEX_TYPES),
        "eligible_count": eligible_count,
        "generated_count": generated_count,
        "skipped_count": skipped_count,
        "warnings": warnings or [],
        "errors": errors or [],
    }


def _default_model(provider: str) -> str:
    if provider == "openrouter":
        return "openai/text-embedding-3-small"
    return "local-hash-embedding-v1"


def _default_dimension(provider: str) -> int:
    if provider == "openrouter":
        return 1536
    return DEFAULT_LOCAL_HASH_DIMENSION
