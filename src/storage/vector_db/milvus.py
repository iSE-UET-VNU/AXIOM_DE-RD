"""Milvus vector database adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import os


DEFAULT_COLLECTION_NAME = "axiom_text_chunks"
DEFAULT_METRIC_TYPE = "COSINE"


@dataclass(frozen=True)
class MilvusConfig:
    uri: str
    token: str | None = None
    collection_name: str = DEFAULT_COLLECTION_NAME
    dimension: int = 1536
    metric_type: str = DEFAULT_METRIC_TYPE
    drop_collection: bool = False

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "MilvusConfig":
        token_env = str(config.get("token_env", "MILVUS_TOKEN"))
        token = config.get("token") or os.getenv(token_env)
        uri = str(config.get("uri") or os.getenv("MILVUS_URI") or "")
        if not uri:
            raise RuntimeError("Milvus uri is required. Set storage.vector_db.uri or MILVUS_URI.")

        return cls(
            uri=uri,
            token=str(token) if token else None,
            collection_name=str(config.get("collection_name", DEFAULT_COLLECTION_NAME)),
            dimension=int(config.get("dimension", 1536)),
            metric_type=str(config.get("metric_type", DEFAULT_METRIC_TYPE)),
            drop_collection=bool(config.get("drop_collection", False)),
        )


class MilvusVectorDB:
    """Small Milvus adapter for text chunk vector records."""

    def __init__(self, config: MilvusConfig) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise RuntimeError("Missing pymilvus package. Install it with: pip install pymilvus") from exc

        self.config = config
        self.client = MilvusClient(uri=config.uri, token=config.token)

    def upsert_vectors(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        records = list(records)
        self._ensure_collection()
        rows = milvus_rows_from_vector_records(records)
        if not rows:
            return self._report(0)

        try:
            self.client.upsert(collection_name=self.config.collection_name, data=rows)
        except Exception:
            ids = [row["vector_id"] for row in rows]
            self.client.delete(
                collection_name=self.config.collection_name,
                filter=f"vector_id in {ids!r}",
            )
            self.client.insert(collection_name=self.config.collection_name, data=rows)

        return self._report(len(rows))

    def _ensure_collection(self) -> None:
        from pymilvus import DataType

        if self.config.drop_collection and self.client.has_collection(self.config.collection_name):
            self.client.drop_collection(self.config.collection_name)

        if self.client.has_collection(self.config.collection_name):
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("vector_id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("embedding", datatype=DataType.FLOAT_VECTOR, dim=self.config.dimension)
        schema.add_field("record_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field("index_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field("source_object_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field("document_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field("chunk_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field("chunk_index", datatype=DataType.INT64)
        schema.add_field("table_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field("image_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field("source_block_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field("page", datatype=DataType.INT64)
        schema.add_field("source_uri", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field("title", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field("document_type", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field("language", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field("text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field("start_char", datatype=DataType.INT64)
        schema.add_field("end_char", datatype=DataType.INT64)
        schema.add_field("embedding_model", datatype=DataType.VARCHAR, max_length=256)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type=self.config.metric_type,
        )
        self.client.create_collection(
            collection_name=self.config.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _report(self, upserted: int) -> dict[str, Any]:
        return {
            "provider": "milvus",
            "collection_name": self.config.collection_name,
            "status": "passed",
            "upserted": upserted,
            "dimension": self.config.dimension,
            "metric_type": self.config.metric_type,
            "errors": [],
            "warnings": [],
        }


def milvus_rows_from_vector_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map AXIOM vector records to flat Milvus rows."""
    rows = []
    for record in records:
        rows.append(
            {
                "vector_id": _text(record.get("vector_id")),
                "embedding": record.get("embedding") or [],
                "record_id": _text(record.get("record_id")),
                "index_type": _text(record.get("index_type")),
                "source_object_id": _text(record.get("source_object_id")),
                "document_id": _text(record.get("document_id")),
                "chunk_id": _text(record.get("chunk_id")),
                "chunk_index": _int(record.get("chunk_index")),
                "table_id": _text(record.get("table_id")),
                "image_id": _text(record.get("image_id")),
                "source_block_id": _text(record.get("source_block_id")),
                "page": _int(record.get("page")),
                "source_uri": _text(record.get("source_uri")),
                "title": _text(record.get("title")),
                "document_type": _text(record.get("document_type")),
                "language": _text(record.get("language")),
                "text": _text(record.get("text")),
                "start_char": _int(record.get("start_char")),
                "end_char": _int(record.get("end_char")),
                "embedding_model": _text(record.get("embedding_model")),
            }
        )
    return rows


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _int(value: Any) -> int:
    if value is None:
        return 0
    return int(value)
