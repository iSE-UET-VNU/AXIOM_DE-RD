"""Data contracts for the chunking + embedding stage.

``doc_id`` is the enriched record's ``source_object_id``; this stage never
re-derives it. ``chunk_id`` is content-addressed as
``sha256(doc_id + field_path + "start:end")`` with offsets within the field, so
the same config over the same document reproduces identical ids. Synthetic nodes
(RAPTOR summaries, section headers) carry ``start = end = -1`` and set ``level``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json

RECORD_CONTRACT_VERSION = "chunk-embed-v1"


def chunk_identity(doc_id: str, field_path: str, start: int, end: int, suffix: str = "") -> str:
    return _sha256(f"{doc_id}{field_path}{start}:{end}{suffix}")


def params_hash(params: dict[str, Any]) -> str:
    return _sha256(json.dumps(params, sort_keys=True, ensure_ascii=False))[:12]


@dataclass
class Unit:
    """A chunker's output when it yields derived text or tree nodes instead of
    faithful spans. ``parent``/``children`` reference other units by list index;
    the field router assigns final chunk ids and resolves those references."""

    text: str
    start: int = -1
    end: int = -1
    chunk_type: str = "text_chunk"
    level: int = 0
    parent: int | None = None
    children: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    field_path: str
    chunk_type: str
    text: str
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)
    level: int = 0
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        doc_id: str,
        field_path: str,
        chunk_type: str,
        text: str,
        start: int,
        end: int,
        metadata: dict[str, Any] | None = None,
        level: int = 0,
        parent_id: str | None = None,
        children: list[str] | None = None,
    ) -> "Chunk":
        return cls(
            chunk_id=chunk_identity(doc_id, field_path, start, end),
            doc_id=doc_id,
            field_path=field_path,
            chunk_type=chunk_type,
            text=text,
            start=start,
            end=end,
            metadata=dict(metadata or {}),
            level=level,
            parent_id=parent_id,
            children=list(children or []),
        )

    def embedding_text(self) -> str:
        override = self.metadata.get("embed_text")
        if override:
            return str(override)
        caption = self.metadata.get("caption")
        if caption and caption not in self.text:
            return f"{caption}\n{self.text}"
        return self.text

    def to_record(self, config_hash: str) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "doc_id": self.doc_id,
            "field_path": self.field_path,
            "content": self.text,
            "char_start": self.start,
            "char_end": self.end,
            "level": self.level,
            "parent_id": self.parent_id,
            "children": self.children,
            "metadata": {
                **self.metadata,
                "config_hash": config_hash,
                "contract_version": RECORD_CONTRACT_VERSION,
            },
        }


@dataclass(frozen=True)
class ChunkEmbedConfig:
    chunker: str = "recursive"
    chunker_params: dict[str, Any] = field(default_factory=dict)
    max_rows_per_chunk: int = 20
    embedder: str = "openrouter_te3s"
    embedder_params: dict[str, Any] = field(default_factory=dict)
    retrieval_profile: str = "hybrid_default"
    llm: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Any) -> "ChunkEmbedConfig":
        if not isinstance(value, dict):
            return cls()
        return cls(
            chunker=str(value.get("chunker", cls.chunker)),
            chunker_params=dict(value.get("chunker_params") or {}),
            max_rows_per_chunk=max(1, int(value.get("max_rows_per_chunk", cls.max_rows_per_chunk))),
            embedder=str(value.get("embedder", cls.embedder)),
            embedder_params=dict(value.get("embedder_params") or {}),
            retrieval_profile=str(value.get("retrieval_profile", cls.retrieval_profile)),
            llm=dict(value.get("llm") or {}),
        )

    def resolved(self) -> dict[str, Any]:
        return {
            "chunker": self.chunker,
            "chunker_params": self.chunker_params,
            "max_rows_per_chunk": self.max_rows_per_chunk,
            "embedder": self.embedder,
            "embedder_params": self.embedder_params,
            "retrieval_profile": self.retrieval_profile,
            "llm": self.llm,
            "contract_version": RECORD_CONTRACT_VERSION,
        }

    def config_hash(self) -> str:
        return _sha256(json.dumps(self.resolved(), sort_keys=True, ensure_ascii=False))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
