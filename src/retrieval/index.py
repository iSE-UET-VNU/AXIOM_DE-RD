"""A built corpus representation: chunks, a sparse index, optional vectors.


"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np

from .protocol import ChunkRecord, QueryEncoding
from .sparse import BM25Index


@dataclass
class LocalIndex:

    index_id: str
    records: list[ChunkRecord] = field(default_factory=list)
    bm25: BM25Index | None = None
    vectors: np.ndarray | None = None
    embedder: Any = None
    embeddings_model: str = ""
    metric: str = "cosine"
    normalized: bool = False

    def __post_init__(self) -> None:
        self._by_id = {record.chunk_id: record for record in self.records}
        self._positions = {record.chunk_id: i for i, record in enumerate(self.records)}
        if self.vectors is not None:
            matrix = np.asarray(self.vectors, dtype=np.float32)
            if self.metric == "cosine" and not self.normalized:
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                matrix = matrix / np.clip(norms, 1e-12, None)
                self.normalized = True
            self.vectors = matrix

    # -- protocol ----------------------------------------------------------

    def encode_query(self, text: str) -> QueryEncoding:
        dense = None
        if self.embedder is not None:
            encode = getattr(self.embedder, "embed_query", None) or self.embedder.embed
            dense = list(np.asarray(encode([text]), dtype=np.float32)[0])
        return QueryEncoding(text=text, dense=dense)

    def chunks(self, ids: Sequence[str]) -> Mapping[str, ChunkRecord]:
        return {cid: self._by_id[cid] for cid in ids if cid in self._by_id}

    def doc_ids(self) -> Sequence[str]:
        seen: list[str] = []
        for record in self.records:
            if record.doc_id not in seen:
                seen.append(record.doc_id)
        return seen

    # -- helpers for retrievers -------------------------------------------

    @property
    def has_dense(self) -> bool:
        return self.vectors is not None

    def position(self, chunk_id: str) -> int:
        return self._positions[chunk_id]

    def scope_mask(self, scope: Sequence[str] | None) -> np.ndarray | None:
        """Boolean mask over chunk positions for a document-id scope.
        """
        if scope is None:
            return None
        wanted = set(scope)
        return np.array([record.doc_id in wanted for record in self.records], dtype=bool)

    def record_at(self, position: int) -> ChunkRecord:
        return self.records[position]

    def scope_positions(self, scope: Sequence[str] | None) -> set[int] | None:
        if scope is None:
            return None
        key = frozenset(scope)
        if not hasattr(self, "_scope_positions"):
            self._scope_positions: dict[frozenset[str], set[int]] = {}
            self._by_doc: dict[str, list[int]] = {}
            for position, record in enumerate(self.records):
                self._by_doc.setdefault(record.doc_id, []).append(position)
        cached = self._scope_positions.get(key)
        if cached is None:
            cached = {p for doc in key for p in self._by_doc.get(doc, ())}
            self._scope_positions[key] = cached
        return cached


class ArtifactMisaligned(RuntimeError):
    """Vectors and chunk ids disagree.
    """


def load_artifacts(artifact_dir: Path, embedder: Any = None) -> LocalIndex:
    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))

    records: list[ChunkRecord] = []
    for line in (artifact_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(
            ChunkRecord(
                chunk_id=payload["chunk_id"],
                doc_id=payload["doc_id"],
                text=payload["text"],
                meta={k: v for k, v in payload.items() if k not in {"chunk_id", "doc_id", "text"}},
            )
        )

    ids_path = artifact_dir / "chunk_ids.json"
    if ids_path.exists():
        chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if chunk_ids != [record.chunk_id for record in records]:
            raise ArtifactMisaligned(
                f"chunk_ids.json does not match chunks.jsonl order in {artifact_dir}."
            )

    vectors = None
    vector_path = artifact_dir / "vectors.npy"
    if vector_path.exists():
        vectors = np.load(vector_path)
        if vectors.shape[0] != len(records):
            raise ArtifactMisaligned(
                f"{vectors.shape[0]} vectors for {len(records)} chunks in {artifact_dir}. "
                "Row i must be the embedding of chunk i."
            )
        expected_dim = int(manifest.get("dim") or 0)
        if expected_dim and vectors.shape[1] != expected_dim:
            raise ArtifactMisaligned(
                f"Manifest declares dim={expected_dim}, vectors are {vectors.shape[1]}."
            )

    analyzer = str(manifest.get("analyzer_id") or manifest.get("analyzer") or "")
    if analyzer == "auto":
        raise ArtifactMisaligned(
            f"{artifact_dir} was built before the analyzer was resolved at build "
            "time. Its postings and its queries may use different tokenizers; "
            "rebuild rather than load."
        )

    return LocalIndex(
        index_id=str(
            manifest.get("index_id") or manifest.get("config_hash") or artifact_dir.name
        ),
        records=records,
        bm25=BM25Index.load(artifact_dir / "bm25.json"),
        vectors=vectors,
        embedder=embedder,
        embeddings_model=", ".join(manifest.get("embeddings_models") or []),
        metric=str(manifest.get("metric") or "cosine"),
        normalized=bool(manifest.get("normalized")),
    )
