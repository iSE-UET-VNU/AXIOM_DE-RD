"""BM25 artifacts
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping
import logging

from .sparse import BM25Index
from .upstream import CorpusClient

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
INDEX_NAME = "bm25.json"
CHUNKS_NAME = "chunks.jsonl"


REQUIRED_MANIFEST_KEYS = ("config_hash", "run_id", "embeddings_models", "analyzer")


class Staleness(StrEnum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"


class ArtifactError(RuntimeError):
    """Artifacts are missing, ambiguous or malformed. Always fatal at startup."""


@dataclass(frozen=True)
class ArtifactSet:
    config_hash: str
    run_ids: frozenset[str]
    embeddings_models: tuple[str, ...]
    analyzer: str
    index: BM25Index
    chunk_text: Mapping[str, str]
    chunk_doc: Mapping[str, str]
    chunk_source: Mapping[str, str]
    by_embedding_id: Mapping[str, str]

    @property
    def chunk_count(self) -> int:
        raise NotImplementedError

    @property
    def doc_count(self) -> int:
        raise NotImplementedError


def load(artifact_dir: Path, wanted_hash: str = "") -> ArtifactSet:
    """Load exactly one artifact set.
    Validates manifest *shape*, not just presence — see REQUIRED_MANIFEST_KEYS.
    """
    raise NotImplementedError


def resolve_embeddings_model(artifacts: ArtifactSet, override: str = "") -> str:
    """Decide which model to filter dense hits by.
    Raises ArtifactError when the run stored several models and no override
    picks one, rather than guessing.
    """
    raise NotImplementedError


def check_request_model(artifacts: ArtifactSet, requested: str | None) -> str | None:
    """Compare the caller's ``embeddings_model`` against what the artifacts say.
    """
    raise NotImplementedError


def check_staleness(artifacts: ArtifactSet, corpus: CorpusClient) -> Staleness:
    """Compare what we indexed against what corpus-service holds..
    """
    raise NotImplementedError
