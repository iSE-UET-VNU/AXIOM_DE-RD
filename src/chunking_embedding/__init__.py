"""Chunking + embedding stage — enriched documents into retrieval records.

Importing this package registers the built-in chunkers and embedders (one file
each under chunkers/ and embedders/).  The public ``run`` entrypoint consumes
enrichment-stage records; file-based batch execution is exposed separately as
``run_cli``.
"""

from pathlib import Path
from typing import Any

from .contracts import Chunk, ChunkEmbedConfig, chunk_identity
from .registry import (
    chunker_aliases,
    chunker_info,
    chunker_names,
    embedder_names,
    get_chunker,
    get_embedder,
)
from . import chunkers as _chunkers  # noqa: F401  (registers chunker plugins)
from . import embedders as _embedders  # noqa: F401  (registers embedder plugins)
from .embedders import create_embedder, sanitize_text
from .document_view import DocumentView, document_from_enriched_data
from .fields import FieldContext, route_document
from .lexical import build_corpus_statistics, build_lexical_payload
from .stage import ChunkEmbedResult, ChunkingEmbeddingOutput, chunk_and_embed, run


__all__ = [
    "Chunk",
    "ChunkEmbedConfig",
    "ChunkEmbedResult",
    "ChunkingEmbeddingOutput",
    "DocumentView",
    "FieldContext",
    "chunk_and_embed",
    "build_corpus_statistics",
    "build_lexical_payload",
    "chunk_identity",
    "chunker_aliases",
    "chunker_info",
    "chunker_names",
    "create_embedder",
    "document_from_enriched_data",
    "embedder_names",
    "get_chunker",
    "get_embedder",
    "route_document",
    "run",
    "run_cli",
    "sanitize_text",
]


def run_cli(
    config: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Run the artifact CLI lazily so package imports stay file-system free."""
    from .cli import run_from_artifacts

    return run_from_artifacts(config, input_dir, output_dir, force=force)
