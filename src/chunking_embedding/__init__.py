"""Chunking + embedding stage — parsed documents into chunk records and vectors.

Importing this package registers the built-in chunkers and embedders (one file
each under chunkers/ and embedders/), so a notebook can drive the same code:

    from src.chunking_embedding import chunker_names, embedder_names, run
"""

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
from .fields import FieldContext, route_document

_RUNNER_EXPORTS = ("ChunkEmbedResult", "chunk_and_embed", "run")


def __getattr__(name: str):
    if name in _RUNNER_EXPORTS:
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Chunk",
    "ChunkEmbedConfig",
    "ChunkEmbedResult",
    "FieldContext",
    "chunk_and_embed",
    "chunk_identity",
    "chunker_aliases",
    "chunker_info",
    "chunker_names",
    "create_embedder",
    "embedder_names",
    "get_chunker",
    "get_embedder",
    "route_document",
    "run",
    "sanitize_text",
]
