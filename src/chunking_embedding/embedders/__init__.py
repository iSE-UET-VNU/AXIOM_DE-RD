"""Embedders — one module per backend, auto-registered via @embedder."""

from __future__ import annotations

from typing import Any

from ..registry import get_embedder, load_plugins
from ._sanitize import CJK_CHAR_LIMIT, CJK_RATIO_THRESHOLD, DEFAULT_CHAR_LIMIT, sanitize_text

load_plugins(__name__, __path__)


def create_embedder(name: str, params: dict[str, Any]) -> Any:
    return get_embedder(name)(**params)


__all__ = [
    "CJK_CHAR_LIMIT",
    "CJK_RATIO_THRESHOLD",
    "DEFAULT_CHAR_LIMIT",
    "create_embedder",
    "sanitize_text",
]
