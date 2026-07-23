"""Structure baseline: split on blank-line paragraph boundaries."""

from __future__ import annotations

from ..registry import chunker
from ..text import Span, paragraph_spans


@chunker("paragraph")
def paragraph(text: str) -> list[Span]:
    return paragraph_spans(text)
