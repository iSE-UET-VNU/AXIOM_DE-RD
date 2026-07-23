"""Structure baseline: fixed word windows with overlap."""

from __future__ import annotations

from ..registry import chunker
from ..text import Span, word_windows


@chunker("fixed_overlap", aliases=("fixed_512_ol",))
def fixed_overlap(text: str, n_words: int = 512, overlap: int = 128) -> list[Span]:
    return word_windows(text, n_words=n_words, overlap=overlap)


fixed_512_ol = fixed_overlap
