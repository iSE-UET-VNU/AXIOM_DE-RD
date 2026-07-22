"""Structure baseline: group a fixed number of sentences."""

from __future__ import annotations

from ..registry import chunker
from ..text import Span, sentence_spans


@chunker("sentence_group", aliases=("sentence_5",))
def sentence_group(text: str, n: int = 5) -> list[Span]:
    ss = sentence_spans(text)
    return [(ss[i][0], ss[min(i + n, len(ss)) - 1][1]) for i in range(0, len(ss), n)]


sentence_5 = sentence_group
