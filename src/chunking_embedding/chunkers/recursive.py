"""Structure baseline: recursively pack paragraphs/sentences under a char target."""

from __future__ import annotations

from ..registry import chunker
from ..text import Span, paragraph_spans, sentence_spans


@chunker("recursive", aliases=("recursive_400",))
def recursive(text: str, target: int = 400, hard_max: int | None = None) -> list[Span]:
    if not text.strip():
        return []
    hard_max = hard_max or target * 2
    spans: list[Span] = []
    cur_start: int | None = None
    cur_end: int | None = None

    for para_start, para_end in paragraph_spans(text):
        if para_end - para_start > hard_max:
            _flush(spans, cur_start, cur_end)
            cur_start, cur_end = None, None
            spans.extend(_pack_sentences(text, para_start, para_end, target, hard_max))
            continue
        if cur_start is None:
            cur_start, cur_end = para_start, para_end
            continue
        assert cur_end is not None
        if para_end - cur_start <= target:
            cur_end = para_end
        else:
            spans.append((cur_start, cur_end))
            cur_start, cur_end = para_start, para_end

    _flush(spans, cur_start, cur_end)
    return spans


def _pack_sentences(text: str, start: int, end: int, target: int, hard_max: int) -> list[Span]:
    spans: list[Span] = []
    cur_start: int | None = None
    cur_end: int | None = None
    for rel_start, rel_end in sentence_spans(text[start:end]):
        sent_start, sent_end = start + rel_start, start + rel_end
        if sent_end - sent_start > hard_max:
            _flush(spans, cur_start, cur_end)
            cur_start, cur_end = None, None
            spans.extend(_hard_split(sent_start, sent_end, hard_max))
            continue
        if cur_start is None:
            cur_start, cur_end = sent_start, sent_end
            continue
        assert cur_end is not None
        if sent_end - cur_start <= target:
            cur_end = sent_end
        else:
            spans.append((cur_start, cur_end))
            cur_start, cur_end = sent_start, sent_end
    _flush(spans, cur_start, cur_end)
    return spans


def _hard_split(start: int, end: int, hard_max: int) -> list[Span]:
    return [(i, min(i + hard_max, end)) for i in range(start, end, hard_max)]


def _flush(spans: list[Span], start: int | None, end: int | None) -> None:
    if start is not None and end is not None and start < end:
        spans.append((start, end))


recursive_400 = recursive
