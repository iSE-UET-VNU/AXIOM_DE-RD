"""Built-in text chunking strategies."""

from __future__ import annotations

from ..fields import Span, paragraph_spans, sentence_spans, word_windows
from ..registry import chunker


@chunker("fixed_overlap", aliases=("fixed_512_ol",))
def fixed_overlap(text: str, n_words: int = 512, overlap: int = 128) -> list[Span]:
    """Split text into overlapping fixed-size word windows."""
    return word_windows(text, n_words=n_words, overlap=overlap)


@chunker("paragraph")
def paragraph(text: str) -> list[Span]:
    """Split text on blank-line paragraph boundaries."""
    return paragraph_spans(text)


@chunker("sentence_group", aliases=("sentence_5",))
def sentence_group(text: str, n: int = 5) -> list[Span]:
    """Group a fixed number of consecutive sentences."""
    spans = sentence_spans(text)
    return [
        (spans[index][0], spans[min(index + n, len(spans)) - 1][1])
        for index in range(0, len(spans), n)
    ]


@chunker("recursive", aliases=("recursive_400",))
def recursive(text: str, target: int = 400, hard_max: int | None = None) -> list[Span]:
    """Recursively pack paragraphs and sentences under a character target."""
    if not text.strip():
        return []
    hard_max = hard_max or target * 2
    spans: list[Span] = []
    current_start: int | None = None
    current_end: int | None = None

    for paragraph_start, paragraph_end in paragraph_spans(text):
        if paragraph_end - paragraph_start > hard_max:
            _flush(spans, current_start, current_end)
            current_start, current_end = None, None
            spans.extend(
                _pack_sentences(text, paragraph_start, paragraph_end, target, hard_max)
            )
            continue
        if current_start is None:
            current_start, current_end = paragraph_start, paragraph_end
            continue
        assert current_end is not None
        if paragraph_end - current_start <= target:
            current_end = paragraph_end
        else:
            spans.append((current_start, current_end))
            current_start, current_end = paragraph_start, paragraph_end

    _flush(spans, current_start, current_end)
    return spans


def _pack_sentences(
    text: str,
    start: int,
    end: int,
    target: int,
    hard_max: int,
) -> list[Span]:
    spans: list[Span] = []
    current_start: int | None = None
    current_end: int | None = None
    for relative_start, relative_end in sentence_spans(text[start:end]):
        sentence_start = start + relative_start
        sentence_end = start + relative_end
        if sentence_end - sentence_start > hard_max:
            _flush(spans, current_start, current_end)
            current_start, current_end = None, None
            spans.extend(_hard_split(sentence_start, sentence_end, hard_max))
            continue
        if current_start is None:
            current_start, current_end = sentence_start, sentence_end
            continue
        assert current_end is not None
        if sentence_end - current_start <= target:
            current_end = sentence_end
        else:
            spans.append((current_start, current_end))
            current_start, current_end = sentence_start, sentence_end
    _flush(spans, current_start, current_end)
    return spans


def _hard_split(start: int, end: int, hard_max: int) -> list[Span]:
    return [(index, min(index + hard_max, end)) for index in range(start, end, hard_max)]


def _flush(spans: list[Span], start: int | None, end: int | None) -> None:
    if start is not None and end is not None and start < end:
        spans.append((start, end))


fixed_512_ol = fixed_overlap
sentence_5 = sentence_group
recursive_400 = recursive

__all__ = [
    "fixed_512_ol",
    "fixed_overlap",
    "paragraph",
    "recursive",
    "recursive_400",
    "sentence_5",
    "sentence_group",
]
