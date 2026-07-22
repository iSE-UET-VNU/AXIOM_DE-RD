"""Character-span text primitives shared by the chunkers."""

from __future__ import annotations

import re

Span = tuple[int, int]

SENTENCE_END_CHARS = ".!?。！？:\"')"

_WORD_RE = re.compile(r"\S+")
_SENT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")
_PARA_RE = re.compile(r"\n{2,}")


def sentence_spans(text: str) -> list[Span]:
    return _split_spans(text, _SENT_RE)


def paragraph_spans(text: str) -> list[Span]:
    return _split_spans(text, _PARA_RE)


def find_span(text: str, piece: str, cursor: int) -> tuple[Span | None, int]:
    """Map a library splitter's returned substring back to a source char span.

    Splitters return verbatim substrings modulo whitespace/joins; anchor on the
    head and tail when an exact match past the cursor is unavailable.
    """
    s = piece.strip()
    if not s:
        return None, cursor
    i = text.find(s, cursor)
    if i >= 0:
        return (i, i + len(s)), i + len(s)
    i = text.find(s[:60], cursor)
    if i < 0:
        return None, cursor
    j = text.find(s[-40:], i)
    j = j + 40 if j >= 0 else min(len(text), i + len(s))
    return (i, j), j


def word_windows(text: str, n_words: int, overlap: int) -> list[Span]:
    words = list(_WORD_RE.finditer(text))
    if not words:
        return []
    spans: list[Span] = []
    step = max(1, n_words - overlap)
    i = 0
    while i < len(words):
        j = min(i + n_words, len(words))
        spans.append((words[i].start(), words[j - 1].end()))
        if j == len(words):
            break
        i += step
    return spans


def _split_spans(text: str, pattern: re.Pattern[str]) -> list[Span]:
    out: list[Span] = []
    pos = 0
    for match in pattern.finditer(text):
        if text[pos:match.start()].strip():
            out.append((pos, match.start()))
        pos = match.end()
    if text[pos:].strip():
        out.append((pos, len(text)))
    return out
