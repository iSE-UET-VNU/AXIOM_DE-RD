"""Input sanitization shared by the embedders.

The embedding API rejects inputs over 8191 tokens. Truncate on real token counts.
"""

from __future__ import annotations

from typing import Any, Sequence
import logging

logger = logging.getLogger(__name__)

# Hiragana/Katakana, CJK ext A, CJK Unified, Hangul, CJK compatibility.
_CJK_RANGES = ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
               (0xAC00, 0xD7AF), (0xF900, 0xFAFF))
CJK_RATIO_THRESHOLD = 0.20
# Conservative enough to hold even at the worst observed 1.17 tokens/char.
CJK_CHAR_LIMIT = 6_000
DEFAULT_CHAR_LIMIT = 24_000
MAX_TOKENS = 8_000
# Providers cap tokens per request as well as per input; stay clear of both.
MAX_REQUEST_TOKENS = 250_000
TOKEN_CHECK_CHARS = 3_000
ENCODING = "cl100k_base"

_encoder: Any | None = None
_encoder_loaded = False


def _get_encoder() -> Any | None:
    global _encoder, _encoder_loaded
    if not _encoder_loaded:
        _encoder_loaded = True
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding(ENCODING)
        except Exception:
            _encoder = None
            logger.warning(
                "tiktoken unavailable; falling back to estimated token counts. "
                "Batches will be smaller than necessary. Install tiktoken for exact counts."
            )
    return _encoder


def token_batches(
    indices: Sequence[int],
    texts: Sequence[str],
    batch_size: int,
    max_request_tokens: int = MAX_REQUEST_TOKENS,
) -> tuple[list[list[int]], int]:
    """Group ``indices`` so each request fits both the input-count and the
    tokens-per-request cap.  Returns the batches and the total token count.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    budget = 0
    total = 0
    for index in indices:
        tokens = count_tokens(texts[index])
        total += tokens
        if current and (len(current) >= batch_size or budget + tokens > max_request_tokens):
            batches.append(current)
            current, budget = [], 0
        current.append(index)
        budget += tokens
    if current:
        batches.append(current)
    return batches, total


def count_tokens(text: str) -> int:
    """Exact count when tiktoken is available, else an upper bound.
    """
    encoder = _get_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    cjk = sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES))
    return max(1, int(cjk * 1.5 + (len(text) - cjk) * 0.5) + 1)


def sanitize_text(text: str) -> str:
    """truncate to stay under the embedding API's 8191-token cap."""
    if not text or not text.strip():
        return " "
    cjk = sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES))
    limit = CJK_CHAR_LIMIT if cjk / len(text) > CJK_RATIO_THRESHOLD else DEFAULT_CHAR_LIMIT
    clipped = text[:limit]
    if len(clipped) < TOKEN_CHECK_CHARS:
        return clipped
    encoder = _get_encoder()
    if encoder is None:
        return clipped
    tokens = encoder.encode(clipped)
    if len(tokens) <= MAX_TOKENS:
        return clipped
    return encoder.decode(tokens[:MAX_TOKENS])
