"""Input sanitization shared by the embedders."""

from __future__ import annotations

# Hiragana/Katakana, CJK ext A, CJK Unified, Hangul, CJK compatibility.
_CJK_RANGES = ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
               (0xAC00, 0xD7AF), (0xF900, 0xFAFF))
CJK_RATIO_THRESHOLD = 0.20
CJK_CHAR_LIMIT = 7_000
DEFAULT_CHAR_LIMIT = 30_000


def sanitize_text(text: str) -> str:
    """Empty -> " "; truncate under the embedding API's ~8191-token cap, tighter
    for CJK-heavy text where tokens run about one per character."""
    if not text or not text.strip():
        return " "
    cjk = sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES))
    limit = CJK_CHAR_LIMIT if cjk / len(text) > CJK_RATIO_THRESHOLD else DEFAULT_CHAR_LIMIT
    return text[:limit]
