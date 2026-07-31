"""Lexical statistics used by BM25 retrieval.

Term frequency and chunk length are stored with each retrieval item. Document
statistics are derived from those item-level payloads and stored in that same
document, so every output file is self-contained for BM25 retrieval.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
import re
import unicodedata


LEXICAL_CONTRACT_VERSION = "lexical-v1"
CORPUS_STATS_CONTRACT_VERSION = "bm25-corpus-v1"
ANALYZER_NAME = "unicode_word_v1"

# Unicode letters and digits, optionally joined by an apostrophe. Underscores
# are separators so identifiers such as ``invoice_number`` become two terms.
_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", flags=re.UNICODE)


def analyze(text: str) -> list[str]:
    """Normalize and tokenize text deterministically for indexing and queries."""
    normalized = unicodedata.normalize("NFC", str(text)).casefold()
    normalized = normalized.replace("\u2019", "'")
    return _TOKEN_PATTERN.findall(normalized)


def build_lexical_payload(text: str) -> dict[str, Any] | None:
    """Return item-level raw term frequencies and token length.

    Empty text has no lexical payload and is therefore excluded from the BM25
    corpus statistics.
    """
    tokens = analyze(text)
    if not tokens:
        return None
    frequencies = Counter(tokens)
    return {
        "contract_version": LEXICAL_CONTRACT_VERSION,
        "analyzer": ANALYZER_NAME,
        "tf": dict(sorted(frequencies.items())),
        "dl": len(tokens),
    }


def content_text(value: Any) -> str:
    """Flatten consumer retrieval content into analyzer input text.

    Mapping keys are structural labels and are intentionally excluded; string
    and numeric values retain their input order.
    """
    parts: list[str] = []

    def visit(child: Any) -> None:
        if child is None or isinstance(child, bool):
            return
        if isinstance(child, str):
            if child.strip():
                parts.append(child)
            return
        if isinstance(child, (int, float)):
            parts.append(str(child))
            return
        if isinstance(child, Mapping):
            for nested in child.values():
                visit(nested)
            return
        if isinstance(child, Sequence) and not isinstance(child, (bytes, bytearray)):
            for nested in child:
                visit(nested)

    visit(value)
    return "\n".join(parts)


def build_corpus_statistics(
    lexical_payloads: Iterable[Mapping[str, Any] | None],
    *,
    scope: str = "document",
) -> dict[str, Any]:
    """Aggregate DF, N, and average chunk length for one BM25 scope."""
    document_frequency: Counter[str] = Counter()
    total_dl = 0
    chunk_count = 0

    for lexical in lexical_payloads:
        if not isinstance(lexical, Mapping):
            continue
        tf = lexical.get("tf")
        dl = lexical.get("dl")
        if not isinstance(tf, Mapping) or not isinstance(dl, int) or dl < 0:
            continue

        chunk_count += 1
        total_dl += dl
        # DF counts a term at most once per chunk, regardless of its TF.
        document_frequency.update(str(term) for term in tf)

    return {
        "contract_version": CORPUS_STATS_CONTRACT_VERSION,
        "scope": scope,
        "analyzer": ANALYZER_NAME,
        "N": chunk_count,
        "avgdl": round(total_dl / chunk_count, 6) if chunk_count else 0.0,
        "df": dict(sorted(document_frequency.items())),
    }
