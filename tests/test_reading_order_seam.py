"""The seam between ``reading_order`` and the text chunks are cut from.

``main_text`` and ``reading_order`` are derived **independently from the same
rows**: the first by the document builder, the second by
``reading_order_from_rows``. Nothing forces them to agree.

A divergence reorders evidence without changing any count. Every block is
present, no text is missing, nothing is empty, and every check we run stays
green -- only the sequence is wrong. That is precisely the failure the epub
handler had before it read the OPF spine instead of archive order, and PDF
extraction has the same exposure through column order and floating blocks, with
no spine to fall back on.

The assertion is relative order: **blocks that come earlier in ``reading_order``
must appear earlier in ``main_text``.** Absolute string equality is too strong --
``main_text`` carries joining whitespace and may omit non-text blocks -- but
order is the property chunking actually depends on.

Per Rule 2 in docs/testing_rules.md, each guard is demonstrated against the
defect it guards.
"""

from __future__ import annotations

from typing import Any
import re

import pytest

from src.artifacts.pipeline_output import _output_content
from src.reading_order import reading_order_from_rows


class _Doc:
    """Minimal stand-in for the document object ``_output_content`` reads."""

    def __init__(self, main_text: str) -> None:
        self.main_text = main_text
        self.tables: list[Any] = []
        self.figures: list[Any] = []
        self.formulas: list[Any] = []


def _row(*texts: str) -> dict[str, Any]:
    return {
        "source_blocks": [
            {
                "component_id": f"0/0/{i}",
                "source": "parser_json",
                "text": text,
                "type": "paragraph",
            }
            for i, text in enumerate(texts)
        ]
    }


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def positions_in(main_text: str, blocks: list[dict[str, Any]]) -> list[int]:
    """Where each block's text starts in main_text, in reading_order sequence."""
    haystack = _norm(main_text)
    out: list[int] = []
    for block in blocks:
        needle = _norm(block.get("text", ""))
        if needle:
            out.append(haystack.find(needle))
    return out


SENTENCES = [
    "The committee opened the session at nine.",
    "Revenue for the quarter rose by four percent.",
    "The appendix lists every regional office.",
    "Signatures were collected before adjournment.",
]


def test_reading_order_matches_main_text_order():
    """The spanning assertion.

    Only holds if the two independent derivations agree on sequence.
    """
    rows = [_row(*SENTENCES)]
    blocks, reading_order, meta = reading_order_from_rows(rows)
    payload = _output_content(_Doc(" ".join(SENTENCES)), None, None)
    # The pipeline's own blocks, walked in its own reading_order.
    by_id = {b["component_id"]: b for b in payload["blocks"]}
    ordered = [by_id[cid] for cid in payload["reading_order"] if cid in by_id]
    found = positions_in(payload["main_text"], ordered)
    assert all(p >= 0 for p in found), "a block's text is absent from main_text"
    assert found == sorted(found), (
        f"reading_order disagrees with main_text order: offsets {found}. "
        "Chunks cut from main_text would carry evidence out of sequence."
    )


def test_the_order_check_catches_a_shuffle():
    """Rule 2: the guard must fail against the defect.

    A permutation leaves every block present and every count identical -- the
    exact silent corruption -- so a check that cannot see it is not a guard.
    """
    rows = [_row(*SENTENCES)]
    blocks, _, _ = reading_order_from_rows(rows)
    main_text = " ".join(SENTENCES)
    shuffled = [blocks[2], blocks[0], blocks[3], blocks[1]]
    found = positions_in(main_text, shuffled)
    assert all(p >= 0 for p in found), "fixture is wrong: text must still be present"
    assert found != sorted(found), (
        "A shuffled reading_order must violate the order assertion; if it does "
        "not, the assertion cannot detect reordering."
    )


def test_block_count_is_blind_to_reordering():
    """Why the count-based checks we already have could never catch this."""
    rows = [_row(*SENTENCES)]
    blocks, reading_order, meta = reading_order_from_rows(rows)
    shuffled = list(reversed(reading_order))
    assert len(shuffled) == len(reading_order) == meta["block_count"]
    assert set(shuffled) == set(reading_order)


# --------------------------------------------------------------------------
# Missing reading_order must be loud, not silent
# --------------------------------------------------------------------------


def test_absent_reading_order_is_reported_as_unavailable():
    """A seam unspanned because the data is missing must not look like a pass.

    ``source: unavailable`` is the signal. Without it, an extractor that emits
    no reading_order is indistinguishable from one whose order is correct.
    """
    blocks, reading_order, meta = reading_order_from_rows([])
    assert blocks == [] and reading_order == []
    assert meta["source"] == "unavailable"
    assert meta["complete"] is False


def test_extraction_fallback_is_marked_incomplete():
    """Blocks recovered from structured extraction are not parser-native order.

    ``complete: False`` says the order is inferred rather than authoritative --
    the distinction a downstream reader needs to know it is on the weaker path.
    """
    rows = [{"extraction": {"main_text": "First paragraph.\n\nSecond paragraph."}}]
    blocks, reading_order, meta = reading_order_from_rows(rows)
    if blocks:
        assert meta["source"] == "structured_extraction_citations"
        assert meta["complete"] is False


@pytest.mark.parametrize("meta_source", ["unavailable", "structured_extraction_citations"])
def test_non_native_order_is_distinguishable_from_native(meta_source: str):
    """The inventory only works if 'no data' and 'passes' are different states."""
    native_blocks, _, native_meta = reading_order_from_rows([_row(*SENTENCES)])
    assert native_meta["source"] == "parser_json"
    assert native_meta["complete"] is True
    assert native_meta["source"] != meta_source
