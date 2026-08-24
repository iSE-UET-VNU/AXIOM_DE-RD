"""A corpus that ships text but no blocks must chunk, not vanish.

ViDoRe ships one markdown string per page and no block structure. The chunker
read only ``blocks``, so every chunked arm indexed nothing and scored 0.0
without raising -- a number where an error belonged.
"""

from __future__ import annotations

import pytest

from src.evaluation.chunking import chunk_corpus, chunk_document

PARAMS = {"n_words": 40, "overlap": 8}
WORDS = " ".join(f"w{i}" for i in range(300))


def page(text: str, blocks: list | None = None) -> dict:
    return {"doc_id": "sub::doc#page=1", "title": "Doc", "text": text, "blocks": blocks or []}


def test_a_blockless_page_still_chunks():
    chunks = chunk_document(page(WORDS), "fixed_overlap", PARAMS)
    assert chunks, "text-only page produced no chunks"


def test_the_chunks_carry_the_page_text():
    chunks = chunk_document(page(WORDS), "fixed_overlap", PARAMS)
    assert "w0" in chunks[0].text
    assert "w299" in chunks[-1].text


def test_blocks_still_win_when_present():
    """The fallback must not override a corpus that does ship structure."""
    blocks = [{"text": "from the blocks"}]
    chunks = chunk_document(page("from the text field", blocks), "fixed_overlap", PARAMS)
    assert "blocks" in chunks[0].text
    assert "text field" not in chunks[0].text


def test_an_empty_page_still_produces_nothing():
    assert chunk_document(page("   "), "fixed_overlap", PARAMS) == []


def test_prefix_applies_to_a_blockless_page():
    chunks = chunk_document(page(WORDS), "fixed_overlap", PARAMS, prefix=True)
    assert chunks[0].text.startswith("Doc")


def test_a_blockless_corpus_chunks_every_document():
    docs = [page(WORDS) | {"doc_id": f"sub::doc#page={i}"} for i in range(5)]
    chunks = chunk_corpus(docs, "fixed_overlap", PARAMS)
    assert len({c.doc_id for c in chunks}) == 5


@pytest.mark.parametrize("strategy", ["fixed_overlap", "blocks"])
def test_chunk_ids_stay_unique_per_document(strategy):
    chunks = chunk_document(page(WORDS, [{"text": WORDS}]), strategy, PARAMS)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
