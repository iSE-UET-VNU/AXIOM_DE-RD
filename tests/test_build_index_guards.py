"""An index with nothing in it must raise, and chunking must not lose modality.

An empty index scores 0.0 on every question without error, which is the most
believable possible result for "this arm is worse". The chunked branch also
dropped the modality tag, silently emptying the per-modality breakdown.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.evaluation.run_retrieval import build_index

WORDS = " ".join(f"w{i}" for i in range(300))


def doc(doc_id: str, text: str, modality: str = "Text") -> SimpleNamespace:
    return SimpleNamespace(
        doc_id=doc_id, text=text, page="1", modality=modality,
        meta={"title": "Doc", "subset": "sub"},
    )


def bench(docs: list) -> SimpleNamespace:
    return SimpleNamespace(corpus=lambda: iter(docs))


def test_an_empty_corpus_raises():
    with pytest.raises(SystemExit, match="no indexable"):
        build_index(bench([]), None, "idx")


def test_a_corpus_of_blank_text_raises():
    with pytest.raises(SystemExit, match="no indexable"):
        build_index(bench([doc("a", "   "), doc("b", "")]), None, "idx")


def test_the_refusal_names_the_chunker():
    """A chunker that silently drops every document is the case worth naming."""
    with pytest.raises(SystemExit, match="fixed_overlap"):
        build_index(bench([doc("a", "  ")]), None, "idx", chunker="fixed_overlap",
                    params={"n_words": 40, "overlap": 8})


def test_a_chunked_index_is_not_empty():
    index = build_index(bench([doc("sub::d#page=1", WORDS)]), None, "idx",
                        chunker="fixed_overlap", params={"n_words": 40, "overlap": 8})
    assert len(index.records) > 1


def test_chunked_records_keep_their_modality():
    index = build_index(bench([doc("sub::d#page=1", WORDS, modality="Chart")]), None, "idx",
                        chunker="fixed_overlap", params={"n_words": 40, "overlap": 8})
    assert {c.meta.get("modality") for c in index.records} == {"Chart"}


def test_unchunked_records_keep_their_modality():
    index = build_index(bench([doc("sub::d#page=1", WORDS, modality="Chart")]), None, "idx")
    assert {c.meta.get("modality") for c in index.records} == {"Chart"}
