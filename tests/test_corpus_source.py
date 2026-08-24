"""BenchmarkCorpus must be a pure extraction of the path it replaces.

Units byte-identical and corpus_identity equal to today's corpus_token, or every
cached run silently invalidates -- the gate would pass while the cache misses,
which reads as "the refactor was free" and costs a full re-embed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.evaluation.corpus_source import BenchmarkCorpus, CorpusSource
from src.evaluation.run_retrieval import corpus_token

WORDS = " ".join(f"w{i}" for i in range(300))


def doc(doc_id: str, text: str = WORDS, modality: str = "Text") -> SimpleNamespace:
    return SimpleNamespace(doc_id=doc_id, text=text, page="1", modality=modality,
                           meta={"title": "Doc", "subset": "sub"})


def bench(docs: list, identity: str | None = None) -> SimpleNamespace:
    b = SimpleNamespace(corpus=lambda: iter(docs))
    if identity is not None:
        b.corpus_identity = lambda: identity
    return b


def test_units_carry_the_document_identity():
    records = list(BenchmarkCorpus(bench([doc("sub::d#page=1")])).units())
    assert [r.doc_id for r in records] == ["sub::d#page=1"]
    assert [r.chunk_id for r in records] == ["sub::d#page=1"]


def test_units_carry_modality_and_page():
    record = next(iter(BenchmarkCorpus(bench([doc("a", modality="Chart")])).units()))
    assert record.meta["modality"] == "Chart"
    assert record.page == "1"


def test_units_carry_the_benchmark_meta():
    record = next(iter(BenchmarkCorpus(bench([doc("a")])).units()))
    assert record.meta["subset"] == "sub"


def test_blank_units_are_dropped():
    records = list(BenchmarkCorpus(bench([doc("a", text="  "), doc("b")])).units())
    assert [r.doc_id for r in records] == ["b"]


def test_corpus_identity_matches_corpus_token_exactly():
    """The cache key. Any drift here invalidates every cached run."""
    b = bench([doc("a")], identity="physics-french")
    assert BenchmarkCorpus(b).corpus_identity() == corpus_token(b)
    assert BenchmarkCorpus(b).corpus_identity() == "physics-french"


def test_corpus_identity_falls_back_to_builtin():
    b = bench([doc("a")])
    assert BenchmarkCorpus(b).corpus_identity() == corpus_token(b) == "builtin"


def test_corpus_identity_hashes_a_corpus_file(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"doc_id": "a"}\n')
    b = bench([doc("a")])
    b.corpus_path = path
    assert BenchmarkCorpus(b).corpus_identity() == corpus_token(b)
    assert BenchmarkCorpus(b).corpus_identity().startswith("corpus-")


def test_chunking_happens_inside_the_source():
    source = BenchmarkCorpus(bench([doc("sub::d#page=1")]),
                             chunker="fixed_overlap", params={"n_words": 40, "overlap": 8})
    assert len(list(source.units())) > 1


def test_chunked_units_keep_modality():
    source = BenchmarkCorpus(bench([doc("sub::d#page=1", modality="Chart")]),
                             chunker="fixed_overlap", params={"n_words": 40, "overlap": 8})
    assert {r.meta["modality"] for r in source.units()} == {"Chart"}


def test_chunking_does_not_change_corpus_identity():
    """The chunker is its own key in index_id; folding it in here would double-count."""
    b = bench([doc("a")], identity="physics-french")
    plain = BenchmarkCorpus(b).corpus_identity()
    chunked = BenchmarkCorpus(b, chunker="fixed_overlap",
                              params={"n_words": 40, "overlap": 8}).corpus_identity()
    assert plain == chunked


def test_benchmark_corpus_satisfies_the_protocol():
    assert isinstance(BenchmarkCorpus(bench([doc("a")])), CorpusSource)
