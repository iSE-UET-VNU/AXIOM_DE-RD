"""Scope must be honoured by scoring, not by filtering afterwards.

The bug this pins: BM25 over-fetched a global top-N and filtered to scope after,
so whenever a scope's units missed that window it returned fewer than k and
under-retrieved. It cost ~6.5pp on MMDocIR page-level recall and produced a
plausible number rather than an error.

Fusion arms are the dangerous case. If one leg scopes correctly and the other
truncates, the fused ranking inherits the weaker leg's truncation silently, and
the contamination flatters whichever leg is correct.

Every arm is audited by the same property: **given a scope, an arm must return
min(k, units_in_scope) hits, all in scope** -- never fewer because of where the
scope sat in a global ranking.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval import retrievers
from src.retrieval.index import LocalIndex
from src.retrieval.protocol import ChunkRecord
from src.retrieval.sparse import BM25Index

ARMS = ["bm25", "dense", "rrf", "alpha0.7"]
FILLER = "annual report commentary appendix notes disclosure section paragraph"


def _index(n_docs: int = 120, per_doc: int = 5) -> LocalIndex:
    """A corpus where the scoped document ranks BELOW the over-fetch window.

    The fixture has to be adversarial or the audit proves nothing: an earlier
    version gave every unit the same filler, so ties put the scoped document
    inside a 200-unit window and the buggy path passed too.

    Here ``doc0`` mentions the query terms **once**, while every other document
    repeats them five times. BM25 term frequency ranks all 595 other units above
    doc0's 5, so a global top-200 window cannot contain them. Post-filtering that
    window returns zero; scoping inside the scorer returns all five.
    """
    records = []
    for d in range(n_docs):
        for c in range(per_doc):
            # doc0 is deliberately the weakest match in the collection.
            body = FILLER if d == 0 else " ".join([FILLER] * 5)
            records.append(
                ChunkRecord(chunk_id=f"d{d}c{c}", doc_id=f"doc{d}", text=f"{body} unit {c}")
            )
    rng = np.random.default_rng(3)
    return LocalIndex(
        "audit",
        records,
        bm25=BM25Index("auto").build(
            [{"chunk_id": r.chunk_id, "doc_id": r.doc_id, "text": r.text} for r in records]
        ),
        vectors=rng.normal(size=(len(records), 16)).astype("float32"),
        embedder=type(
            "E", (), {"embed": lambda self, t: rng.normal(size=(len(t), 16)).astype("float32")}
        )(),
    )


@pytest.mark.parametrize("arm", ARMS)
def test_scope_returns_full_k_even_when_scope_ranks_low_globally(arm: str):
    """The exact failure: the scoped document is buried in the global ranking."""
    index = _index()
    scope = ["doc0"]  # 5 units, all scoring low on the filler query
    hits = retrievers.build(arm, index).retrieve(FILLER, 5, scope=scope)
    assert len(hits) == 5, f"{arm} under-retrieved: {len(hits)}/5 in-scope units"
    assert {h.doc_id for h in hits} == {"doc0"}


@pytest.mark.parametrize("arm", ARMS)
def test_scope_caps_at_units_available(arm: str):
    """Asking for more than the scope holds returns everything in it, not fewer."""
    index = _index()
    hits = retrievers.build(arm, index).retrieve(FILLER, 50, scope=["doc3"])
    assert len(hits) == 5
    assert {h.doc_id for h in hits} == {"doc3"}


@pytest.mark.parametrize("arm", ARMS)
def test_multi_document_scope(arm: str):
    index = _index()
    scope = ["doc1", "doc2"]
    hits = retrievers.build(arm, index).retrieve(FILLER, 10, scope=scope)
    assert len(hits) == 10
    assert {h.doc_id for h in hits} <= set(scope)


@pytest.mark.parametrize("arm", ARMS)
def test_unscoped_is_unaffected(arm: str):
    index = _index()
    hits = retrievers.build(arm, index).retrieve(FILLER, 10, scope=None)
    assert len(hits) == 10


def test_fusion_legs_are_both_scoped():
    """A fused arm must not inherit an unscoped leg.

    Checked by construction rather than by reading: every chunk RRF returns must
    also be reachable from both of its legs under the same scope.
    """
    index = _index()
    scope = ["doc0"]
    fused = retrievers.RrfRetriever(index)
    sparse_ids = {c for c, _ in fused.sparse.raw(FILLER, 100, scope)}
    dense_ids = {c for c, _ in fused.dense.raw(FILLER, 100, scope)}
    in_scope = {r.chunk_id for r in index.records if r.doc_id == "doc0"}
    assert sparse_ids <= in_scope and sparse_ids
    assert dense_ids <= in_scope and dense_ids
    assert {h.chunk_id for h in fused.retrieve(FILLER, 5, scope)} <= in_scope


def test_scope_positions_cache_is_correct_not_just_fast():
    index = _index()
    first = index.scope_positions(["doc0", "doc1"])
    second = index.scope_positions(["doc1", "doc0"])
    assert first == second
    assert all(index.record_at(p).doc_id in {"doc0", "doc1"} for p in first)
    assert len(first) == 10
