"""Analyzer symmetry, the run-record contract, and the harness round trip.

Each test here pins a failure that was measured, not imagined. The analyzer
cases in particular reproduce two silent-zero bugs: bigrams applied on only one
side, and an analyzer chosen per text rather than per index.
"""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from src.chunking_embedding.lexical import analyze, build_lexical_payload
from src.retrieval import runs
from src.retrieval.protocol import QueryEncoding, ScoredChunk, UnsupportedEncoding
from src.retrieval.sparse import BM25Index, analyze_cjk, analyze_french, resolve_analyzer

MIXED = (
    "Annual fire safety maintenance record for the building. The department "
    "reviewed all equipment. Reference: 建筑消防设施故障维修记录表. All findings "
    "were logged and the report was filed with the authority."
)


def build_index(text: str = MIXED) -> BM25Index:
    return BM25Index(analyzer_name="auto").build(
        [{"chunk_id": "c1", "doc_id": "d1", "text": text}]
    )


# --------------------------------------------------------------------------
# Analyzer symmetry
# --------------------------------------------------------------------------


def test_query_side_bigrams_cannot_match_whole_phrase_storage():
    """Why we build our own index instead of scoring corpus-service's tf.

    The pipeline writes tf with the whole-phrase analyzer. Splitting only the
    query into bigrams produces an empty intersection, so the CJK sparse leg
    would score zero while reporting success.
    """
    stored = build_lexical_payload("建筑消防设施故障维修记录表")["tf"]
    assert set(analyze_cjk("消防设施")) & set(stored) == set()


def test_analyzer_is_resolved_per_index_not_per_text():
    """A corpus with any CJK indexes and queries with the same tokenizer."""
    index = build_index()
    assert index.analyzer_name == "cjk_bigram"
    assert index.search("消防设施", 5), "CJK query must reach a CJK chunk"
    assert index.search("fire safety", 5), "Latin query must still work"


def test_presence_not_proportion_decides():
    """The mixed document is only ~7% CJK; a ratio threshold would miss it."""
    assert resolve_analyzer([MIXED]) == "cjk_bigram"
    assert resolve_analyzer(["fire safety maintenance record"]) == "plain"


@pytest.mark.parametrize(
    "text",
    ["fire safety maintenance", "Đại học Công nghệ tuyển sinh", "invoice_number 2026"],
)
def test_bigrams_leave_non_cjk_identical(text: str):
    """Bigrams are safe to apply corpus-wide: they only split CJK-majority tokens."""
    assert analyze(text) == analyze_cjk(text)


def test_elision_hides_the_noun_from_the_plain_analyzer():
    """The measured French loss: ``l'énergie`` is one token, so ``énergie`` misses.

    Worth +1.31 NDCG@10 on ViDoRe V3 physics on its own, more than stopwords and
    stemming together.
    """
    assert "énergie" not in analyze("L'énergie de l'atome")
    assert "énerg" in analyze_french("L'énergie de l'atome")


def test_french_analyzer_matches_query_to_document():
    index = BM25Index(analyzer_name="french").build(
        [{"chunk_id": "c1", "doc_id": "d1", "text": "L'énergie de l'atome est quantifiée."}]
    )
    assert index.search("énergie quantifiée", 5)


def test_unresolved_analyzer_fails_loudly(tmp_path: Path):
    with pytest.raises(ValueError, match="never resolved"):
        _ = BM25Index(analyzer_name="auto").analyzer


def test_resolved_analyzer_survives_save_and_load(tmp_path: Path):
    """An index that forgot its analyzer would tokenize queries differently."""
    path = tmp_path / "bm25.json"
    build_index().save(path)
    reloaded = BM25Index.load(path)
    assert reloaded.analyzer_name == "cjk_bigram"
    assert reloaded.search("消防设施", 5)


# --------------------------------------------------------------------------
# Query encoding
# --------------------------------------------------------------------------


def test_missing_encoding_raises_instead_of_falling_back():
    encoding = QueryEncoding(text="q", dense=[0.1, 0.2])
    assert encoding.require("dense") == [0.1, 0.2]
    with pytest.raises(UnsupportedEncoding):
        encoding.require("multi_vector")


# --------------------------------------------------------------------------
# Run records
# --------------------------------------------------------------------------


def test_params_hash_ignores_key_order():
    assert runs.params_hash({"k1": 1.2, "b": 0.75}) == runs.params_hash({"b": 0.75, "k1": 1.2})


def test_query_set_hash_distinguishes_subsets_but_not_order():
    assert runs.query_set_hash(["1", "2", "3"]) == runs.query_set_hash(["3", "1", "2"])
    assert runs.query_set_hash(["1", "2"]) != runs.query_set_hash(["1", "2", "3"])


def test_run_record_round_trip(tmp_path: Path):
    record = runs.RunRecord.build(
        "1", "câu hỏi", "bm25", "idx", "abc123",
        [ScoredChunk("c1", "d1", 1.5, 1, "text")],
    )
    path = tmp_path / "run.jsonl"
    runs.write(path, [record])
    assert runs.read(path)[0] == record


def test_run_record_is_consumed_by_the_answer_harness(tmp_path: Path):
    """The contract that matters: one shape, both measurement paths.

    ``run_answer.load_run`` is the harness side; if this breaks, the in-memory
    and Methods-Hub runs are no longer measured by identical code.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # research/ is not distributed with the module -- it imports src/, never the
    # reverse. A skip here is visible in the pytest summary; a silent pass is
    # what docs/testing_rules.md forbids, so do not stub load_run.
    pytest.importorskip(
        "src.evaluation.run_answer",
        reason="reader side of this seam lives in research/, absent from this checkout",
    )
    from src.evaluation.run_answer import load_run

    path = tmp_path / "run.jsonl"
    # Adversarial by construction: every field holds a distinct value, chunk_id
    # and doc_id differ from each other, and rank order is NOT the order the
    # values sort in. A uniform fixture would pass under a field swap or a
    # re-sort, which is the failure this seam can actually have.
    runs.write(
        path,
        [
            runs.RunRecord.build(
                "1", "q", "rrf", "idx", "h",
                [
                    ScoredChunk("chunk_zulu", "doc_alpha.pdf", 0.11, 1, "text_for_zulu"),
                    ScoredChunk("chunk_alpha", "doc_zulu.pdf", 0.99, 2, "text_for_alpha"),
                ],
            )
        ],
    )
    loaded = load_run(path)["1"]
    assert [c.chunk_id for c in loaded] == ["chunk_zulu", "chunk_alpha"], "rank order lost"
    assert [c.doc_id for c in loaded] == ["doc_alpha.pdf", "doc_zulu.pdf"]
    assert [c.text for c in loaded] == ["text_for_zulu", "text_for_alpha"]
    # Scores must survive as written: the low score is FIRST, so a reader that
    # re-sorts by score instead of trusting rank would reverse the ranking.
    assert [c.score for c in loaded] == [0.11, 0.99]


# --------------------------------------------------------------------------
# Retrievers
# --------------------------------------------------------------------------


def _index():
    import numpy as np

    from src.retrieval.index import LocalIndex
    from src.retrieval.protocol import ChunkRecord

    texts = [
        "fire safety maintenance record for the building",
        "建筑消防设施故障维修记录表 annual review",
        "Đại học Công nghệ tuyển sinh năm 2026",
        "unrelated content about chess openings",
    ]
    records = [ChunkRecord(f"c{i}", f"doc{i}.pdf", t) for i, t in enumerate(texts)]
    rng = np.random.default_rng(0)
    return LocalIndex(
        "test-idx",
        records,
        bm25=BM25Index("auto").build(
            [{"chunk_id": r.chunk_id, "doc_id": r.doc_id, "text": r.text} for r in records]
        ),
        vectors=rng.normal(size=(4, 8)).astype("float32"),
        embedder=type(
            "E", (), {"embed": lambda self, t: rng.normal(size=(len(t), 8)).astype("float32")}
        )(),
    )


@pytest.mark.parametrize("name", ["bm25", "dense", "rrf", "alpha0.7"])
def test_arms_satisfy_the_protocol(name: str):
    from src.retrieval.protocol import Index, Retriever
    from src.retrieval import retrievers

    index = _index()
    assert isinstance(index, Index)
    assert isinstance(retrievers.build(name, index), Retriever)


@pytest.mark.parametrize("name", ["bm25", "dense", "rrf", "alpha0.7"])
def test_scope_never_leaks_out_of_scope_documents(name: str):
    """Fusion arms must scope BOTH legs.

    Scoping only the dense leg would fuse a restricted ranking with an
    unrestricted one, and out-of-scope documents would reappear in the output of
    a two-stage retriever that believed it had filtered them.
    """
    from src.retrieval import retrievers

    scope = ["doc0.pdf", "doc1.pdf"]
    hits = retrievers.build(name, _index()).retrieve("fire safety", 5, scope=scope)
    assert {hit.doc_id for hit in hits} <= set(scope)


def test_retrievers_never_embed_directly():
    """Rule 2: query encoding belongs to the index.

    An embedder reachable from a retriever is how a required prefix gets
    omitted, so the arms must route through ``index.encode_query``.
    """
    from src.retrieval import retrievers

    for name in ("bm25", "dense", "rrf", "alpha0.7"):
        arm = retrievers.build(name, _index())
        assert not hasattr(arm, "embedder"), f"{name} holds an embedder directly"


def test_bm25_overfetches_before_scoping():
    """Top-k then filter would return fewer than k in-scope hits."""
    from src.retrieval import retrievers

    index = _index()
    hits = retrievers.BM25Retriever(index).retrieve("annual review", 1, scope=["doc1.pdf"])
    assert [h.doc_id for h in hits] == ["doc1.pdf"]
