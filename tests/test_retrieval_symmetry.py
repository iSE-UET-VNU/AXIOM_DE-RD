"""The invariant behind the analyzer bug, generalized.

The bug class is: **the same text takes a different path at index time than at
query time.** The CJK failure was one instance. At least three more are waiting
in this codebase -- E5-style query prefixes, normalization applied on one side
only, and case folding -- and all of them fail the same way: no exception, no
log, an empty or wrongly-ordered result.

So the tests here are properties over any index, not assertions about one
tokenizer:

1. Text processing resolves through the frozen manifest config, identically on
   both sides, and an unresolved ``auto`` raises rather than guessing.
2. A term that appears in an indexed document retrieves that document. Three
   languages, every arm. This single round trip would have caught the original
   bug on the day it was written.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from src.chunking_embedding.embedders import create_embedder
from src.retrieval import retrievers
from src.retrieval.build_artifacts import build
from src.retrieval.index import ArtifactMisaligned, load_artifacts
from src.retrieval.sparse import BM25Index

# One document per script, each with a term we will query back out of it.
CORPUS = {
    "en": ("fire safety maintenance record for the building", "maintenance"),
    "vi": ("Đại học Công nghệ tuyển sinh ngành công nghệ thông tin", "tuyển sinh"),
    "zh": ("建筑消防设施故障维修记录表 annual equipment review", "消防设施"),
}
ARMS = ["bm25", "dense", "rrf", "alpha0.7"]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory) -> Path:
    """Build a real artifact set through the real build path."""
    root = tmp_path_factory.mktemp("artifacts")
    documents = root / "out" / "RUN" / "documents"
    documents.mkdir(parents=True)
    for i, (lang, (text, _)) in enumerate(CORPUS.items()):
        (documents / f"{lang}.json").write_text(
            json.dumps(
                {
                    "document": {"document_id": f"{lang}.pdf", "file_name": f"{lang}.pdf"},
                    "retrieval": {
                        "items": [
                            {
                                "item_id": f"c{i}",
                                "type": "text",
                                "content": {"text": text},
                                "embeddings": [{"model": "local_hash"}],
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
    return build(
        "RUN",
        root / "artifacts",
        "testcfg",
        "auto",
        output_root=root / "out",
        embedder=create_embedder("local_hash", {}),
        embedder_id="local_hash",
    )


# --------------------------------------------------------------------------
# Property 1: one frozen config, both sides
# --------------------------------------------------------------------------


def test_manifest_records_a_resolved_analyzer(artifacts: Path):
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analyzer_id"] != "auto"
    assert manifest["analyzer_id"] in {"plain", "cjk_bigram"}


def test_index_and_query_use_the_same_analyzer_object(artifacts: Path):
    """Not "the same name" -- the same callable.

    Two functions that agree today can diverge later; identity cannot.
    """
    index = load_artifacts(artifacts)
    assert index.bm25.analyzer is BM25Index(index.bm25.analyzer_name).analyzer


def test_identity_includes_everything_that_changes_results(artifacts: Path):
    """A cached run keyed without the analyzer would survive an analyzer change."""
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    index_id = manifest["index_id"]
    for part in (manifest["analyzer_id"], manifest["embedder_id"], manifest["corpus_hash"]):
        assert part in index_id


def test_unresolved_artifacts_are_refused(artifacts: Path, tmp_path: Path):
    """Pre-fix artifact sets exist on disk; loading one must not be possible."""
    stale = tmp_path / "stale"
    stale.mkdir()
    for name in ("bm25.json", "chunks.jsonl"):
        (stale / name).write_text((artifacts / name).read_text(encoding="utf-8"), encoding="utf-8")
    (stale / "manifest.json").write_text(json.dumps({"analyzer": "auto"}), encoding="utf-8")
    with pytest.raises(ArtifactMisaligned, match="before the analyzer was resolved"):
        load_artifacts(stale)


def test_vector_row_misalignment_is_fatal(artifacts: Path, tmp_path: Path):
    """A shifted row gives a working retriever that is confidently wrong."""
    broken = tmp_path / "broken"
    broken.mkdir()
    for name in ("bm25.json", "chunks.jsonl", "chunk_ids.json", "manifest.json"):
        (broken / name).write_text((artifacts / name).read_text(encoding="utf-8"), encoding="utf-8")
    original = np.load(artifacts / "vectors.npy")
    np.save(broken / "vectors.npy", original[:-1])
    with pytest.raises(ArtifactMisaligned, match="vectors for"):
        load_artifacts(broken)


def test_cosine_index_is_normalized_exactly_once(artifacts: Path):
    index = load_artifacts(artifacts)
    assert index.normalized
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------
# Property 2: round trip, every arm, three scripts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lang", sorted(CORPUS))
@pytest.mark.parametrize("arm", ARMS)
def test_indexed_term_retrieves_its_document(artifacts: Path, lang: str, arm: str):
    """Index a document, query a term inside it, expect that document back.

    The test that would have caught the original bug: before the fix, the `zh`
    case returned nothing for `bm25` while every internal consistency check
    passed.
    """
    index = load_artifacts(artifacts, embedder=create_embedder("local_hash", {}))
    _, term = CORPUS[lang]
    hits = retrievers.build(arm, index).retrieve(term, len(CORPUS))
    assert hits, f"{arm} returned nothing for {lang} term {term!r}"
    if arm in {"bm25", "rrf", "alpha0.7"}:
        # Lexical evidence is exact here, so the source document must be found.
        # Dense is excluded: local_hash is a hashing stub with no semantics.
        assert f"{lang}.pdf" in {hit.doc_id for hit in hits}


@pytest.mark.parametrize("lang", sorted(CORPUS))
def test_sparse_leg_finds_the_exact_source_document_first(artifacts: Path, lang: str):
    index = load_artifacts(artifacts)
    _, term = CORPUS[lang]
    hits = retrievers.BM25Retriever(index).retrieve(term, 1)
    assert [hit.doc_id for hit in hits] == [f"{lang}.pdf"]
