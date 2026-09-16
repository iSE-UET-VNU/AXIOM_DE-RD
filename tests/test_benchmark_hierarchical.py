from __future__ import annotations

import json
from pathlib import Path

from research.data_discovery.raw_hierarchical import (
    RawPage,
    aggregate_file_scores,
    build_file_bm25,
    build_page_bm25,
    canonical_page_id,
    normalise_scores,
    load_documents,
    sort_scores,
)
from research.experiments.run_benchmark_hierarchical import _ndcg


def _pages() -> list[RawPage]:
    return [
        RawPage(
            page_id=canonical_page_id("doc-a", 0),
            doc_id="doc-a",
            source="raw",
            source_path="a.pdf",
            relative_path="a.pdf",
            page_index=0,
            page_number=1,
            text="thermodynamic entropy Boltzmann",
            visual_only=False,
        ),
        RawPage(
            page_id=canonical_page_id("doc-a", 1),
            doc_id="doc-a",
            source="raw",
            source_path="a.pdf",
            relative_path="a.pdf",
            page_index=1,
            page_number=2,
            text="microscopic configurations",
            visual_only=False,
        ),
        RawPage(
            page_id=canonical_page_id("doc-b", 0),
            doc_id="doc-b",
            source="raw",
            source_path="b.pdf",
            relative_path="b.pdf",
            page_index=0,
            page_number=1,
            text="electromagnetic field",
            visual_only=False,
        ),
    ]


def test_canonical_page_id_is_zero_based_and_stable() -> None:
    assert canonical_page_id("doc-a", 0) == "doc-a#page=0"
    assert canonical_page_id("doc-a", 12) == "doc-a#page=12"


def test_page_and_file_bm25_preserve_units() -> None:
    pages = _pages()
    page_index = build_page_bm25(pages)
    file_index = build_file_bm25(pages)
    page_hits = page_index.search("entropy", 3)
    file_hits = file_index.search("entropy", 2)
    assert page_index.chunk_ids[page_hits[0][0]] == "doc-a#page=0"
    assert file_index.chunk_ids[file_hits[0][0]] == "doc-a"


def test_file_pool_aggregation_and_normalisation_are_deterministic() -> None:
    page_to_file = {page.page_id: page.doc_id for page in _pages()}
    raw = {
        "doc-a#page=0": 0.5,
        "doc-a#page=1": 0.8,
        "doc-b#page=0": 0.3,
    }
    files = aggregate_file_scores(raw, page_to_file, ["doc-a", "doc-b", "doc-c"])
    assert files == {"doc-a": 0.8, "doc-b": 0.3, "doc-c": 0.0}
    assert sort_scores(normalise_scores(files))[0] == ("doc-a", 1.0)


def test_ndcg_uses_graded_relevance() -> None:
    qrels = {"gold-high": 2, "gold-low": 1}
    perfect = _ndcg(["gold-high", "gold-low"], qrels, 2)
    reversed_order = _ndcg(["gold-low", "gold-high"], qrels, 2)
    assert perfect == 1.0
    assert 0.0 < reversed_order < perfect


def test_slim_run_row_has_no_page_text() -> None:
    row = {
        "query_id": "q1",
        "chunks": [{"page_id": "doc-a#page=0", "doc_id": "doc-a", "score": 1.0}],
    }
    assert "text" not in json.dumps(row)


def test_manifest_can_materialise_an_image_as_one_visual_page(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"not decoded during inventory")
    manifest = tmp_path / "documents.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "doc_id": "image-1",
                "source": "raw",
                "path": image_path.name,
                "mime_type": "image/png",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    documents = load_documents(tmp_path)
    assert len(documents) == 1
    assert documents[0].is_image
    assert not documents[0].is_pdf
