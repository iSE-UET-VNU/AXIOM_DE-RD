"""Reconstructing a ViDoRe unit id from run_pipeline output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.evaluation.benchmarks import load  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import unit_id  # noqa: E402

SUBSET = "physics"
DOC = "Autrement_Ch-2-Des-particules-aux-systemes"


def canonical_doc(file_name: str) -> str:
    """Basename then strip .pdf; the real run emits ``pdfs/<name>.pdf``."""
    base = file_name.rsplit("/", 1)[-1]
    return base[:-4] if base.lower().endswith(".pdf") else base


def pipeline_units(document: dict, subset: str) -> list[str]:
    """The reconstruction under test."""
    stem = canonical_doc(document["document"]["file_name"])
    pages = sorted({b["page"] for b in document["content"]["blocks"]})
    return [unit_id(subset, stem, page) for page in pages]


@pytest.fixture
def pipeline_document() -> dict:
    """run_pipeline's real shape: document_id is a hash, page is 0-based."""
    return {
        "document": {
            "document_id": "2e6e1a4844bed00c",
            # The real run emits a directory prefix; stripping .pdf alone matched 0/42.
            "file_name": f"pdfs/{DOC}.pdf",
            "content_type": "application/pdf",
        },
        "content": {
            "main_text": "...",
            "blocks": [
                {"component_id": "/page/0/Text/0", "page": 0, "block_index": 0,
                 "type": "Text", "text": "cover"},
                {"component_id": "/page/1/Text/0", "page": 1, "block_index": 1,
                 "type": "Text", "text": "des particules"},
                {"component_id": "/page/1/Table/1", "page": 1, "block_index": 2,
                 "type": "Table", "text": "| a | b |"},
                {"component_id": "/page/2/Text/0", "page": 2, "block_index": 3,
                 "type": "Text", "text": "aux systemes"},
            ],
            "reading_order": ["/page/0/Text/0", "/page/1/Text/0",
                              "/page/1/Table/1", "/page/2/Text/0"],
            "reading_order_meta": {"source": "parser_json", "complete": True,
                                   "block_count": 4},
        },
    }


@pytest.fixture
def vidore_root(tmp_path: Path) -> Path:
    """A physics subset in the released schema, pages 0..2 of one document."""
    path = tmp_path / SUBSET
    path.mkdir(parents=True)
    pq.write_table(
        pa.table({
            "corpus_id": [40, 41, 42],
            "doc_id": [DOC] * 3,
            "markdown": ["cover", "des particules", "aux systemes"],
            "page_number_in_doc": [0, 1, 2],
        }),
        path / "corpus.parquet",
    )
    pq.write_table(
        pa.table({
            "query_id": [0], "query": ["what are systems?"], "language": ["english"],
            "query_types": [["extractive"]], "query_format": ["question"],
            "content_type": [["Text"]], "source_type": ["summary"],
            "answer": ["systems"],
        }),
        path / "queries.parquet",
    )
    bbox = pa.list_(pa.struct([("annotator", pa.int64()), ("x1", pa.int64()),
                               ("x2", pa.int64()), ("y1", pa.int64()), ("y2", pa.int64())]))
    pq.write_table(
        pa.table({
            "query_id": [0], "corpus_id": [42], "score": [2],
            "content_type": [["Text"]],
            "bounding_boxes": pa.array([[{"annotator": 0, "x1": 1, "x2": 9,
                                          "y1": 1, "y2": 9}]], type=bbox),
        }),
        path / "qrels.parquet",
    )
    pq.write_table(
        pa.table({"file_name": [f"{DOC}.pdf"], "doc_id": [DOC],
                  "doc_type": ["slides"], "doc_language": ["french"],
                  "page_number": [3]}),
        path / "documents_metadata.parquet",
    )
    return tmp_path


def test_reconstructed_units_match_the_adapters_own(pipeline_document, vidore_root):
    """The seam: their output must land on the ids ViDoRe's corpus.parquet produces."""
    benchmark = load("vidore_v3", root=vidore_root, subset=SUBSET, language="english")
    theirs = set(pipeline_units(pipeline_document, SUBSET))
    ours = {d.doc_id for d in benchmark.corpus()}
    assert theirs == ours


def test_a_one_based_page_index_breaks_every_unit(pipeline_document, vidore_root):
    """Rule 2. Off-by-one gives zero overlap, which reads as a broken retriever."""
    shifted = json.loads(json.dumps(pipeline_document))
    for block in shifted["content"]["blocks"]:
        block["page"] += 1
    benchmark = load("vidore_v3", root=vidore_root, subset=SUBSET, language="english")
    ours = {d.doc_id for d in benchmark.corpus()}
    assert set(pipeline_units(shifted, SUBSET)).isdisjoint(ours) or \
        set(pipeline_units(shifted, SUBSET)) != ours


def test_keeping_the_pdf_suffix_breaks_every_unit(pipeline_document, vidore_root):
    """Rule 2 for the canonical_doc class: 0/189 file_names equal their doc_id."""
    stem_kept = pipeline_document["document"]["file_name"]
    benchmark = load("vidore_v3", root=vidore_root, subset=SUBSET, language="english")
    ours = {d.doc_id for d in benchmark.corpus()}
    naive = {unit_id(SUBSET, stem_kept, b["page"])
             for b in pipeline_document["content"]["blocks"]}
    assert naive.isdisjoint(ours)


def test_keeping_the_directory_prefix_breaks_every_unit(pipeline_document, vidore_root):
    """Measured on the real run: strip .pdf without basename matched 0/42 docs."""
    name = pipeline_document["document"]["file_name"]
    suffix_only = name[:-4] if name.lower().endswith(".pdf") else name
    benchmark = load("vidore_v3", root=vidore_root, subset=SUBSET, language="english")
    ours = {d.doc_id for d in benchmark.corpus()}
    naive = {unit_id(SUBSET, suffix_only, b["page"])
             for b in pipeline_document["content"]["blocks"]}
    assert naive.isdisjoint(ours)


def test_document_id_is_not_the_vidore_doc_id(pipeline_document, vidore_root):
    """Their identity is a content hash; routing through it matches nothing."""
    doc_hash = pipeline_document["document"]["document_id"]
    benchmark = load("vidore_v3", root=vidore_root, subset=SUBSET, language="english")
    ours = {d.doc_id for d in benchmark.corpus()}
    assert {unit_id(SUBSET, doc_hash, b["page"])
            for b in pipeline_document["content"]["blocks"]}.isdisjoint(ours)


def test_gold_resolves_against_the_reconstructed_units(pipeline_document, vidore_root):
    """A gold page must be reachable from their output, or coverage is a lie."""
    benchmark = load("vidore_v3", root=vidore_root, subset=SUBSET, language="english")
    gold = set(benchmark.qrels()["physics::0"])
    assert gold <= set(pipeline_units(pipeline_document, SUBSET))


def test_missing_pages_are_countable_not_silent(pipeline_document, vidore_root):
    """A page their parse dropped is a coverage loss, not a denominator change."""
    partial = json.loads(json.dumps(pipeline_document))
    partial["content"]["blocks"] = [b for b in partial["content"]["blocks"] if b["page"] != 2]
    benchmark = load("vidore_v3", root=vidore_root, subset=SUBSET, language="english")
    ours = {d.doc_id for d in benchmark.corpus()}
    theirs = set(pipeline_units(partial, SUBSET))
    assert ours - theirs == {unit_id(SUBSET, DOC, 2)}
