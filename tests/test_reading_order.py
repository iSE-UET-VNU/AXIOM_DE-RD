"""Tests for reading-order normalization of parser blocks and citations."""

from __future__ import annotations

from src.reading_order import (
    citation_ids,
    component_path_parts,
    parse_document_json,
    reading_order_from_rows,
    source_blocks_from_extraction,
    source_blocks_from_parser_json,
)


# ── citation_ids ──────────────────────────────────────────────────────────────

def test_citation_ids_keeps_only_non_empty_strings() -> None:
    assert citation_ids(["  /page/0/Text/1  ", "", "   ", None, 7]) == ["/page/0/Text/1"]


def test_citation_ids_rejects_non_list() -> None:
    assert citation_ids(None) == []
    assert citation_ids("/page/0/Text/1") == []
    assert citation_ids({"a": 1}) == []


# ── component_path_parts ──────────────────────────────────────────────────────

def test_component_path_parts_extracts_page_index_and_type() -> None:
    assert component_path_parts("/page/3/Figure/12") == (3, 12, "Figure")


def test_component_path_parts_rejects_malformed_paths() -> None:
    for bad in ("", "page/0/Text/1", "/page/0/Text", "/page/x/Text/1",
                "/page/0/Text/1/extra", "/bad/0/Text/1", "/page/0//1"):
        assert component_path_parts(bad) is None, bad


# ── parse_document_json ───────────────────────────────────────────────────────

def test_parse_document_json_accepts_mapping_and_json_string() -> None:
    assert parse_document_json({"a": 1}) == {"a": 1}
    assert parse_document_json('{"a": 1}') == {"a": 1}


def test_parse_document_json_returns_empty_for_unusable_input() -> None:
    assert parse_document_json("not json") == {}
    assert parse_document_json("[1, 2]") == {}  # valid JSON, wrong shape
    assert parse_document_json("") == {}
    assert parse_document_json(None) == {}


# ── source_blocks_from_parser_json ────────────────────────────────────────────

def test_parser_blocks_are_flattened_and_sorted_by_page_then_block_index() -> None:
    document = {
        "pages": [
            {
                "id": "/page/1/Page/0",
                "block_type": "Page",
                "children": [
                    {"id": "/page/1/Text/3", "block_type": "Text", "text": "second"},
                    {"id": "/page/1/Text/1", "block_type": "Text", "text": "first"},
                ],
            },
            {
                "id": "/page/0/Page/0",
                "block_type": "Page",
                "children": [
                    {"id": "/page/0/Title/0", "block_type": "Title", "text": "Title"},
                ],
            },
        ]
    }
    blocks = source_blocks_from_parser_json(document)
    assert [b["component_id"] for b in blocks] == [
        "/page/0/Title/0",
        "/page/1/Text/1",
        "/page/1/Text/3",
    ]
    # container blocks (Page/Document) are never emitted as content
    assert all(b["type"] not in ("Page", "Document") for b in blocks)
    assert {b["source"] for b in blocks} == {"parser_json"}


def test_parser_block_text_falls_back_to_stripped_html() -> None:
    document = {"children": [
        {"id": "/page/0/Text/0", "block_type": "Text",
         "html": "<p>hello <b>bold</b> world</p>"},
    ]}
    [block] = source_blocks_from_parser_json(document)
    assert block["text"] == "hello bold world"
    assert block["html"] == "<p>hello <b>bold</b> world</p>"


def test_parser_block_carries_geometry_and_hierarchy_when_present() -> None:
    document = {"children": [
        {"id": "/page/0/Text/0", "block_type": "Text", "text": "t",
         "bbox": [0, 1, 2, 3], "polygon": [[0, 0]], "section_hierarchy": {"1": "Intro"}},
    ]}
    [block] = source_blocks_from_parser_json(document)
    assert block["bbox"] == [0, 1, 2, 3]
    assert block["polygon"] == [[0, 0]]
    assert block["section_hierarchy"] == {"1": "Intro"}


def test_semantic_text_recorded_when_it_differs_from_parser_text() -> None:
    document = {"children": [
        {"id": "/page/0/Text/0", "block_type": "Text", "text": "parser version"},
    ]}
    extraction = {
        "main_text": "semantic version",
        "main_text_citations": ["/page/0/Text/0"],
    }
    [block] = source_blocks_from_parser_json(document, extraction)
    assert block["text"] == "parser version"
    assert block["semantic_text"] == "semantic version"


def test_cited_blocks_missing_from_parser_json_are_merged_in() -> None:
    document = {"children": [
        {"id": "/page/0/Text/0", "block_type": "Text", "text": "parser text"},
    ]}
    extraction = {
        "figures": [{"caption": "A figure", "caption_citations": ["/page/0/Figure/9"]}],
    }
    blocks = source_blocks_from_parser_json(document, extraction)
    by_id = {b["component_id"]: b for b in blocks}
    assert by_id["/page/0/Text/0"]["source"] == "parser_json"
    assert by_id["/page/0/Figure/9"]["source"] == "structured_extraction_citations"
    assert by_id["/page/0/Figure/9"]["text"] == "A figure"


def test_empty_or_unparseable_document_yields_no_blocks() -> None:
    assert source_blocks_from_parser_json({}) == []
    assert source_blocks_from_parser_json("not json") == []


# ── source_blocks_from_extraction ─────────────────────────────────────────────

def test_main_text_paragraphs_pair_with_citations_positionally() -> None:
    extraction = {
        "main_text": "para one\n\npara two",
        "main_text_citations": ["/page/0/Text/0", "/page/0/Text/1"],
    }
    blocks = source_blocks_from_extraction(extraction)
    assert [(b["component_id"], b["text"]) for b in blocks] == [
        ("/page/0/Text/0", "para one"),
        ("/page/0/Text/1", "para two"),
    ]
    assert {b["source"] for b in blocks} == {"structured_extraction_citations"}


def test_main_text_is_skipped_when_paragraph_and_citation_counts_disagree() -> None:
    """Misaligned pairing would attribute text to the wrong block, so emit nothing."""
    extraction = {
        "main_text": "para one\n\npara two",
        "main_text_citations": ["/page/0/Text/0"],
    }
    assert source_blocks_from_extraction(extraction) == []


def test_formula_citations_resolve_from_per_item_meta() -> None:
    extraction = {
        "formulas": ["E=mc^2", "a+b"],
        "formulas_meta": {"items": [
            {"citations": ["/page/0/Equation/1"]},
            {"citations": ["/page/0/Equation/2"]},
        ]},
    }
    blocks = source_blocks_from_extraction(extraction)
    assert [(b["component_id"], b["text"], b["type"]) for b in blocks] == [
        ("/page/0/Equation/1", "E=mc^2", "Equation"),
        ("/page/0/Equation/2", "a+b", "Equation"),
    ]


def test_formula_citations_fall_back_to_aligned_top_level_list() -> None:
    extraction = {
        "formulas": ["E=mc^2", "a+b"],
        "formulas_citations": ["/page/0/Equation/1", "/page/0/Equation/2"],
    }
    assert [b["text"] for b in source_blocks_from_extraction(extraction)] == ["E=mc^2", "a+b"]


def test_formula_top_level_citations_ignored_when_length_mismatches() -> None:
    extraction = {
        "formulas": ["only one"],
        "formulas_citations": ["/page/0/Equation/1", "/page/0/Equation/2"],
    }
    assert source_blocks_from_extraction(extraction) == []


def test_table_and_figure_fields_emit_blocks_per_cited_field() -> None:
    extraction = {
        "tables": [{"content": "| a |", "content_citations": ["/page/0/Table/0"]}],
        "figures": [{"description": "OCR text", "description_citations": ["/page/1/Figure/0"]}],
    }
    blocks = source_blocks_from_extraction(extraction)
    assert [(b["component_id"], b["text"]) for b in blocks] == [
        ("/page/0/Table/0", "| a |"),
        ("/page/1/Figure/0", "OCR text"),
    ]


def test_title_uses_its_own_citations() -> None:
    extraction = {"title": "Doc Title", "title_citations": ["/page/0/SectionHeader/0"]}
    [block] = source_blocks_from_extraction(extraction)
    assert (block["component_id"], block["text"]) == ("/page/0/SectionHeader/0", "Doc Title")


def test_unparseable_component_ids_are_dropped() -> None:
    extraction = {"title": "T", "title_citations": ["not-a-path", "/bad/0/X/1"]}
    assert source_blocks_from_extraction(extraction) == []


# ── reading_order_from_rows ───────────────────────────────────────────────────

def _native_row(component_id: str = "/page/0/Text/0", text: str = "a") -> dict:
    return {"source_blocks": [
        {"component_id": component_id, "source": "parser_json", "text": text},
    ]}


def test_rows_with_native_parser_blocks_report_complete_reading_order() -> None:
    blocks, order, meta = reading_order_from_rows([_native_row()])
    assert order == ["/page/0/Text/0"]
    assert meta == {"source": "parser_json", "complete": True, "block_count": 1}
    assert blocks[0]["text"] == "a"


def test_rows_without_source_blocks_fall_back_to_extraction_citations() -> None:
    rows = [{"extraction": {
        "main_text": "one\n\ntwo",
        "main_text_citations": ["/page/0/Text/0", "/page/0/Text/1"],
    }}]
    _, order, meta = reading_order_from_rows(rows)
    assert order == ["/page/0/Text/0", "/page/0/Text/1"]
    assert meta == {
        "source": "structured_extraction_citations",
        "complete": False,
        "block_count": 2,
    }


def test_mixed_provenance_within_a_row_is_not_reported_as_complete() -> None:
    rows = [{"source_blocks": [
        {"component_id": "/page/0/Text/0", "source": "parser_json", "text": "a"},
        {"component_id": "/page/0/Text/1", "source": "structured_extraction_citations", "text": "b"},
    ]}]
    _, _, meta = reading_order_from_rows(rows)
    assert meta["source"] == "structured_extraction_citations"
    assert meta["complete"] is False


def test_no_usable_rows_report_unavailable() -> None:
    for rows in ([], [{}], [{"extraction": {}}], ["not a dict", None]):
        _, order, meta = reading_order_from_rows(list(rows))
        assert order == []
        assert meta == {"source": "unavailable", "complete": False, "block_count": 0}


def test_duplicate_component_ids_across_rows_keep_the_first_occurrence() -> None:
    rows = [_native_row(text="FIRST"), _native_row(text="SECOND")]
    blocks, order, meta = reading_order_from_rows(rows)
    assert order == ["/page/0/Text/0"]
    assert blocks[0]["text"] == "FIRST"
    assert meta["block_count"] == 1


def test_blocks_from_multiple_rows_are_merged_into_one_global_order() -> None:
    rows = [_native_row("/page/2/Text/0", "late"), _native_row("/page/0/Text/5", "early")]
    _, order, meta = reading_order_from_rows(rows)
    assert order == ["/page/0/Text/5", "/page/2/Text/0"]
    assert meta["complete"] is True
