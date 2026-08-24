"""The block chunker reads ``kind``; chandra2 emits ``type``. Map, don't assume.

Left unmapped every block falls through to "paragraph", so the heading-aware
arm silently degrades to a flat pack and reports a number that looks like
structured chunking. Types here are the real labels chandra2 emits on the
physics run, not invented ones.
"""

from __future__ import annotations

from src.evaluation.pipeline_pages import BOILERPLATE, page_blocks

CHANDRA_TYPES = ["Text", "Image", "PageFooter", "PageHeader", "SectionHeader",
                 "ListGroup", "Figure", "EquationBlock", "Caption", "Table", "Diagram"]


def block(component_id, page, index, type_, text):
    return {"component_id": component_id, "page": page, "block_index": index,
            "type": type_, "text": text}


def document(blocks, order=None):
    return {"content": {"blocks": blocks, "reading_order": order or []}}


def test_a_section_header_becomes_a_heading():
    doc = document([block("c0", "0", "0", "SectionHeader", "Chapitre 1")])
    assert page_blocks(doc)[0][0]["kind"] == "heading"


def test_a_table_is_not_a_paragraph():
    doc = document([block("c0", "0", "0", "Table", "| a | b |")])
    assert page_blocks(doc)[0][0]["kind"] == "table"


def test_body_text_is_a_paragraph():
    doc = document([block("c0", "0", "0", "Text", "corps du texte")])
    assert page_blocks(doc)[0][0]["kind"] == "paragraph"


def test_every_real_chandra_type_maps_to_a_known_kind():
    """An unmapped type is the silent failure; enumerate the real ones."""
    known = {"heading", "table", "paragraph", "caption", "figure", "boilerplate"}
    doc = document([block(f"c{i}", "0", str(i), t, f"text {i}")
                    for i, t in enumerate(CHANDRA_TYPES)])
    assert {b["kind"] for b in page_blocks(doc)[0]} <= known


def test_page_headers_and_footers_are_marked_boilerplate():
    doc = document([block("c0", "0", "0", "PageHeader", "logo"),
                    block("c1", "0", "1", "PageFooter", "p. 3")])
    assert {b["kind"] for b in page_blocks(doc)[0]} == {BOILERPLATE}


def test_pages_are_keyed_by_int_not_the_string_chandra_emits():
    doc = document([block("c0", "0", "0", "Text", "a"), block("c1", "1", "0", "Text", "b")])
    assert sorted(page_blocks(doc)) == [0, 1]


def test_reading_order_wins_over_block_index():
    doc = document(
        [block("c0", "0", "0", "Text", "second"), block("c1", "0", "1", "Text", "first")],
        order=["c1", "c0"],
    )
    assert [b["text"] for b in page_blocks(doc)[0]] == ["first", "second"]


def test_blocks_without_a_page_are_dropped():
    doc = document([{"component_id": "c0", "type": "Text", "text": "orphan"}])
    assert page_blocks(doc) == {}


def test_the_text_survives_the_mapping():
    doc = document([block("c0", "0", "0", "Text", "corps")])
    assert page_blocks(doc)[0][0]["text"] == "corps"
