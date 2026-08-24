"""Page reassembly from run_pipeline blocks."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.pipeline_pages import canonical_doc, page_texts, pages

DOC = "Autrement_Ch-2-Des-particules-aux-systemes"

DOCUMENT = {
    "document": {"document_id": "07451b426815af70", "file_name": f"pdfs/{DOC}.pdf"},
    "content": {
        # Deliberately out of file order: reading_order, not list order, decides.
        "blocks": [
            {"component_id": "/page/1/Text/1", "page": 1, "block_index": 3, "text": "second"},
            {"component_id": "/page/0/Text/0", "page": 0, "block_index": 0, "text": "cover"},
            {"component_id": "/page/1/Text/0", "page": 1, "block_index": 1, "text": "first"},
            {"component_id": "/page/1/Figure/2", "page": 1, "block_index": 2, "text": "   "},
        ],
        "reading_order": ["/page/0/Text/0", "/page/1/Text/0",
                          "/page/1/Figure/2", "/page/1/Text/1"],
    },
}


def test_basename_and_suffix_are_both_stripped():
    assert canonical_doc(f"pdfs/{DOC}.pdf") == DOC
    assert canonical_doc(f"{DOC}.pdf") == DOC
    assert canonical_doc(f"a/b/{DOC}.PDF") == DOC


def test_blocks_group_by_page():
    text = page_texts(DOCUMENT)
    assert set(text) == {0, 1}
    assert text[0] == "cover"


def test_reading_order_decides_within_a_page():
    """List order puts 'second' first; reading_order is the authority."""
    assert page_texts(DOCUMENT)[1] == "first\nsecond"


def test_blank_blocks_do_not_leave_separators():
    """A figure with no text must not pad the page with empty lines."""
    assert "\n\n" not in page_texts(DOCUMENT)[1]


def test_blocks_absent_from_reading_order_still_appear():
    """An incomplete reading_order must not silently drop text."""
    partial = json.loads(json.dumps(DOCUMENT))
    partial["content"]["reading_order"] = ["/page/1/Text/0"]
    assert "second" in page_texts(partial)[1]


def test_pages_across_a_run_are_keyed_by_doc_and_page(tmp_path: Path):
    run = tmp_path / "run"
    (run / "documents").mkdir(parents=True)
    (run / "documents" / "a.json").write_text(json.dumps(DOCUMENT), encoding="utf-8")
    assert set(pages(run)) == {(DOC, 0), (DOC, 1)}
