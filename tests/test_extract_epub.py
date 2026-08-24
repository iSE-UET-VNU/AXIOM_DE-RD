"""EPUB extraction, with spine order as the property that matters."""

from __future__ import annotations

from pathlib import Path
import zipfile

from src.evaluation.extract import extract, supported_extensions

CONTAINER = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="OEBPS/book.opf"
    media-type="application/oebps-package+xml"/></rootfiles>
</container>"""

OPF = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
    <item id="fm" href="front.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="fm"/><itemref idref="c1"/><itemref idref="c2"/></spine>
</package>"""


def _page(title: str, body: str) -> str:
    return (
        f"<html><head><title>x</title><style>p{{color:red}}</style></head>"
        f"<body><h1>{title}</h1><p>{body}</p>"
        f"<script>ignored()</script></body></html>"
    )


def _epub(tmp_path: Path) -> Path:
    path = tmp_path / "book.epub"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", CONTAINER)
        archive.writestr("OEBPS/book.opf", OPF)
        # Written in an order that does NOT match the spine.
        archive.writestr("OEBPS/ch2.xhtml", _page("Chương 2", "Nội dung hai"))
        archive.writestr("OEBPS/ch1.xhtml", _page("Chương 1", "Nội dung một"))
        archive.writestr("OEBPS/front.xhtml", _page("Lời nói đầu", "Mở đầu"))
    return path


def test_epub_is_supported():
    assert ".epub" in supported_extensions()


def test_blocks_follow_spine_order_not_archive_order(tmp_path: Path):
    """Archive order would put chapter 2 before the front matter."""
    doc = extract(_epub(tmp_path))
    headings = [b.text for b in doc.blocks if b.kind == "heading"]
    assert headings == ["Lời nói đầu", "Chương 1", "Chương 2"]


def test_script_and_style_are_dropped(tmp_path: Path):
    doc = extract(_epub(tmp_path))
    assert "ignored()" not in doc.text
    assert "color:red" not in doc.text


def test_headings_and_paragraphs_are_distinguished(tmp_path: Path):
    doc = extract(_epub(tmp_path))
    kinds = {b.kind for b in doc.blocks}
    assert kinds == {"heading", "paragraph"}
    assert "Nội dung một" in doc.text


def test_diacritics_survive_extraction(tmp_path: Path):
    """Vietnamese content is the point of the epub shelf."""
    assert "Chương" in extract(_epub(tmp_path)).text


def test_missing_container_falls_back_to_any_opf(tmp_path: Path):
    path = tmp_path / "loose.epub"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("OEBPS/book.opf", OPF)
        archive.writestr("OEBPS/ch1.xhtml", _page("Chương 1", "một"))
        archive.writestr("OEBPS/ch2.xhtml", _page("Chương 2", "hai"))
        archive.writestr("OEBPS/front.xhtml", _page("Đầu", "mở"))
    assert extract(path).ok
