"""Reassemble run_pipeline block output into page text, for page-level benchmarks.

ViDoRe's retrieval unit is the page. A pipeline run emits blocks and, downstream,
chunks; comparing its chunks against ``corpus.markdown`` would vary parser,
chunker and unit size at once. Grouping blocks by page restores the one-variable
comparison.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator
import json


BOILERPLATE = "boilerplate"

# chandra2 labels blocks by ``type``; the block chunker dispatches on ``kind``.
KIND_BY_TYPE = {
    "SectionHeader": "heading",
    "Title": "heading",
    "Table": "table",
    "TableGroup": "table",
    "Caption": "caption",
    "Figure": "figure",
    "Image": "figure",
    "Diagram": "figure",
    "PageHeader": BOILERPLATE,
    "PageFooter": BOILERPLATE,
    "PageNumber": BOILERPLATE,
}


def block_kind(block: dict[str, Any]) -> str:
    return KIND_BY_TYPE.get(str(block.get("type") or ""), "paragraph")


def page_blocks(document: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Blocks per page in reading order, with ``kind`` mapped from chandra2 ``type``."""
    content = document.get("content") or {}
    order = {cid: i for i, cid in enumerate(content.get("reading_order") or [])}

    by_page: dict[int, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for position, block in enumerate(content.get("blocks") or []):
        page = block.get("page")
        if page is None:
            continue
        rank = order.get(block.get("component_id"), len(order) + position)
        by_page[int(page)].append((
            rank, int(block.get("block_index") or position),
            {**block, "kind": block_kind(block), "text": str(block.get("text") or "")},
        ))

    return {page: [b for _, _, b in sorted(items, key=lambda i: (i[0], i[1]))]
            for page, items in by_page.items()}


def canonical_doc(file_name: str) -> str:
    """Basename then strip .pdf; runs emit ``pdfs/<name>.pdf``, matching 42/42."""
    base = str(file_name or "").rsplit("/", 1)[-1]
    return base[:-4] if base.lower().endswith(".pdf") else base


def documents(run_dir: Path | str) -> Iterator[dict[str, Any]]:
    for path in sorted(Path(run_dir).glob("documents/*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def page_texts(document: dict[str, Any]) -> dict[int, str]:
    """Block text per page, ordered by reading_order then block_index."""
    blocks = document.get("content", {}).get("blocks") or []
    order = {cid: i for i, cid in
             enumerate(document.get("content", {}).get("reading_order") or [])}

    by_page: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for position, block in enumerate(blocks):
        page = block.get("page")
        if page is None:
            continue
        rank = order.get(block.get("component_id"), len(order) + position)
        by_page[int(page)].append((rank, block.get("block_index", position),
                                   str(block.get("text") or "")))

    return {
        page: "\n".join(text for _, _, text in sorted(items) if text.strip())
        for page, items in by_page.items()
    }


def pages(run_dir: Path | str) -> dict[tuple[str, int], str]:
    """``(doc_id, page) -> text`` across a whole run."""
    out: dict[tuple[str, int], str] = {}
    for document in documents(run_dir):
        doc_id = canonical_doc(document.get("document", {}).get("file_name"))
        for page, text in page_texts(document).items():
            out[(doc_id, page)] = text
    return out
