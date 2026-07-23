"""Normalize parser blocks and structured-extraction citations into reading order."""

from __future__ import annotations

from html.parser import HTMLParser
import json
import re
from typing import Any


_COMPONENT_PATH = re.compile(r"^/page/(\d+)/([^/]+)/(\d+)$")
_NON_CONTENT_BLOCK_TYPES = {"document", "page"}
_NATIVE_BLOCK_SOURCES = {"parser_json", "chandra2_layout"}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def citation_ids(value: Any) -> list[str]:
    """Return non-empty component paths without inventing replacement IDs."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def component_path_parts(component_id: str) -> tuple[int, int, str] | None:
    """Return page, global page-block index, and type from a component path."""
    match = _COMPONENT_PATH.fullmatch(component_id)
    if not match:
        return None
    return int(match.group(1)), int(match.group(3)), match.group(2)


def parse_document_json(value: Any) -> dict[str, Any]:
    """Normalize an SDK JSON result that may arrive as a mapping or JSON string."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def source_blocks_from_parser_json(
    document_json: Any,
    extraction: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Flatten top-level page components from Datalab JSON in parser order."""
    document = parse_document_json(document_json)
    if not document:
        return []

    semantic_by_id = {
        block["component_id"]: block
        for block in source_blocks_from_extraction(extraction or {})
    }
    collected: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return

        component_id = value.get("id") or value.get("component_id")
        parts = component_path_parts(component_id) if isinstance(component_id, str) else None
        if parts is not None:
            page, block_index, path_type = parts
            block_type = str(value.get("block_type") or value.get("type") or path_type)
            if block_type.casefold() not in _NON_CONTENT_BLOCK_TYPES:
                raw_text = _block_text(value)
                semantic = semantic_by_id.get(component_id, {})
                block: dict[str, Any] = {
                    "component_id": component_id,
                    "page": page,
                    "block_index": block_index,
                    "type": block_type,
                    "text": raw_text or semantic.get("text") or "",
                    "source": "parser_json",
                }
                semantic_text = semantic.get("text")
                if raw_text and semantic_text and semantic_text != raw_text:
                    block["semantic_text"] = semantic_text
                for field in ("bbox", "polygon", "section_hierarchy"):
                    if value.get(field) is not None:
                        block[field] = value[field]
                if isinstance(value.get("html"), str) and value["html"]:
                    block["html"] = value["html"]
                collected.append(block)

        for child in value.values():
            if isinstance(child, (dict, list)):
                visit(child)

    visit(document)
    parser_ids = {
        block["component_id"]
        for block in collected
        if isinstance(block.get("component_id"), str)
    }
    collected.extend(
        block for component_id, block in semantic_by_id.items() if component_id not in parser_ids
    )
    return _deduplicate_and_sort(collected)


def source_blocks_from_extraction(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct cited semantic blocks when parser-native JSON is unavailable."""
    candidates: list[tuple[str, str]] = []

    main_text = extraction.get("main_text")
    main_citations = citation_ids(extraction.get("main_text_citations"))
    if main_text:
        paragraphs = _paragraphs(str(main_text))
        if len(paragraphs) == len(main_citations):
            candidates.extend(zip(main_citations, paragraphs))

    candidates.extend(_formula_candidates(extraction))
    candidates.extend(_collection_candidates(extraction, "figures"))
    candidates.extend(_collection_candidates(extraction, "tables"))

    title = extraction.get("title")
    if title:
        candidates.extend(
            (component_id, str(title))
            for component_id in citation_ids(extraction.get("title_citations"))
        )

    blocks: list[dict[str, Any]] = []
    for component_id, text in candidates:
        parts = component_path_parts(component_id)
        if parts is None:
            continue
        page, block_index, block_type = parts
        blocks.append(
            {
                "component_id": component_id,
                "page": page,
                "block_index": block_index,
                "type": block_type,
                "text": text,
                "source": "structured_extraction_citations",
            }
        )
    return _deduplicate_and_sort(blocks)


def reading_order_from_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Build final-output blocks, ordered IDs, and completeness metadata."""
    all_blocks: list[dict[str, Any]] = []
    native_rows = 0
    extraction_rows = 0
    native_sources: set[str] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        row_blocks = row.get("source_blocks")
        normalized_row_blocks = (
            [dict(block) for block in row_blocks if isinstance(block, dict)]
            if isinstance(row_blocks, list)
            else []
        )
        if normalized_row_blocks:
            row_sources = {
                str(block.get("source") or "")
                for block in normalized_row_blocks
            }
            if row_sources and row_sources <= _NATIVE_BLOCK_SOURCES:
                native_rows += 1
                native_sources.update(row_sources)
            else:
                extraction_rows += 1
            all_blocks.extend(normalized_row_blocks)
            continue

        extraction = row.get("extraction")
        if isinstance(extraction, dict):
            fallback = source_blocks_from_extraction(extraction)
            if fallback:
                extraction_rows += 1
                all_blocks.extend(fallback)

    blocks = _deduplicate_and_sort(all_blocks)
    reading_order = [block["component_id"] for block in blocks]
    is_native = bool(blocks) and native_rows > 0 and extraction_rows == 0
    source = (
        next(iter(native_sources))
        if is_native and len(native_sources) == 1
        else "parser_native_mixed"
        if is_native
        else "structured_extraction_citations"
        if blocks
        else "unavailable"
    )
    metadata = {
        "source": source,
        "complete": is_native,
        "block_count": len(blocks),
    }
    return blocks, reading_order, metadata


def _paragraphs(text: str) -> list[str]:
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", text.strip())
        if paragraph.strip()
    ]


def _formula_candidates(extraction: dict[str, Any]) -> list[tuple[str, str]]:
    formulas = extraction.get("formulas")
    if not isinstance(formulas, list):
        return []

    formula_meta = extraction.get("formulas_meta")
    meta_items = formula_meta.get("items") if isinstance(formula_meta, dict) else None
    top_level_citations = citation_ids(extraction.get("formulas_citations"))
    candidates: list[tuple[str, str]] = []
    for index, formula in enumerate(formulas):
        citations: list[str] = []
        if isinstance(meta_items, list) and index < len(meta_items):
            item_meta = meta_items[index]
            if isinstance(item_meta, dict):
                citations = citation_ids(item_meta.get("citations"))
        if not citations and len(top_level_citations) == len(formulas):
            citations = [top_level_citations[index]]
        candidates.extend((component_id, str(formula)) for component_id in citations)
    return candidates


def _collection_candidates(
    extraction: dict[str, Any],
    collection_name: str,
) -> list[tuple[str, str]]:
    items = extraction.get(collection_name)
    if not isinstance(items, list):
        return []

    candidates: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for field in ("description", "caption", "content", "text"):
            value = item.get(field)
            if value in (None, ""):
                continue
            candidates.extend(
                (component_id, str(value))
                for component_id in citation_ids(item.get(f"{field}_citations"))
            )
    return candidates


def _block_text(block: dict[str, Any]) -> str:
    for field in ("text", "raw_text", "markdown", "content"):
        value = block.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    html = block.get("html")
    if not isinstance(html, str) or not html.strip():
        return ""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return " ".join(parser.parts)


def _deduplicate_and_sort(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for block in blocks:
        component_id = block.get("component_id")
        if isinstance(component_id, str) and component_id:
            unique.setdefault(component_id, block)

    indexed = list(enumerate(unique.values()))

    def order_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
        fallback_index, block = item
        parts = component_path_parts(str(block.get("component_id") or ""))
        if parts is None:
            return 10**9, fallback_index, fallback_index
        page, block_index, _ = parts
        return page, block_index, fallback_index

    return [block for _, block in sorted(indexed, key=order_key)]
