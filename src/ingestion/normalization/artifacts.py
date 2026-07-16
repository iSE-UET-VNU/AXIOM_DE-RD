"""Build normalized text, image, table, and observed-schema artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
import json
import re

from ...models import ParsedData, make_id
from ...utils.paths import portable_path_value

NORMALIZED_TEXT_VERSION = "normalized-text-v1"
NORMALIZED_IMAGE_VERSION = "normalized-image-v1"
NORMALIZED_TABLE_VERSION = "normalized-table-v1"
NORMALIZED_DOCUMENT_VERSION = "normalized-document-v1"

TEXT_BLOCK_TYPES = {
    "Caption",
    "ComplexRegion",
    "ListItem",
    "PageFooter",
    "PageHeader",
    "SectionHeader",
    "Text",
}


@dataclass
class NormalizationOutput:
    texts: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)


def normalize_parsed_data(
    parsed_data: list[ParsedData],
    project_root: str | Path | None = None,
) -> NormalizationOutput:
    """Normalize parser outputs into component records and a document registry."""
    output = NormalizationOutput()
    root = Path(project_root).resolve() if project_root else None

    for parsed in parsed_data:
        document = _normalize_document_components(parsed, root)
        output.texts.extend(document["texts"])
        output.images.extend(document["images"])
        output.tables.extend(document["tables"])
        output.documents.append(document["document"])

    return output


def _normalize_document_components(parsed: ParsedData, project_root: Path | None) -> dict[str, Any]:
    lift = _load_lift_artifacts(parsed, project_root)
    extraction = _extraction_from_parsed(parsed, lift)
    document_json = lift.get("document_json")
    convert_markdown = _text_or_none(lift.get("convert_markdown"))
    parsed_tables = _tables_from_parsed_tables(parsed, project_root)

    if isinstance(document_json, dict):
        texts = _texts_from_document_json(parsed, document_json, project_root)
        lift_tables = _tables_from_document_json(parsed, document_json, extraction, project_root)
        images = _images_from_document_json(parsed, document_json, extraction, project_root)
    else:
        texts = []
        lift_tables = []
        images = []

    if not texts:
        texts = _fallback_texts(parsed, extraction, convert_markdown, project_root)
    # Structured tables emitted by a parser backend are already canonical at
    # the cell level and therefore take precedence over provider fallbacks.
    # Lift does not currently populate ParsedData.tables, so its behavior is
    # unchanged.
    tables = parsed_tables or lift_tables
    if not tables:
        tables = _fallback_tables(parsed, extraction, project_root)
    if not tables and convert_markdown:
        tables = _tables_from_markdown(parsed, convert_markdown, project_root)
    if not images:
        images = _fallback_images(parsed, extraction, project_root)
    else:
        images = _append_unmatched_image_files(parsed, images, project_root)

    formula_texts = _formula_texts(parsed, extraction, project_root)
    texts.extend(formula_texts)

    return {
        "texts": texts,
        "images": images,
        "tables": tables,
        "document": _document_record(
            parsed,
            extraction,
            texts,
            images,
            tables,
            len(formula_texts),
            project_root,
        ),
    }


def _load_lift_artifacts(parsed: ParsedData, project_root: Path | None) -> dict[str, Any]:
    raw_outputs = parsed.metadata.get("raw_lift_outputs")
    if not isinstance(raw_outputs, dict):
        return {}

    artifacts: dict[str, Any] = {}
    extraction_path = _resolve_path(raw_outputs.get("extract_extraction_schema_json"), project_root)
    document_path = _resolve_path(raw_outputs.get("extract_json"), project_root)
    convert_markdown_path = _resolve_path(raw_outputs.get("convert_markdown"), project_root)

    if extraction_path:
        artifacts["extraction_schema"] = _read_json(extraction_path)
    if document_path:
        artifacts["document_json"] = _read_json(document_path)
        artifacts["document_json_path"] = str(document_path)
    if convert_markdown_path:
        artifacts["convert_markdown"] = convert_markdown_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    return artifacts


def _texts_from_document_json(
    parsed: ParsedData,
    document_json: dict[str, Any],
    project_root: Path | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, block in enumerate(_walk_blocks(document_json)):
        block_type = _text_or_none(block.get("block_type"))
        if block_type not in TEXT_BLOCK_TYPES:
            continue

        text = _block_text(block)
        if not text:
            continue

        source_block_id = _text_or_none(block.get("id")) or f"text-{index}"
        records.append(
            {
                "contract_version": NORMALIZED_TEXT_VERSION,
                "text_id": make_id(parsed.object_id, "text", source_block_id, index),
                "document_id": parsed.object_id,
                "source_uri": parsed.source_uri,
                "source_block_id": source_block_id,
                "page": _optional_int(block.get("page")),
                "role": _text_role(block_type),
                "text": text,
                "embedding_text": text,
                "section_path": _section_path(block),
                "source_artifact": "extract.document.json",
            }
        )
    return _portable_records(records, project_root)


def _tables_from_document_json(
    parsed: ParsedData,
    document_json: dict[str, Any],
    extraction: dict[str, Any],
    project_root: Path | None,
) -> list[dict[str, Any]]:
    semantic_tables = _list_of_dicts(extraction.get("tables"))
    records: list[dict[str, Any]] = []

    for index, block in enumerate(_blocks_by_type(document_json, "Table")):
        source_block_id = _text_or_none(block.get("id")) or f"table-{index}"
        html = _text_or_none(block.get("html")) or ""
        rows = _parse_html_table(html)
        semantic = _match_semantic_item(semantic_tables, source_block_id) or _item_at(semantic_tables, index)
        caption = _text_or_none((semantic or {}).get("caption")) or _nearby_caption(document_json, block)
        markdown = _markdown_table(rows) or _text_or_none((semantic or {}).get("content")) or ""

        records.append(
            {
                "contract_version": NORMALIZED_TABLE_VERSION,
                "table_id": make_id(parsed.object_id, "table", source_block_id, index),
                "document_id": parsed.object_id,
                "source_uri": parsed.source_uri,
                "source_block_id": source_block_id,
                "page": _optional_int(block.get("page")),
                "caption": caption,
                "html": html,
                "rows": rows,
                "markdown": markdown,
                "embedding_text": _table_embedding_text(caption, rows, markdown),
                "row_count": len(rows),
                "column_count": max((len(row) for row in rows), default=0),
                "headers": rows[0] if rows else [],
                "semantic": semantic or {},
                "source_artifact": "extract.document.json",
            }
        )

    return _portable_records(records, project_root)


def _images_from_document_json(
    parsed: ParsedData,
    document_json: dict[str, Any],
    extraction: dict[str, Any],
    project_root: Path | None,
) -> list[dict[str, Any]]:
    figures = _list_of_dicts(extraction.get("figures"))
    image_files = _image_file_map(parsed)
    records: list[dict[str, Any]] = []

    for index, block in enumerate(_blocks_by_type(document_json, "Picture")):
        source_block_id = _text_or_none(block.get("id")) or f"image-{index}"
        html = _text_or_none(block.get("html")) or ""
        image_markup = _parse_image_markup(html)
        semantic = _match_semantic_item(figures, source_block_id) or _item_at(figures, index) or {}
        visible_caption = _visible_caption(semantic)
        generated_description = _generated_description(semantic, image_markup.get("alt"))
        image_name = image_markup.get("src") or _first_image_name(block)
        image_path = image_files.get(str(image_name)) if image_name else None
        # images.jsonl is an asset registry. Semantic Picture blocks for which
        # the parser did not return a decoded file must not become dangling
        # image records; their source data remains available in the work
        # bundle's raw document JSON.
        if not image_path:
            continue
        embedding_text = _image_embedding_text(visible_caption, generated_description)

        records.append(
            {
                "contract_version": NORMALIZED_IMAGE_VERSION,
                "image_id": make_id(parsed.object_id, "image", source_block_id, index),
                "document_id": parsed.object_id,
                "source_uri": parsed.source_uri,
                "source_block_id": source_block_id,
                "page": _optional_int(block.get("page")),
                "image_name": image_name,
                "image_path": image_path,
                "visible_caption": visible_caption,
                "generated_description": generated_description,
                "caption_is_visible": bool(visible_caption),
                "description_source": _description_source(semantic, generated_description),
                "embedding_text": embedding_text,
                "semantic": semantic,
                "source_artifact": "extract.document.json",
            }
        )

    return _portable_records(records, project_root)


def _fallback_texts(
    parsed: ParsedData,
    extraction: dict[str, Any],
    convert_markdown: str | None,
    project_root: Path | None,
) -> list[dict[str, Any]]:
    text = _first_text(
        extraction.get("main_text"),
        extraction.get("text"),
        extraction.get("markdown"),
        extraction.get("content"),
        parsed.text,
        convert_markdown,
    )
    if not text:
        return []
    return _portable_records(
        [
            {
                "contract_version": NORMALIZED_TEXT_VERSION,
                "text_id": make_id(parsed.object_id, "text", "fallback", 0),
                "document_id": parsed.object_id,
                "source_uri": parsed.source_uri,
                "source_block_id": None,
                "page": None,
                "role": "body",
                "text": text,
                "embedding_text": text,
                "section_path": [],
                "source_artifact": "convert.md" if text == convert_markdown else "parsed.extraction",
            }
        ],
        project_root,
    )


def _fallback_tables(
    parsed: ParsedData,
    extraction: dict[str, Any],
    project_root: Path | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, table in enumerate(_list_of_dicts(extraction.get("tables"))):
        source_block_id = _first_text(*_citation_values(table)) or f"table-{index}"
        caption = _text_or_none(table.get("caption")) or ""
        markdown = _text_or_none(table.get("content")) or ""
        rows = _parse_markdown_table(markdown)
        records.append(
            {
                "contract_version": NORMALIZED_TABLE_VERSION,
                "table_id": make_id(parsed.object_id, "table", source_block_id, index),
                "document_id": parsed.object_id,
                "source_uri": parsed.source_uri,
                "source_block_id": source_block_id,
                "page": _page_from_block_id(source_block_id),
                "caption": caption,
                "html": "",
                "rows": rows,
                "markdown": _markdown_table(rows) or markdown,
                "embedding_text": _table_embedding_text(caption, rows, markdown),
                "row_count": len(rows),
                "column_count": max((len(row) for row in rows), default=0),
                "headers": rows[0] if rows else [],
                "semantic": table,
                "source_artifact": "extract.extraction_schema.json",
            }
        )
    return _portable_records(records, project_root)


def _tables_from_parsed_tables(
    parsed: ParsedData,
    project_root: Path | None,
) -> list[dict[str, Any]]:
    """Normalize provider-neutral ``ParsedTable`` values.

    ``getattr`` intentionally keeps this normalizer compatible with ParsedData
    instances produced before the tables field was introduced.
    """
    records: list[dict[str, Any]] = []
    for index, table in enumerate(getattr(parsed, "tables", None) or []):
        name = _text_or_none(_table_value(table, "name")) or f"Table {index + 1}"
        source_ref = _text_or_none(_table_value(table, "source_ref")) or f"table-{index}"
        headers = [_cell_text(value) for value in (_table_value(table, "headers") or [])]
        data_rows = [
            [_cell_text(value) for value in row]
            for row in (_table_value(table, "rows") or [])
            if isinstance(row, (list, tuple))
        ]
        rows = ([headers] if headers else []) + data_rows
        metadata = _table_value(table, "metadata")
        semantic = dict(metadata) if isinstance(metadata, dict) else {}
        semantic.update(
            {
                "parser": _parser_provider(parsed),
                "table_name": name,
                "source_ref": source_ref,
            }
        )
        markdown = _markdown_table(rows)

        records.append(
            {
                "contract_version": NORMALIZED_TABLE_VERSION,
                "table_id": make_id(parsed.object_id, "table", source_ref, index),
                "document_id": parsed.object_id,
                "source_uri": parsed.source_uri,
                "source_block_id": source_ref,
                "page": None,
                "caption": name,
                "html": "",
                "rows": rows,
                "markdown": markdown,
                "embedding_text": _table_embedding_text(name, rows, markdown),
                "row_count": len(rows),
                "column_count": max((len(row) for row in rows), default=0),
                "headers": headers,
                "semantic": semantic,
                "source_artifact": "parsed.tables",
            }
        )
    return _portable_records(records, project_root)


def _tables_from_markdown(
    parsed: ParsedData,
    markdown: str,
    project_root: Path | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    lines = markdown.splitlines()
    index = 0
    table_index = 0
    while index < len(lines):
        if not _is_markdown_table_line(lines[index]):
            index += 1
            continue
        start = index
        while index < len(lines) and _is_markdown_table_line(lines[index]):
            index += 1
        table_lines = lines[start:index]
        if len(table_lines) < 2 or not _is_markdown_separator_line(table_lines[1]):
            continue
        raw_markdown = "\n".join(table_lines)
        rows = _parse_markdown_table(raw_markdown)
        if not rows:
            continue
        caption = _markdown_table_caption(lines, start)
        source_block_id = f"markdown-table-{table_index}"
        records.append(
            {
                "contract_version": NORMALIZED_TABLE_VERSION,
                "table_id": make_id(parsed.object_id, "table", source_block_id, table_index),
                "document_id": parsed.object_id,
                "source_uri": parsed.source_uri,
                "source_block_id": source_block_id,
                "page": None,
                "caption": caption,
                "html": "",
                "rows": rows,
                "markdown": _markdown_table(rows) or raw_markdown,
                "embedding_text": _table_embedding_text(caption, rows, raw_markdown),
                "row_count": len(rows),
                "column_count": max((len(row) for row in rows), default=0),
                "headers": rows[0] if rows else [],
                "semantic": {"caption": caption, "content": raw_markdown},
                "source_artifact": "convert.md",
            }
        )
        table_index += 1
    return _portable_records(records, project_root)


def _fallback_images(
    parsed: ParsedData,
    extraction: dict[str, Any],
    project_root: Path | None,
) -> list[dict[str, Any]]:
    image_files = _image_file_items(parsed)
    figures = _list_of_dicts(extraction.get("figures"))
    records: list[dict[str, Any]] = []
    for index, image_item in enumerate(image_files):
        figure = _item_at(figures, index) or {}
        source_block_id = _first_text(*_citation_values(figure)) or f"image-{index}"
        visible_caption = _visible_caption(figure)
        generated_description = _generated_description(figure, None)
        image_name = image_item.get("name")
        image_path = image_item.get("path")
        records.append(
            {
                "contract_version": NORMALIZED_IMAGE_VERSION,
                "image_id": make_id(parsed.object_id, "image", source_block_id, index),
                "document_id": parsed.object_id,
                "source_uri": parsed.source_uri,
                "source_block_id": source_block_id,
                "page": _page_from_block_id(source_block_id),
                "image_name": image_name,
                "image_path": image_path,
                "visible_caption": visible_caption,
                "generated_description": generated_description,
                "caption_is_visible": bool(visible_caption),
                "description_source": _description_source(figure, generated_description),
                "embedding_text": _image_embedding_text(visible_caption, generated_description),
                "semantic": figure,
                "source_artifact": "extract.extraction_schema.json",
            }
        )
    return _portable_records(records, project_root)


def _formula_texts(
    parsed: ParsedData,
    extraction: dict[str, Any],
    project_root: Path | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    formulas = extraction.get("formulas")
    if not isinstance(formulas, list):
        return records
    for index, formula in enumerate(formulas):
        text = _text_or_none(formula.get("text") if isinstance(formula, dict) else formula)
        if not text:
            continue
        source_block_id = f"formula-{index}"
        records.append(
            {
                "contract_version": NORMALIZED_TEXT_VERSION,
                "text_id": make_id(parsed.object_id, "text", source_block_id, index),
                "document_id": parsed.object_id,
                "source_uri": parsed.source_uri,
                "source_block_id": source_block_id,
                "page": None,
                "role": "formula",
                "text": text,
                "embedding_text": text,
                "section_path": [],
                "source_artifact": "extract.extraction_schema.json",
            }
        )
    return _portable_records(records, project_root)


def _append_unmatched_image_files(
    parsed: ParsedData,
    images: list[dict[str, Any]],
    project_root: Path | None,
) -> list[dict[str, Any]]:
    used_names = {
        str(image.get("image_name")).replace("\\", "/")
        for image in images
        if image.get("image_name")
    }
    used_basenames = {Path(name).name for name in used_names}
    records = list(images)
    for index, item in enumerate(_image_file_items(parsed), start=len(records)):
        name = str(item.get("name") or "")
        if name in used_names or Path(name).name in used_basenames:
            continue
        source_block_id = f"image-{index}"
        records.append(
            {
                "contract_version": NORMALIZED_IMAGE_VERSION,
                "image_id": make_id(parsed.object_id, "image", source_block_id, index),
                "document_id": parsed.object_id,
                "source_uri": parsed.source_uri,
                "source_block_id": source_block_id,
                "page": None,
                "image_name": name,
                "image_path": item.get("path"),
                "visible_caption": "",
                "generated_description": "",
                "caption_is_visible": False,
                "description_source": None,
                "embedding_text": "",
                "semantic": {},
                "source_artifact": "convert.md",
            }
        )
    return _portable_records(records, project_root)


def _document_record(
    parsed: ParsedData,
    extraction: dict[str, Any],
    texts: list[dict[str, Any]],
    images: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    formula_count: int,
    project_root: Path | None,
) -> dict[str, Any]:
    has_text = any(_text_or_none(item.get("text")) for item in texts)
    has_table_content = any(
        _text_or_none(cell)
        for table in tables
        for row in table.get("rows", [])
        if isinstance(row, list)
        for cell in row
    )
    has_image_content = any(image.get("image_path") for image in images)
    record = {
        "contract_version": NORMALIZED_DOCUMENT_VERSION,
        "document_id": parsed.object_id,
        "source_uri": parsed.source_uri,
        "source_format": parsed.source_format,
        "document_type": _text_or_none(extraction.get("document_type")),
        "language": _text_or_none(extraction.get("language")),
        "title": _text_or_none(extraction.get("title")),
        "component_counts": {
            "texts": len(texts),
            "images": len(images),
            "tables": len(tables),
            "formulas": formula_count,
        },
        "parser": {
            "provider": _parser_provider(parsed),
            "mode": parsed.metadata.get("mode"),
            "status": parsed.metadata.get("status") or "success",
        },
        "page_count": parsed.metadata.get("page_count"),
        "work_artifact_uri": _work_artifact_uri(parsed),
        "quality": {
            "has_text": has_text,
            "has_content": bool(has_text or has_table_content or has_image_content),
            "missing_image_assets": sum(1 for image in images if not image.get("image_path")),
        },
    }
    return portable_path_value(record, project_root)


def _extraction_from_parsed(parsed: ParsedData, lift: dict[str, Any]) -> dict[str, Any]:
    if isinstance(lift.get("extraction_schema"), dict):
        return lift["extraction_schema"]
    for row in parsed.rows:
        extraction = row.get("extraction") if isinstance(row, dict) else None
        if isinstance(extraction, dict):
            return extraction
    return {}


def _parser_provider(parsed: ParsedData) -> str | None:
    value = parsed.metadata.get("backend") or parsed.metadata.get("parser")
    if isinstance(value, dict):
        value = value.get("provider") or value.get("backend")
    return _text_or_none(value)


def _table_value(table: Any, field_name: str) -> Any:
    if isinstance(table, dict):
        return table.get(field_name)
    return getattr(table, field_name, None)


def _cell_text(value: Any) -> str:
    return "" if value is None else str(value)


def _walk_blocks(value: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "block_type" in value or "html" in value or "id" in value:
            blocks.append(value)
        children = value.get("children")
        if isinstance(children, list):
            for child in children:
                blocks.extend(_walk_blocks(child))
    elif isinstance(value, list):
        for item in value:
            blocks.extend(_walk_blocks(item))
    return blocks


def _blocks_by_type(document_json: dict[str, Any], block_type: str) -> list[dict[str, Any]]:
    return [block for block in _walk_blocks(document_json) if block.get("block_type") == block_type]


def _block_text(block: dict[str, Any]) -> str:
    for field in ("text", "markdown"):
        value = _text_or_none(block.get(field))
        if value:
            return _normalize_space(value)
    html = _text_or_none(block.get("html"))
    if html:
        return _normalize_space(_html_text(html))
    return ""


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _html_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return " ".join(parser.parts)


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(_normalize_space(" ".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(cell for cell in self._row):
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None and data.strip():
            self._cell.append(data.strip())


def _parse_html_table(html: str) -> list[list[str]]:
    if not html:
        return []
    parser = _HTMLTableParser()
    parser.feed(html)
    return parser.rows


class _HTMLImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.attrs: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img" and not self.attrs:
            self.attrs = {name: value or "" for name, value in attrs}


def _parse_image_markup(html: str) -> dict[str, str]:
    if not html:
        return {}
    parser = _HTMLImageParser()
    parser.feed(html)
    return {"src": parser.attrs.get("src", ""), "alt": parser.attrs.get("alt", "")}


def _parse_markdown_table(markdown: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [_normalize_space(cell) for cell in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    return rows


def _is_markdown_table_line(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_markdown_separator_line(value: str) -> bool:
    stripped = value.strip().strip("|")
    cells = [cell.strip().replace(" ", "") for cell in stripped.split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _markdown_table_caption(lines: list[str], start: int) -> str:
    for line in reversed(lines[max(0, start - 3):start]):
        value = line.strip()
        if value and not value.startswith("!["):
            return _normalize_space(value.lstrip("#").strip())
    return ""


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    lines = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _table_embedding_text(caption: str, rows: list[list[str]], markdown: str) -> str:
    parts = []
    if caption:
        parts.append(f"Table: {caption}.")
    elif rows or markdown:
        parts.append("Table.")
    for index, row in enumerate(rows[:50], start=1):
        cells = [cell for cell in row if cell]
        if cells:
            parts.append(f"Row {index}: {'; '.join(cells)}.")
    if len(rows) > 50:
        parts.append(f"{len(rows) - 50} additional row(s) omitted from embedding text.")
    if len(parts) == 1 and markdown:
        parts.append(_normalize_space(markdown))
    return " ".join(parts).strip()


def _image_embedding_text(visible_caption: str, generated_description: str) -> str:
    caption = _normalize_space(visible_caption)
    description = _normalize_space(generated_description)
    if caption and description and caption.lower() != description.lower():
        return f"Image caption: {caption}. Description: {description}"
    text = caption or description
    return f"Image: {text}" if text else ""


def _visible_caption(figure: dict[str, Any]) -> str:
    caption = _text_or_none(figure.get("caption")) or ""
    if not caption:
        return ""
    caption_meta = figure.get("caption_meta")
    if isinstance(caption_meta, dict) and caption_meta.get("extraction_status") == "NOT_RESOLVABLE":
        return ""
    return caption


def _generated_description(figure: dict[str, Any], alt_text: str | None) -> str:
    description = _text_or_none(figure.get("description")) or ""
    alt = _text_or_none(alt_text) or ""
    return description or alt


def _description_source(figure: dict[str, Any], generated_description: str) -> str | None:
    if not generated_description:
        return None
    citations = _citation_values(figure)
    if not citations or any("/Picture/" in citation for citation in citations):
        return "vlm_generated"
    return "datalab_figure_description"


def _match_semantic_item(items: list[dict[str, Any]], source_block_id: str) -> dict[str, Any] | None:
    for item in items:
        if source_block_id in _citation_values(item):
            return item
    return None


def _citation_values(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key, value in item.items():
        if key.endswith("_citations") and isinstance(value, list):
            values.extend(str(citation) for citation in value if citation)
    return values


def _nearby_caption(document_json: dict[str, Any], block: dict[str, Any]) -> str:
    blocks = _walk_blocks(document_json)
    try:
        index = blocks.index(block)
    except ValueError:
        return ""
    for previous in reversed(blocks[max(0, index - 3):index]):
        if previous.get("block_type") == "SectionHeader":
            return _block_text(previous)
    return ""


def _image_file_map(parsed: ParsedData) -> dict[str, str]:
    paths: dict[str, str] = {}
    basename_paths: dict[str, set[str]] = {}
    for item in _image_file_items(parsed):
        name = str(item["name"])
        path = str(item["path"])
        paths[name] = path
        basename_paths.setdefault(Path(name.replace("\\", "/")).name, set()).add(path)
    for basename, candidates in basename_paths.items():
        if len(candidates) == 1:
            paths.setdefault(basename, next(iter(candidates)))
    return paths


def _image_file_items(parsed: ParsedData) -> list[dict[str, str]]:
    image_files = parsed.metadata.get("image_files")
    if not isinstance(image_files, list):
        return []
    items: list[dict[str, str]] = []
    for item in image_files:
        if not isinstance(item, dict) or item.get("status") != "saved":
            continue
        name = _text_or_none(item.get("name"))
        path = _text_or_none(item.get("path"))
        if name and path:
            items.append({"name": name, "path": path})
    return sorted(items, key=lambda item: (item["name"], item["path"]))


def _first_image_name(block: dict[str, Any]) -> str | None:
    images = block.get("images")
    if isinstance(images, dict) and images:
        return str(next(iter(images.keys())))
    return None


def _section_path(block: dict[str, Any]) -> list[str]:
    hierarchy = block.get("section_hierarchy")
    if isinstance(hierarchy, list):
        return [_normalize_space(str(item)) for item in hierarchy if str(item).strip()]
    if isinstance(hierarchy, dict):
        return [_normalize_space(str(value)) for value in hierarchy.values() if str(value).strip()]
    return []


def _text_role(block_type: str | None) -> str:
    return {
        "Caption": "caption",
        "ComplexRegion": "region",
        "ListItem": "list_item",
        "PageFooter": "page_footer",
        "PageHeader": "page_header",
        "SectionHeader": "heading",
        "Text": "paragraph",
    }.get(block_type or "", "text")


def _resolve_path(value: Any, project_root: Path | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute() and project_root:
        path = project_root / path
    return path if path.exists() else None


def _work_artifact_uri(parsed: ParsedData) -> str | None:
    raw_output_path = _text_or_none(parsed.metadata.get("raw_output_path"))
    if raw_output_path:
        return str(Path(raw_output_path).parent)
    raw_outputs = parsed.metadata.get("raw_lift_outputs")
    if isinstance(raw_outputs, dict):
        for value in raw_outputs.values():
            path = _text_or_none(value)
            if path:
                return str(Path(path).parent)
    return None


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _portable_records(records: list[dict[str, Any]], project_root: Path | None) -> list[dict[str, Any]]:
    return portable_path_value(records, project_root)


def _item_at(items: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    return items[index] if 0 <= index < len(items) else None


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _text_or_none(value)
        if text:
            return text
    return None


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _page_from_block_id(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"/page/(\d+)/", value)
    return int(match.group(1)) if match else None


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")
