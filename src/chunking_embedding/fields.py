"""Per-field routing: chunk each extraction field on its own, never the raw JSON.

main_text runs the configured chunker; each table is one chunk, split into
row-groups (header repeated) past ``max_rows_per_chunk``; each figure is one
chunk of caption + OCR text; each formula is atomic. Empty fields are skipped.

Chunkers that declare ``needs`` receive a ``Resources`` handle (embedder / llm)
and may return either character spans (wrapped faithfully) or pre-built
``Chunk`` objects (derived text or tree nodes used as-is).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import re

from .contracts import Chunk, Unit, chunk_identity, params_hash
from .registry import chunker_needs, get_chunker
from .text import Span

_PAGE_RE = re.compile(r"^/page/(\d+)/")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


@dataclass(frozen=True)
class Resources:
    embedder: Any = None
    llm: Any = None


@dataclass(frozen=True)
class FieldContext:
    doc_id: str
    chunker_name: str
    chunker_params: dict[str, Any] = field(default_factory=dict)
    max_rows_per_chunk: int = 20
    language: str | None = None

    @property
    def chunker(self) -> Callable[..., list[Span]]:
        return get_chunker(self.chunker_name)

    def base_metadata(self) -> dict[str, Any]:
        return {
            "chunker": self.chunker_name,
            "chunker_params_hash": params_hash(self.chunker_params),
            "language": self.language,
        }


def route_document(extraction: dict[str, Any], ctx: FieldContext, resources: Resources | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for field_name, route in _ROUTES:
        value = extraction.get(field_name)
        if not _is_empty(value):
            chunks.extend(route(value, extraction, ctx, resources))
    return chunks


def route_main_text(value: Any, extraction: dict[str, Any], ctx: FieldContext, resources: Resources | None = None) -> list[Chunk]:
    text = str(value)
    base_meta = {
        **ctx.base_metadata(),
        "component_type": "main_text",
        "page": _single_page(extraction.get("main_text_citations")),
        "field_chars": len(text),
    }
    result = _run_chunker(ctx, text, resources)
    return _chunks_from_result(result, ctx.doc_id, "main_text", text, base_meta)


def route_tables(value: Any, extraction: dict[str, Any], ctx: FieldContext, resources: Resources | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, table in enumerate(value):
        if not isinstance(table, dict):
            continue
        content = str(table.get("content") or "")
        if not content.strip():
            continue
        field_path = f"tables[{i}]"
        caption = str(table.get("caption") or "").strip()
        page = _single_page(table.get("content_citations") or table.get("caption_citations"))
        metadata = {
            **ctx.base_metadata(),
            "component_type": "table",
            "caption": caption or None,
            "page": page,
            "field_chars": len(content),
        }
        header_lines, row_lines = _split_markdown_table(content)
        if not row_lines or len(row_lines) <= ctx.max_rows_per_chunk:
            chunks.append(
                Chunk.create(
                    ctx.doc_id, field_path, "table_chunk", content, 0, len(content),
                    metadata={**metadata, "n_rows": len(row_lines)},
                )
            )
            continue
        header_text = content[header_lines[0][0]:header_lines[-1][1]] if header_lines else ""
        for group_idx in range(0, len(row_lines), ctx.max_rows_per_chunk):
            group = row_lines[group_idx:group_idx + ctx.max_rows_per_chunk]
            start, end = group[0][0], group[-1][1]
            text = f"{header_text}\n{content[start:end]}" if header_text else content[start:end]
            chunks.append(
                Chunk.create(
                    ctx.doc_id, field_path, "table_chunk", text, start, end,
                    metadata={
                        **metadata,
                        "row_group": group_idx // ctx.max_rows_per_chunk,
                        "n_rows": len(group),
                        "header_repeated": bool(header_text),
                    },
                )
            )
    return chunks


def route_figures(value: Any, extraction: dict[str, Any], ctx: FieldContext, resources: Resources | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, figure in enumerate(value):
        if not isinstance(figure, dict):
            continue
        caption = str(figure.get("caption") or "").strip()
        ocr_text = str(figure.get("description") or figure.get("ocr_text") or "").strip()
        composed = "\n\n".join(part for part in (caption, ocr_text) if part)
        if not composed:
            continue
        page = _single_page(figure.get("caption_citations") or figure.get("description_citations"))
        chunks.append(
            Chunk.create(
                ctx.doc_id, f"figures[{i}]", "figure_chunk", composed, 0, len(composed),
                metadata={
                    **ctx.base_metadata(),
                    "component_type": "figure",
                    "composed_from": [n for n, part in (("caption", caption), ("description", ocr_text)) if part],
                    "page": page,
                    "field_chars": len(composed),
                },
            )
        )
    return chunks


def route_formulas(value: Any, extraction: dict[str, Any], ctx: FieldContext, resources: Resources | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, formula in enumerate(value):
        if isinstance(formula, dict):
            content = str(formula.get("content") or formula.get("latex") or formula.get("text") or "")
        else:
            content = str(formula or "")
        if not content.strip():
            continue
        chunks.append(
            Chunk.create(
                ctx.doc_id, f"formulas[{i}]", "text_chunk", content, 0, len(content),
                metadata={
                    **ctx.base_metadata(),
                    "component_type": "formula",
                    "page": None,
                    "field_chars": len(content),
                },
            )
        )
    return chunks


_ROUTES: list[tuple[str, Callable[..., list[Chunk]]]] = [
    ("main_text", route_main_text),
    ("tables", route_tables),
    ("figures", route_figures),
    ("formulas", route_formulas),
]


def _run_chunker(ctx: FieldContext, text: str, resources: Resources | None) -> list[Any]:
    needs = chunker_needs(ctx.chunker_name)
    fn = get_chunker(ctx.chunker_name)
    if not needs:
        return fn(text, **ctx.chunker_params)
    if resources is None:
        raise RuntimeError(f"chunker {ctx.chunker_name!r} needs {needs} but no resources were provided")
    missing = [n for n in needs if getattr(resources, n, None) is None]
    if missing:
        raise RuntimeError(f"chunker {ctx.chunker_name!r} needs {missing} which the run did not supply")
    return fn(text, resources=resources, **ctx.chunker_params)


def _chunks_from_result(result: list[Any], doc_id: str, field_path: str, text: str, base_meta: dict[str, Any]) -> list[Chunk]:
    ids = [
        chunk_identity(doc_id, field_path, item.start, item.end, f"#{i}") if isinstance(item, Unit) else None
        for i, item in enumerate(result)
    ]
    chunks: list[Chunk] = []
    for i, item in enumerate(result):
        if isinstance(item, Unit):
            parent_id = ids[item.parent] if item.parent is not None and 0 <= item.parent < len(ids) else None
            children = [ids[j] for j in item.children if 0 <= j < len(ids) and ids[j]]
            chunks.append(
                Chunk(
                    chunk_id=ids[i], doc_id=doc_id, field_path=field_path, chunk_type=item.chunk_type,
                    text=item.text, start=item.start, end=item.end,
                    metadata={**base_meta, **item.meta}, level=item.level, parent_id=parent_id, children=children,
                )
            )
        else:
            a, b = item
            chunks.append(Chunk.create(doc_id, field_path, "text_chunk", text[a:b], a, b, metadata=dict(base_meta)))
    return chunks


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _single_page(citations: Any) -> int | None:
    if not isinstance(citations, list):
        return None
    pages = {
        int(match.group(1))
        for item in citations
        if isinstance(item, str) and (match := _PAGE_RE.match(item))
    }
    return pages.pop() if len(pages) == 1 else None


def _split_markdown_table(content: str) -> tuple[list[Span], list[Span]]:
    lines: list[Span] = []
    pos = 0
    for raw in content.split("\n"):
        end = pos + len(raw)
        if raw.strip():
            lines.append((pos, end))
        pos = end + 1
    if not lines:
        return [], []
    header: list[Span] = [lines[0]]
    rest = lines[1:]
    if rest and _TABLE_SEPARATOR_RE.match(content[rest[0][0]:rest[0][1]]):
        header.append(rest[0])
        rest = rest[1:]
    return header, rest
