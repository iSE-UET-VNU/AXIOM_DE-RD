"""Benchmark chunking arms.

Flat arms delegate to the production registry in ``src.chunking_embedding`` so
the benchmark measures shippable code, not a parallel implementation. The
structure arm packs extracted blocks without crossing a heading and without ever
splitting a table row, which the flat arms cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import hashlib
import sys
from pathlib import Path

from src.utils.paths import repo_root

PROJECT_ROOT = repo_root(__file__)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking_embedding.registry import get_chunker, load_plugins  # noqa: E402
from src.chunking_embedding import chunkers as _chunkers  # noqa: E402,F401

load_plugins("src.chunking_embedding.chunkers", _chunkers.__path__)

ROW_KINDS = frozenset({"table_row", "table_header"})


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    kind: str = "text"

    @property
    def index_text(self) -> str:
        return self.text


def chunk_document(
    record: dict[str, Any],
    strategy: str,
    params: dict[str, Any] | None = None,
    prefix: bool = False,
) -> list[Chunk]:
    """Chunk one corpus record with the named strategy."""
    params = dict(params or {})
    blocks = record.get("blocks") or []
    doc_id = record["doc_id"]
    title = record.get("title") or ""

    if strategy == "blocks":
        pieces = _block_pack(
            blocks,
            int(params.get("target", 1200)),
            carry_header=bool(params.pop("carry_header", False)),
            overlap=int(params.get("overlap", 0)),
        )
    else:
        # ViDoRe ships page text and no blocks; without this the arm indexes nothing.
        text = "\n\n".join(block["text"] for block in blocks if block.get("text"))
        if not text.strip():
            text = record.get("text") or ""
        spans = get_chunker(strategy)(text, **params)
        pieces = [(text[start:end], _section_at(blocks, text, start)) for start, end in spans]

    chunks: list[Chunk] = []
    for position, (body, section) in enumerate(pieces):
        body = (body or "").strip()
        if not body:
            continue
        indexed = _with_prefix(body, title, section) if prefix else body
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(doc_id, strategy, position, body),
                doc_id=doc_id,
                text=indexed,
                kind="table" if section == "__table__" else "text",
            )
        )
    return chunks


def _block_pack(
    blocks: list[dict[str, Any]],
    target: int,
    carry_header: bool = False,
    overlap: int = 0,
) -> list[tuple[str, str | None]]:
    """Pack consecutive blocks, breaking at headings and never splitting a row.

    With ``carry_header`` the active table header is repeated at the top of every
    chunk of rows, so a chunk carries the column names its values belong to
    instead of losing them to the previous chunk.
    """
    out: list[tuple[str, str | None]] = []
    buffer: list[str] = []
    size = 0
    section: str | None = None
    header: str | None = None

    def flush(carry_tail: bool = False) -> None:
        """Emit the buffer. With ``overlap`` the trailing whole blocks are repeated
        at the head of the next chunk, giving fixed_overlap's redundancy without
        fixed_overlap's mid-row cuts."""
        nonlocal buffer, size
        if not buffer:
            return
        out.append(("\n".join(buffer), section))
        tail: list[str] = []
        if carry_tail and overlap > 0:
            budget = 0
            for text in reversed(buffer):
                if budget + len(text) > overlap:
                    break
                tail.insert(0, text)
                budget += len(text)
        buffer, size = tail, sum(len(t) for t in tail)

    def open_rows() -> None:
        if carry_header and header and not buffer:
            buffer.append(header)

    for block in blocks:
        text = (block.get("text") or "").strip()
        if not text:
            continue
        kind = block.get("kind") or "paragraph"
        if kind == "heading":
            flush()
            section = block.get("section") or text[:120]
            header = None
            buffer.append(text)
            size += len(text)
            continue
        if kind == "table_header":
            flush()
            header = text
            buffer.append(text)
            size += len(text)
            continue
        if size and size + len(text) > target:
            flush(carry_tail=True)
            if kind == "table_row":
                open_rows()
        buffer.append(text)
        size += len(text)
    flush()
    return out


def _section_at(blocks: list[dict[str, Any]], text: str, start: int) -> str | None:
    if not blocks:
        return None
    cursor = 0
    current: str | None = None
    for block in blocks:
        body = block.get("text") or ""
        if not body:
            continue
        if (block.get("kind") or "") == "heading":
            current = block.get("section") or body[:120]
        cursor += len(body) + 2
        if cursor > start:
            return current
    return current


def _with_prefix(body: str, title: str, section: str | None) -> str:
    head = " | ".join(part for part in (title, section) if part and part != "__table__")
    return f"{head}\n{body}" if head else body


def _chunk_id(doc_id: str, strategy: str, position: int, body: str) -> str:
    digest = hashlib.sha1(f"{doc_id}|{strategy}|{position}|{body[:200]}".encode()).hexdigest()
    return digest[:16]


def chunk_corpus(
    records: Iterable[dict[str, Any]],
    strategy: str,
    params: dict[str, Any] | None = None,
    prefix: bool = False,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for record in records:
        chunks.extend(chunk_document(record, strategy, params, prefix))
    return chunks
