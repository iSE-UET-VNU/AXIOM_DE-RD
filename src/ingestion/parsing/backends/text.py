"""Local parser for plain and structured text files."""

from __future__ import annotations

import json
from pathlib import Path
import re

from ....models import DataObject, ParsedData, ParseResult
from ..contracts import AXIOM_NATIVE_BLOCK_SOURCE


TEXT_EXTENSIONS = frozenset(
    {".txt", ".md", ".json", ".jsonl", ".yaml", ".yml"}
)
_MARKDOWN_HEADING = re.compile(
    r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$",
    re.MULTILINE,
)


class TextParserBackend:
    """Parse UTF-8 text into the canonical document contract."""

    supported_extensions = TEXT_EXTENSIONS
    backend_name = "text"

    def parse(self, path: str | Path, data_object: DataObject) -> ParseResult:
        file_path = Path(path)
        try:
            parsed = self._parse(file_path, data_object)
        except Exception as exc:
            return ParseResult.failed(
                data_object.object_id,
                self.backend_name,
                exc,
                route=self.backend_name,
            )
        return ParseResult.success(
            data_object.object_id,
            self.backend_name,
            parsed,
            route=self.backend_name,
        )

    def _parse(self, path: Path, data_object: DataObject) -> ParsedData:
        extension = path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(
                f"Unsupported text extension: {extension or '<none>'}"
            )

        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if extension == ".json":
            json.loads(text)

        markdown_title = _markdown_title(text) if extension == ".md" else None
        title = markdown_title or path.stem
        component_id = "/page/0/Text/0"
        source_blocks = (
            [
                {
                    "component_id": component_id,
                    "page": 0,
                    "block_index": 0,
                    "type": "Text",
                    "text": text,
                    "source": AXIOM_NATIVE_BLOCK_SOURCE,
                    "parser_source": "text_native",
                }
            ]
            if text
            else []
        )
        citations = [component_id] if source_blocks else []
        extraction = {
            "document_type": "text",
            "language": None,
            "title": title,
            "title_citations": citations if markdown_title else [],
            "main_text": text,
            "main_text_citations": citations,
            "tables": [],
            "figures": [],
            "formulas": [],
            "formulas_citations": [],
        }
        return ParsedData(
            object_id=data_object.object_id,
            source_uri=data_object.uri,
            source_format=extension.lstrip("."),
            rows=[
                {
                    "extraction": extraction,
                    "text": text,
                    "source_blocks": source_blocks,
                    "reading_order": citations,
                }
            ],
            text=text,
            metadata={
                "parser": self.backend_name,
                "page_count": None,
                "logical_page_count": 1,
                "pagination_source": "unavailable",
                "source_block_count": len(source_blocks),
                "table_count": 0,
                "figure_count": 0,
                "formula_count": 0,
                "reading_order_source": "text_native",
                "reading_order_complete": True,
            },
        )


def _markdown_title(text: str) -> str | None:
    match = _MARKDOWN_HEADING.search(text)
    return match.group(1).strip() if match else None
