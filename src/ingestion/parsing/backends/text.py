"""Local text, JSON, and CSV parser backend."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from ....models import DataObject, ParsedData, ParsedTable, ParseResult
from ._tabular import normalize_headers, row_is_empty

TEXT_EXTENSIONS = frozenset(
    {".txt", ".md", ".csv", ".json", ".jsonl", ".yaml", ".yml"}
)


class TextParserBackend:
    """Parse UTF-8 text, JSON, and CSV inputs locally."""

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
                route="text",
            )
        return ParseResult.success(
            data_object.object_id,
            self.backend_name,
            parsed,
            route="text",
        )

    def _parse(self, path: Path, data_object: DataObject) -> ParsedData:
        extension = path.suffix.lower()
        source_format = extension.lstrip(".")
        if extension == ".csv":
            return self._parse_csv(path, data_object, source_format)

        text = _read_utf8_text(path)
        rows = _json_rows(json.loads(text)) if extension == ".json" else (
            [{"text": text}] if text else []
        )
        return ParsedData(
            object_id=data_object.object_id,
            source_uri=data_object.uri,
            source_format=source_format,
            rows=rows,
            text=text,
            metadata={"parser": self.backend_name},
        )

    def _parse_csv(
        self,
        path: Path,
        data_object: DataObject,
        source_format: str,
    ) -> ParsedData:
        text = _read_utf8_text(path)
        delimiter = _detect_csv_delimiter(text)
        raw_rows = [
            list(row)
            for row in csv.reader(StringIO(text, newline=""), delimiter=delimiter)
        ]
        header_index = _first_non_empty_row_index(raw_rows)

        if header_index is None:
            headers: list[str] = []
            table_rows: list[list[str]] = []
        else:
            relevant_rows = raw_rows[header_index:]
            while len(relevant_rows) > 1 and row_is_empty(relevant_rows[-1]):
                relevant_rows.pop()
            width = max((len(row) for row in relevant_rows), default=0)
            headers = normalize_headers(relevant_rows[0], width=width)
            table_rows = [row + [""] * (width - len(row)) for row in relevant_rows[1:]]

        table = ParsedTable(
            name=path.stem,
            source_ref=f"csv:0:{path.stem}",
            headers=headers,
            rows=table_rows,
            metadata={
                "parser": self.backend_name,
                "delimiter": delimiter,
                "header_row_number": header_index + 1 if header_index is not None else None,
            },
        )
        rows = [dict(zip(headers, row, strict=True)) for row in table_rows] if headers else []
        return ParsedData(
            object_id=data_object.object_id,
            source_uri=data_object.uri,
            source_format=source_format,
            rows=rows,
            text=None,
            tables=[table],
            metadata={
                "parser": self.backend_name,
                "table_count": 1,
                "delimiter": delimiter,
            },
        )


def _json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item if isinstance(item, dict) else {"value": item} for item in payload]
    if isinstance(payload, dict):
        return [payload]
    return [{"value": payload}]


def _read_utf8_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _first_non_empty_row_index(rows: list[list[str]]) -> int | None:
    return next((index for index, row in enumerate(rows) if not row_is_empty(row)), None)


def _detect_csv_delimiter(text: str) -> str:
    if not text:
        return ","
    try:
        return csv.Sniffer().sniff(text[:65536], delimiters=",;\t|").delimiter
    except csv.Error:
        return _best_effort_csv_delimiter(text)


def _best_effort_csv_delimiter(text: str) -> str:
    best_delimiter = ","
    best_score = (0, 0, 0)
    for delimiter in (",", ";", "\t", "|"):
        try:
            rows = list(csv.reader(StringIO(text, newline=""), delimiter=delimiter))
        except csv.Error:
            continue
        widths = [len(row) for row in rows if not row_is_empty(row)]
        multi_column_widths = [width for width in widths if width > 1]
        if not multi_column_widths:
            continue
        frequencies = {
            width: multi_column_widths.count(width) for width in set(multi_column_widths)
        }
        score = (
            len(multi_column_widths),
            max(frequencies.values(), default=0),
            sum(width - 1 for width in multi_column_widths),
        )
        if score > best_score:
            best_score = score
            best_delimiter = delimiter
    return best_delimiter
