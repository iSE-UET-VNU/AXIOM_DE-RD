from dataclasses import dataclass, field
from typing import Any

AXIOM_NATIVE_BLOCK_SOURCE = "parser_json"


@dataclass
class DataObject:
    object_id: str
    uri: str
    content_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedTable:
    name: str
    source_ref: str
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedData:
    object_id: str
    source_uri: str
    source_format: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tables: list[ParsedTable] = field(default_factory=list)
