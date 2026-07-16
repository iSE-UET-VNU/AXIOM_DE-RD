"""Public contracts and composition API for ingestion parsing."""

from ...models import ParsedTable, ParseResult, ParseStatus
from .backends import (
    DOCUMENT_EXTENSIONS,
    TABLE_EXTENSIONS,
    TEXT_EXTENSIONS,
    DocumentParser,
    TableParser,
    TextParserBackend,
)
from .contracts import DocumentProvider, ParserBackend
from .chandra2 import CHANDRA2_EXTENSIONS, Chandra2Config, Chandra2Provider
from .lift import LiftAPIConfig, LiftAPIParserClient
from .router import ParserRouter
from .service import ParsingService

__all__ = [
    "DOCUMENT_EXTENSIONS",
    "CHANDRA2_EXTENSIONS",
    "TABLE_EXTENSIONS",
    "TEXT_EXTENSIONS",
    "DocumentProvider",
    "DocumentParser",
    "Chandra2Config",
    "Chandra2Provider",
    "LiftAPIConfig",
    "LiftAPIParserClient",
    "ParsedTable",
    "ParseResult",
    "ParseStatus",
    "ParserBackend",
    "ParserRouter",
    "ParsingService",
    "TableParser",
    "TextParserBackend",
]
