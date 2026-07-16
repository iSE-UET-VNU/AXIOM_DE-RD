"""Built-in parser backends."""

from .document import DOCUMENT_EXTENSIONS, DocumentParser
from .table import TABLE_EXTENSIONS, TableParser
from .text import TEXT_EXTENSIONS, TextParserBackend

__all__ = [
    "DOCUMENT_EXTENSIONS",
    "TABLE_EXTENSIONS",
    "TEXT_EXTENSIONS",
    "DocumentParser",
    "TableParser",
    "TextParserBackend",
]
