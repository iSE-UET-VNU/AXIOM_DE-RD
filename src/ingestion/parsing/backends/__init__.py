"""Built-in parser backends."""

from .document import DOCUMENT_EXTENSIONS, DocumentParser
from .pptx import PPTX_EXTENSIONS, PptxConfig, PptxParserBackend
from .table import TABLE_EXTENSIONS, TableParser
from .text import TEXT_EXTENSIONS, TextParserBackend
from .word import WORD_EXTENSIONS, WordConfig, WordParserBackend

__all__ = [
    "DOCUMENT_EXTENSIONS",
    "PPTX_EXTENSIONS",
    "TABLE_EXTENSIONS",
    "TEXT_EXTENSIONS",
    "WORD_EXTENSIONS",
    "DocumentParser",
    "PptxConfig",
    "PptxParserBackend",
    "TableParser",
    "TextParserBackend",
    "WordConfig",
    "WordParserBackend",
]
