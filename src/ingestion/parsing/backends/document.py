"""Document parser backend with an injectable extraction provider."""

from __future__ import annotations

from pathlib import Path

from ....models import DataObject, ParseResult
from ..contracts import DocumentProvider

DOCUMENT_EXTENSIONS = frozenset(
    {".pdf", ".pptx", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff"}
)


class DocumentParser:
    """Defer documents by default or delegate them to an injected provider."""

    supported_extensions = DOCUMENT_EXTENSIONS
    backend_name = "document"

    def __init__(
        self,
        provider: DocumentProvider | None = None,
        *,
        fallback_to_deferred: bool = False,
    ) -> None:
        self.provider = provider
        self.provider_name = getattr(provider, "provider_name", "deferred")
        self.fallback_to_deferred = fallback_to_deferred

    def parse(self, path: str | Path, data_object: DataObject) -> ParseResult:
        provider = self.provider
        if provider is None:
            return ParseResult.deferred(
                data_object.object_id,
                self.backend_name,
                route="document",
                reason="document_parser_not_implemented",
            )
        if Path(path).suffix.lower() not in provider.supported_extensions:
            return ParseResult.deferred(
                data_object.object_id,
                self.provider_name,
                route="document",
                reason="document_provider_does_not_support_extension",
            )

        try:
            parsed = provider.parse_file(path, data_object)
            parsed.metadata.setdefault("parser", self.provider_name.replace("_", "-"))
            parsed.metadata["backend"] = self.provider_name
        except Exception as exc:
            if self.fallback_to_deferred:
                return ParseResult.deferred(
                    data_object.object_id,
                    self.provider_name,
                    route="document",
                    reason="document_provider_failed_fallback_deferred",
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            return ParseResult.failed(
                data_object.object_id,
                self.provider_name,
                exc,
                route="document",
            )
        return ParseResult.success(
            data_object.object_id,
            self.provider_name,
            parsed,
            route="document",
        )
