"""Configuration and file-scoped execution for ingestion parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...models import DataObject, ParseResult, ParseStatus
from .backends import DocumentParser, TableParser, TextParserBackend
from .chandra2 import Chandra2Config, Chandra2Provider
from .lift import LiftAPIConfig, LiftAPIParserClient
from .router import ParserRouter


class ParsingService:
    """Compose parser backends and execute one source object safely."""

    def __init__(self, router: ParserRouter) -> None:
        self.router = router

    @classmethod
    def from_config(
        cls,
        parser_config: dict[str, Any] | None = None,
    ) -> "ParsingService":
        config = parser_config if isinstance(parser_config, dict) else {}
        document_parser = _build_document_parser(
            _document_provider_name(config),
            config,
        )
        return cls(
            ParserRouter(
                [
                    TextParserBackend(),
                    TableParser(),
                    document_parser,
                ]
            )
        )

    def parse(
        self,
        path: str | Path,
        data_object: DataObject,
    ) -> ParseResult:
        """Resolve and execute one backend without leaking file-scoped errors."""
        backend = self.router.resolve(path)
        if backend is None:
            extension = Path(path).suffix.lower() or "<none>"
            return ParseResult.unsupported(data_object.object_id, extension)

        try:
            result = backend.parse(path, data_object)
        except Exception as exc:
            return ParseResult.failed(
                data_object.object_id,
                backend.backend_name,
                exc,
                route=backend.backend_name,
            )

        if result.status == ParseStatus.SUCCESS and result.parsed_data is None:
            return ParseResult.failed(
                result.source_object_id,
                result.backend,
                ValueError("A successful ParseResult must include parsed_data."),
                route=result.route,
                reason="missing_parsed_data",
                error_type="InvalidParseResult",
            )
        return result

    def parse_many(
        self,
        documents: list[tuple[str | Path, DataObject]],
    ) -> list[ParseResult]:
        """Parse many inputs while sharing a continuous Chandra2 page queue."""

        results: list[ParseResult | None] = [None] * len(documents)
        chandra_groups: dict[
            int,
            tuple[DocumentParser, list[tuple[int, str | Path, DataObject]]],
        ] = {}

        for index, (path, data_object) in enumerate(documents):
            backend = self.router.resolve(path)
            provider = (
                backend.provider if isinstance(backend, DocumentParser) else None
            )
            if (
                isinstance(provider, Chandra2Provider)
                and provider.config.continuous_page_queue
                and Path(path).suffix.lower() in provider.supported_extensions
            ):
                key = id(backend)
                if key not in chandra_groups:
                    chandra_groups[key] = (backend, [])
                chandra_groups[key][1].append((index, path, data_object))
                continue
            results[index] = self.parse(path, data_object)

        for backend, group in chandra_groups.values():
            provider = backend.provider
            assert isinstance(provider, Chandra2Provider)
            try:
                parsed_documents = provider.parse_files(
                    [(path, data_object) for _, path, data_object in group]
                )
                if len(parsed_documents) != len(group):
                    raise RuntimeError(
                        "Chandra2 returned a different number of documents than inputs."
                    )
            except Exception as exc:
                for index, _, data_object in group:
                    results[index] = _document_provider_failure(
                        backend,
                        data_object,
                        exc,
                    )
                continue

            for (index, _, data_object), parsed in zip(group, parsed_documents):
                parsed.metadata.setdefault("parser", "chandra2")
                parsed.metadata["backend"] = "chandra2"
                results[index] = ParseResult.success(
                    data_object.object_id,
                    "chandra2",
                    parsed,
                    route="document",
                )

        if any(result is None for result in results):
            raise RuntimeError("ParsingService did not produce a result for every input.")
        return [result for result in results if result is not None]


def _build_document_parser(
    provider_name: str,
    parser_config: dict[str, Any],
) -> DocumentParser:
    fallback_to_deferred = _fallback_to_deferred(provider_name, parser_config)
    match provider_name:
        case "deferred":
            return DocumentParser()
        case "lift_api":
            provider_config = LiftAPIConfig.from_mapping(
                _provider_config(parser_config, "lift_api")
            )
            provider = LiftAPIParserClient(provider_config)
        case "chandra2":
            provider_config = Chandra2Config.from_mapping(
                _provider_config(parser_config, "chandra2")
            )
            provider = Chandra2Provider(provider_config)
        case _:
            raise ValueError(
                "Unsupported document provider "
                f"{provider_name!r}. Expected deferred, lift_api, or chandra2."
            )
    return DocumentParser(
        provider,
        fallback_to_deferred=fallback_to_deferred,
    )


def _document_provider_name(parser_config: dict[str, Any]) -> str:
    top_level_value = parser_config.get("provider")
    top_level_provider = (
        _normalize_provider_name(str(top_level_value))
        if top_level_value is not None
        else ""
    )
    document_config = parser_config.get("document")
    document_config = document_config if isinstance(document_config, dict) else {}
    document_provider = _normalize_provider_name(
        str(document_config.get("provider") or "")
    )

    # Routing is always enabled internally. The top-level provider remains the
    # backwards-compatible switch for the visual/document backend.
    if top_level_provider and top_level_provider != "router":
        return top_level_provider
    if document_provider:
        return document_provider
    return "lift_api"


def _normalize_provider_name(value: str) -> str:
    provider = value.strip().lower().replace("-", "_")
    if provider in {"lift", "lift_api"}:
        return "lift_api"
    if provider in {"chandra", "chandra2", "chandra_2"}:
        return "chandra2"
    return provider


def _provider_config(
    parser_config: dict[str, Any],
    provider_name: str,
) -> dict[str, Any]:
    config = parser_config.get(provider_name)
    return config if isinstance(config, dict) else {}


def _fallback_to_deferred(
    provider_name: str,
    parser_config: dict[str, Any],
) -> bool:
    document_config = parser_config.get("document")
    document_config = document_config if isinstance(document_config, dict) else {}
    if "fallback_to_deferred" in document_config:
        return bool(document_config["fallback_to_deferred"])
    if provider_name == "lift_api":
        lift_config = _provider_config(parser_config, "lift_api")
        return bool(lift_config.get("fallback_to_local", False))
    return False


def _document_provider_failure(
    backend: DocumentParser,
    data_object: DataObject,
    exc: Exception,
) -> ParseResult:
    if backend.fallback_to_deferred:
        return ParseResult.deferred(
            data_object.object_id,
            backend.provider_name,
            route="document",
            reason="document_provider_failed_fallback_deferred",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
    return ParseResult.failed(
        data_object.object_id,
        backend.provider_name,
        exc,
        route="document",
    )
