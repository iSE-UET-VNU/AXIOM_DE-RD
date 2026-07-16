from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ingestion.parsing import (
    DocumentParser,
    ParseStatus,
    ParserRouter,
    ParsingService,
    TableParser,
    TextParserBackend,
)
from src.ingestion.runner import run as run_ingestion
from src.models import DataObject, ParsedData, ParseResult


def _data_object(path: Path, object_id: str = "object-1") -> DataObject:
    return DataObject(
        object_id=object_id,
        uri=path.as_posix(),
        content_type="application/octet-stream",
        metadata={"format": path.suffix.lstrip(".").lower()},
    )


class ParserRouterV1Tests(unittest.TestCase):
    def test_routes_supported_extensions_case_insensitively(self) -> None:
        router = ParsingService.from_config().router

        expected_routes = {
            "notes.TXT": TextParserBackend,
            "readme.Md": TextParserBackend,
            "records.CSV": TextParserBackend,
            "records.JsOnL": TextParserBackend,
            "workbook.XLSX": TableParser,
            "legacy.XlS": TableParser,
            "scan.PDF": DocumentParser,
            "slides.PpTx": DocumentParser,
            "photo.TiFf": DocumentParser,
        }
        for name, backend_type in expected_routes.items():
            with self.subTest(name=name):
                self.assertIsInstance(router.resolve(Path(name)), backend_type)

    def test_resolve_selects_a_backend_without_executing_it(self) -> None:
        class CountingBackend:
            backend_name = "counting"
            supported_extensions = frozenset({".count"})

            def __init__(self) -> None:
                self.calls = 0

            def parse(self, path: str | Path, data_object: DataObject) -> ParseResult:
                self.calls += 1
                return ParseResult(
                    source_object_id=data_object.object_id,
                    backend=self.backend_name,
                    status=ParseStatus.DEFERRED,
                )

        backend = CountingBackend()
        router = ParserRouter([backend])

        self.assertIs(router.resolve(Path("source.COUNT")), backend)
        self.assertEqual(backend.calls, 0)

    def test_duplicate_extensions_are_rejected_when_building_router(self) -> None:
        first = TextParserBackend()
        duplicate = TextParserBackend()

        with self.assertRaisesRegex(ValueError, "Duplicate parser extension"):
            ParserRouter([first, duplicate])

    def test_unknown_extension_returns_unsupported_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "archive.zip"
            path.write_bytes(b"not a supported input")

            service = ParsingService.from_config()
            result = service.parse(path, _data_object(path))

        self.assertEqual(result.status, ParseStatus.UNSUPPORTED)
        self.assertIsNone(result.parsed_data)
        self.assertIsNone(service.router.resolve(path))
        self.assertTrue(result.reason)

    def test_top_level_lift_alias_only_changes_the_document_backend(self) -> None:
        router = ParsingService.from_config({"provider": "lift_api"}).router

        document_backend = router.resolve(Path("scan.pdf"))
        self.assertIsInstance(document_backend, DocumentParser)
        self.assertEqual(document_backend.provider_name, "lift_api")
        self.assertIsInstance(router.resolve(Path("notes.txt")), TextParserBackend)
        self.assertIsInstance(router.resolve(Path("book.xlsx")), TableParser)

    def test_default_service_uses_lift_for_documents(self) -> None:
        router = ParsingService.from_config().router

        document_backend = router.resolve(Path("scan.pdf"))
        self.assertIsInstance(document_backend, DocumentParser)
        self.assertEqual(document_backend.provider_name, "lift_api")

    def test_document_provider_aliases_are_normalized(self) -> None:
        for alias in ("chandra", "chandra2", "chandra_2", "chandra-2"):
            with self.subTest(alias=alias):
                service = ParsingService.from_config(
                    {"document": {"provider": alias}}
                )
                backend = service.router.resolve(Path("scan.pdf"))
                self.assertIsInstance(backend, DocumentParser)
                self.assertEqual(backend.provider_name, "chandra2")

    def test_unknown_document_provider_is_rejected_at_startup(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported document provider"):
            ParsingService.from_config({"document": {"provider": "unknown"}})

    def test_document_parser_is_deferred_without_calling_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.pdf"
            path.write_bytes(b"%PDF-1.4\n")

            result = ParsingService.from_config(
                {"document": {"provider": "deferred"}}
            ).parse(path, _data_object(path))

        self.assertEqual(result.status, ParseStatus.DEFERRED)
        self.assertEqual(result.backend, "document")
        self.assertEqual(result.reason, "document_parser_not_implemented")
        self.assertIsNone(result.parsed_data)

    def test_explicit_lift_document_provider_uses_injected_client(self) -> None:
        class FakeLiftClient:
            provider_name = "lift_api"
            supported_extensions = frozenset({".pdf"})

            def __init__(self) -> None:
                self.calls: list[tuple[Path, DataObject]] = []

            def parse_file(self, path: str | Path, data_object: DataObject) -> ParsedData:
                self.calls.append((Path(path), data_object))
                return ParsedData(
                    object_id=data_object.object_id,
                    source_uri=data_object.uri,
                    source_format="pdf",
                    text="lift result",
                    metadata={"parser": "lift-api", "status": "complete"},
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.pdf"
            path.write_bytes(b"%PDF-1.4\n")
            data_object = _data_object(path)
            client = FakeLiftClient()
            backend = DocumentParser(client)

            result = backend.parse(path, data_object)

        self.assertEqual(result.status, ParseStatus.SUCCESS)
        self.assertEqual(result.route, "document")
        self.assertEqual(result.backend, "lift_api")
        self.assertEqual(result.parsed_data.text, "lift result")
        self.assertEqual(client.calls, [(path, data_object)])

    def test_provider_failure_honors_fallback_to_deferred(self) -> None:
        class FailingLiftClient:
            provider_name = "lift_api"
            supported_extensions = frozenset({".pdf"})

            def parse_file(self, path: str | Path, data_object: DataObject) -> ParsedData:
                raise RuntimeError("provider unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scan.pdf"
            path.write_bytes(b"%PDF-1.4\n")
            data_object = _data_object(path)
            deferred = DocumentParser(
                FailingLiftClient(),
                fallback_to_deferred=True,
            ).parse(path, data_object)
            failed = DocumentParser(
                FailingLiftClient(),
                fallback_to_deferred=False,
            ).parse(path, data_object)

        self.assertEqual(deferred.status, ParseStatus.DEFERRED)
        self.assertEqual(
            deferred.reason,
            "document_provider_failed_fallback_deferred",
        )
        self.assertEqual(deferred.backend, "lift_api")
        self.assertEqual(failed.status, ParseStatus.FAILED)
        self.assertEqual(failed.backend, "lift_api")

    def test_document_parser_does_not_call_provider_for_unsupported_extension(self) -> None:
        class PdfOnlyProvider:
            provider_name = "pdf_only"
            supported_extensions = frozenset({".pdf"})

            def __init__(self) -> None:
                self.calls = 0

            def parse_file(self, path: str | Path, data_object: DataObject) -> ParsedData:
                self.calls += 1
                raise AssertionError("provider should not be called")

        path = Path("slides.pptx")
        provider = PdfOnlyProvider()
        result = DocumentParser(provider).parse(path, _data_object(path))

        self.assertEqual(result.status, ParseStatus.DEFERRED)
        self.assertEqual(
            result.reason,
            "document_provider_does_not_support_extension",
        )
        self.assertEqual(result.backend, "pdf_only")
        self.assertEqual(provider.calls, 0)

    def test_service_is_the_canonical_parsing_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "note.txt"
            path.write_text("service text", encoding="utf-8")

            result = ParsingService.from_config().parse(path, _data_object(path))

        self.assertEqual(result.status, ParseStatus.SUCCESS)
        self.assertIsInstance(result.parsed_data, ParsedData)
        self.assertEqual(result.parsed_data.text, "service text")

    def test_service_isolates_custom_backend_exceptions(self) -> None:
        class RaisingBackend:
            backend_name = "raising"
            supported_extensions = frozenset({".boom"})

            def parse(self, path: str | Path, data_object: DataObject) -> ParseResult:
                raise RuntimeError("backend exploded")

        path = Path("source.boom")
        result = ParsingService(ParserRouter([RaisingBackend()])).parse(
            path, _data_object(path)
        )

        self.assertEqual(result.status, ParseStatus.FAILED)
        self.assertEqual(result.backend, "raising")
        self.assertEqual(result.error["message"], "backend exploded")

    def test_service_rejects_success_without_parsed_data(self) -> None:
        class InvalidBackend:
            backend_name = "invalid"
            supported_extensions = frozenset({".invalid"})

            def parse(self, path: str | Path, data_object: DataObject) -> ParseResult:
                return ParseResult(
                    source_object_id=data_object.object_id,
                    backend=self.backend_name,
                    status=ParseStatus.SUCCESS,
                    route=self.backend_name,
                )

        path = Path("source.invalid")
        result = ParsingService(ParserRouter([InvalidBackend()])).parse(
            path, _data_object(path)
        )

        self.assertEqual(result.status, ParseStatus.FAILED)
        self.assertEqual(result.reason, "missing_parsed_data")

    def test_mixed_directory_isolates_parse_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "good.txt").write_text("usable text", encoding="utf-8")
            (root / "broken.json").write_text('{"broken":', encoding="utf-8")
            (root / "pending.pdf").write_bytes(b"%PDF-1.4\n")
            (root / "unknown.zip").write_bytes(b"zip")

            output = run_ingestion(
                root,
                parser_config={
                    "provider": "router",
                    "document": {"provider": "deferred"},
                },
            )

        self.assertEqual(len(output.data_objects), 4)
        self.assertEqual(len(output.parsed_data), 1)
        self.assertEqual(output.parsed_data[0].text, "usable text")
        self.assertEqual(
            {result.status for result in output.parse_results},
            {
                ParseStatus.SUCCESS,
                ParseStatus.FAILED,
                ParseStatus.DEFERRED,
                ParseStatus.UNSUPPORTED,
            },
        )
        failed = next(
            result for result in output.parse_results if result.status is ParseStatus.FAILED
        )
        self.assertEqual(failed.source_object_id, next(
            item.object_id for item in output.data_objects if item.metadata["relative_uri"] == "broken.json"
        ))
        self.assertTrue(failed.error)
        self.assertEqual(len(output.normalized_texts), 1)

    def test_dotfiles_are_inventoried_and_reported_as_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".source-metadata"
            path.write_text("metadata", encoding="utf-8")

            output = run_ingestion(temp_dir)

        self.assertEqual(len(output.data_objects), 1)
        self.assertEqual(len(output.parse_results), 1)
        self.assertEqual(output.parse_results[0].status, ParseStatus.UNSUPPORTED)

    def test_corrupt_workbook_does_not_block_a_later_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a-broken.xlsx").write_bytes(b"not an Excel workbook")
            (root / "z-valid.txt").write_text("still parsed", encoding="utf-8")

            output = run_ingestion(root)

        result_by_name = {
            data_object.metadata["relative_uri"]: result
            for data_object, result in zip(output.data_objects, output.parse_results, strict=True)
        }
        self.assertEqual(result_by_name["a-broken.xlsx"].status, ParseStatus.FAILED)
        self.assertEqual(result_by_name["z-valid.txt"].status, ParseStatus.SUCCESS)
        self.assertEqual(len(output.errors), 1)

    def test_inventory_error_is_file_scoped_and_later_files_continue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            blocked = root / "a-blocked.txt"
            blocked.write_text("blocked", encoding="utf-8")
            (root / "z-valid.txt").write_text("valid", encoding="utf-8")
            real_stat = Path.stat

            def flaky_stat(path: Path, *args: object, **kwargs: object):
                if path.name == blocked.name:
                    raise PermissionError("simulated inventory failure")
                return real_stat(path, *args, **kwargs)

            with patch.object(Path, "stat", flaky_stat):
                output = run_ingestion(root)

        result_by_name = {
            data_object.metadata["relative_uri"]: result
            for data_object, result in zip(output.data_objects, output.parse_results, strict=True)
        }
        self.assertEqual(result_by_name[blocked.name].status, ParseStatus.FAILED)
        self.assertEqual(result_by_name[blocked.name].backend, "inventory")
        self.assertEqual(result_by_name["z-valid.txt"].status, ParseStatus.SUCCESS)
        self.assertEqual(output.errors[0]["stage"], "ingestion.inventory")

    def test_downstream_stage_errors_do_not_rewrite_successful_parse_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "source.txt").write_text("parsed", encoding="utf-8")

            with patch(
                "src.ingestion.runner.build_initial_schema",
                side_effect=RuntimeError("schema failure"),
            ):
                schema_error = run_ingestion(root)
            with patch(
                "src.ingestion.runner.normalize_parsed_data",
                side_effect=RuntimeError("normalization failure"),
            ):
                normalization_error = run_ingestion(root)

        self.assertEqual(schema_error.parse_results[0].status, ParseStatus.SUCCESS)
        self.assertEqual(len(schema_error.parsed_data), 1)
        self.assertEqual(schema_error.errors[0]["stage"], "ingestion.schema_inference")
        self.assertEqual(len(schema_error.normalized_texts), 1)
        self.assertEqual(normalization_error.parse_results[0].status, ParseStatus.SUCCESS)
        self.assertEqual(len(normalization_error.parsed_data), 1)
        self.assertEqual(normalization_error.errors[0]["stage"], "ingestion.normalization")


if __name__ == "__main__":
    unittest.main()
