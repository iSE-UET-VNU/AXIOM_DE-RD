from __future__ import annotations

from collections import Counter
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import cleaning, enrichment, indexing_cataloging
from src.ingestion.parsing import (
    Chandra2Config,
    Chandra2Provider,
    DocumentParser,
    ParseStatus,
    ParsingService,
)
from src.ingestion.parsing.chandra2 import _ChandraRuntime
from src.ingestion.runner import run as run_ingestion
from src.models import DataObject
from src.utils.config import resolve_parser_config


class _FakeBatchInput:
    def __init__(self, *, image: object, prompt_type: str) -> None:
        self.image = image
        self.prompt_type = prompt_type


class _FakeManager:
    def __init__(
        self,
        results: dict[object, object],
        *,
        error: Exception | None = None,
    ) -> None:
        self.results = results
        self.error = error
        self.calls: list[tuple[list[_FakeBatchInput], dict[str, object]]] = []

    def generate(
        self,
        batch: list[_FakeBatchInput],
        **kwargs: object,
    ) -> list[object]:
        self.calls.append((batch, kwargs))
        if self.error:
            raise self.error
        return [self.results[item.image] for item in batch]


def _runtime(pages: list[object], manager: _FakeManager) -> _ChandraRuntime:
    return _ChandraRuntime(
        load_file=lambda path, config: pages,
        inference_manager=lambda **kwargs: manager,
        batch_input_item=_FakeBatchInput,
    )


def _result(markdown: str, token_count: int, *, error: bool = False) -> object:
    return SimpleNamespace(
        markdown=markdown,
        token_count=token_count,
        error=error,
    )


def _data_object(path: Path, object_id: str = "document-1") -> DataObject:
    return DataObject(
        object_id=object_id,
        uri=path.as_posix(),
        content_type="application/pdf",
        metadata={"format": path.suffix.lstrip(".").lower()},
    )


class Chandra2ProviderTests(unittest.TestCase):
    def test_default_lift_service_does_not_load_chandra_runtime(self) -> None:
        with patch("src.ingestion.parsing.chandra2._load_runtime") as load_runtime:
            service = ParsingService.from_config()

        load_runtime.assert_not_called()
        backend = service.router.resolve(Path("scan.pdf"))
        self.assertIsInstance(backend, DocumentParser)
        self.assertEqual(backend.provider_name, "lift_api")

    def test_service_builds_chandra_provider_from_config(self) -> None:
        service = ParsingService.from_config(
            {
                "document": {
                    "provider": "chandra2",
                    "fallback_to_deferred": True,
                },
                "chandra2": {"batch_size": 7},
            }
        )

        backend = service.router.resolve(Path("scan.pdf"))
        self.assertIsInstance(backend, DocumentParser)
        self.assertIsInstance(backend.provider, Chandra2Provider)
        self.assertEqual(backend.provider.config.batch_size, 7)
        self.assertTrue(backend.fallback_to_deferred)

    def test_local_alias_builds_hf_provider_with_safe_batch_default(self) -> None:
        service = ParsingService.from_config(
            {
                "document": {"provider": "chandra2"},
                "chandra2": {"method": "local"},
            }
        )

        backend = service.router.resolve(Path("scan.pdf"))
        self.assertIsInstance(backend, DocumentParser)
        self.assertIsInstance(backend.provider, Chandra2Provider)
        self.assertEqual(backend.provider.config.method, "hf")
        self.assertEqual(backend.provider.config.batch_size, 1)

    def test_chandra_defers_pptx_without_loading_runtime(self) -> None:
        with patch("src.ingestion.parsing.chandra2._load_runtime") as load_runtime:
            service = ParsingService.from_config(
                {"document": {"provider": "chandra2"}}
            )
            path = Path("slides.pptx")
            result = service.parse(path, _data_object(path))

        load_runtime.assert_not_called()
        self.assertEqual(result.status, ParseStatus.DEFERRED)
        self.assertEqual(
            result.reason,
            "document_provider_does_not_support_extension",
        )

    def test_lift_fallback_alias_remains_compatible(self) -> None:
        service = ParsingService.from_config(
            {
                "document": {"provider": "lift_api"},
                "lift_api": {"fallback_to_local": True},
            }
        )

        backend = service.router.resolve(Path("scan.pdf"))
        self.assertIsInstance(backend, DocumentParser)
        self.assertTrue(backend.fallback_to_deferred)

    def test_parses_pages_in_order_and_writes_debug_artifacts(self) -> None:
        pages = ["page-1", "page-2", "page-3"]
        manager = _FakeManager(
            {
                "page-1": _result("First", 3),
                "page-2": _result("Second", 4),
                "page-3": _result("Third", 5),
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "scan.pdf"
            path.write_bytes(b"%PDF-1.4\n")
            provider = Chandra2Provider(
                Chandra2Config(batch_size=2, output_dir=str(root / "outputs")),
                _runtime_loader=lambda: _runtime(pages, manager),
            )

            with patch.dict("os.environ", {"VLLM_MODEL_NAME": "chandra-test"}):
                parsed = provider.parse_file(path, _data_object(path))
            bundle = root / "outputs" / "scan--document-1"
            metadata = json.loads(
                (bundle / "metadata.json").read_text(encoding="utf-8")
            )

        self.assertEqual(parsed.text, "First\n\nSecond\n\nThird")
        self.assertEqual(parsed.rows, [{"text": parsed.text}])
        self.assertEqual(parsed.metadata["page_count"], 3)
        self.assertEqual(parsed.metadata["token_count"], 12)
        self.assertEqual([len(batch) for batch, _ in manager.calls], [2, 1])
        self.assertEqual(
            [item.prompt_type for batch, _ in manager.calls for item in batch],
            ["ocr_layout", "ocr_layout", "ocr_layout"],
        )
        self.assertEqual(
            manager.calls[0][1],
            {
                "include_images": False,
                "include_headers_footers": False,
                "max_output_tokens": 12384,
                "max_workers": 4,
                "max_retries": 6,
            },
        )
        self.assertEqual(metadata["page_count"], 3)
        self.assertEqual(metadata["token_count"], 12)
        self.assertEqual(metadata["method"], "vllm")
        self.assertEqual(metadata["model_name"], "chandra-test")

    def test_hf_method_runs_in_process_without_vllm_options(self) -> None:
        manager = _FakeManager({"page-1": _result("Local result", 2)})
        manager_methods: list[str] = []
        runtime = _ChandraRuntime(
            load_file=lambda path, config: ["page-1"],
            inference_manager=lambda **kwargs: (
                manager_methods.append(str(kwargs["method"])) or manager
            ),
            batch_input_item=_FakeBatchInput,
        )
        provider = Chandra2Provider(
            Chandra2Config.from_mapping(
                {"method": "hf", "save_raw_outputs": False}
            ),
            _runtime_loader=lambda: runtime,
        )

        with patch.dict(
            "os.environ", {"MODEL_CHECKPOINT": "datalab-to/chandra-ocr-2-test"}
        ):
            parsed = provider.parse_file(
                Path("scan.pdf"), _data_object(Path("scan.pdf"))
            )

        self.assertEqual(manager_methods, ["hf"])
        self.assertEqual(parsed.text, "Local result")
        self.assertEqual(parsed.metadata["method"], "hf")
        self.assertEqual(
            parsed.metadata["model_name"], "datalab-to/chandra-ocr-2-test"
        )
        self.assertEqual(len(manager.calls), 1)
        self.assertNotIn("max_workers", manager.calls[0][1])
        self.assertNotIn("max_retries", manager.calls[0][1])

    def test_any_failed_page_fails_the_document_without_partial_artifacts(self) -> None:
        pages = ["page-1", "page-2"]
        manager = _FakeManager(
            {
                "page-1": _result("First", 3),
                "page-2": _result("", 0, error=True),
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "scan.pdf"
            path.write_bytes(b"%PDF-1.4\n")
            provider = Chandra2Provider(
                Chandra2Config(output_dir=str(root / "outputs")),
                _runtime_loader=lambda: _runtime(pages, manager),
            )

            with self.assertRaisesRegex(RuntimeError, r"page\(s\): 2"):
                provider.parse_file(path, _data_object(path))

            self.assertFalse((root / "outputs").exists())

    def test_config_rejects_invalid_batch_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            Chandra2Config.from_mapping({"batch_size": 0})
        with self.assertRaisesRegex(ValueError, "max_retries"):
            Chandra2Config.from_mapping({"max_retries": -1})
        with self.assertRaisesRegex(ValueError, "method"):
            Chandra2Config.from_mapping({"method": "unknown"})

    def test_parser_config_resolves_chandra_output_into_run_work_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            resolved = resolve_parser_config(
                root,
                {"document": {"provider": "chandra2"}},
                root / "work",
                "run-1",
            )

        self.assertEqual(
            Path(resolved["chandra2"]["output_dir"]),
            root / "work" / "run-1" / "chandra2",
        )
        self.assertEqual(Path(resolved["chandra2"]["project_root"]), root)


class Chandra2EndToEndTests(unittest.TestCase):
    def test_pdf_reaches_canonical_text_and_index_records(self) -> None:
        manager = _FakeManager(
            {
                "page-1": _result("# Invoice", 4),
                "page-2": _result("Total: 100", 3),
            }
        )
        runtime = _runtime(["page-1", "page-2"], manager)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            raw.mkdir()
            (raw / "invoice.pdf").write_bytes(b"%PDF-1.4\n")
            parser_config = {
                "document": {"provider": "chandra2"},
                "chandra2": {
                    "output_dir": str(root / "work"),
                    "project_root": str(root),
                },
            }
            with patch(
                "src.ingestion.parsing.chandra2._load_runtime",
                return_value=runtime,
            ):
                ingested = run_ingestion(
                    raw,
                    parser_config=parser_config,
                    project_root=root,
                )
            cleaned = cleaning.run(ingested.parsed_data, ingested.initial_schemas)
            enriched = enrichment.run(cleaned.cleaned_data, cleaned.cleaned_schemas)
            indexed = indexing_cataloging.run(
                enriched.enriched_data,
                enriched.enriched_schemas,
                indexing_config={"embeddings": {"enabled": False}},
                normalized_texts=ingested.normalized_texts,
                normalized_documents=ingested.documents,
            )

        self.assertEqual(ingested.parse_results[0].status, ParseStatus.SUCCESS)
        self.assertEqual(ingested.parse_results[0].backend, "chandra2")
        self.assertEqual(ingested.parsed_data[0].text, "# Invoice\n\nTotal: 100")
        self.assertEqual(len(ingested.normalized_texts), 1)
        self.assertEqual(len(ingested.documents), 1)
        self.assertEqual(
            Counter(record.index_type for record in indexed.index_records),
            Counter({"document": 1, "text_chunk": 1, "catalog": 1}),
        )

    def test_chandra_failure_is_file_scoped(self) -> None:
        manager = _FakeManager({}, error=ConnectionError("vLLM unavailable"))
        runtime = _runtime(["page-1"], manager)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.pdf").write_bytes(b"%PDF-1.4\n")
            (root / "b.txt").write_text("still parsed", encoding="utf-8")
            with patch(
                "src.ingestion.parsing.chandra2._load_runtime",
                return_value=runtime,
            ):
                output = run_ingestion(
                    root,
                    parser_config={
                        "document": {"provider": "chandra2"},
                        "chandra2": {"save_raw_outputs": False},
                    },
                )

        result_by_name = {
            data_object.metadata["relative_uri"]: result
            for data_object, result in zip(
                output.data_objects,
                output.parse_results,
                strict=True,
            )
        }
        self.assertEqual(result_by_name["a.pdf"].status, ParseStatus.FAILED)
        self.assertEqual(result_by_name["a.pdf"].backend, "chandra2")
        self.assertEqual(result_by_name["b.txt"].status, ParseStatus.SUCCESS)
        self.assertEqual(len(output.parsed_data), 1)
        self.assertEqual(len(output.errors), 1)


if __name__ == "__main__":
    unittest.main()
