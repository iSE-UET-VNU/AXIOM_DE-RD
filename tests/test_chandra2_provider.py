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
)
from src.ingestion.parsing.backends import DocumentParser
from src.ingestion.parsing.chandra2 import _ChandraRuntime, _replace_tables
from src.ingestion.parsing.service import ParsingService
from src.ingestion.runner import run as run_ingestion
from src.ingestion.schema_inference import build_initial_schema
from src.models import DataObject, ParseStatus
from src.reading_order import reading_order_from_rows
from src.utils.config import resolve_parser_config


class _FakeBatchInput:
    def __init__(
        self,
        *,
        image: object,
        prompt_type: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.image = image
        self.prompt_type = prompt_type
        self.prompt = prompt


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
        generated = []
        for item in batch:
            if item.prompt is not None:
                generated.append(self.results["__table_result__"])
            elif item.image in self.results:
                generated.append(self.results[item.image])
            else:
                generated.append(self.results["__page_result__"])
        return generated


class _FakeImage:
    def __init__(self, payload: bytes = b"image") -> None:
        self.payload = payload

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.payload)


class _FakePageImage:
    size = (1000, 1400)

    def crop(self, box: tuple[int, int, int, int]) -> "_FakeCropImage":
        return _FakeCropImage((box[2] - box[0], box[3] - box[1]))


class _FakeCropImage(_FakeImage):
    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__(b"table-crop")
        self.size = size

    def copy(self) -> "_FakeCropImage":
        return _FakeCropImage(self.size)

    def resize(
        self,
        size: tuple[int, int],
        resample: object | None = None,
    ) -> "_FakeCropImage":
        return _FakeCropImage(size)


def _runtime(pages: list[object], manager: _FakeManager) -> _ChandraRuntime:
    return _ChandraRuntime(
        load_file=lambda path, config: pages,
        inference_manager=lambda **kwargs: manager,
        batch_input_item=_FakeBatchInput,
    )


def _result(
    markdown: str,
    token_count: int,
    *,
    error: bool = False,
    raw: str = "",
    html: str = "",
    chunks: list[dict[str, object]] | None = None,
    page_box: list[int] | None = None,
    images: dict[str, object] | None = None,
) -> object:
    return SimpleNamespace(
        markdown=markdown,
        token_count=token_count,
        error=error,
        raw=raw,
        html=html,
        chunks=chunks or [],
        page_box=page_box or [],
        images=images or {},
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
        self.assertEqual(parsed.rows[0]["text"], parsed.text)
        self.assertEqual(parsed.rows[0]["extraction"]["main_text"], parsed.text)
        self.assertEqual(parsed.rows[0]["source_blocks"], [])
        self.assertEqual(parsed.rows[0]["reading_order"], [])
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
                "include_images": True,
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
        self.assertFalse(parsed.metadata["reading_order_complete"])

    def test_adapts_layout_blocks_components_and_images_to_ingestion_contract(
        self,
    ) -> None:
        page_one_chunks = [
            {
                "bbox": [100, 50, 900, 120],
                "label": "Section-Header",
                "content": "<h1>Invoice 2026</h1>",
            },
            {
                "bbox": [100, 150, 900, 450],
                "label": "Table",
                "content": (
                    '<table><tr><th>Item</th><th>Value</th></tr>'
                    '<tr><td rowspan="2">A</td><td>1</td></tr>'
                    '<tr><td colspan="1">2</td></tr></table>'
                ),
            },
            {
                "bbox": [100, 455, 900, 490],
                "label": "Caption",
                "content": "<p>Table 1. Totals</p>",
            },
            {
                "bbox": [100, 520, 500, 800],
                "label": "Image",
                "content": '<img alt="Revenue chart"/>',
            },
            {
                "bbox": [100, 805, 500, 840],
                "label": "Caption",
                "content": "<p>Figure 1. Revenue</p>",
            },
            {
                "bbox": [600, 700, 900, 850],
                "label": "Text",
                "content": '<img alt="Handwritten signature"/>',
            },
        ]
        page_two_chunks = [
            {
                "bbox": [150, 200, 850, 300],
                "label": "Equation-Block",
                "content": "<math>E = mc^2</math>",
            }
        ]
        manager = _FakeManager(
            {
                "page-1": _result(
                    "# Invoice 2026\n\n| Item | Value |\n|---|---|",
                    20,
                    raw="<div>raw page 1</div>",
                    html=(
                        "<h1>Invoice 2026</h1>"
                        '<img alt="Revenue chart" src="chart.webp"/>'
                    ),
                    chunks=page_one_chunks,
                    page_box=[0, 0, 1000, 1400],
                    images={"chart.webp": _FakeImage(b"chart")},
                ),
                "page-2": _result(
                    "E = mc^2",
                    5,
                    raw="<div>raw page 2</div>",
                    html="<math>E = mc^2</math>",
                    chunks=page_two_chunks,
                    page_box=[0, 0, 1000, 1400],
                ),
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "invoice.pdf"
            path.write_bytes(b"%PDF-1.4\n")
            provider = Chandra2Provider(
                Chandra2Config(
                    batch_size=1,
                    output_dir=str(root / "outputs"),
                    project_root=str(root),
                ),
                _runtime_loader=lambda: _runtime(
                    ["page-1", "page-2"],
                    manager,
                ),
            )

            parsed = provider.parse_file(path, _data_object(path))
            bundle = root / "outputs" / "invoice--document-1"
            chunks_payload = json.loads(
                (bundle / "chunks.json").read_text(encoding="utf-8")
            )
            raw_metadata = json.loads(
                (bundle / "metadata.json").read_text(encoding="utf-8")
            )
            saved_chart = (bundle / "images" / "chart.webp").read_bytes()
            artifact_names = {
                item.relative_to(bundle).as_posix()
                for item in bundle.rglob("*")
                if item.is_file()
            }

        row = parsed.rows[0]
        extraction = row["extraction"]
        self.assertEqual(extraction["title"], "Invoice 2026")
        self.assertIsNone(extraction["language"])
        self.assertIsNone(extraction["document_type"])
        self.assertEqual(len(extraction["tables"]), 1)
        self.assertIn('rowspan="2"', extraction["tables"][0]["content"])
        self.assertIn('colspan="1"', extraction["tables"][0]["content"])
        self.assertEqual(
            extraction["tables"][0]["caption"],
            "Table 1. Totals",
        )
        self.assertEqual(len(extraction["figures"]), 2)
        self.assertEqual(
            extraction["figures"][0]["caption"],
            "Figure 1. Revenue",
        )
        self.assertEqual(extraction["figures"][1]["caption"], "")
        self.assertEqual(extraction["formulas"], ["E = mc^2"])
        self.assertEqual(parsed.tables, [])
        initial_schema = build_initial_schema(parsed)
        self.assertEqual(
            initial_schema.metadata["component_counts"]["tables"],
            1,
        )
        self.assertEqual(
            initial_schema.metadata["component_counts"]["figures"],
            2,
        )

        self.assertEqual(len(row["source_blocks"]), 7)
        first_block = row["source_blocks"][0]
        self.assertEqual(
            first_block["component_id"],
            "/page/0/SectionHeader/0",
        )
        self.assertEqual(first_block["bbox"], [100, 50, 900, 120])
        self.assertEqual(
            first_block["polygon"],
            [[100, 50], [900, 50], [900, 120], [100, 120]],
        )
        self.assertEqual(first_block["page_box"], [0, 0, 1000, 1400])
        self.assertEqual(
            row["reading_order"][-1],
            "/page/1/EquationBlock/0",
        )
        final_blocks, final_order, final_order_meta = reading_order_from_rows(
            parsed.rows
        )
        self.assertEqual(len(final_blocks), 7)
        self.assertEqual(final_order, row["reading_order"])
        self.assertEqual(final_order_meta["source"], "chandra2_layout")
        self.assertTrue(final_order_meta["complete"])
        self.assertTrue(parsed.metadata["reading_order_complete"])
        self.assertEqual(parsed.metadata["reading_order_source"], "chandra2_layout")
        self.assertEqual(parsed.metadata["table_count"], 1)
        self.assertEqual(parsed.metadata["figure_count"], 2)
        self.assertEqual(parsed.metadata["formula_count"], 1)
        self.assertEqual(parsed.metadata["image_count"], 1)
        self.assertEqual(parsed.metadata["image_files"][0]["status"], "saved")
        self.assertEqual(
            parsed.metadata["image_files"][1]["status"],
            "unavailable",
        )

        self.assertEqual(saved_chart, b"chart")
        self.assertEqual(len(chunks_payload["pages"]), 2)
        self.assertEqual(raw_metadata["source_block_count"], 7)
        self.assertTrue(
            {
                "result.md",
                "result.html",
                "raw.html",
                "chunks.json",
                "metadata.json",
                "pages/page_0001.raw.html",
                "pages/page_0001.clean.html",
                "pages/page_0001.chunks.json",
                "pages/page_0002.raw.html",
                "pages/page_0002.clean.html",
                "pages/page_0002.chunks.json",
                "images/chart.webp",
            }.issubset(artifact_names)
        )

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
                {
                    "method": "hf",
                    "include_images": False,
                    "save_raw_outputs": False,
                }
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
        self.assertFalse(manager.calls[0][1]["include_images"])
        self.assertNotIn("max_workers", manager.calls[0][1])
        self.assertNotIn("max_retries", manager.calls[0][1])

    def test_refines_detected_table_crop_and_preserves_merged_cells(self) -> None:
        original = "<table><tr><td>A</td><td></td></tr></table>"
        refined = (
            '<table><tr><td rowspan="2">A</td><td>1</td></tr>'
            '<tr><td colspan="1">2</td></tr></table>'
        )
        manager = _FakeManager(
            {
                "__page_result__": _result(
                    f"Before\n\n{original}\n\nAfter",
                    10,
                    raw=f'<div data-label="Table">{original}</div>',
                    html=f"<p>Before</p>{original}<p>After</p>",
                    chunks=[
                        {
                            "bbox": [100, 200, 900, 800],
                            "label": "Table",
                            "content": original,
                        }
                    ],
                    page_box=[0, 0, 1000, 1400],
                ),
                "__table_result__": _result(
                    refined,
                    7,
                    raw=f"model prose that is ignored\n{refined}",
                ),
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "table.pdf"
            path.write_bytes(b"%PDF-1.4\n")
            provider = Chandra2Provider(
                Chandra2Config.from_mapping(
                    {
                        "method": "hf",
                        "refine_tables": True,
                        "output_dir": str(root / "outputs"),
                        "project_root": str(root),
                    }
                ),
                _runtime_loader=lambda: _runtime([_FakePageImage()], manager),
            )

            parsed = provider.parse_file(path, _data_object(path))
            bundle = root / "outputs" / "table--document-1"
            saved_markdown = (bundle / "result.md").read_text(encoding="utf-8")
            chunks = json.loads(
                (bundle / "chunks.json").read_text(encoding="utf-8")
            )
            first_pass_raw = (bundle / "raw.html").read_text(encoding="utf-8")
            artifact_names = {
                item.relative_to(bundle).as_posix()
                for item in bundle.rglob("*")
                if item.is_file()
            }

        refinement = parsed.metadata["table_refinement"]
        self.assertEqual(len(manager.calls), 2)
        self.assertIsNone(manager.calls[0][0][0].prompt)
        self.assertIsNotNone(manager.calls[1][0][0].prompt)
        self.assertEqual(manager.calls[1][1]["max_output_tokens"], 4096)
        self.assertFalse(manager.calls[1][1]["include_images"])
        self.assertNotIn("max_workers", manager.calls[1][1])
        self.assertIn('rowspan="2"', parsed.text)
        self.assertIn('rowspan="2"', saved_markdown)
        self.assertIn(
            'rowspan="2"',
            parsed.rows[0]["extraction"]["tables"][0]["content"],
        )
        self.assertIn(
            'rowspan="2"',
            chunks["pages"][0]["blocks"][0]["content"],
        )
        self.assertIn(original, first_pass_raw)
        self.assertEqual(parsed.metadata["token_count"], 17)
        self.assertEqual(refinement["attempted"], 1)
        self.assertEqual(refinement["succeeded"], 1)
        self.assertEqual(refinement["failed"], 0)
        self.assertEqual(refinement["records"][0]["rowspan_count"], 1)
        self.assertTrue(
            {
                "table_refinement/page_0001_table_0001.crop.png",
                "table_refinement/page_0001_table_0001.raw.html",
                "table_refinement/page_0001_table_0001.table.html",
            }.issubset(artifact_names)
        )

    def test_failed_table_refinement_falls_back_to_first_pass_table(self) -> None:
        original = "<table><tr><td>Keep me</td></tr></table>"
        manager = _FakeManager(
            {
                "__page_result__": _result(
                    original,
                    3,
                    html=original,
                    chunks=[
                        {
                            "bbox": [100, 100, 900, 500],
                            "label": "Table",
                            "content": original,
                        }
                    ],
                    page_box=[0, 0, 1000, 1400],
                ),
                "__table_result__": _result("", 0, error=True),
            }
        )
        provider = Chandra2Provider(
            Chandra2Config.from_mapping(
                {
                    "method": "hf",
                    "refine_tables": True,
                    "save_raw_outputs": False,
                    "include_images": False,
                }
            ),
            _runtime_loader=lambda: _runtime([_FakePageImage()], manager),
        )

        parsed = provider.parse_file(Path("table.pdf"), _data_object(Path("table.pdf")))

        self.assertEqual(parsed.text, original)
        self.assertEqual(
            parsed.rows[0]["extraction"]["tables"][0]["content"],
            original,
        )
        self.assertEqual(parsed.metadata["table_refinement"]["failed"], 1)

    def test_table_replacement_uses_layout_ordinal(self) -> None:
        first = "<table><tr><td>First</td></tr></table>"
        second = "<table><tr><td>Second</td></tr></table>"
        refined_second = (
            '<table><tr><td rowspan="2">Second</td></tr>'
            "<tr></tr></table>"
        )

        updated = _replace_tables(
            f"{first}\n{second}",
            [(1, refined_second)],
        )

        self.assertIn(first, updated)
        self.assertNotIn(second, updated)
        self.assertIn(refined_second, updated)

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
        self.assertFalse(
            Chandra2Config.from_mapping({"include_images": False}).include_images
        )

    def test_parser_config_resolves_chandra_output_into_parser_assets_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parser_assets_dir = root / "ingested" / "run-1" / "assets"
            resolved = resolve_parser_config(
                root,
                {"provider": "chandra2", "chandra2": {"method": "hf"}},
                parser_assets_dir,
            )

        self.assertEqual(
            Path(resolved["chandra2"]["output_dir"]),
            parser_assets_dir,
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
                "provider": "chandra2",
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
                    raw / "invoice.pdf",
                    parser_config=parser_config,
                    project_root=root,
                )
            cleaned = cleaning.run(ingested.parsed_data, ingested.initial_schemas)
            enriched = enrichment.run(cleaned.cleaned_data, cleaned.cleaned_schemas)
            indexed = indexing_cataloging.run(
                enriched.enriched_data,
                enriched.enriched_schemas,
                indexing_config={"embeddings": {"enabled": False}},
            )

        self.assertEqual(len(ingested.data_objects), 1)
        self.assertEqual(ingested.parsed_data[0].text, "# Invoice\n\nTotal: 100")
        self.assertEqual(ingested.parsed_data[0].metadata["parser"], "chandra2")
        self.assertEqual(
            Counter(record.index_type for record in indexed.index_records),
            Counter({"document": 1, "text_chunk": 1, "catalog": 1}),
        )

    def test_chandra_failure_propagates_from_main_file_runner(self) -> None:
        manager = _FakeManager({}, error=ConnectionError("vLLM unavailable"))
        runtime = _runtime(["page-1"], manager)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "a.pdf"
            path.write_bytes(b"%PDF-1.4\n")
            with patch(
                "src.ingestion.parsing.chandra2._load_runtime",
                return_value=runtime,
            ):
                with self.assertRaisesRegex(
                    ConnectionError,
                    "vLLM unavailable",
                ):
                    run_ingestion(
                        path,
                        parser_config={
                            "provider": "chandra2",
                            "chandra2": {"save_raw_outputs": False},
                        },
                    )


if __name__ == "__main__":
    unittest.main()
