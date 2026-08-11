"""AXIOM provider for the full KDL-Frontier-Parser-nano pipeline."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
from PIL import Image

from ...models import DataObject, ParsedData
from .contracts import AXIOM_NATIVE_BLOCK_SOURCE
from .kdl_frontier_engine import NanoEngine

logger = logging.getLogger(__name__)

KDL_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".jfif"})


@dataclass(frozen=True)
class KDLConfig:
    """Runtime settings for KDL's two-stage vLLM pipeline."""

    method: str = "vllm"
    endpoint_url: str = "http://127.0.0.1:8000/v1"
    model: str = "kdl-frontier-parser-nano"
    dpi: int = 144
    request_timeout_seconds: float = 3600.0
    max_retries: int = 2
    max_pages: int = 400
    max_workers: int = 32
    render_processes: int = 32
    bbox_max_workers: int = 32
    layout_max_output_tokens: int = 6000
    text_max_output_tokens: int = 2048
    table_max_output_tokens: int = 5500
    picture_max_output_tokens: int = 4096
    formula_max_output_tokens: int = 128
    continuous_page_queue: bool = True
    save_raw_outputs: bool = True
    output_dir: str | None = "data/work/kdl"
    project_root: str | None = None

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None) -> "KDLConfig":
        values = config or {}
        method = str(values.get("method", "vllm")).strip().lower()
        if method != "vllm":
            raise ValueError("kdl.method must be 'vllm'")
        endpoint = str(
            values.get("endpoint_url")
            or os.getenv("VLLM_API_BASE")
            or os.getenv("KDL_NANO_ENDPOINT_URL")
            or "http://127.0.0.1:8000/v1"
        ).strip().rstrip("/")
        max_workers = _positive_int(values, "max_workers", 32)
        return cls(
            method=method,
            endpoint_url=endpoint,
            model=str(
                values.get("model")
                or os.getenv("VLLM_MODEL_NAME")
                or os.getenv("KDL_NANO_MODEL")
                or "kdl-frontier-parser-nano"
            ).strip(),
            dpi=_positive_int(values, "dpi", 144),
            request_timeout_seconds=_positive_float(
                values,
                "request_timeout_seconds",
                float(values.get("timeout_seconds", 3600.0)),
            ),
            max_retries=_non_negative_int(values, "max_retries", 2),
            max_pages=_positive_int(values, "max_pages", 400),
            max_workers=max_workers,
            render_processes=_positive_int(
                values, "render_processes", 32
            ),
            bbox_max_workers=_positive_int(values, "bbox_max_workers", 32),
            layout_max_output_tokens=_positive_int(
                values, "layout_max_output_tokens", 6000
            ),
            text_max_output_tokens=_positive_int(
                values, "text_max_output_tokens", 2048
            ),
            table_max_output_tokens=_positive_int(
                values, "table_max_output_tokens", 5500
            ),
            picture_max_output_tokens=_positive_int(
                values, "picture_max_output_tokens", 4096
            ),
            formula_max_output_tokens=_positive_int(
                values, "formula_max_output_tokens", 128
            ),
            continuous_page_queue=bool(values.get("continuous_page_queue", True)),
            save_raw_outputs=bool(values.get("save_raw_outputs", True)),
            output_dir=_optional_string(values.get("output_dir", "data/work/kdl")),
            project_root=_optional_string(values.get("project_root")),
        )


@dataclass
class _KDLDocument:
    path: Path
    data_object: DataObject
    page_count: int
    started: float = field(default_factory=time.monotonic)
    pages: list[list[dict[str, Any]] | None] = field(default_factory=list)
    remaining: int = 0
    routing_context: Any | None = None
    failure: Exception | None = None
    reported: bool = False


class KDLProvider:
    """Run ParseBench's complete KDL pipeline and emit canonical ParsedData."""

    provider_name = "kdl"
    inference_mode = "kdl_two_stage_vllm"
    supported_extensions = KDL_EXTENSIONS

    def __init__(self, config: KDLConfig, *, text_router: Any | None = None) -> None:
        self.config = config
        self._text_router = text_router

    def parse_file(self, path: str | Path, data_object: DataObject) -> ParsedData:
        outcome = self.parse_files_with_errors([(path, data_object)])[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def parse_files(
        self, documents: list[tuple[str | Path, DataObject]]
    ) -> list[ParsedData]:
        outcomes = self.parse_files_with_errors(documents)
        failures = [str(item) for item in outcomes if isinstance(item, Exception)]
        if failures:
            raise RuntimeError("KDL failed to parse document(s): " + "; ".join(failures))
        return [item for item in outcomes if isinstance(item, ParsedData)]

    def parse_files_with_errors(
        self,
        documents: list[tuple[str | Path, DataObject]],
        *,
        on_document_complete: Callable[[int, ParsedData | Exception], None] | None = None,
    ) -> list[ParsedData | Exception]:
        if not documents:
            return []
        if not self.config.continuous_page_queue:
            outcomes: list[ParsedData | Exception] = []
            for index, (path, data_object) in enumerate(documents):
                try:
                    outcome = self._parse_one(path, data_object)
                except Exception as exc:
                    outcome = exc
                outcomes.append(outcome)
                if on_document_complete is not None:
                    on_document_complete(index, outcome)
            return outcomes
        return asyncio.run(
            self._parse_continuous(documents, on_document_complete)
        )

    def _parse_one(self, path: str | Path, data_object: DataObject) -> ParsedData:
        started = time.monotonic()
        file_path = _validate_path(path)
        page_count = _page_count(file_path, self.config.max_pages)
        routing_context = self._prepare_routing_context(file_path)
        images = [
            _render_page(str(file_path), page_index, self.config.dpi)
            for page_index in range(page_count)
        ]
        engine = self._engine()
        try:
            if self._text_router is None:
                raw = asyncio.run(engine.parse_pages(images))
            else:
                raw = asyncio.run(
                    engine.parse_pages(
                        images,
                        routing_context=routing_context,
                    )
                )
        finally:
            for image in images:
                image.close()
        self._attach_routing_metadata(raw, routing_context)
        return self._to_parsed_data(
            file_path, data_object, raw, started, page_count
        )

    async def _parse_continuous(
        self,
        documents: list[tuple[str | Path, DataObject]],
        on_document_complete: Callable[[int, ParsedData | Exception], None] | None,
    ) -> list[ParsedData | Exception]:
        prepared: list[_KDLDocument] = []
        for path, data_object in documents:
            file_path = Path(path).resolve()
            try:
                file_path = _validate_path(file_path)
                count = _page_count(file_path, self.config.max_pages)
                routing_context = self._prepare_routing_context(file_path)
            except Exception as exc:
                prepared.append(_KDLDocument(file_path, data_object, 0, failure=exc))
                continue
            prepared.append(
                _KDLDocument(
                    file_path,
                    data_object,
                    count,
                    pages=[None] * count,
                    remaining=count,
                    routing_context=routing_context,
                )
            )

        outcomes: list[ParsedData | Exception | None] = [None] * len(prepared)
        engine = self._engine()
        layout_semaphore = asyncio.Semaphore(self.config.max_workers)
        bbox_semaphore = asyncio.Semaphore(self.config.bbox_max_workers)
        job_queue: asyncio.Queue[tuple[int, int] | None] = asyncio.Queue(
            maxsize=self.config.max_workers
        )
        loop = asyncio.get_running_loop()

        def report(index: int, outcome: ParsedData | Exception) -> None:
            document = prepared[index]
            if document.reported:
                return
            document.reported = True
            outcomes[index] = outcome
            if on_document_complete is not None:
                on_document_complete(index, outcome)

        async def finalize(index: int) -> None:
            document = prepared[index]
            if document.failure is not None:
                report(index, document.failure)
                return
            elements = [
                element
                for page in document.pages
                for element in (page or [])
            ]
            try:
                raw = engine.finalize_elements(elements)
                self._attach_routing_metadata(raw, document.routing_context)
                parsed = self._to_parsed_data(
                    document.path,
                    document.data_object,
                    raw,
                    document.started,
                    document.page_count,
                )
            except Exception as exc:
                report(index, exc)
            else:
                report(index, parsed)

        async def producer(worker_count: int) -> None:
            active = [
                index for index, document in enumerate(prepared)
                if document.failure is None
            ]
            next_page = {index: 0 for index in active}
            while active:
                next_active: list[int] = []
                for index in active:
                    page_index = next_page[index]
                    await job_queue.put((index, page_index))
                    page_index += 1
                    next_page[index] = page_index
                    if page_index < prepared[index].page_count:
                        next_active.append(index)
                active = next_active
            for _ in range(worker_count):
                await job_queue.put(None)

        async def worker(executor: ProcessPoolExecutor) -> None:
            async with httpx.AsyncClient(
                timeout=self.config.request_timeout_seconds
            ) as client:
                while True:
                    job = await job_queue.get()
                    if job is None:
                        job_queue.task_done()
                        return
                    document_index, page_index = job
                    document = prepared[document_index]
                    image: Image.Image | None = None
                    try:
                        if document.failure is None:
                            image = await loop.run_in_executor(
                                executor,
                                _render_page,
                                str(document.path),
                                page_index,
                                self.config.dpi,
                            )
                            if self._text_router is None:
                                document.pages[page_index] = await engine._parse_page(
                                    client,
                                    layout_semaphore,
                                    bbox_semaphore,
                                    image,
                                    page_index + 1,
                                )
                            else:
                                document.pages[page_index] = await engine._parse_page(
                                    client,
                                    layout_semaphore,
                                    bbox_semaphore,
                                    image,
                                    page_index + 1,
                                    routing_context=document.routing_context,
                                )
                    except Exception as exc:
                        if document.failure is None:
                            document.failure = RuntimeError(
                                f"page {page_index + 1}: {type(exc).__name__}: {exc}"
                            )
                    finally:
                        if image is not None:
                            image.close()
                        document.remaining -= 1
                        if document.remaining == 0:
                            await finalize(document_index)
                        job_queue.task_done()

        for index, document in enumerate(prepared):
            if document.failure is not None:
                report(index, document.failure)

        total_pages = sum(document.page_count for document in prepared)
        worker_count = min(self.config.max_workers, max(1, total_pages))
        with ProcessPoolExecutor(max_workers=self.config.render_processes) as executor:
            workers = [asyncio.create_task(worker(executor)) for _ in range(worker_count)]
            await producer(worker_count)
            await job_queue.join()
            await asyncio.gather(*workers)

        for index, outcome in enumerate(outcomes):
            if outcome is None:
                report(index, RuntimeError("KDL did not finalize the document."))
        return [item for item in outcomes if item is not None]

    def _engine(self) -> NanoEngine:
        return NanoEngine(
            self.config.endpoint_url,
            self.config.model,
            self.config.bbox_max_workers,
            self.config.request_timeout_seconds,
            page_max_concurrent=self.config.max_workers,
            max_retries=self.config.max_retries,
            max_output_tokens={
                "layout": self.config.layout_max_output_tokens,
                "text": self.config.text_max_output_tokens,
                "table": self.config.table_max_output_tokens,
                "picture": self.config.picture_max_output_tokens,
                "formula": self.config.formula_max_output_tokens,
            },
            text_router=self._text_router,
        )

    def _prepare_routing_context(self, file_path: Path) -> Any | None:
        if self._text_router is None:
            return None
        return self._text_router.prepare_document(file_path)

    def _attach_routing_metadata(
        self,
        raw: dict[str, Any],
        routing_context: Any | None,
    ) -> None:
        if self._text_router is None:
            return
        raw["_routing"] = self._text_router.routing_metadata(routing_context)

    def _to_parsed_data(
        self,
        file_path: Path,
        data_object: DataObject,
        raw: dict[str, Any],
        started: float,
        page_count: int,
    ) -> ParsedData:
        source_blocks = _source_blocks(raw.get("pages") or [])
        markdown = str(raw.get("markdown") or "")
        extraction = _build_extraction(markdown, source_blocks)
        raw_paths = self._write_raw_outputs(file_path, data_object, raw)
        label_counts = Counter(str(block["raw_label"]) for block in source_blocks)
        metadata: dict[str, Any] = {
            "parser": self.provider_name,
            "method": self.config.method,
            "model_name": self.config.model,
            "inference_mode": self.inference_mode,
            "continuous_page_queue": self.config.continuous_page_queue,
            "max_workers": self.config.max_workers,
            "render_processes": self.config.render_processes,
            "bbox_max_workers": self.config.bbox_max_workers,
            "request_timeout_seconds": self.config.request_timeout_seconds,
            "max_retries": self.config.max_retries,
            "max_output_tokens": {
                "layout": self.config.layout_max_output_tokens,
                "text": self.config.text_max_output_tokens,
                "table": self.config.table_max_output_tokens,
                "picture": self.config.picture_max_output_tokens,
                "formula": self.config.formula_max_output_tokens,
            },
            "page_count": page_count,
            "latency_seconds": round(time.monotonic() - started, 3),
            "raw_kdl_outputs": raw_paths,
            "raw_output_path": raw_paths.get("result_markdown"),
            "raw_metadata_path": raw_paths.get("result_json"),
            "label_counts": dict(sorted(label_counts.items())),
            "source_block_count": len(source_blocks),
            "table_count": len(extraction["tables"]),
            "figure_count": len(extraction["figures"]),
            "formula_count": len(extraction["formulas"]),
            "reading_order_source": "kdl_layout",
            "reading_order_complete": bool(source_blocks),
        }
        routing_metadata = raw.get("_routing")
        if isinstance(routing_metadata, dict):
            metadata.update(routing_metadata)

        return ParsedData(
            object_id=data_object.object_id,
            source_uri=data_object.uri,
            source_format=str(
                data_object.metadata.get("format", file_path.suffix.lstrip("."))
            ),
            rows=[
                {
                    "extraction": extraction,
                    "text": extraction["main_text"],
                    "source_blocks": source_blocks,
                    "reading_order": [block["component_id"] for block in source_blocks],
                }
            ],
            text=markdown,
            metadata=metadata,
        )

    def _write_raw_outputs(
        self,
        file_path: Path,
        data_object: DataObject,
        raw: dict[str, Any],
    ) -> dict[str, str]:
        if not self.config.save_raw_outputs or not self.config.output_dir:
            return {}
        bundle = Path(self.config.output_dir) / data_object.object_id
        bundle.mkdir(parents=True, exist_ok=True)
        markdown_path = bundle / "result.md"
        json_path = bundle / "result.json"
        markdown_path.write_text(str(raw.get("markdown") or ""), encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {"source_file": str(file_path), **raw},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "result_markdown": _portable_path(markdown_path, self.config.project_root),
            "result_json": _portable_path(json_path, self.config.project_root),
        }


def _render_page(path: str, page_index: int, dpi: int) -> Image.Image:
    file_path = Path(path)
    if file_path.suffix.lower() != ".pdf":
        if page_index != 0:
            raise IndexError(f"Image has no page {page_index + 1}")
        with Image.open(file_path) as image:
            return image.convert("RGB").copy()
    import fitz

    with fitz.open(str(file_path)) as document:
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
        with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
            return image.convert("RGB").copy()


def _page_count(path: Path, max_pages: int) -> int:
    if path.suffix.lower() != ".pdf":
        return 1
    import fitz

    with fitz.open(str(path)) as document:
        count = int(document.page_count)
    if count <= 0:
        raise ValueError("Document rendered to zero pages.")
    if count > max_pages:
        raise ValueError(f"Document has {count} pages > max_pages={max_pages}.")
    return count


def _validate_path(path: str | Path) -> Path:
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Source file not found: {file_path}")
    if file_path.suffix.lower() not in KDL_EXTENSIONS:
        raise ValueError(f"Unsupported KDL input: {file_path.suffix or '<none>'}")
    return file_path


def _source_blocks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for page in sorted(pages, key=lambda item: int(item.get("page_number") or 0)):
        page_index = int(page.get("page_number") or 1) - 1
        elements = sorted(
            page.get("elements") or [],
            key=lambda item: int(item.get("layout_order") or 0),
        )
        for block_index, element in enumerate(elements):
            raw_label = str(element.get("category") or "Text")
            block_type = _canonical_block_type(raw_label)
            content = str(element.get("content") or "")
            block: dict[str, Any] = {
                "component_id": f"/page/{page_index}/{block_type}/{block_index}",
                "page": page_index,
                "block_index": block_index,
                "type": block_type,
                "raw_label": raw_label,
                "text": content.strip(),
                "source": AXIOM_NATIVE_BLOCK_SOURCE,
                "parser_source": "kdl_layout",
                "html": content if content.lstrip().startswith("<table") else "",
                "section_hierarchy": {},
            }
            if element.get("recognition_source"):
                block["recognition_source"] = str(element["recognition_source"])
            bbox = element.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                numeric = [float(value) for value in bbox]
                block["bbox"] = numeric
                block["polygon"] = [
                    [numeric[0], numeric[1]], [numeric[2], numeric[1]],
                    [numeric[2], numeric[3]], [numeric[0], numeric[3]],
                ]
                block["page_bbox"] = [0.0, 0.0, 1.0, 1.0]
            blocks.append(block)
    return blocks


def _build_extraction(markdown: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    title_block = next(
        (block for block in blocks if block["raw_label"] == "Title" and block["text"]),
        None,
    )
    tables: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    formulas: list[str] = []
    formula_citations: list[str] = []
    for block in blocks:
        component_id = block["component_id"]
        content = block["html"] or block["text"]
        if block["type"] == "Table":
            tables.append(
                {
                    "caption": None,
                    "caption_citations": [],
                    "content": content,
                    "content_citations": [component_id],
                    "content_format": "html" if block["html"] else "markdown",
                }
            )
        elif block["type"] == "Figure":
            figures.append(
                {
                    "caption": None,
                    "caption_citations": [],
                    "description": block["text"],
                    "description_citations": [component_id],
                }
            )
        elif block["type"] == "EquationBlock" and block["text"]:
            formulas.append(block["text"])
            formula_citations.append(component_id)
    return {
        "document_type": None,
        "language": None,
        "title": title_block["text"] if title_block else None,
        "title_citations": [title_block["component_id"]] if title_block else [],
        "main_text": markdown.strip(),
        "main_text_citations": [
            block["component_id"] for block in blocks if block["text"]
        ],
        "tables": tables,
        "figures": figures,
        "formulas": formulas,
        "formulas_citations": formula_citations,
    }


def _canonical_block_type(category: str) -> str:
    if category in {"Title", "Section-header"}:
        return "SectionHeader"
    if category == "Table" or category == "Chart":
        return "Table"
    if category in {"Picture", "Flowchart"}:
        return "Figure"
    if category == "Formula":
        return "EquationBlock"
    if category == "Caption":
        return "Caption"
    return "Text"


def _portable_path(path: Path, project_root: str | None) -> str:
    resolved = path.resolve()
    if project_root:
        try:
            return resolved.relative_to(Path(project_root).resolve()).as_posix()
        except ValueError:
            pass
    return str(resolved)


def _positive_int(config: dict[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, default))
    if value <= 0:
        raise ValueError(f"kdl.{key} must be positive")
    return value


def _positive_float(config: dict[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if value <= 0:
        raise ValueError(f"kdl.{key} must be positive")
    return value


def _non_negative_int(config: dict[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, default))
    if value < 0:
        raise ValueError(f"kdl.{key} must be non-negative")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["KDLConfig", "KDLProvider"]
