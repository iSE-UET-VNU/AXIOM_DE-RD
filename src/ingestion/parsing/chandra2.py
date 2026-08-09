"""Chandra2 document provider backed by self-hosted inference."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from html.parser import HTMLParser
import hashlib
import json
import logging
import math
import multiprocessing
import os
from queue import Empty
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...models import DataObject, ParsedData
from ...utils.paths import portable_path
from .contracts import AXIOM_NATIVE_BLOCK_SOURCE

logger = logging.getLogger(__name__)

CHANDRA2_EXTENSIONS = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff"}
)

TABLE_OCR_PROMPT = """
OCR this image as exactly one HTML table.
Return only one complete <table>...</table>; no prose, Markdown, JSON, or outer div.
Reconstruct the complete logical row and column grid from the visible borders.
A missing internal horizontal border means a cell spans rows: use rowspan.
A missing internal vertical border means a cell spans columns: use colspan.
Do not create a new column unless a visible boundary or clear subheader supports it.
Place each merged cell at its upper-left grid position and emit its content once.
Do not emit td/th cells for positions covered by an earlier rowspan or colspan.
Preserve visible text and genuinely empty cells.
Use only table, thead, tbody, tr, th, td, b, i, br, span, sup, sub, and math.
""".strip()


@dataclass(frozen=True)
class Chandra2Config:
    """Runtime and artifact settings for Chandra2 inference."""

    method: str = "vllm"
    continuous_page_queue: bool = False
    batch_size: int = 28
    max_workers: int = 4
    render_processes: int = 4
    request_batch_size: int = 1
    request_timeout_seconds: float = 3600.0
    max_output_tokens: int = 12384
    max_retries: int = 2
    include_images: bool = True
    include_headers_footers: bool = False
    save_raw_outputs: bool = True
    refine_tables: bool = False
    table_prompt: str = TABLE_OCR_PROMPT
    table_max_workers: int = 4
    table_max_output_tokens: int = 8192
    table_crop_margin_ratio: float = 0.02
    table_crop_min_short_side: int = 1536
    table_crop_max_long_side: int = 3072
    table_crop_max_pixels: int = 3072 * 2048
    output_dir: str | None = "data/work/chandra2"
    project_root: str | None = None

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None) -> "Chandra2Config":
        config = config or {}
        method = _inference_method(config.get("method", "vllm"))
        continuous_page_queue = bool(config.get("continuous_page_queue", False))
        if continuous_page_queue and method != "vllm":
            raise ValueError(
                "chandra2.continuous_page_queue is only supported with method='vllm'"
            )
        max_workers = _positive_int(config, "max_workers", 4)
        request_batch_size = _positive_int(config, "request_batch_size", 1)
        if continuous_page_queue and request_batch_size > max_workers:
            raise ValueError(
                "chandra2.request_batch_size must not exceed max_workers; "
                "max_workers is also the rendered-image pool capacity"
            )
        return cls(
            method=method,
            continuous_page_queue=continuous_page_queue,
            batch_size=_positive_int(
                config,
                "batch_size",
                1 if method == "hf" else 28,
            ),
            max_workers=max_workers,
            render_processes=_positive_int(
                config,
                "render_processes",
                min(max_workers, 4),
            ),
            request_batch_size=request_batch_size,
            request_timeout_seconds=_positive_float(
                config,
                "request_timeout_seconds",
                3600.0,
            ),
            max_output_tokens=_positive_int(config, "max_output_tokens", 12384),
            max_retries=_non_negative_int(config, "max_retries", 2),
            include_images=bool(config.get("include_images", True)),
            include_headers_footers=bool(
                config.get("include_headers_footers", False)
            ),
            save_raw_outputs=bool(config.get("save_raw_outputs", True)),
            refine_tables=bool(config.get("refine_tables", False)),
            table_prompt=_non_empty_str(
                config.get("table_prompt", TABLE_OCR_PROMPT),
                "table_prompt",
            ),
            table_max_workers=_positive_int(
                config,
                "table_max_workers",
                _positive_int(config, "max_workers", 4),
            ),
            table_max_output_tokens=_positive_int(
                config, "table_max_output_tokens", 8192
            ),
            table_crop_margin_ratio=_non_negative_float(
                config, "table_crop_margin_ratio", 0.02
            ),
            table_crop_min_short_side=_positive_int(
                config, "table_crop_min_short_side", 1536
            ),
            table_crop_max_long_side=_positive_int(
                config, "table_crop_max_long_side", 3072
            ),
            table_crop_max_pixels=_positive_int(
                config, "table_crop_max_pixels", 3072 * 2048
            ),
            output_dir=_optional_str(config.get("output_dir", "data/work/chandra2")),
            project_root=_optional_str(config.get("project_root")),
        )


@dataclass(frozen=True)
class _ChandraRuntime:
    load_file: Callable[[str, dict[str, Any]], list[Any]]
    inference_manager: Callable[..., Any]
    batch_input_item: Callable[..., Any]
    page_count: Callable[[str], int] | None = None
    load_pages: Callable[[str, int, int], list[Any]] | None = None
    process_rendering: bool = False


@dataclass
class _PreparedDocument:
    file_path: Path
    data_object: DataObject
    page_count: int = 0
    pages: list[Any | None] = field(default_factory=list)
    results: list[Any | None] = field(default_factory=list)
    preloaded_pages: list[Any] | None = None
    next_page: int = 0
    completed_pages: int = 0
    in_flight_pages: int = 0
    failure: Exception | None = None


@dataclass(frozen=True)
class _PageJob:
    document_id: str
    document_index: int
    page_index: int


@dataclass
class _TableRefinementJob:
    page_index: int
    table_index: int
    block: dict[str, Any]
    model_crop: Any
    prefix: str
    record: dict[str, Any]


class Chandra2Provider:
    """Convert PDF and image pages to ordered Markdown with Chandra2."""

    provider_name = "chandra2"
    supported_extensions = CHANDRA2_EXTENSIONS

    def __init__(
        self,
        config: Chandra2Config,
        *,
        _runtime_loader: Callable[[], _ChandraRuntime] | None = None,
    ) -> None:
        self.config = config
        self._runtime_loader = _runtime_loader or _load_runtime
        self._runtime: _ChandraRuntime | None = None
        self._manager: Any = None

    def parse_file(self, path: str | Path, data_object: DataObject) -> ParsedData:
        if self.config.continuous_page_queue:
            return self.parse_files([(path, data_object)])[0]

        started = time.monotonic()
        runtime = self._get_runtime()
        file_path, pages = self._load_document(path, runtime)
        results: list[Any] = []
        for start in range(0, len(pages), self.config.batch_size):
            generated = self._generate_pages(
                runtime,
                pages[start : start + self.config.batch_size],
            )
            failed_pages = [
                start + index + 1
                for index, result in enumerate(generated)
                if bool(getattr(result, "error", False))
            ]
            if failed_pages:
                pages_text = ", ".join(str(page) for page in failed_pages)
                raise RuntimeError(f"Chandra2 failed to parse page(s): {pages_text}")
            results.extend(generated)

        return self._finalize_document(
            runtime,
            file_path,
            data_object,
            pages,
            results,
            started=started,
            inference_mode=(
                "batched_vllm" if self.config.method == "vllm" else "static_hf"
            ),
        )

    def parse_files(
        self,
        documents: list[tuple[str | Path, DataObject]],
    ) -> list[ParsedData]:
        """Parse multiple documents and raise after preserving successful outputs."""

        outcomes = self.parse_files_with_errors(documents)
        failures = [
            f"{Path(path).name}: {outcome}"
            for (path, _), outcome in zip(documents, outcomes)
            if isinstance(outcome, Exception)
        ]
        if failures:
            raise RuntimeError(
                "Chandra2 failed to parse document(s): " + "; ".join(failures)
            )
        return [outcome for outcome in outcomes if isinstance(outcome, ParsedData)]

    def parse_files_with_errors(
        self,
        documents: list[tuple[str | Path, DataObject]],
        *,
        on_document_complete: (
            Callable[[int, ParsedData | Exception], None] | None
        ) = None,
    ) -> list[ParsedData | Exception]:
        """Parse a batch while keeping failures scoped to their source document."""

        if not documents:
            return []
        if not self.config.continuous_page_queue:
            outcomes: list[ParsedData | Exception] = []
            for path, data_object in documents:
                try:
                    outcomes.append(self.parse_file(path, data_object))
                except Exception as exc:
                    outcomes.append(exc)
                if on_document_complete is not None:
                    on_document_complete(len(outcomes) - 1, outcomes[-1])
            return outcomes
        return self._parse_files_continuous(
            documents,
            on_document_complete=on_document_complete,
        )

    def _parse_files_continuous(
        self,
        documents: list[tuple[str | Path, DataObject]],
        *,
        on_document_complete: (
            Callable[[int, ParsedData | Exception], None] | None
        ) = None,
    ) -> list[ParsedData | Exception]:
        """Render and infer one page per worker without limiting active PDFs."""

        started = time.monotonic()
        runtime = self._get_runtime()
        self._get_manager()
        if runtime.process_rendering:
            return self._parse_files_multiprocess(
                documents,
                runtime,
                started,
                on_document_complete=on_document_complete,
            )

        prepared = [
            _PreparedDocument(Path(path), data_object)
            for path, data_object in documents
        ]
        parsed_documents: list[ParsedData | None] = [None] * len(prepared)
        reported_documents: set[int] = set()
        schedulable_documents: list[int] = []
        round_robin_cursor = 0

        def report(document_index: int, outcome: ParsedData | Exception) -> None:
            if document_index in reported_documents:
                return
            reported_documents.add(document_index)
            if on_document_complete is not None:
                on_document_complete(document_index, outcome)

        # Counting pages opens each PDF briefly but does not render any page.
        # Every valid document then participates in one global round-robin queue.
        for document_index, document in enumerate(prepared):
            self._activate_stream_document(document, runtime)
            if document.failure is None:
                schedulable_documents.append(document_index)

        finalize_worker_count = min(
            self.config.table_max_workers,
            len(prepared),
        )
        with (
            ThreadPoolExecutor(max_workers=self.config.max_workers) as page_executor,
            ThreadPoolExecutor(max_workers=finalize_worker_count) as finalize_executor,
            ThreadPoolExecutor(
                max_workers=self.config.table_max_workers
            ) as refinement_executor,
        ):
            page_futures: dict[Future[tuple[Any, Any | None]], _PageJob] = {}
            finalize_futures: dict[Future[ParsedData], int] = {}

            while True:
                while len(page_futures) < self.config.max_workers:
                    scheduled = self._next_schedulable_document(
                        prepared,
                        schedulable_documents,
                        round_robin_cursor,
                    )
                    if scheduled is None:
                        break
                    document_index, round_robin_cursor = scheduled
                    document = prepared[document_index]
                    page_index = document.next_page
                    document.next_page += 1
                    job = _PageJob(
                        document_id=document.data_object.object_id,
                        document_index=document_index,
                        page_index=page_index,
                    )
                    try:
                        future = page_executor.submit(
                            self._render_and_generate_page,
                            runtime,
                            document,
                            page_index,
                        )
                    except Exception as exc:
                        document.failure = RuntimeError(
                            f"{page_index + 1}: failed to submit page: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        document.next_page = document.page_count
                        if document.in_flight_pages == 0:
                            self._discard_stream_document(document)
                            self._remove_schedulable_document(
                                schedulable_documents,
                                document_index,
                            )
                        continue
                    page_futures[future] = job
                    document.in_flight_pages += 1

                if not page_futures:
                    break

                completed, _ = wait(
                    tuple(page_futures),
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    job = page_futures.pop(future)
                    document = prepared[job.document_index]
                    document.in_flight_pages -= 1
                    document.completed_pages += 1
                    retained_image: Any | None = None
                    try:
                        result, retained_image = future.result()
                        if document.failure is None:
                            _write_stream_page_artifacts(
                                self.config,
                                document.file_path,
                                document.data_object,
                                job.page_index,
                                document.page_count,
                                result,
                            )
                            document.results[job.page_index] = result
                            if self.config.refine_tables:
                                document.pages[job.page_index] = retained_image
                                retained_image = None
                    except Exception as exc:
                        if document.failure is None:
                            document.failure = RuntimeError(
                                f"{job.page_index + 1}: {exc}"
                            )
                        document.next_page = document.page_count
                    finally:
                        _release_image(retained_image)

                    if document.failure is not None:
                        if document.in_flight_pages == 0:
                            self._discard_stream_document(document)
                            self._remove_schedulable_document(
                                schedulable_documents,
                                job.document_index,
                            )
                        continue
                    if document.completed_pages == document.page_count:
                        future = finalize_executor.submit(
                            self._finalize_stream_document,
                            runtime,
                            document,
                            started,
                            refinement_executor,
                        )
                        finalize_futures[future] = job.document_index
                        self._remove_schedulable_document(
                            schedulable_documents,
                            job.document_index,
                        )

                completed_finalizations = [
                    future
                    for future in finalize_futures
                    if future.done()
                ]
                for future in completed_finalizations:
                    document_index = finalize_futures.pop(future)
                    document = prepared[document_index]
                    try:
                        parsed = future.result()
                    except Exception as exc:
                        document.failure = exc
                        report(document_index, exc)
                    else:
                        parsed_documents[document_index] = parsed
                        report(document_index, parsed)

            for future in as_completed(tuple(finalize_futures)):
                document_index = finalize_futures.pop(future)
                document = prepared[document_index]
                try:
                    parsed_documents[document_index] = future.result()
                except Exception as exc:
                    document.failure = exc
                    report(document_index, exc)
                else:
                    report(document_index, parsed_documents[document_index])

        outcomes: list[ParsedData | Exception] = []
        for index, document in enumerate(prepared):
            parsed = parsed_documents[index]
            if parsed is not None:
                outcomes.append(parsed)
            elif document.failure is not None:
                outcomes.append(document.failure)
            else:
                outcomes.append(
                    RuntimeError("Chandra2 did not finalize the document.")
                )
            report(index, outcomes[-1])
        return outcomes

    def _parse_files_multiprocess(
        self,
        documents: list[tuple[str | Path, DataObject]],
        runtime: _ChandraRuntime,
        started: float,
        *,
        on_document_complete: (
            Callable[[int, ParsedData | Exception], None] | None
        ) = None,
    ) -> list[ParsedData | Exception]:
        """Render whole documents in isolated PDFium processes."""

        prepared = [
            _PreparedDocument(Path(path), data_object)
            for path, data_object in documents
        ]
        parsed_documents: list[ParsedData | None] = [None] * len(prepared)
        reported_documents: set[int] = set()

        def report(document_index: int, outcome: ParsedData | Exception) -> None:
            if document_index in reported_documents:
                return
            reported_documents.add(document_index)
            if on_document_complete is not None:
                on_document_complete(document_index, outcome)

        render_document_indexes: list[int] = []
        for document_index, document in enumerate(prepared):
            try:
                self._validate_document_path(document.file_path)
            except Exception as exc:
                document.failure = exc
            else:
                render_document_indexes.append(document_index)

        if not render_document_indexes:
            outcomes = [
                document.failure
                or RuntimeError("Chandra2 could not schedule the document.")
                for document in prepared
            ]
            for document_index, outcome in enumerate(outcomes):
                report(document_index, outcome)
            return outcomes

        context = multiprocessing.get_context("spawn")
        task_queue = context.Queue()
        rendered_queue = context.Queue(maxsize=self.config.max_workers)
        image_slots = context.BoundedSemaphore(self.config.max_workers)
        cancelled = context.Array("b", len(prepared), lock=False)
        render_process_count = min(
            self.config.render_processes,
            len(render_document_indexes),
        )
        render_processes = [
            context.Process(
                target=_render_document_worker,
                args=(task_queue, rendered_queue, image_slots, cancelled),
                name=f"chandra2-render-{index + 1}",
            )
            for index in range(render_process_count)
        ]
        for process in render_processes:
            process.start()
        for document_index in render_document_indexes:
            task_queue.put(
                (
                    document_index,
                    str(prepared[document_index].file_path),
                )
            )
        for _ in render_processes:
            task_queue.put(None)

        finalize_worker_count = min(
            self.config.table_max_workers,
            len(prepared),
        )
        render_done: set[int] = set()
        finalizing: set[int] = set()
        dead_process_timeouts = 0
        page_futures: dict[Future[list[Any]], list[_PageJob]] = {}
        future_images: dict[Future[list[Any]], list[Any]] = {}
        pending_pages: list[tuple[_PageJob, Any, Any]] = []
        finalize_futures: dict[Future[ParsedData], int] = {}
        request_batch_size = self.config.request_batch_size
        max_inflight_requests = max(
            1,
            self.config.max_workers // request_batch_size,
        )
        batch_client: Any | None = None

        def mark_failure(document_index: int, error: Exception) -> None:
            document = prepared[document_index]
            if document.failure is None:
                document.failure = error
            cancelled[document_index] = 1

        def maybe_finalize(
            document_index: int,
            executor: Executor,
            refinement_executor: Executor,
        ) -> None:
            document = prepared[document_index]
            if (
                document_index in finalizing
                or document.failure is not None
                or document_index not in render_done
                or document.page_count <= 0
                or document.in_flight_pages != 0
                or document.completed_pages != document.page_count
            ):
                return
            future = executor.submit(
                self._finalize_stream_document,
                runtime,
                document,
                started,
                refinement_executor,
            )
            finalize_futures[future] = document_index
            finalizing.add(document_index)

        try:
            if request_batch_size > 1:
                batch_client = self._open_vllm_batch_client()
            with (
                ThreadPoolExecutor(
                    max_workers=max_inflight_requests
                ) as page_executor,
                ThreadPoolExecutor(
                    max_workers=finalize_worker_count
                ) as finalize_executor,
                ThreadPoolExecutor(
                    max_workers=self.config.table_max_workers
                ) as refinement_executor,
            ):
                def submit_pending_batches(*, force: bool = False) -> None:
                    while (
                        pending_pages
                        and len(page_futures) < max_inflight_requests
                        and (force or len(pending_pages) >= request_batch_size)
                    ):
                        take = min(request_batch_size, len(pending_pages))
                        selected = pending_pages[:take]
                        del pending_pages[:take]

                        active: list[tuple[_PageJob, Any, Any]] = []
                        for job, input_item, image in selected:
                            document = prepared[job.document_index]
                            if document.failure is None:
                                active.append((job, input_item, image))
                                continue
                            document.in_flight_pages -= 1
                            _release_image(image)
                            image_slots.release()

                        if not active:
                            continue

                        jobs = [item[0] for item in active]
                        inputs = [item[1] for item in active]
                        images = [item[2] for item in active]
                        if request_batch_size > 1:
                            future = page_executor.submit(
                                self._generate_vllm_request_batch,
                                batch_client,
                                inputs,
                            )
                        else:
                            future = page_executor.submit(
                                self._generate_batch,
                                inputs,
                                max_workers=1,
                            )
                        page_futures[future] = jobs
                        future_images[future] = images

                while (
                    len(render_done) < len(render_document_indexes)
                    or page_futures
                    or pending_pages
                    or finalize_futures
                ):
                    received_event = False
                    try:
                        event = rendered_queue.get(timeout=0.05)
                        received_event = True
                    except Empty:
                        event = None

                    if event is not None:
                        event_type, document_index, page_index, payload = event
                        document = prepared[document_index]
                        if event_type == "started":
                            page_count = int(payload)
                            if page_count <= 0:
                                mark_failure(
                                    document_index,
                                    RuntimeError(
                                        "Chandra2 could not load any pages from "
                                        f"the document: {document.file_path.name}"
                                    ),
                                )
                            else:
                                document.page_count = page_count
                                document.pages = [None] * page_count
                                document.results = [None] * page_count
                        elif event_type == "page":
                            if document.failure is not None:
                                image_slots.release()
                            else:
                                image: Any | None = None
                                try:
                                    image = _deserialize_rendered_image(payload)
                                    input_item = runtime.batch_input_item(
                                        image=image,
                                        prompt_type="ocr_layout",
                                    )
                                except Exception as exc:
                                    _release_image(image)
                                    image_slots.release()
                                    mark_failure(
                                        document_index,
                                        RuntimeError(
                                            f"{page_index + 1}: failed to submit page: "
                                            f"{type(exc).__name__}: {exc}"
                                        ),
                                    )
                                else:
                                    job = _PageJob(
                                        document_id=document.data_object.object_id,
                                        document_index=document_index,
                                        page_index=page_index,
                                    )
                                    pending_pages.append((job, input_item, image))
                                    document.in_flight_pages += 1
                                    submit_pending_batches()
                        elif event_type == "error":
                            error_type, message = payload
                            page_text = (
                                f"{page_index + 1}: " if page_index >= 0 else ""
                            )
                            mark_failure(
                                document_index,
                                RuntimeError(
                                    f"{page_text}failed to render page: "
                                    f"{error_type}: {message}"
                                ),
                            )
                        elif event_type == "done":
                            render_done.add(document_index)
                            maybe_finalize(
                                document_index,
                                finalize_executor,
                                refinement_executor,
                            )
                        else:
                            mark_failure(
                                document_index,
                                RuntimeError(
                                    f"Unknown Chandra2 render event: {event_type}"
                                ),
                            )

                    completed = [
                        future for future in page_futures if future.done()
                    ]
                    for future in completed:
                        jobs = page_futures.pop(future)
                        images = future_images.pop(future)
                        try:
                            generated = future.result()
                            if len(generated) != len(jobs):
                                raise RuntimeError(
                                    "Chandra2 returned a different number of "
                                    "results than request-batch pages."
                                )
                        except Exception as exc:
                            generated = [exc] * len(jobs)

                        for job, image, result in zip(jobs, images, generated):
                            document = prepared[job.document_index]
                            document.in_flight_pages -= 1
                            document.completed_pages += 1
                            try:
                                if isinstance(result, Exception):
                                    raise result
                                if bool(getattr(result, "error", False)):
                                    raise RuntimeError(
                                        "Chandra2 marked the page as failed"
                                    )
                                if document.failure is None:
                                    _write_stream_page_artifacts(
                                        self.config,
                                        document.file_path,
                                        document.data_object,
                                        job.page_index,
                                        document.page_count,
                                        result,
                                    )
                                    document.results[job.page_index] = result
                                    if self.config.refine_tables:
                                        document.pages[job.page_index] = image
                                        image = None
                            except Exception as exc:
                                mark_failure(
                                    job.document_index,
                                    RuntimeError(
                                        f"{job.page_index + 1}: failed page: "
                                        f"{type(exc).__name__}: {exc}"
                                    ),
                                )
                            finally:
                                _release_image(image)
                                image_slots.release()

                            maybe_finalize(
                                job.document_index,
                                finalize_executor,
                                refinement_executor,
                            )

                        submit_pending_batches(
                            force=(
                                len(render_done)
                                == len(render_document_indexes)
                            )
                        )

                    all_renderers_stopped = not any(
                        process.is_alive() for process in render_processes
                    )
                    if (
                        not received_event
                        and all_renderers_stopped
                        and len(render_done) < len(render_document_indexes)
                    ):
                        dead_process_timeouts += 1
                        if dead_process_timeouts >= 20:
                            missing = set(render_document_indexes) - render_done
                            for document_index in missing:
                                mark_failure(
                                    document_index,
                                    RuntimeError(
                                        "Chandra2 render process exited before the "
                                        "document completed."
                                    ),
                                )
                                render_done.add(document_index)
                    else:
                        dead_process_timeouts = 0

                    submit_pending_batches(
                        force=(
                            len(render_done) == len(render_document_indexes)
                        )
                    )

                    completed_finalizations = [
                        future
                        for future in finalize_futures
                        if future.done()
                    ]
                    for future in completed_finalizations:
                        document_index = finalize_futures.pop(future)
                        document = prepared[document_index]
                        try:
                            parsed = future.result()
                        except Exception as exc:
                            document.failure = exc
                            report(document_index, exc)
                        else:
                            parsed_documents[document_index] = parsed
                            report(document_index, parsed)

                for future in as_completed(tuple(finalize_futures)):
                    document_index = finalize_futures.pop(future)
                    document = prepared[document_index]
                    try:
                        parsed = future.result()
                    except Exception as exc:
                        document.failure = exc
                        report(document_index, exc)
                    else:
                        parsed_documents[document_index] = parsed
                        report(document_index, parsed)
        finally:
            if batch_client is not None:
                batch_client.close()
            for process in render_processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            task_queue.close()
            task_queue.cancel_join_thread()
            rendered_queue.close()
            rendered_queue.cancel_join_thread()

        outcomes: list[ParsedData | Exception] = []
        for index, document in enumerate(prepared):
            parsed = parsed_documents[index]
            if parsed is not None:
                outcomes.append(parsed)
                report(index, parsed)
                continue
            if document.failure is not None:
                self._discard_stream_document(document)
                outcomes.append(document.failure)
                report(index, document.failure)
                continue
            self._discard_stream_document(document)
            error = RuntimeError(
                "Chandra2 did not finalize the document."
            )
            outcomes.append(error)
            report(index, error)
        return outcomes

    def _activate_stream_document(
        self,
        document: _PreparedDocument,
        runtime: _ChandraRuntime,
    ) -> None:
        try:
            self._validate_document_path(document.file_path)
            if runtime.page_count is not None and runtime.load_pages is not None:
                document.page_count = int(runtime.page_count(str(document.file_path)))
            else:
                document.preloaded_pages = list(
                    runtime.load_file(
                        str(document.file_path),
                        {},
                    )
                )
                document.page_count = len(document.preloaded_pages)
            if document.page_count <= 0:
                raise RuntimeError(
                    f"Chandra2 could not load any pages from the document: "
                    f"{document.file_path.name}"
                )
            document.pages = [None] * document.page_count
            document.results = [None] * document.page_count
        except Exception as exc:
            document.failure = exc

    def _load_stream_pages(
        self,
        document: _PreparedDocument,
        runtime: _ChandraRuntime,
        start_page: int,
        count: int,
    ) -> list[Any]:
        if document.preloaded_pages is not None:
            return list(
                document.preloaded_pages[start_page : start_page + count]
            )
        if runtime.load_pages is None:
            raise RuntimeError("Chandra2 runtime has no streaming page loader.")
        return runtime.load_pages(str(document.file_path), start_page, count)

    def _render_and_generate_page(
        self,
        runtime: _ChandraRuntime,
        document: _PreparedDocument,
        page_index: int,
    ) -> tuple[Any, Any | None]:
        images: list[Any] = []
        retained_image: Any | None = None
        try:
            try:
                images = self._load_stream_pages(
                    document,
                    runtime,
                    page_index,
                    1,
                )
                if document.preloaded_pages is not None:
                    document.preloaded_pages[page_index] = None
                if len(images) != 1:
                    raise RuntimeError(
                        f"expected 1 rendered page, got {len(images)}"
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"failed to render page: {type(exc).__name__}: {exc}"
                ) from exc

            image = images[0]
            try:
                input_item = runtime.batch_input_item(
                    image=image,
                    prompt_type="ocr_layout",
                )
                result = self._generate_batch(
                    [input_item],
                    max_workers=1,
                )[0]
                if bool(getattr(result, "error", False)):
                    raise RuntimeError("Chandra2 marked the page as failed")
            except Exception as exc:
                raise RuntimeError(
                    f"failed page: {type(exc).__name__}: {exc}"
                ) from exc

            if self.config.refine_tables:
                retained_image = image
            return result, retained_image
        finally:
            for image in images:
                if image is not retained_image:
                    _release_image(image)

    @staticmethod
    def _discard_stream_document(document: _PreparedDocument) -> None:
        images = [*document.pages, *(document.preloaded_pages or [])]
        released: set[int] = set()
        for image in images:
            identity = id(image)
            if image is None or identity in released:
                continue
            released.add(identity)
            _release_image(image)
        document.pages.clear()
        document.results.clear()
        document.preloaded_pages = None

    def _finalize_stream_document(
        self,
        runtime: _ChandraRuntime,
        document: _PreparedDocument,
        started: float,
        refinement_executor: Executor,
    ) -> ParsedData:
        try:
            return self._finalize_document(
                runtime,
                document.file_path,
                document.data_object,
                document.pages,
                list(document.results),
                started=started,
                inference_mode="continuous_vllm",
                refinement_executor=refinement_executor,
            )
        finally:
            for image in document.pages:
                _release_image(image)
            document.pages.clear()
            document.results.clear()
            document.preloaded_pages = None

    @staticmethod
    def _next_schedulable_document(
        prepared: list[_PreparedDocument],
        active_documents: list[int],
        cursor: int,
    ) -> tuple[int, int] | None:
        if not active_documents:
            return None
        for offset in range(len(active_documents)):
            position = (cursor + offset) % len(active_documents)
            document_index = active_documents[position]
            document = prepared[document_index]
            if (
                document.failure is None
                and document.next_page < document.page_count
            ):
                return document_index, (position + 1) % len(active_documents)
        return None

    @staticmethod
    def _remove_schedulable_document(
        schedulable_documents: list[int],
        document_index: int,
    ) -> None:
        try:
            schedulable_documents.remove(document_index)
        except ValueError:
            pass

    def _validate_document_path(self, file_path: Path) -> None:
        if file_path.suffix.lower() not in self.supported_extensions:
            raise RuntimeError(
                f"Chandra2 does not support file type: {file_path.suffix}"
            )

    def _load_document(
        self,
        path: str | Path,
        runtime: _ChandraRuntime,
    ) -> tuple[Path, list[Any]]:
        file_path = Path(path)
        self._validate_document_path(file_path)
        pages = runtime.load_file(str(file_path), {})
        if not pages:
            raise RuntimeError(
                f"Chandra2 could not load any pages from the document: {file_path.name}"
            )
        return file_path, pages

    def _generate_pages(
        self,
        runtime: _ChandraRuntime,
        pages: list[Any],
    ) -> list[Any]:
        return self._generate_batch(
            [
                runtime.batch_input_item(image=image, prompt_type="ocr_layout")
                for image in pages
            ]
        )

    def _open_vllm_batch_client(self) -> Any:
        import httpx

        from chandra.settings import settings

        headers = {
            "Authorization": f"Bearer {settings.VLLM_API_KEY}",
            "ngrok-skip-browser-warning": "true",
        }
        return httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(
                self.config.request_timeout_seconds,
                connect=min(60.0, self.config.request_timeout_seconds),
            ),
        )

    def _generate_vllm_request_batch(
        self,
        client: Any,
        batch: list[Any],
    ) -> list[Any]:
        """Send many independent page conversations in one vLLM HTTP request."""

        from chandra.model.schema import BatchOutputItem
        from chandra.model.util import detect_repeat_token, scale_to_fit
        from chandra.model.vllm import image_to_base64
        from chandra.output import (
            extract_images,
            parse_chunks,
            parse_html,
            parse_markdown,
        )
        from chandra.prompts import PROMPT_MAPPING
        from chandra.settings import settings

        conversations: list[list[dict[str, Any]]] = []
        for item in batch:
            prompt = item.prompt or PROMPT_MAPPING[item.prompt_type]
            scaled_image = scale_to_fit(item.image)
            try:
                image_b64 = image_to_base64(scaled_image)
            finally:
                if scaled_image is not item.image:
                    _release_image(scaled_image)
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        "data:image/png;base64," + image_b64
                                    )
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )

        api_base = str(settings.VLLM_API_BASE).rstrip("/")
        model_name = settings.VLLM_MODEL_NAME
        if model_name is None:
            response = client.get(f"{api_base}/models")
            response.raise_for_status()
            model_name = response.json()["data"][0]["id"]

        raw_results: list[dict[str, Any]] = [
            {"raw": "", "token_count": 0, "error": True}
            for _ in batch
        ]
        pending = list(range(len(batch)))
        for attempt in range(self.config.max_retries + 1):
            if not pending:
                break
            if attempt == 0:
                logger.info(
                    "Submitting one vLLM HTTP request containing %s page(s)",
                    len(pending),
                )
            payload = {
                "model": model_name,
                "messages": [conversations[index] for index in pending],
                "max_tokens": self.config.max_output_tokens,
                "temperature": min(0.2 * attempt, 0.8),
                "top_p": 0.1 if attempt == 0 else 0.95,
                "return_token_ids": True,
            }
            try:
                response = client.post(
                    f"{api_base}/chat/completions/batch",
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
                choices = body.get("choices")
                if not isinstance(choices, list):
                    raise RuntimeError(
                        "vLLM batch response does not contain a choices list"
                    )
            except Exception as exc:
                retryable = _vllm_batch_error_is_retryable(exc)
                if retryable and attempt < self.config.max_retries:
                    logger.warning(
                        "vLLM request batch of %s page(s) failed; retrying "
                        "attempt %s/%s: %s",
                        len(pending),
                        attempt + 1,
                        self.config.max_retries,
                        _vllm_batch_error_text(exc),
                    )
                    time.sleep(2 * (attempt + 1))
                    continue
                logger.error(
                    "vLLM request batch of %s page(s) failed: %s",
                    len(pending),
                    _vllm_batch_error_text(exc),
                )
                break

            aggregate_tokens = int(
                (body.get("usage") or {}).get("completion_tokens") or 0
            )
            fallback_tokens = aggregate_tokens // max(1, len(choices))
            seen_local_indexes: set[int] = set()
            retry_indexes: list[int] = []
            for choice in choices:
                try:
                    local_index = int(choice["index"])
                    if local_index < 0 or local_index >= len(pending):
                        continue
                    global_index = pending[local_index]
                    content = choice["message"]["content"]
                except (IndexError, KeyError, TypeError, ValueError):
                    continue
                if local_index in seen_local_indexes:
                    continue
                seen_local_indexes.add(local_index)
                raw = _vllm_message_text(content)
                token_ids = choice.get("token_ids")
                token_count = (
                    len(token_ids)
                    if isinstance(token_ids, list)
                    else fallback_tokens
                )
                has_repeat = detect_repeat_token(raw) or (
                    len(raw) > 50
                    and detect_repeat_token(raw, cut_from_end=50)
                )
                if has_repeat and attempt < self.config.max_retries:
                    retry_indexes.append(global_index)
                    continue
                raw_results[global_index] = {
                    "raw": raw,
                    "token_count": token_count,
                    "error": False,
                }

            missing_indexes = [
                global_index
                for local_index, global_index in enumerate(pending)
                if local_index not in seen_local_indexes
            ]
            pending = retry_indexes + missing_indexes
            if pending and attempt < self.config.max_retries:
                logger.warning(
                    "Retrying %s incomplete or repeating output(s) from a "
                    "vLLM request batch",
                    len(pending),
                )

        outputs: list[Any] = []
        for item, result in zip(batch, raw_results):
            raw = str(result["raw"] or "")
            if result["error"]:
                outputs.append(
                    BatchOutputItem(
                        markdown="",
                        html="",
                        chunks={},
                        raw=raw,
                        page_box=[0, 0, item.image.width, item.image.height],
                        token_count=int(result["token_count"]),
                        images={},
                        error=True,
                    )
                )
                continue
            try:
                chunks = parse_chunks(raw, item.image)
                outputs.append(
                    BatchOutputItem(
                        markdown=parse_markdown(
                            raw,
                            include_headers_footers=(
                                self.config.include_headers_footers
                            ),
                            include_images=self.config.include_images,
                        ),
                        html=parse_html(
                            raw,
                            include_headers_footers=(
                                self.config.include_headers_footers
                            ),
                            include_images=self.config.include_images,
                        ),
                        chunks=chunks,
                        raw=raw,
                        page_box=[0, 0, item.image.width, item.image.height],
                        token_count=int(result["token_count"]),
                        images=extract_images(raw, chunks, item.image),
                        error=False,
                    )
                )
            except Exception as exc:
                logger.error(
                    "Failed to post-process one vLLM batch choice: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                outputs.append(
                    BatchOutputItem(
                        markdown="",
                        html="",
                        chunks={},
                        raw=raw,
                        page_box=[0, 0, item.image.width, item.image.height],
                        token_count=int(result["token_count"]),
                        images={},
                        error=True,
                    )
                )
        return outputs

    def _generate_batch(
        self,
        batch: list[Any],
        *,
        max_workers: int | None = None,
    ) -> list[Any]:
        generate_options: dict[str, Any] = {
            "include_images": self.config.include_images,
            "include_headers_footers": self.config.include_headers_footers,
            "max_output_tokens": self.config.max_output_tokens,
        }
        if self.config.method == "vllm":
            generate_options.update(
                max_workers=(
                    self.config.max_workers
                    if max_workers is None
                    else max_workers
                ),
                max_retries=self.config.max_retries,
            )
        generated = list(self._get_manager().generate(batch, **generate_options))
        if len(generated) != len(batch):
            raise RuntimeError(
                "Chandra2 returned a different number of results than input pages."
            )
        return generated

    def _finalize_document(
        self,
        runtime: _ChandraRuntime,
        file_path: Path,
        data_object: DataObject,
        pages: list[Any],
        results: list[Any],
        *,
        started: float,
        inference_mode: str,
        refinement_executor: Executor | None = None,
    ) -> ParsedData:
        page_payloads, raw_block_count = _page_payloads(results)
        page_markdown = [
            str(getattr(result, "markdown", "") or "") for result in results
        ]
        page_clean_html = [
            str(getattr(result, "html", "") or "") for result in results
        ]
        bundle_dir = _output_bundle_dir(
            self.config,
            file_path,
            data_object,
            results,
        )
        table_refinement = _refine_table_blocks(
            self.config,
            runtime,
            self._get_manager(),
            pages,
            page_payloads,
            page_markdown,
            page_clean_html,
            bundle_dir,
            executor=refinement_executor,
        )
        markdown = "\n\n".join(page_markdown)
        latency_seconds = round(time.monotonic() - started, 3)
        first_pass_token_count = sum(
            int(getattr(result, "token_count", 0) or 0) for result in results
        )
        token_count = first_pass_token_count + int(
            table_refinement["token_count"]
        )
        source_blocks = _source_blocks_from_pages(page_payloads)
        reading_order_complete = bool(source_blocks) and (
            len(source_blocks) == raw_block_count
            and all(page.get("blocks") for page in page_payloads)
        )
        extraction = _build_extraction(markdown, source_blocks)

        saved_images = _write_images(
            self.config,
            bundle_dir,
            results,
        )
        image_files = _align_image_files(
            extraction.get("figures", []),
            source_blocks,
            results,
            saved_images,
        )
        for page_index, page in enumerate(page_payloads):
            page["image_files"] = [
                image["name"]
                for image in saved_images
                if image["page"] == page_index and image["status"] == "saved"
            ]
        raw_output_paths = _write_raw_outputs(
            self.config,
            bundle_dir,
            file_path,
            data_object,
            markdown,
            results,
            page_clean_html,
            page_payloads,
            source_blocks,
            extraction,
            image_files,
            reading_order_complete,
            latency_seconds,
            token_count,
            table_refinement,
            inference_mode,
        )
        label_counts = Counter(
            str(block.get("raw_label") or block.get("type") or "Block")
            for block in source_blocks
        )

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
                    "reading_order": [
                        block["component_id"] for block in source_blocks
                    ],
                }
            ],
            text=markdown,
            metadata={
                "parser": "chandra2",
                "method": self.config.method,
                "inference_mode": inference_mode,
                "continuous_page_queue": self.config.continuous_page_queue,
                "max_workers": self.config.max_workers,
                "render_processes": self.config.render_processes,
                "request_batch_size": self.config.request_batch_size,
                "model_name": _model_name(self.config.method),
                "page_count": len(results),
                "token_count": token_count,
                "first_pass_token_count": first_pass_token_count,
                "table_refinement": table_refinement,
                "latency_seconds": latency_seconds,
                "raw_output_path": raw_output_paths.get("result_markdown"),
                "raw_metadata_path": raw_output_paths.get("metadata"),
                "raw_chandra_outputs": raw_output_paths,
                "label_counts": dict(sorted(label_counts.items())),
                "source_block_count": len(source_blocks),
                "table_count": len(extraction["tables"]),
                "figure_count": len(extraction["figures"]),
                "formula_count": len(extraction["formulas"]),
                "image_count": sum(
                    item.get("status") == "saved" for item in image_files
                ),
                "image_files": image_files,
                "reading_order_source": (
                    "chandra2_layout" if source_blocks else "unavailable"
                ),
                "reading_order_complete": reading_order_complete,
            },
        )

    def _get_runtime(self) -> _ChandraRuntime:
        if self._runtime is None:
            self._runtime = self._runtime_loader()
        return self._runtime

    def _get_manager(self) -> Any:
        if self._manager is None:
            try:
                self._manager = self._get_runtime().inference_manager(
                    method=self.config.method
                )
            except ImportError as exc:
                if self.config.method == "hf":
                    raise RuntimeError(
                        "Chandra2 local inference requires the HuggingFace extras. "
                        'Install them with: pip install -e ".[chandra2-local]"'
                    ) from exc
                raise
        return self._manager


def _load_runtime() -> _ChandraRuntime:
    try:
        import pypdfium2 as pdfium

        from chandra.input import load_file
        from chandra.model import InferenceManager
        from chandra.model.schema import BatchInputItem
    except ImportError as exc:
        raise RuntimeError(
            "Missing Chandra2 dependency. Install it with: pip install -e .[chandra2]"
        ) from exc

    def page_count(filepath: str) -> int:
        if Path(filepath).suffix.lower() != ".pdf":
            return 1
        document = pdfium.PdfDocument(filepath)
        try:
            return len(document)
        finally:
            document.close()

    def load_pages(filepath: str, start_page: int, count: int) -> list[Any]:
        if Path(filepath).suffix.lower() != ".pdf":
            return load_file(filepath, {}) if start_page == 0 and count else []
        end_page = start_page + count - 1
        page_range = (
            str(start_page)
            if start_page == end_page
            else f"{start_page}-{end_page}"
        )
        return load_file(filepath, {"page_range": page_range})

    return _ChandraRuntime(
        load_file,
        InferenceManager,
        BatchInputItem,
        page_count=page_count,
        load_pages=load_pages,
        process_rendering=True,
    )


def _render_document_worker(
    task_queue: Any,
    rendered_queue: Any,
    image_slots: Any,
    cancelled: Any,
) -> None:
    """Render complete documents in an isolated spawned process."""

    while True:
        task = task_queue.get()
        if task is None:
            return
        document_index, filepath = task
        try:
            if Path(filepath).suffix.lower() == ".pdf":
                _render_pdf_document(
                    document_index,
                    filepath,
                    rendered_queue,
                    image_slots,
                    cancelled,
                )
            else:
                _render_image_document(
                    document_index,
                    filepath,
                    rendered_queue,
                    image_slots,
                    cancelled,
                )
        except Exception as exc:
            rendered_queue.put(
                (
                    "error",
                    document_index,
                    -1,
                    (type(exc).__name__, str(exc)),
                )
            )
        finally:
            rendered_queue.put(("done", document_index, -1, None))


def _render_pdf_document(
    document_index: int,
    filepath: str,
    rendered_queue: Any,
    image_slots: Any,
    cancelled: Any,
) -> None:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(filepath)
    try:
        document.init_forms()
        page_count = len(document)
        rendered_queue.put(("started", document_index, -1, page_count))
        for page_index in range(page_count):
            if cancelled[document_index]:
                break
            image_slots.acquire()
            if cancelled[document_index]:
                image_slots.release()
                break
            image: Any | None = None
            try:
                image = _render_pdfium_page(document, page_index)
                payload = _serialize_rendered_image(image)
                rendered_queue.put(
                    ("page", document_index, page_index, payload)
                )
            except Exception as exc:
                image_slots.release()
                rendered_queue.put(
                    (
                        "error",
                        document_index,
                        page_index,
                        (type(exc).__name__, str(exc)),
                    )
                )
                break
            finally:
                _release_image(image)
    finally:
        document.close()


def _render_pdfium_page(document: Any, page_index: int) -> Any:
    from chandra.input import flatten
    from chandra.settings import settings

    page = document[page_index]
    try:
        min_page_dim = min(page.get_width(), page.get_height())
    finally:
        page.close()

    scale_dpi = (settings.MIN_PDF_IMAGE_DIM / min_page_dim) * 72
    scale_dpi = max(scale_dpi, settings.IMAGE_DPI)
    page = document[page_index]
    try:
        flatten(page)
    finally:
        page.close()

    page = document[page_index]
    bitmap: Any | None = None
    source_image: Any | None = None
    try:
        bitmap = page.render(scale=scale_dpi / 72)
        source_image = bitmap.to_pil()
        return source_image.convert("RGB")
    finally:
        _release_image(source_image)
        _release_image(bitmap)
        page.close()


def _render_image_document(
    document_index: int,
    filepath: str,
    rendered_queue: Any,
    image_slots: Any,
    cancelled: Any,
) -> None:
    from chandra.input import load_image

    rendered_queue.put(("started", document_index, -1, 1))
    if cancelled[document_index]:
        return
    image_slots.acquire()
    if cancelled[document_index]:
        image_slots.release()
        return
    image: Any | None = None
    try:
        image = load_image(filepath)
        rendered_queue.put(
            (
                "page",
                document_index,
                0,
                _serialize_rendered_image(image),
            )
        )
    except Exception:
        image_slots.release()
        raise
    finally:
        _release_image(image)


def _serialize_rendered_image(image: Any) -> tuple[str, tuple[int, int], bytes]:
    return image.mode, tuple(image.size), image.tobytes()


def _deserialize_rendered_image(
    payload: tuple[str, tuple[int, int], bytes],
) -> Any:
    from PIL import Image

    mode, size, data = payload
    return Image.frombytes(mode, size, data)


def _release_image(image: Any) -> None:
    close = getattr(image, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _refine_table_blocks(
    config: Chandra2Config,
    runtime: _ChandraRuntime,
    manager: Any,
    page_images: list[Any],
    page_payloads: list[dict[str, Any]],
    page_markdown: list[str],
    page_clean_html: list[str],
    bundle_dir: Path | None,
    *,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Run a focused OCR pass for each layout block labelled as a table."""

    summary: dict[str, Any] = {
        "enabled": config.refine_tables,
        "max_workers": config.table_max_workers,
        "prompt_sha256": hashlib.sha256(
            config.table_prompt.encode("utf-8")
        ).hexdigest(),
        "attempted": 0,
        "succeeded": 0,
        "rejected": 0,
        "failed": 0,
        "token_count": 0,
        "records": [],
    }
    if not config.refine_tables:
        return summary

    artifact_dir = (
        bundle_dir / "table_refinement"
        if bundle_dir is not None and config.save_raw_outputs
        else None
    )
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[_TableRefinementJob] = []
    for page_index, (page_image, page_payload) in enumerate(
        zip(page_images, page_payloads)
    ):
        page_box = _numeric_box(page_payload.get("page_box"))
        blocks = page_payload.get("blocks")
        if not isinstance(blocks, list):
            continue

        table_number = 0
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            if _normalize_label(str(block.get("label") or "")) != "Table":
                continue

            table_number += 1
            summary["attempted"] += 1
            bbox = _numeric_box(block.get("bbox"))
            record: dict[str, Any] = {
                "page": page_index,
                "block_index": block_index,
                "bbox": bbox,
                "status": "failed",
            }
            summary["records"].append(record)
            try:
                if not bbox or not page_box:
                    raise ValueError("table bbox or page_box is unavailable")
                crop_box = _crop_box_from_bbox(
                    page_image,
                    bbox,
                    page_box,
                    config.table_crop_margin_ratio,
                )
                crop = page_image.crop(crop_box)
                model_crop = _resize_table_crop(crop, config)
                record["crop_box_pixels"] = list(crop_box)
                record["model_image_size"] = list(model_crop.size)

                prefix = f"page_{page_index + 1:04d}_table_{table_number:04d}"
                if artifact_dir is not None:
                    crop_path = artifact_dir / f"{prefix}.crop.png"
                    model_crop.save(crop_path)
                    record["crop_path"] = portable_path(
                        crop_path, config.project_root
                    )

                jobs.append(
                    _TableRefinementJob(
                        page_index=page_index,
                        table_index=table_number - 1,
                        block=block,
                        model_crop=model_crop,
                        prefix=prefix,
                        record=record,
                    )
                )
            except Exception as exc:
                summary["failed"] += 1
                record["error"] = f"{type(exc).__name__}: {exc}"

    replacements_by_page: dict[int, list[tuple[int, str]]] = {}

    def consume(job: _TableRefinementJob, result: Any) -> None:
        try:
            _apply_table_refinement_result(
                config,
                job,
                result,
                artifact_dir,
                summary,
                replacements_by_page,
            )
        except Exception as exc:
            summary["failed"] += 1
            job.record["error"] = f"{type(exc).__name__}: {exc}"

    if jobs and config.method == "vllm":
        owned_executor: ThreadPoolExecutor | None = None
        active_executor = executor
        if active_executor is None:
            owned_executor = ThreadPoolExecutor(
                max_workers=min(config.table_max_workers, len(jobs))
            )
            active_executor = owned_executor
        try:
            future_to_job = {
                active_executor.submit(
                    _generate_table_refinement,
                    config,
                    runtime,
                    manager,
                    job.model_crop,
                ): job
                for job in jobs
            }
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    consume(job, future.result())
                except Exception as exc:
                    summary["failed"] += 1
                    job.record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if owned_executor is not None:
                owned_executor.shutdown(wait=True)
    else:
        for job in jobs:
            try:
                result = _generate_table_refinement(
                    config,
                    runtime,
                    manager,
                    job.model_crop,
                )
                consume(job, result)
            except Exception as exc:
                summary["failed"] += 1
                job.record["error"] = f"{type(exc).__name__}: {exc}"

    for page_index, replacements in replacements_by_page.items():
        if replacements:
            page_markdown[page_index] = _replace_tables(
                page_markdown[page_index], sorted(replacements)
            )
            page_clean_html[page_index] = _replace_tables(
                page_clean_html[page_index], sorted(replacements)
            )

    summary["complete"] = summary["failed"] == 0
    if artifact_dir is not None:
        summary["artifact_dir"] = portable_path(
            artifact_dir, config.project_root
        )
    return summary


def _generate_table_refinement(
    config: Chandra2Config,
    runtime: _ChandraRuntime,
    manager: Any,
    model_crop: Any,
) -> Any:
    batch = [
        runtime.batch_input_item(
            image=model_crop,
            prompt=config.table_prompt,
        )
    ]
    options: dict[str, Any] = {
        "include_images": False,
        "include_headers_footers": False,
        "max_output_tokens": config.table_max_output_tokens,
    }
    if config.method == "vllm":
        options.update(
            max_workers=1,
            max_retries=config.max_retries,
        )
    generated = list(manager.generate(batch, **options))
    if len(generated) != 1:
        raise RuntimeError("table OCR returned no single result")
    result = generated[0]
    if bool(getattr(result, "error", False)):
        raise RuntimeError("table OCR result is marked as failed")
    return result


def _apply_table_refinement_result(
    config: Chandra2Config,
    job: _TableRefinementJob,
    result: Any,
    artifact_dir: Path | None,
    summary: dict[str, Any],
    replacements_by_page: dict[int, list[tuple[int, str]]],
) -> None:
    raw_response = str(getattr(result, "raw", "") or "")
    response = (
        raw_response
        or str(getattr(result, "html", "") or "")
        or str(getattr(result, "markdown", "") or "")
    )
    refined_table = _extract_first_table(response)
    if not refined_table:
        raise ValueError("table OCR did not return a complete HTML table")

    primary_table = _extract_first_table(str(job.block.get("content") or ""))
    accepted, guard = _minimal_table_refinement_guard(
        primary_table or "",
        refined_table,
    )
    result_tokens = int(getattr(result, "token_count", 0) or 0)
    summary["token_count"] += result_tokens
    job.record.update(
        token_count=result_tokens,
        rowspan_count=_span_count(refined_table, "rowspan"),
        colspan_count=_span_count(refined_table, "colspan"),
        guard=guard,
    )
    if artifact_dir is not None:
        raw_path = artifact_dir / f"{job.prefix}.raw.html"
        table_path = artifact_dir / f"{job.prefix}.table.html"
        raw_path.write_text(response, encoding="utf-8")
        table_path.write_text(refined_table, encoding="utf-8")
        job.record["raw_path"] = portable_path(raw_path, config.project_root)
        job.record["table_path"] = portable_path(table_path, config.project_root)

    if not accepted:
        summary["rejected"] += 1
        job.record["status"] = "rejected"
        return

    job.block["content"] = refined_table
    replacements_by_page.setdefault(job.page_index, []).append(
        (job.table_index, refined_table)
    )
    summary["succeeded"] += 1
    job.record["status"] = "succeeded"


def _crop_box_from_bbox(
    image: Any,
    bbox: list[int | float],
    page_box: list[int | float],
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    image_width, image_height = image.size
    page_x0, page_y0, page_x1, page_y1 = page_box
    page_width = float(page_x1) - float(page_x0)
    page_height = float(page_y1) - float(page_y0)
    if page_width <= 0 or page_height <= 0:
        raise ValueError("invalid page_box dimensions")

    x0, y0, x1, y1 = bbox
    left = (float(x0) - float(page_x0)) * image_width / page_width
    top = (float(y0) - float(page_y0)) * image_height / page_height
    right = (float(x1) - float(page_x0)) * image_width / page_width
    bottom = (float(y1) - float(page_y0)) * image_height / page_height
    margin = max(right - left, bottom - top) * margin_ratio
    crop_box = (
        max(0, math.floor(left - margin)),
        max(0, math.floor(top - margin)),
        min(image_width, math.ceil(right + margin)),
        min(image_height, math.ceil(bottom + margin)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError("table bbox maps to an empty crop")
    return crop_box


def _resize_table_crop(image: Any, config: Chandra2Config) -> Any:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("empty table crop")
    scale = min(
        config.table_crop_min_short_side / min(width, height),
        config.table_crop_max_long_side / max(width, height),
        math.sqrt(config.table_crop_max_pixels / (width * height)),
    )
    if abs(scale - 1.0) < 0.01:
        return image.copy()
    size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    try:
        from PIL import Image

        return image.resize(size, Image.Resampling.LANCZOS)
    except (ImportError, AttributeError):
        return image.resize(size)


def _extract_first_table(value: str) -> str | None:
    match = re.search(r"<table\b[^>]*>.*?</table\s*>", value, re.I | re.S)
    return match.group(0).strip() if match else None


def _replace_tables(value: str, replacements: list[tuple[int, str]]) -> str:
    """Replace tables by their page-local layout order."""

    by_index = dict(replacements)
    index = 0

    def replace_at_index(match: re.Match[str]) -> str:
        nonlocal index
        replacement = by_index.get(index, match.group(0))
        index += 1
        return replacement

    return re.sub(
        r"<table\b[^>]*>.*?</table\s*>",
        replace_at_index,
        value,
        flags=re.I | re.S,
    )


def _span_count(value: str, attribute: str) -> int:
    return len(re.findall(rf"\b{re.escape(attribute)}\s*=", value, re.I))


def _minimal_table_refinement_guard(
    primary: str,
    candidate: str,
) -> tuple[bool, dict[str, Any]]:
    """Accept structural repairs without allowing schema or content drift."""

    primary_shape = _table_shape(primary)
    candidate_shape = _table_shape(candidate)
    detail: dict[str, Any] = {
        "profile": "minimal",
        "primary_shape": primary_shape,
        "candidate_shape": candidate_shape,
    }
    if primary_shape is None or candidate_shape is None:
        detail["reason"] = "unparseable"
        return False, detail

    primary_rows, primary_columns = primary_shape
    candidate_rows, candidate_columns = candidate_shape
    primary_tokens = _table_tokens(primary)
    candidate_tokens = _table_tokens(candidate)
    overlap = sum((primary_tokens & candidate_tokens).values())
    precision = overlap / max(1, sum(candidate_tokens.values()))
    recall = overlap / max(1, sum(primary_tokens.values()))
    detail.update(
        token_precision=precision,
        token_recall=recall,
    )

    if candidate_columns > primary_columns:
        detail["reason"] = "added_columns"
        return False, detail

    contracting_grid = (
        candidate_rows < primary_rows
        and candidate_columns < primary_columns
        and precision >= 0.99
    )
    if (precision < 0.99 or recall < 0.99) and not contracting_grid:
        detail["reason"] = "text_not_preserved"
        return False, detail

    detail["reason"] = "accepted"
    return True, detail


def _table_tokens(value: str) -> Counter[str]:
    inspected = _inspect_html(value)
    text = unicodedata.normalize(
        "NFKC",
        " ".join(inspected.parts),
    ).casefold()
    return Counter(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))


def _table_shape(value: str) -> tuple[int, int] | None:
    parser = _TableShapeParser()
    try:
        parser.feed(value)
        parser.close()
    except (ValueError, TypeError):
        return None
    if not parser.rows or not any(parser.rows):
        return None

    occupied: set[tuple[int, int]] = set()
    max_columns = 0
    for row_index, cells in enumerate(parser.rows):
        column_index = 0
        for rowspan, colspan in cells:
            while (row_index, column_index) in occupied:
                column_index += 1
            for spanned_row in range(row_index, row_index + rowspan):
                for spanned_column in range(
                    column_index,
                    column_index + colspan,
                ):
                    occupied.add((spanned_row, spanned_column))
            column_index += colspan
        row_columns = [
            column + 1
            for row, column in occupied
            if row == row_index
        ]
        max_columns = max(max_columns, max(row_columns, default=0))
    return len(parser.rows), max_columns


def _write_stream_page_artifacts(
    config: Chandra2Config,
    file_path: Path,
    data_object: DataObject,
    page_index: int,
    page_count: int,
    result: Any,
) -> None:
    """Checkpoint one completed page before the whole document is ready."""

    if not config.save_raw_outputs or not config.output_dir:
        return
    bundle_dir = Path(config.output_dir) / (
        f"{_safe_slug(file_path.stem)}--{data_object.object_id}"
    )
    pages_dir = bundle_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"page_{page_index + 1:04d}"
    page_payloads, _ = _page_payloads([result])
    page_payload = page_payloads[0]
    page_payload["page_number"] = page_index + 1

    _atomic_write_text(
        pages_dir / f"{prefix}.raw.html",
        str(getattr(result, "raw", "") or ""),
    )
    _atomic_write_text(
        pages_dir / f"{prefix}.clean.html",
        str(getattr(result, "html", "") or ""),
    )
    _atomic_write_text(
        pages_dir / f"{prefix}.first-pass.md",
        str(getattr(result, "markdown", "") or ""),
    )
    _atomic_write_text(
        pages_dir / f"{prefix}.chunks.json",
        _json_text(page_payload),
    )
    logger.info(
        "Persisted completed Chandra page %s/%s: %s",
        page_index + 1,
        page_count,
        file_path.name,
    )


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_raw_outputs(
    config: Chandra2Config,
    bundle_dir: Path | None,
    file_path: Path,
    data_object: DataObject,
    markdown: str,
    results: list[Any],
    page_clean_html: list[str],
    page_payloads: list[dict[str, Any]],
    source_blocks: list[dict[str, Any]],
    extraction: dict[str, Any],
    image_files: list[dict[str, Any]],
    reading_order_complete: bool,
    latency_seconds: float,
    token_count: int,
    table_refinement: dict[str, Any],
    inference_mode: str,
) -> dict[str, str]:
    if not config.save_raw_outputs or bundle_dir is None:
        return {}

    pages_dir = bundle_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = bundle_dir / "result.md"
    clean_html_path = bundle_dir / "result.html"
    raw_html_path = bundle_dir / "raw.html"
    chunks_path = bundle_dir / "chunks.json"
    metadata_path = bundle_dir / "metadata.json"

    raw_sections: list[str] = []
    clean_sections: list[str] = []
    for page_index, (result, clean_html, page_payload) in enumerate(
        zip(results, page_clean_html, page_payloads),
        start=1,
    ):
        prefix = f"page_{page_index:04d}"
        raw_html = str(getattr(result, "raw", "") or "")
        raw_sections.append(
            f'<section data-page-number="{page_index}">\n{raw_html}\n</section>'
        )
        clean_sections.append(
            f'<section data-page-number="{page_index}">\n{clean_html}\n</section>'
        )
        (pages_dir / f"{prefix}.raw.html").write_text(
            raw_html,
            encoding="utf-8",
        )
        (pages_dir / f"{prefix}.clean.html").write_text(
            clean_html,
            encoding="utf-8",
        )
        (pages_dir / f"{prefix}.chunks.json").write_text(
            _json_text(page_payload),
            encoding="utf-8",
        )

    markdown_path.write_text(markdown, encoding="utf-8")
    clean_html_path.write_text("\n\n".join(clean_sections), encoding="utf-8")
    raw_html_path.write_text("\n\n".join(raw_sections), encoding="utf-8")
    chunks_path.write_text(
        _json_text(
            {
                "source_object_id": data_object.object_id,
                "input_path": str(file_path),
                "pages": page_payloads,
            }
        ),
        encoding="utf-8",
    )

    paths = {
        "result_markdown": portable_path(markdown_path, config.project_root),
        "result_html": portable_path(clean_html_path, config.project_root),
        "raw_html": portable_path(raw_html_path, config.project_root),
        "chunks": portable_path(chunks_path, config.project_root),
        "pages": portable_path(pages_dir, config.project_root),
    }
    label_counts = Counter(
        str(block.get("raw_label") or block.get("type") or "Block")
        for block in source_blocks
    )
    metadata_path.write_text(
        _json_text(
            {
                "source_object_id": data_object.object_id,
                "file_name": file_path.name,
                "method": config.method,
                "inference_mode": inference_mode,
                "continuous_page_queue": config.continuous_page_queue,
                "max_workers": config.max_workers,
                "render_processes": config.render_processes,
                "request_batch_size": config.request_batch_size,
                "model_name": _model_name(config.method),
                "page_count": len(results),
                "token_count": token_count,
                "latency_seconds": latency_seconds,
                "label_counts": dict(sorted(label_counts.items())),
                "source_block_count": len(source_blocks),
                "table_count": len(extraction.get("tables", [])),
                "figure_count": len(extraction.get("figures", [])),
                "formula_count": len(extraction.get("formulas", [])),
                "image_files": image_files,
                "reading_order_source": (
                    "chandra2_layout" if source_blocks else "unavailable"
                ),
                "reading_order_complete": reading_order_complete,
                "table_refinement": table_refinement,
                "raw_chandra_outputs": paths,
            }
        ),
        encoding="utf-8",
    )
    paths["metadata"] = portable_path(
        metadata_path,
        config.project_root,
    )
    return paths


class _TableShapeParser(HTMLParser):
    """Read the logical dimensions of the first HTML table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[int, int]]] = []
        self._table_depth = 0
        self._current_row: list[tuple[int, int]] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.casefold()
        if tag == "table":
            self._table_depth += 1
            return
        if self._table_depth != 1:
            return
        if tag == "tr":
            self._current_row = []
            self.rows.append(self._current_row)
            return
        if tag not in {"td", "th"} or self._current_row is None:
            return
        attributes = {
            key.casefold(): str(value or "")
            for key, value in attrs
        }
        self._current_row.append(
            (
                _positive_span(attributes.get("rowspan")),
                _positive_span(attributes.get("colspan")),
            )
        )

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "tr" and self._table_depth == 1:
            self._current_row = None
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1


def _positive_span(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


class _HTMLInspector(HTMLParser):
    """Collect readable text, image references, and math from model HTML."""

    _SEPARATORS = set(
        "br caption div h1 h2 h3 h4 h5 li ol p pre table td th tr ul".split()
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.images: list[dict[str, str]] = []
        self.math: list[str] = []
        self._in_math = False
        self._math_parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.casefold()
        if tag in self._SEPARATORS:
            self.parts.append(" ")
        if tag == "img":
            attributes = {key.casefold(): str(value or "") for key, value in attrs}
            alt = attributes.get("alt", "").strip()
            src = attributes.get("src", "").strip()
            self.images.append({"alt": alt, "src": src})
            if alt:
                self.parts.append(f" {alt} ")
        if tag == "math":
            self._in_math = True
            self._math_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "math" or not self._in_math:
            return
        value = _normalize_text("".join(self._math_parts))
        if value:
            self.math.append(value)
        self._in_math = False
        self._math_parts = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)
            if self._in_math:
                self._math_parts.append(data)


def _page_payloads(
    results: list[Any],
) -> tuple[list[dict[str, Any]], int]:
    pages: list[dict[str, Any]] = []
    raw_block_count = 0
    for page_number, result in enumerate(results, start=1):
        raw_blocks = list(getattr(result, "chunks", None) or [])
        raw_block_count += len(raw_blocks)
        blocks = [dict(block) for block in raw_blocks if isinstance(block, dict)]
        pages.append(
            {
                "page_number": page_number,
                "page_box": _numeric_box(getattr(result, "page_box", None)) or [],
                "token_count": int(getattr(result, "token_count", 0) or 0),
                "blocks": blocks,
                "image_files": [],
            }
        )
    return pages, raw_block_count


def _source_blocks_from_pages(
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_blocks: list[dict[str, Any]] = []
    for page in pages:
        page_index = int(page["page_number"]) - 1
        page_box = _numeric_box(page.get("page_box"))
        blocks = page.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block_index, raw_block in enumerate(blocks):
            if not isinstance(raw_block, dict):
                continue
            raw_label = str(raw_block.get("label") or "Block").strip() or "Block"
            block_type = _normalize_label(raw_label)
            html = str(raw_block.get("content") or "")
            inspected = _inspect_html(html)
            block: dict[str, Any] = {
                "component_id": (
                    f"/page/{page_index}/{block_type}/{block_index}"
                ),
                "page": page_index,
                "block_index": block_index,
                "type": block_type,
                "raw_label": raw_label,
                "text": _normalize_text("".join(inspected.parts)),
                "source": AXIOM_NATIVE_BLOCK_SOURCE,
                "parser_source": "chandra2_layout",
                "html": html,
                "section_hierarchy": {},
            }
            bbox = _numeric_box(raw_block.get("bbox"))
            if bbox:
                block["bbox"] = bbox
                block["polygon"] = _polygon_from_box(bbox)
            if page_box:
                block["page_bbox"] = page_box
            source_blocks.append(block)
    return source_blocks


def _build_extraction(
    markdown: str,
    source_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    title_block = next(
        (
            block
            for block in source_blocks
            if block.get("type") == "SectionHeader"
            and str(block.get("text") or "").strip()
        ),
        None,
    )
    title = str(title_block.get("text")) if title_block else None
    title_citations = [title_block["component_id"]] if title_block else []
    text = markdown.strip() or "\n\n".join(
        str(block.get("text") or "").strip()
        for block in source_blocks
        if str(block.get("text") or "").strip()
    )
    text_citations = [
        block["component_id"]
        for block in source_blocks
        if str(block.get("text") or "").strip()
    ]

    tables: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    formulas: list[str] = []
    formula_citations: list[str] = []
    used_caption_ids: set[str] = set()

    for index, block in enumerate(source_blocks):
        block_type = str(block.get("type") or "")
        html = str(block.get("html") or "")
        inspected = _inspect_html(html)
        component_id = str(block["component_id"])

        if block_type == "Table":
            caption, caption_citations = _adjacent_caption(
                source_blocks,
                index,
                prefer_next=False,
                used_caption_ids=used_caption_ids,
            )
            tables.append(
                {
                    "caption": caption,
                    "caption_citations": caption_citations,
                    "content": html or str(block.get("text") or ""),
                    "content_citations": [component_id],
                    "content_format": "html" if html else "text",
                }
            )

        is_figure = block_type in {"Image", "Figure", "Diagram"} or bool(
            inspected.images
        )
        if is_figure:
            caption, caption_citations = _adjacent_caption(
                source_blocks,
                index,
                prefer_next=True,
                used_caption_ids=used_caption_ids,
            )
            alt_text = " ".join(
                dict.fromkeys(
                    _normalize_text(image.get("alt", ""))
                    for image in inspected.images
                    if image.get("alt")
                )
            )
            description = alt_text or str(block.get("text") or "").strip()
            figures.append(
                {
                    "caption": caption,
                    "caption_citations": caption_citations,
                    "description": description,
                    "description_citations": [component_id],
                }
            )

        block_formulas = list(inspected.math)
        if block_type == "EquationBlock" and not block_formulas:
            formula_text = str(block.get("text") or "").strip()
            if formula_text:
                block_formulas.append(formula_text)
        for formula in block_formulas:
            if formula and formula not in formulas:
                formulas.append(formula)
                formula_citations.append(component_id)

    return {
        "document_type": None,
        "language": None,
        "title": title,
        "title_citations": title_citations,
        "main_text": text,
        "main_text_citations": text_citations,
        "tables": tables,
        "figures": figures,
        "formulas": formulas,
        "formulas_citations": formula_citations,
    }


def _adjacent_caption(
    blocks: list[dict[str, Any]],
    index: int,
    *,
    prefer_next: bool,
    used_caption_ids: set[str],
) -> tuple[str, list[str]]:
    candidates = (index + 1, index - 1) if prefer_next else (index - 1, index + 1)
    page = blocks[index].get("page")
    for candidate_index in candidates:
        if not 0 <= candidate_index < len(blocks):
            continue
        candidate = blocks[candidate_index]
        if candidate.get("page") != page or candidate.get("type") != "Caption":
            continue
        component_id = str(candidate.get("component_id") or "")
        if component_id in used_caption_ids:
            continue
        caption = str(candidate.get("text") or "").strip()
        if caption:
            used_caption_ids.add(component_id)
            return caption, [component_id]
    return "", []


def _output_bundle_dir(
    config: Chandra2Config,
    file_path: Path,
    data_object: DataObject,
    results: list[Any],
) -> Path | None:
    if not config.output_dir:
        return None
    has_images = config.include_images and any(
        bool(getattr(result, "images", None)) for result in results
    )
    if not config.save_raw_outputs and not has_images:
        return None
    bundle_dir = Path(config.output_dir) / (
        f"{_safe_slug(file_path.stem)}--{data_object.object_id}"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    return bundle_dir


def _write_images(
    config: Chandra2Config,
    bundle_dir: Path | None,
    results: list[Any],
) -> list[dict[str, Any]]:
    if not config.include_images:
        return []

    records: list[dict[str, Any]] = []
    used_names: set[str] = set()
    image_dir = bundle_dir / "images" if bundle_dir else None
    if image_dir:
        image_dir.mkdir(parents=True, exist_ok=True)

    for page_number, result in enumerate(results, start=1):
        for image_index, (original_name, image) in enumerate(
            dict(getattr(result, "images", None) or {}).items(),
            start=1,
        ):
            file_name = Path(str(original_name)).name
            if not file_name:
                file_name = f"page-{page_number}-image-{image_index}.webp"
            if file_name.casefold() in used_names:
                file_name = f"page-{page_number}-{image_index}-{file_name}"
            used_names.add(file_name.casefold())
            record: dict[str, Any] = {
                "page": page_number - 1,
                "original_name": str(original_name),
                "name": file_name,
                "path": None,
                "status": "not_saved",
            }
            if image_dir is not None:
                output_path = image_dir / file_name
                try:
                    image.save(output_path)
                except Exception as exc:
                    record["status"] = "save_failed"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                else:
                    record["status"] = "saved"
                    record["path"] = portable_path(
                        output_path,
                        config.project_root,
                    )
            records.append(record)
    return records


def _align_image_files(
    figures: Any,
    source_blocks: list[dict[str, Any]],
    results: list[Any],
    saved_images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(figures, list):
        return []
    blocks_by_id = {block["component_id"]: block for block in source_blocks}
    clean_refs_by_page = {
        page: _inspect_html(str(getattr(result, "html", "") or "")).images
        for page, result in enumerate(results)
    }
    image_names = {
        index: {
            Path(str(record.get("original_name") or "")).name.casefold(),
            Path(str(record.get("name") or "")).name.casefold(),
        }
        for index, record in enumerate(saved_images)
    }
    used: set[int] = set()
    aligned: list[dict[str, Any]] = []

    for figure_index, figure in enumerate(figures):
        citations = figure.get("description_citations", [])
        component_id = str(citations[0]) if citations else ""
        block = blocks_by_id.get(component_id, {})
        page = int(block.get("page", 0))
        raw_refs = _inspect_html(str(block.get("html") or "")).images
        raw_alts = {
            _normalize_text(ref["alt"]).casefold()
            for ref in raw_refs
            if ref["alt"]
        }
        candidates = {
            Path(ref["src"]).name.casefold()
            for ref in raw_refs
            if ref["src"]
        }
        candidates.update(
            Path(ref["src"]).name.casefold()
            for ref in clean_refs_by_page.get(page, [])
            if ref["src"]
            and _normalize_text(ref["alt"]).casefold() in raw_alts
        )
        if not candidates and block.get("type") in {"Image", "Figure", "Diagram"}:
            candidates.update(
                Path(ref["src"]).name.casefold()
                for ref in clean_refs_by_page.get(page, [])
                if ref["src"]
            )

        match = next(
            (
                index
                for index, record in enumerate(saved_images)
                if index not in used
                and record["page"] == page
                and candidates & image_names[index]
            ),
            None,
        )
        if match is None:
            aligned.append(
                {
                    "name": None,
                    "path": None,
                    "status": "unavailable",
                    "source_ref": component_id or f"figure:{figure_index}",
                }
            )
            continue

        used.add(match)
        record = saved_images[match]
        aligned.append(
            {
                "name": record["name"],
                "path": record["path"],
                "status": record["status"],
                "source_ref": component_id,
            }
        )

    aligned.extend(
        {
            "name": record["name"],
            "path": record["path"],
            "status": record["status"],
            "source_ref": None,
        }
        for index, record in enumerate(saved_images)
        if index not in used
    )
    return aligned


def _inspect_html(value: str) -> _HTMLInspector:
    inspector = _HTMLInspector()
    inspector.feed(value)
    inspector.close()
    return inspector


def _normalize_label(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value)
    return "".join(part[:1].upper() + part[1:] for part in parts) or "Block"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _numeric_box(value: Any) -> list[int | float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    return list(value)


def _polygon_from_box(
    box: list[int | float],
) -> list[list[int | float]]:
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _positive_int(config: dict[str, Any], name: str, default: int) -> int:
    value = int(config.get(name, default))
    if value <= 0:
        raise ValueError(f"chandra2.{name} must be greater than zero")
    return value


def _positive_float(config: dict[str, Any], name: str, default: float) -> float:
    value = float(config.get(name, default))
    if value <= 0:
        raise ValueError(f"chandra2.{name} must be greater than zero")
    return value


def _vllm_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
        )
    return "" if content is None else str(content)


def _vllm_batch_error_text(error: Exception) -> str:
    response = getattr(error, "response", None)
    if response is None:
        return f"{type(error).__name__}: {error}"
    status_code = getattr(response, "status_code", "unknown")
    try:
        detail = str(response.text).strip().replace("\n", " ")[:500]
    except Exception:
        detail = ""
    suffix = f": {detail}" if detail else ""
    return f"HTTP {status_code}{suffix}"


def _vllm_batch_error_is_retryable(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if response is None:
        return True
    try:
        status_code = int(response.status_code)
    except (AttributeError, TypeError, ValueError):
        return True
    return status_code in {408, 409, 425, 429} or status_code >= 500


def _inference_method(value: Any) -> str:
    method = str(value).strip().lower().replace("-", "_")
    if method == "vllm":
        return "vllm"
    if method in {"hf", "local", "huggingface", "hugging_face"}:
        return "hf"
    raise ValueError(
        "chandra2.method must be 'vllm' or 'hf' (alias: 'local')"
    )


def _model_name(method: str) -> str:
    if method == "hf":
        return os.getenv("MODEL_CHECKPOINT", "datalab-to/chandra-ocr-2")
    return os.getenv("VLLM_MODEL_NAME", "chandra")


def _non_negative_int(config: dict[str, Any], name: str, default: int) -> int:
    value = int(config.get(name, default))
    if value < 0:
        raise ValueError(f"chandra2.{name} must be zero or greater")
    return value


def _non_negative_float(
    config: dict[str, Any],
    name: str,
    default: float,
) -> float:
    value = float(config.get(name, default))
    if value < 0:
        raise ValueError(f"chandra2.{name} must be zero or greater")
    return value


def _non_empty_str(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"chandra2.{name} must not be empty")
    return text


def _optional_str(value: Any) -> str | None:
    return str(value) if value not in {None, ""} else None


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._") or "document"
