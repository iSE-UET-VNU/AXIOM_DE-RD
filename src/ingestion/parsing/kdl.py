"""AXIOM provider for the full KDL-Frontier-Parser-nano pipeline."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import multiprocessing
import os
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any, Callable

import httpx
from PIL import Image

from ...models import DataObject, ParsedData
from .contracts import AXIOM_NATIVE_BLOCK_SOURCE
from .kdl_frontier_engine import (
    NanoEngine,
    NanoUsage,
    SequenceLimiter,
    _nano_group_by_bucket,
    layout_recognition_bucket,
    preprocess_for_vlm,
)

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
    request_workers: int = 8
    request_batch_size: int = 1
    max_model_sequences: int = 32
    progress_every_batches: int = 10
    layout_max_output_tokens: int = 6000
    text_max_output_tokens: int = 2048
    table_max_output_tokens: int = 5500
    picture_max_output_tokens: int = 4096
    formula_max_output_tokens: int = 128
    continuous_page_queue: bool = True
    scheduler: str = "parsebench_document"
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
            or os.getenv("KDL_NANO_ENDPOINT_URL")
            or os.getenv("VLLM_API_BASE")
            or "http://127.0.0.1:8000/v1"
        ).strip().rstrip("/")
        max_workers = _positive_int(values, "max_workers", 32)
        request_batch_size = _positive_int(
            values,
            "request_batch_size",
            int(os.getenv("KDL_NANO_BATCH_SIZE", "1")),
        )
        max_model_sequences = _positive_int(
            values,
            "max_model_sequences",
            int(os.getenv("KDL_NANO_MAX_MODEL_SEQUENCES", "32")),
        )
        if request_batch_size > max_model_sequences:
            raise ValueError(
                "kdl.request_batch_size must not exceed max_model_sequences"
            )
        scheduler = str(
            values.get("scheduler", "parsebench_document")
        ).strip().lower()
        if scheduler not in {"parsebench_document", "global_two_phase"}:
            raise ValueError(
                "kdl.scheduler must be 'parsebench_document' or "
                "'global_two_phase'"
            )
        return cls(
            method=method,
            endpoint_url=endpoint,
            model=str(
                values.get("model")
                or os.getenv("KDL_NANO_MODEL")
                or os.getenv("VLLM_MODEL_NAME")
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
            request_workers=_positive_int(values, "request_workers", 8),
            request_batch_size=request_batch_size,
            max_model_sequences=max_model_sequences,
            progress_every_batches=_positive_int(
                values, "progress_every_batches", 10
            ),
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
            scheduler=scheduler,
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
    layout_pages: list[dict[str, list[dict[str, Any]]] | None] = field(
        default_factory=list
    )
    remaining: int = 0
    routing_context: Any | None = None
    usage: NanoUsage = field(default_factory=NanoUsage)
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
        if self.config.scheduler == "global_two_phase":
            return asyncio.run(
                self._parse_continuous(documents, on_document_complete)
            )
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
                    layout_pages=[None] * count,
                    remaining=count,
                    routing_context=routing_context,
                )
            )

        if self.config.scheduler == "parsebench_document":
            return await self._parse_continuous_by_document(
                prepared,
                on_document_complete,
            )
        if self.config.scheduler == "global_two_phase":
            return await self._parse_global_two_phase(
                prepared,
                on_document_complete,
            )
        raise AssertionError(f"Unsupported KDL scheduler: {self.config.scheduler}")

    async def _parse_continuous_by_document(
        self,
        prepared: list[_KDLDocument],
        on_document_complete: Callable[[int, ParsedData | Exception], None] | None,
    ) -> list[ParsedData | Exception]:
        """Match ParseBench's scheduler: concurrent documents, serial KDL calls.

        Each worker owns one document at a time. Pages are processed in order,
        and layout plus every recognition batch share one per-document semaphore.
        With max_workers=8 and request_batch_size=4 this permits at most eight
        HTTP requests and 32 model sequences at once.
        """

        outcomes: list[ParsedData | Exception | None] = [None] * len(prepared)
        engine = self._engine()
        sequence_limiter = SequenceLimiter(self.config.max_model_sequences)
        valid_indexes = [
            index for index, document in enumerate(prepared)
            if document.failure is None
        ]
        worker_count = min(
            self.config.max_workers,
            self.config.bbox_max_workers,
            max(1, len(valid_indexes)),
        )
        job_queue: asyncio.Queue[int | None] = asyncio.Queue(
            maxsize=worker_count
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
                raw["usage"] = document.usage.snapshot()
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

        async def producer() -> None:
            for index in valid_indexes:
                await job_queue.put(index)
            for _ in range(worker_count):
                await job_queue.put(None)

        async def worker(executor: ProcessPoolExecutor) -> None:
            async with httpx.AsyncClient(
                timeout=self.config.request_timeout_seconds
            ) as client:
                while True:
                    document_index = await job_queue.get()
                    if document_index is None:
                        job_queue.task_done()
                        return
                    document = prepared[document_index]
                    # ParseBench's KDL provider uses max_concurrent=1 inside
                    # each document. The same semaphore gates both stages.
                    request_semaphore = asyncio.Semaphore(1)
                    try:
                        for page_index in range(document.page_count):
                            image: Image.Image | None = None
                            try:
                                image = await loop.run_in_executor(
                                    executor,
                                    _render_page,
                                    str(document.path),
                                    page_index,
                                    self.config.dpi,
                                )
                                kwargs: dict[str, Any] = {
                                    "sequence_limiter": sequence_limiter,
                                    "usage": document.usage,
                                }
                                if self._text_router is not None:
                                    kwargs["routing_context"] = (
                                        document.routing_context
                                    )
                                document.pages[page_index] = await engine._parse_page(
                                    client,
                                    request_semaphore,
                                    request_semaphore,
                                    image,
                                    page_index + 1,
                                    **kwargs,
                                )
                            finally:
                                if image is not None:
                                    image.close()
                    except Exception as exc:
                        document.failure = RuntimeError(
                            f"page {page_index + 1}: {type(exc).__name__}: {exc}"
                        )
                    finally:
                        await finalize(document_index)
                        job_queue.task_done()

        for index, document in enumerate(prepared):
            if document.failure is not None:
                report(index, document.failure)

        with ProcessPoolExecutor(max_workers=self.config.render_processes) as executor:
            workers = [
                asyncio.create_task(worker(executor))
                for _ in range(worker_count)
            ]
            await producer()
            await job_queue.join()
            await asyncio.gather(*workers)

        for index, outcome in enumerate(outcomes):
            if outcome is None:
                report(index, RuntimeError("KDL did not finalize the document."))
        return [item for item in outcomes if item is not None]

    async def _parse_global_two_phase(
        self,
        prepared: list[_KDLDocument],
        on_document_complete: Callable[[int, ParsedData | Exception], None] | None,
    ) -> list[ParsedData | Exception]:
        """Corpus-wide layout barrier followed by global bbox recognition."""

        outcomes: list[ParsedData | Exception | None] = [None] * len(prepared)
        raw_by_document: dict[int, dict[str, Any]] = {}
        engine = self._engine()
        usage = NanoUsage()
        request_semaphore = asyncio.Semaphore(self.config.request_workers)
        sequence_limiter = SequenceLimiter(self.config.max_model_sequences)
        telemetry: dict[str, Any] = {
            "scheduler": "global_two_phase",
            "request_workers": self.config.request_workers,
            "queue_capacity": self.config.max_model_sequences,
            "progress_every_batches": self.config.progress_every_batches,
            "layout_rendered_pages": 0,
            "layout_batches_submitted": 0,
            "layout_batches_completed": 0,
            "layout_batch_failures": 0,
            "recognition_rendered_pages": 0,
            "recognition_jobs": 0,
            "recognition_batches_submitted": 0,
            "recognition_batches_completed": 0,
            "recognition_batch_failures": 0,
            "queue_peak": 0,
            "documents_finalized_during_recognition": 0,
            "persistence_queue_peak": 0,
        }
        wall_started = time.perf_counter()

        persistence_queue: asyncio.Queue[tuple[int, ParsedData | Exception] | None] = (
            asyncio.Queue()
        )
        persistence_errors: list[Exception] = []
        persistence_executor: ThreadPoolExecutor | None = None
        persistence_task: asyncio.Task[None] | None = None
        loop = asyncio.get_running_loop()

        if on_document_complete is not None:
            persistence_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="kdl-persistence",
            )

            async def persist_completed_documents() -> None:
                assert persistence_executor is not None
                while True:
                    item = await persistence_queue.get()
                    try:
                        if item is None:
                            return
                        index, outcome = item
                        try:
                            await loop.run_in_executor(
                                persistence_executor,
                                on_document_complete,
                                index,
                                outcome,
                            )
                        except Exception as exc:  # pragma: no cover - caller-specific
                            persistence_errors.append(exc)
                    finally:
                        persistence_queue.task_done()

            persistence_task = asyncio.create_task(persist_completed_documents())

        recognition_active = False
        total_pages = sum(
            prepared[index].page_count for index in range(len(prepared))
        )
        logger.info(
            "KDL global_two_phase started: documents=%s pages=%s "
            "request_workers=%s batch_size=%s sequence_capacity=%s",
            len(prepared),
            total_pages,
            self.config.request_workers,
            self.config.request_batch_size,
            self.config.max_model_sequences,
        )

        def publish(index: int, outcome: ParsedData | Exception) -> None:
            document = prepared[index]
            if document.reported:
                return
            document.reported = True
            outcomes[index] = outcome
            if recognition_active:
                telemetry["documents_finalized_during_recognition"] += 1
            if on_document_complete is not None:
                persistence_queue.put_nowait((index, outcome))
                telemetry["persistence_queue_peak"] = max(
                    telemetry["persistence_queue_peak"],
                    persistence_queue.qsize(),
                )

        def finalize_document(index: int) -> None:
            document = prepared[index]
            if document.reported:
                return
            if document.failure is not None:
                publish(index, document.failure)
                return
            elements = [
                element
                for page in document.pages
                for element in (page or [])
            ]
            try:
                raw = engine.finalize_elements(elements)
                self._attach_routing_metadata(raw, document.routing_context)
                raw_by_document[index] = raw
                parsed = self._to_parsed_data(
                    document.path,
                    document.data_object,
                    raw,
                    document.started,
                    document.page_count,
                    write_raw_outputs=False,
                )
            except Exception as exc:
                publish(index, exc)
            else:
                publish(index, parsed)

        valid_indexes = [
            index for index, document in enumerate(prepared)
            if document.failure is None
        ]
        if valid_indexes:
            layout_started = time.perf_counter()
            await self._run_global_layout_phase(
                prepared,
                valid_indexes,
                engine,
                usage,
                request_semaphore,
                sequence_limiter,
                telemetry,
            )
            telemetry["layout_phase_latency_ms"] = round(
                (time.perf_counter() - layout_started) * 1000.0, 3
            )
            logger.info(
                "KDL layout phase completed: pages=%s batches=%s failures=%s "
                "queue_peak=%s elapsed_ms=%s",
                telemetry["layout_rendered_pages"],
                telemetry["layout_batches_completed"],
                telemetry["layout_batch_failures"],
                telemetry["queue_peak"],
                telemetry["layout_phase_latency_ms"],
            )

            routing_started = time.perf_counter()
            job_map, crop_tasks = await self._prepare_global_recognition(
                prepared,
                valid_indexes,
            )
            telemetry["routing_phase_latency_ms"] = round(
                (time.perf_counter() - routing_started) * 1000.0, 3
            )
            telemetry["recognition_jobs"] = len(job_map)
            logger.info(
                "KDL recognition routing completed: jobs=%s elapsed_ms=%s",
                telemetry["recognition_jobs"],
                telemetry["routing_phase_latency_ms"],
            )

            remaining_jobs = {index: 0 for index in valid_indexes}
            completed_job_ids: set[str] = set()
            for job_id in job_map:
                document_index = int(job_id.split(":", 1)[0])
                remaining_jobs[document_index] += 1

            def complete_jobs(elements: list[dict[str, Any]]) -> None:
                ready_documents: set[int] = set()
                for element in elements:
                    job_id = str(element.get("job_id") or "")
                    if not job_id or job_id in completed_job_ids:
                        continue
                    completed_job_ids.add(job_id)
                    document_index = int(job_id.split(":", 1)[0])
                    remaining_jobs[document_index] -= 1
                    if remaining_jobs[document_index] == 0:
                        ready_documents.add(document_index)
                for document_index in sorted(ready_documents):
                    finalize_document(document_index)

            recognition_started = time.perf_counter()
            recognition_active = True
            for index in valid_indexes:
                if remaining_jobs[index] == 0:
                    finalize_document(index)
            if crop_tasks:
                await self._run_global_recognition_phase(
                    prepared,
                    crop_tasks,
                    job_map,
                    engine,
                    usage,
                    request_semaphore,
                    sequence_limiter,
                    telemetry,
                    complete_jobs,
                )
            recognition_active = False
            telemetry["recognition_phase_latency_ms"] = round(
                (time.perf_counter() - recognition_started) * 1000.0, 3
            )
            logger.info(
                "KDL recognition phase completed: pages=%s batches=%s "
                "failures=%s documents_finalized=%s queue_peak=%s elapsed_ms=%s",
                telemetry["recognition_rendered_pages"],
                telemetry["recognition_batches_completed"],
                telemetry["recognition_batch_failures"],
                telemetry["documents_finalized_during_recognition"],
                telemetry["queue_peak"],
                telemetry["recognition_phase_latency_ms"],
            )

        telemetry["end_to_end_latency_ms"] = round(
            (time.perf_counter() - wall_started) * 1000.0, 3
        )
        for index, document in enumerate(prepared):
            if not document.reported:
                finalize_document(index)

        persistence_drain_started = time.perf_counter()
        if on_document_complete is not None:
            await persistence_queue.join()
            persistence_queue.put_nowait(None)
            await persistence_queue.join()
            assert persistence_task is not None
            await persistence_task
        telemetry["post_recognition_persistence_drain_ms"] = round(
            (time.perf_counter() - persistence_drain_started) * 1000.0,
            3,
        )

        final_usage = usage.snapshot()
        telemetry["usage"] = final_usage
        raw_write_started = time.perf_counter()
        for index, raw in raw_by_document.items():
            outcome = outcomes[index]
            if not isinstance(outcome, ParsedData):
                continue
            raw["usage"] = final_usage
            raw["_global_scheduler"] = telemetry
            raw_paths = self._write_raw_outputs(
                prepared[index].path,
                prepared[index].data_object,
                raw,
            )
            outcome.metadata["kdl_usage"] = final_usage
            outcome.metadata["kdl_global_scheduler"] = telemetry
            outcome.metadata["raw_kdl_outputs"] = raw_paths
            outcome.metadata["raw_output_path"] = raw_paths.get("result_markdown")
            outcome.metadata["raw_metadata_path"] = raw_paths.get("result_json")
        telemetry["raw_output_write_latency_ms"] = round(
            (time.perf_counter() - raw_write_started) * 1000.0,
            3,
        )
        telemetry["provider_end_to_end_latency_ms"] = round(
            (time.perf_counter() - wall_started) * 1000.0,
            3,
        )
        logger.info(
            "KDL global_two_phase finished: documents=%s layout_pages=%s "
            "recognition_jobs=%s elapsed_ms=%s raw_write_ms=%s",
            len(prepared),
            telemetry["layout_rendered_pages"],
            telemetry["recognition_jobs"],
            telemetry["provider_end_to_end_latency_ms"],
            telemetry["raw_output_write_latency_ms"],
        )

        if persistence_executor is not None:
            persistence_executor.shutdown(wait=True)
        if persistence_errors:
            raise RuntimeError(
                "KDL completion persistence failed: " + str(persistence_errors[0])
            )
        return [item for item in outcomes if item is not None]

    async def _run_global_layout_phase(
        self,
        prepared: list[_KDLDocument],
        document_indexes: list[int],
        engine: NanoEngine,
        usage: NanoUsage,
        request_semaphore: asyncio.Semaphore,
        sequence_limiter: SequenceLimiter,
        telemetry: dict[str, Any],
    ) -> None:
        context = multiprocessing.get_context("spawn")
        task_queue = context.Queue()
        rendered_queue = context.Queue(maxsize=self.config.max_model_sequences)
        image_slots = context.BoundedSemaphore(self.config.max_model_sequences)
        cancelled = context.Array("b", len(prepared), lock=False)
        process_count = min(self.config.render_processes, len(document_indexes))
        processes = [
            context.Process(
                target=_kdl_document_worker,
                args=(
                    task_queue,
                    rendered_queue,
                    image_slots,
                    cancelled,
                    self.config.dpi,
                    "layout",
                ),
                name=f"kdl-layout-render-{index + 1}",
            )
            for index in range(process_count)
        ]
        for process in processes:
            process.start()
        for document_index in document_indexes:
            task_queue.put(
                (document_index, str(prepared[document_index].path), None)
            )
        for _ in processes:
            task_queue.put(None)

        pending: list[tuple[int, int, Image.Image]] = []
        in_flight: dict[
            asyncio.Task[list[dict[str, list[dict[str, Any]]]]],
            list[tuple[int, int, Image.Image]],
        ] = {}
        completed_documents: set[int] = set()

        async with httpx.AsyncClient(
            timeout=self.config.request_timeout_seconds
        ) as client:
            async def submit(*, force: bool = False) -> None:
                while (
                    pending
                    and len(in_flight) < self.config.request_workers
                    and (force or len(pending) >= self.config.request_batch_size)
                ):
                    take = min(self.config.request_batch_size, len(pending))
                    batch = pending[:take]
                    del pending[:take]
                    task = asyncio.create_task(
                        engine.layout_batch(
                            client,
                            [
                                (image, page_index + 1)
                                for _, page_index, image in batch
                            ],
                            request_semaphore,
                            sequence_limiter=sequence_limiter,
                            usage=usage,
                        )
                    )
                    in_flight[task] = batch
                    telemetry["layout_batches_submitted"] += 1

            try:
                while (
                    len(completed_documents) < len(document_indexes)
                    or pending
                    or in_flight
                ):
                    event: Any | None = None
                    try:
                        event = await asyncio.to_thread(
                            rendered_queue.get, True, 0.05
                        )
                    except Empty:
                        pass
                    if (
                        event is None
                        and not any(process.is_alive() for process in processes)
                        and len(completed_documents) < len(document_indexes)
                    ):
                        missing = set(document_indexes) - completed_documents
                        for document_index in missing:
                            if prepared[document_index].failure is None:
                                prepared[document_index].failure = RuntimeError(
                                    "KDL layout renderer exited before document completion."
                                )
                        completed_documents.update(missing)
                    if event is not None:
                        event_type, document_index, page_index, payload = event
                        if event_type == "page":
                            image = _deserialize_image(payload)
                            pending.append((document_index, page_index, image))
                            telemetry["layout_rendered_pages"] += 1
                            telemetry["queue_peak"] = max(
                                telemetry["queue_peak"],
                                len(pending) + len(in_flight),
                            )
                        elif event_type == "error":
                            error_type, message = payload
                            if prepared[document_index].failure is None:
                                prepared[document_index].failure = RuntimeError(
                                    f"page {page_index + 1}: {error_type}: {message}"
                                )
                            cancelled[document_index] = 1
                        elif event_type == "done":
                            completed_documents.add(document_index)

                    force = len(completed_documents) == len(document_indexes)
                    await submit(force=force)
                    for task in [task for task in in_flight if task.done()]:
                        batch = in_flight.pop(task)
                        try:
                            grouped_pages = task.result()
                        except Exception as exc:
                            telemetry["layout_batch_failures"] += 1
                            grouped_pages = [
                                {"text": [], "table": [], "picture": [], "formula": []}
                                for _ in batch
                            ]
                            logger.warning("global layout batch failed: %s", exc)
                        telemetry["layout_batches_completed"] += 1
                        for (document_index, page_index, image), grouped in zip(
                            batch, grouped_pages, strict=True
                        ):
                            prepared[document_index].layout_pages[page_index] = grouped
                            image.close()
                            image_slots.release()
                        completed_batches = telemetry["layout_batches_completed"]
                        if (
                            completed_batches % self.config.progress_every_batches == 0
                            or not pending and not in_flight
                        ):
                            logger.info(
                                "KDL layout progress: pages=%s/%s batches_completed=%s "
                                "batches_submitted=%s pending=%s in_flight=%s",
                                telemetry["layout_rendered_pages"],
                                sum(
                                    prepared[index].page_count
                                    for index in document_indexes
                                ),
                                completed_batches,
                                telemetry["layout_batches_submitted"],
                                len(pending),
                                len(in_flight),
                            )
                    await submit(force=force)
            finally:
                for _, _, image in pending:
                    image.close()
                    image_slots.release()
                for task, batch in in_flight.items():
                    if not task.done():
                        task.cancel()
                    for _, _, image in batch:
                        image.close()
                        image_slots.release()
                await asyncio.gather(*in_flight, return_exceptions=True)
                _stop_processes(processes)
                task_queue.close()
                task_queue.cancel_join_thread()
                rendered_queue.close()
                rendered_queue.cancel_join_thread()

    async def _prepare_global_recognition(
        self,
        prepared: list[_KDLDocument],
        document_indexes: list[int],
    ) -> tuple[dict[str, dict[str, Any]], dict[int, dict[int, list[dict[str, Any]]]]]:
        routing_semaphore = asyncio.Semaphore(self.config.request_workers)

        async def route_document(index: int) -> dict[int, set[int]]:
            document = prepared[index]
            if self._text_router is None or document.failure is not None:
                return {}
            page_buckets = {
                page_index + 1: (grouped or {}).get("text", [])
                for page_index, grouped in enumerate(document.layout_pages)
                if (grouped or {}).get("text")
            }
            if not page_buckets:
                return {}
            async with routing_semaphore:
                return await self._text_router.route_document_text_regions(
                    document.routing_context,
                    page_buckets,
                )

        routed_results = await asyncio.gather(
            *(route_document(index) for index in document_indexes),
            return_exceptions=True,
        )
        routed_by_document: dict[int, dict[int, set[int]]] = {}
        for index, result in zip(document_indexes, routed_results, strict=True):
            if isinstance(result, Exception):
                logger.warning(
                    "document-level native text routing failed for %s: %s",
                    prepared[index].path,
                    result,
                )
                routed_by_document[index] = {}
            else:
                routed_by_document[index] = result

        job_map: dict[str, dict[str, Any]] = {}
        crop_tasks: dict[int, dict[int, list[dict[str, Any]]]] = {}
        for document_index in document_indexes:
            document = prepared[document_index]
            if document.failure is not None:
                continue
            document.pages = [None] * document.page_count
            routed_pages = routed_by_document.get(document_index, {})
            document_crop_pages: dict[int, list[dict[str, Any]]] = {}
            for page_index, grouped_value in enumerate(document.layout_pages):
                grouped = grouped_value or {
                    "text": [], "table": [], "picture": [], "formula": []
                }
                page_elements = [
                    element
                    for bucket in ("text", "table", "picture", "formula")
                    for element in grouped[bucket]
                ]
                document.pages[page_index] = page_elements
                candidates: list[dict[str, Any]] = []
                routed_text = routed_pages.get(page_index + 1, set())
                for bucket_index, element in enumerate(grouped["text"]):
                    if bucket_index not in routed_text:
                        candidates.append(element)
                candidates.extend(grouped["table"])
                candidates.extend(grouped["picture"])
                candidates.extend(grouped["formula"])
                table_ids = {id(element) for element in grouped["table"]}
                for sequence, element in enumerate(candidates):
                    job_id = f"{document_index}:{page_index}:{sequence}"
                    element["job_id"] = job_id
                    element["fullpage_table"] = (
                        id(element) in table_ids
                        and len(grouped["table"]) == 1
                    )
                    job_map[job_id] = element
                if candidates:
                    document_crop_pages[page_index] = candidates
            if document_crop_pages:
                crop_tasks[document_index] = document_crop_pages
        return job_map, crop_tasks

    async def _run_global_recognition_phase(
        self,
        prepared: list[_KDLDocument],
        crop_tasks: dict[int, dict[int, list[dict[str, Any]]]],
        job_map: dict[str, dict[str, Any]],
        engine: NanoEngine,
        usage: NanoUsage,
        request_semaphore: asyncio.Semaphore,
        sequence_limiter: SequenceLimiter,
        telemetry: dict[str, Any],
        on_jobs_completed: Callable[[list[dict[str, Any]]], None],
    ) -> None:
        context = multiprocessing.get_context("spawn")
        task_queue = context.Queue()
        crop_queue = context.Queue(maxsize=self.config.max_model_sequences)
        crop_slots = context.BoundedSemaphore(self.config.max_model_sequences)
        cancelled = context.Array("b", len(prepared), lock=False)
        process_count = min(self.config.render_processes, len(crop_tasks))
        processes = [
            context.Process(
                target=_kdl_document_worker,
                args=(
                    task_queue,
                    crop_queue,
                    crop_slots,
                    cancelled,
                    self.config.dpi,
                    "crop",
                ),
                name=f"kdl-crop-render-{index + 1}",
            )
            for index in range(process_count)
        ]
        for process in processes:
            process.start()
        for document_index, pages in crop_tasks.items():
            task_queue.put(
                (document_index, str(prepared[document_index].path), pages)
            )
        for _ in processes:
            task_queue.put(None)

        queues: dict[str, deque[dict[str, Any]]] = {
            "text": deque(),
            "table_fullpage": deque(),
            "table": deque(),
            "picture": deque(),
            "formula": deque(),
        }
        stage_order = tuple(queues)
        stage_cursor = 0
        in_flight: dict[
            asyncio.Task[list[dict[str, Any]]],
            tuple[str, list[dict[str, Any]]],
        ] = {}
        completed_documents: set[int] = set()

        async with httpx.AsyncClient(
            timeout=self.config.request_timeout_seconds
        ) as client:
            async def submit(*, force: bool = False) -> None:
                nonlocal stage_cursor
                while len(in_flight) < self.config.request_workers:
                    selected: str | None = None
                    for offset in range(len(stage_order)):
                        candidate = stage_order[(stage_cursor + offset) % len(stage_order)]
                        if len(queues[candidate]) >= self.config.request_batch_size:
                            selected = candidate
                            stage_cursor = (stage_cursor + offset + 1) % len(stage_order)
                            break
                    if selected is None and force:
                        for offset in range(len(stage_order)):
                            candidate = stage_order[(stage_cursor + offset) % len(stage_order)]
                            if queues[candidate]:
                                selected = candidate
                                stage_cursor = (stage_cursor + offset + 1) % len(stage_order)
                                break
                    if selected is None:
                        return
                    take = min(self.config.request_batch_size, len(queues[selected]))
                    elements = [queues[selected].popleft() for _ in range(take)]
                    model_stage = "table" if selected == "table_fullpage" else selected
                    task = asyncio.create_task(
                        engine.recognize_prepared_batch(
                            client,
                            model_stage,
                            elements,
                            request_semaphore,
                            sequence_limiter=sequence_limiter,
                            usage=usage,
                            image_key=(
                                "fullpage_image"
                                if selected == "table_fullpage"
                                else "preprocessed_image"
                            ),
                            fullpage_table=selected == "table_fullpage",
                        )
                    )
                    in_flight[task] = (selected, elements)
                    telemetry["recognition_batches_submitted"] += 1

            try:
                while (
                    len(completed_documents) < len(crop_tasks)
                    or any(queues.values())
                    or in_flight
                ):
                    event: Any | None = None
                    try:
                        event = await asyncio.to_thread(
                            crop_queue.get, True, 0.05
                        )
                    except Empty:
                        pass
                    if (
                        event is None
                        and not any(process.is_alive() for process in processes)
                        and len(completed_documents) < len(crop_tasks)
                    ):
                        missing = set(crop_tasks) - completed_documents
                        for document_index in missing:
                            if prepared[document_index].failure is None:
                                prepared[document_index].failure = RuntimeError(
                                    "KDL crop renderer exited before document completion."
                                )
                        completed_documents.update(missing)
                    if event is not None:
                        event_type, document_index, page_index, payload = event
                        if event_type == "crop":
                            job_id, stage, image_bytes, fullpage_bytes, crop_size = payload
                            element = job_map[job_id]
                            element["preprocessed_image"] = _deserialize_image(image_bytes)
                            element["crop_size"] = tuple(crop_size)
                            if fullpage_bytes is not None:
                                element["fullpage_image"] = _deserialize_image(
                                    fullpage_bytes
                                )
                                queue_name = "table_fullpage"
                            else:
                                queue_name = stage
                            if (
                                stage == "picture"
                                and min(element["preprocessed_image"].size) < 25
                            ):
                                element["content"] = ""
                                element["recognition_source"] = "kdl_picture"
                                _close_job_images(element)
                                crop_slots.release()
                                on_jobs_completed([element])
                            else:
                                queues[queue_name].append(element)
                            telemetry["queue_peak"] = max(
                                telemetry["queue_peak"],
                                sum(len(queue) for queue in queues.values()),
                            )
                        elif event_type == "skipped":
                            skipped_elements: list[dict[str, Any]] = []
                            for job_id in payload:
                                element = job_map[job_id]
                                stage = _element_stage(element)
                                element["content"] = ""
                                element["recognition_source"] = f"kdl_{stage}"
                                skipped_elements.append(element)
                            on_jobs_completed(skipped_elements)
                        elif event_type == "page_done":
                            telemetry["recognition_rendered_pages"] += 1
                        elif event_type == "error":
                            error_type, message = payload
                            if prepared[document_index].failure is None:
                                prepared[document_index].failure = RuntimeError(
                                    f"page {page_index + 1}: {error_type}: {message}"
                                )
                            cancelled[document_index] = 1
                        elif event_type == "done":
                            completed_documents.add(document_index)

                    force = len(completed_documents) == len(crop_tasks)
                    await submit(force=force)
                    for task in [task for task in in_flight if task.done()]:
                        queue_name, elements = in_flight.pop(task)
                        try:
                            table_fallbacks = task.result()
                        except Exception as exc:
                            telemetry["recognition_batch_failures"] += 1
                            logger.warning("global recognition batch failed: %s", exc)
                            table_fallbacks = []
                            for element in elements:
                                stage = _element_stage(element)
                                element["content"] = ""
                                element["recognition_source"] = f"kdl_{stage}"
                        fallback_ids = {id(element) for element in table_fallbacks}
                        completed_elements: list[dict[str, Any]] = []
                        for element in elements:
                            fullpage = element.pop("fullpage_image", None)
                            if fullpage is not None:
                                fullpage.close()
                            if id(element) in fallback_ids:
                                queues["table"].append(element)
                                continue
                            _close_job_images(element)
                            crop_slots.release()
                            completed_elements.append(element)
                        telemetry["recognition_batches_completed"] += 1
                        on_jobs_completed(completed_elements)
                        completed_batches = telemetry["recognition_batches_completed"]
                        if (
                            completed_batches % self.config.progress_every_batches == 0
                            or not any(queues.values()) and not in_flight
                        ):
                            logger.info(
                                "KDL recognition progress: pages=%s/%s "
                                "batches_completed=%s batches_submitted=%s "
                                "queued_elements=%s in_flight=%s",
                                telemetry["recognition_rendered_pages"],
                                sum(len(pages) for pages in crop_tasks.values()),
                                completed_batches,
                                telemetry["recognition_batches_submitted"],
                                sum(len(queue) for queue in queues.values()),
                                len(in_flight),
                            )
                    await submit(force=force)
            finally:
                for queue in queues.values():
                    while queue:
                        element = queue.popleft()
                        _close_job_images(element)
                        crop_slots.release()
                for task, (_, elements) in in_flight.items():
                    if not task.done():
                        task.cancel()
                    for element in elements:
                        _close_job_images(element)
                        crop_slots.release()
                await asyncio.gather(*in_flight, return_exceptions=True)
                _stop_processes(processes)
                task_queue.close()
                task_queue.cancel_join_thread()
                crop_queue.close()
                crop_queue.cancel_join_thread()

        for element in job_map.values():
            crop_size = element.pop("crop_size", None)
            element.pop("preprocessed_image", None)
            element.pop("fullpage_image", None)
            element.pop("fullpage_table", None)
            element.pop("job_id", None)
            if (
                _element_stage(element) == "picture"
                and crop_size is not None
                and min(crop_size) >= 25
            ):
                page_number = int(element.get("page_number", 1))
                layout_order = int(element.get("layout_order", 0))
                element["picture_path"] = (
                    "artifacts/cropped_pictures/"
                    f"page_{page_number:03d}_picture_{layout_order:03d}.png"
                )

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
            request_batch_size=self.config.request_batch_size,
            max_model_sequences=self.config.max_model_sequences,
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
        *,
        write_raw_outputs: bool = True,
    ) -> ParsedData:
        source_blocks = _source_blocks(raw.get("pages") or [])
        markdown = str(raw.get("markdown") or "")
        extraction = _build_extraction(markdown, source_blocks)
        raw_paths = (
            self._write_raw_outputs(file_path, data_object, raw)
            if write_raw_outputs
            else {}
        )
        label_counts = Counter(str(block["raw_label"]) for block in source_blocks)
        metadata: dict[str, Any] = {
            "parser": self.provider_name,
            "method": self.config.method,
            "model_name": self.config.model,
            "inference_mode": self.inference_mode,
            "continuous_page_queue": self.config.continuous_page_queue,
            "scheduler": self.config.scheduler,
            "max_workers": self.config.max_workers,
            "render_processes": self.config.render_processes,
            "bbox_max_workers": self.config.bbox_max_workers,
            "request_workers": self.config.request_workers,
            "request_batch_size": self.config.request_batch_size,
            "max_model_sequences": self.config.max_model_sequences,
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
        usage = raw.get("usage")
        if isinstance(usage, dict):
            metadata["kdl_usage"] = usage
        scheduler_metadata = raw.get("_global_scheduler")
        if isinstance(scheduler_metadata, dict):
            metadata["kdl_global_scheduler"] = scheduler_metadata
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


def _serialize_image(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _deserialize_image(payload: bytes) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB").copy()


def _render_open_document_page(
    document: Any,
    file_path: Path,
    page_index: int,
    dpi: int,
) -> Image.Image:
    if document is None:
        if page_index != 0:
            raise IndexError(f"Image has no page {page_index + 1}")
        with Image.open(file_path) as image:
            return image.convert("RGB").copy()
    import fitz

    page = document.load_page(page_index)
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
        alpha=False,
    )
    with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
        return image.convert("RGB").copy()


def _kdl_document_worker(
    task_queue: Any,
    output_queue: Any,
    image_slots: Any,
    cancelled: Any,
    dpi: int,
    mode: str,
) -> None:
    """Open one document per task and emit layout pages or bbox crops."""

    while True:
        task = task_queue.get()
        if task is None:
            return
        document_index, raw_path, page_jobs = task
        file_path = Path(raw_path)
        document: Any | None = None
        try:
            if file_path.suffix.lower() == ".pdf":
                import fitz

                document = fitz.open(str(file_path))
                page_indexes = (
                    range(document.page_count)
                    if mode == "layout"
                    else sorted(int(index) for index in page_jobs)
                )
            else:
                page_indexes = (0,)

            for page_index in page_indexes:
                if cancelled[document_index]:
                    break
                page_image: Image.Image | None = None
                try:
                    page_image = _render_open_document_page(
                        document,
                        file_path,
                        page_index,
                        dpi,
                    )
                    if mode == "layout":
                        serialized_page = _serialize_image(page_image)
                        image_slots.acquire()
                        output_queue.put(
                            (
                                "page",
                                document_index,
                                page_index,
                                serialized_page,
                            )
                        )
                        continue

                    candidates = list(page_jobs[page_index])
                    grouped = _nano_group_by_bucket(candidates, page_image)
                    candidate_flags = {
                        str(candidate["job_id"]): bool(
                            candidate.get("fullpage_table")
                        )
                        for candidate in candidates
                    }
                    materialized = [
                        (stage, element)
                        for stage in ("text", "table", "picture", "formula")
                        for element in grouped[stage]
                    ]
                    emitted: set[str] = set()
                    fullpage_bytes: bytes | None = None
                    if any(
                        stage == "table"
                        and candidate_flags.get(str(element.get("job_id")), False)
                        for stage, element in materialized
                    ):
                        fullpage = preprocess_for_vlm(page_image)
                        try:
                            fullpage_bytes = _serialize_image(fullpage)
                        finally:
                            fullpage.close()

                    for stage, element in materialized:
                        job_id = str(element["job_id"])
                        preprocessed = element["preprocessed_image"]
                        cropped = element.get("cropped_image")
                        crop_size = (
                            cropped.size if cropped is not None else preprocessed.size
                        )
                        serialized_crop = _serialize_image(preprocessed)
                        image_slots.acquire()
                        output_queue.put(
                            (
                                "crop",
                                document_index,
                                page_index,
                                (
                                    job_id,
                                    stage,
                                    serialized_crop,
                                    (
                                        fullpage_bytes
                                        if stage == "table"
                                        and candidate_flags.get(job_id, False)
                                        else None
                                    ),
                                    crop_size,
                                ),
                            )
                        )
                        emitted.add(job_id)
                        preprocessed.close()
                        if cropped is not None:
                            cropped.close()
                    skipped = [
                        str(candidate["job_id"])
                        for candidate in candidates
                        if str(candidate["job_id"]) not in emitted
                    ]
                    if skipped:
                        output_queue.put(
                            (
                                "skipped",
                                document_index,
                                page_index,
                                skipped,
                            )
                        )
                    output_queue.put(
                        ("page_done", document_index, page_index, None)
                    )
                except Exception as exc:
                    cancelled[document_index] = 1
                    output_queue.put(
                        (
                            "error",
                            document_index,
                            page_index,
                            (type(exc).__name__, str(exc)),
                        )
                    )
                    break
                finally:
                    if page_image is not None:
                        page_image.close()
        except Exception as exc:
            cancelled[document_index] = 1
            output_queue.put(
                (
                    "error",
                    document_index,
                    -1,
                    (type(exc).__name__, str(exc)),
                )
            )
        finally:
            if document is not None:
                document.close()
            output_queue.put(("done", document_index, -1, None))


def _element_stage(element: dict[str, Any]) -> str:
    return layout_recognition_bucket(str(element.get("category") or "Text"))


def _close_job_images(element: dict[str, Any]) -> None:
    for key in ("preprocessed_image", "fullpage_image"):
        image = element.pop(key, None)
        if image is not None:
            image.close()


def _stop_processes(processes: list[Any]) -> None:
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


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
