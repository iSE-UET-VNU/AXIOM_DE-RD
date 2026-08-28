"""Stateful on-demand-per-query retrieval.

The batch discovery runner parses the union of all selected pages before it
answers any question.  This module implements the online variant used by the
Colab benchmark:

    pdf-inspector page index -> BM25 pages per query ->
    KDL parse of missing pages -> fixed 512-word chunks ->
    text-embedding-3-small -> hybrid retrieval (dense alpha=0.7)

Parsed pages and prepared chunks are persisted.  A shared KDL micro-batcher
allows independent query workers to submit missing pages during a short window
so the hosted model receives useful batches without changing query semantics.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, Iterable, Sequence
import copy
import hashlib
import json
import time

import numpy as np

from .pipeline import DiscoveryHit, PageIndex, run_selected_pages
from src.retrieval.fusion import alpha_fuse
from src.retrieval.index import LocalIndex
from src.retrieval.protocol import ChunkRecord
from src.retrieval.sparse import BM25Index


PIPELINE_VERSION = "on-demand-per-query-v1"


@dataclass(frozen=True)
class PreparedChunk:
    """One fixed-size text chunk and its persisted embedding."""

    record_id: str
    page_id: str
    text: str
    vector: np.ndarray


@dataclass(frozen=True)
class OnDemandQueryResult:
    """Result and timing information for one independent query."""

    query_id: str
    query: str
    hits: list[DiscoveryHit]
    selected_pages: dict[str, list[int]]
    ranked_chunks: list[PreparedChunk]
    timing: dict[str, Any]
    parsed_page_ids: tuple[str, ...] = ()
    cached_page_ids: tuple[str, ...] = ()


@dataclass
class _ParseRequest:
    selected: dict[str, list[int]]
    future: Future
    submitted_at: float


class _KDLBatcher:
    """Collect page requests from query workers and parse one merged batch."""

    def __init__(
        self,
        parse_batch: Callable[[dict[str, list[int]], str], dict[str, Any]],
        page_ids: Callable[[dict[str, Sequence[int]]], set[str]],
        *,
        window_seconds: float,
        max_pages: int,
    ) -> None:
        self._parse_batch = parse_batch
        self._page_ids = page_ids
        self._window_seconds = max(0.0, float(window_seconds))
        self._max_pages = max(1, int(max_pages))
        self._queue: Queue[_ParseRequest | None] = Queue()
        self._closed = Event()
        self._batch_number = 0
        self._thread = Thread(
            target=self._worker_loop,
            name="axiom-kdl-microbatcher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, selected: dict[str, list[int]]) -> dict[str, Any]:
        if self._closed.is_set():
            raise RuntimeError("KDL micro-batcher is closed")
        future: Future = Future()
        self._queue.put(
            _ParseRequest(
                selected=selected,
                future=future,
                submitted_at=time.perf_counter(),
            )
        )
        return future.result()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    def _worker_loop(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                return

            requests = [request]
            merged = _merge_selected([request.selected])
            deferred: list[_ParseRequest] = []
            deadline = time.perf_counter() + self._window_seconds

            while len(self._page_ids(merged)) < self._max_pages:
                timeout = max(0.0, deadline - time.perf_counter())
                if timeout == 0.0:
                    break
                try:
                    candidate = self._queue.get(timeout=timeout)
                except Empty:
                    break
                if candidate is None:
                    # Preserve shutdown ordering.  The current batch is
                    # completed before the sentinel is handled next.
                    self._queue.put(None)
                    break
                candidate_merged = _merge_selected([merged, candidate.selected])
                if len(self._page_ids(candidate_merged)) > self._max_pages:
                    deferred.append(candidate)
                    break
                requests.append(candidate)
                merged = candidate_merged

            for candidate in deferred:
                self._queue.put(candidate)

            self._batch_number += 1
            batch_id = f"kdl-batch-{self._batch_number:05d}"
            batch_started = time.perf_counter()
            try:
                batch_result = self._parse_batch(merged, batch_id)
                parsed_ids = set(batch_result.get("parsed_ids") or set())
                finished = time.perf_counter()
                for item in requests:
                    request_ids = self._page_ids(item.selected)
                    result = dict(batch_result)
                    result["parsed_ids"] = parsed_ids & request_ids
                    result["batch_wait_seconds"] = max(
                        0.0, batch_started - item.submitted_at
                    )
                    result["batch_total_seconds"] = max(
                        0.0, finished - item.submitted_at
                    )
                    item.future.set_result(result)
            except Exception as error:  # noqa: BLE001 - propagated to query
                for item in requests:
                    item.future.set_exception(error)


class OnDemandPerQueryRunner:
    """Run the online pipeline while reusing page and chunk state.

    ``run_query`` is safe to call from multiple threads.  KDL parsing is
    serialized through one micro-batcher, while BM25 search, query embedding,
    and retrieval remain query-local.  Chunk/index mutation is serialized so a
    query never observes a half-built index.
    """

    def __init__(
        self,
        index: PageIndex,
        *,
        parser_config: dict[str, Any],
        chunking_config: dict[str, Any],
        project_root: str | Path,
        work_dir: str | Path,
        cache_dir: str | Path | None = None,
        top_k_pages: int = 10,
        top_k_chunks: int = 10,
        depth: int = 100,
        alpha: float = 0.7,
        query_workers: int = 4,
        microbatch_window_seconds: float = 0.30,
        microbatch_max_pages: int = 32,
        force_reparse: bool = False,
        validate_baseline: bool = True,
    ) -> None:
        if not index.pages:
            raise ValueError("On-demand-per-query requires a non-empty PageIndex")
        if top_k_pages <= 0 or top_k_chunks <= 0 or depth <= 0:
            raise ValueError("top_k_pages, top_k_chunks and depth must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        self.index = index
        self.parser_config = copy.deepcopy(parser_config)
        self.chunking_config = copy.deepcopy(chunking_config)
        self.project_root = Path(project_root).resolve()
        self.work_dir = Path(work_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.work_dir / "on-demand-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.top_k_pages = int(top_k_pages)
        self.top_k_chunks = int(top_k_chunks)
        self.depth = int(depth)
        self.alpha = float(alpha)
        self.query_workers = max(1, int(query_workers))
        self.force_reparse = bool(force_reparse)

        self._state_lock = Lock()
        self._chunk_lock = Lock()
        self._timing_lock = Lock()
        self._page_id_by_location = {
            (str(Path(page.file_path).resolve()), int(page.page_index)): page.page_id
            for page in index.pages
        }
        self._selected_pages: dict[str, set[int]] = {}
        self._selected_page_ids: set[str] = set()
        self._records_by_page: dict[str, dict[str, Any]] = {}
        self._page_texts: dict[str, str] = {}
        self._chunks_by_id: dict[str, PreparedChunk] = {}
        self._chunk_index: LocalIndex | None = None
        self._chunk_embedder: Any | None = None
        self._query_timings: dict[str, dict[str, Any]] = {}
        self._accounted_kdl_batches: set[str] = set()
        self._phase_work = {
            "light_retrieval": 0.0,
            "parsing": 0.0,
            "chunk_embed_index": 0.0,
            "retrieval": 0.0,
        }
        self._stage_started_at: float | None = None
        self._stage_ended_at: float | None = None

        if validate_baseline:
            _validate_baseline_chunking(self.chunking_config)
        self._cache_key = self._make_cache_key()
        self._load_caches()
        if self.force_reparse:
            self._records_by_page.clear()
            self._page_texts.clear()
            self._chunks_by_id.clear()
        if self._chunks_by_id:
            with self._chunk_lock:
                self._rebuild_chunk_index()

        self._batcher = _KDLBatcher(
            self._parse_kdl_batch,
            self._requested_page_ids,
            window_seconds=microbatch_window_seconds,
            max_pages=microbatch_max_pages,
        )
        self.microbatch_window_seconds = float(microbatch_window_seconds)
        self.microbatch_max_pages = max(1, int(microbatch_max_pages))

    @property
    def page_texts(self) -> dict[str, str]:
        """Accurate text available for the successfully parsed pages."""
        with self._state_lock:
            return dict(self._page_texts)

    @property
    def query_timings(self) -> dict[str, dict[str, Any]]:
        with self._timing_lock:
            return copy.deepcopy(self._query_timings)

    @property
    def prepared_chunk_count(self) -> int:
        with self._chunk_lock:
            return len(self._chunks_by_id)

    def close(self) -> None:
        self._batcher.close()

    def __enter__(self) -> "OnDemandPerQueryRunner":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def run_query(self, query: str, *, query_id: str | None = None) -> OnDemandQueryResult:
        """Run one independent query and return ranked chunks plus timings."""
        query_id = str(query_id if query_id is not None else query)
        query = str(query)
        query_started = time.perf_counter()
        with self._timing_lock:
            if self._stage_started_at is None:
                self._stage_started_at = query_started

        light_started = time.perf_counter()
        hits = self.index.search(query, top_k=self.top_k_pages)
        light_bm25_seconds = time.perf_counter() - light_started

        parsed_ids, cached_ids, batch_result = self._ensure_parsed(hits)
        hit_page_ids = {hit.page_id for hit in hits}
        with self._state_lock:
            page_ids_for_query = {
                page_id
                for page_id in hit_page_ids
                if page_id in self._records_by_page
            }
        chunk_stage_seconds, new_page_count, index_seconds = (
            self._ensure_chunks_and_index(page_ids_for_query)
        )
        ranked, retrieval_seconds = self._retrieve_chunks(query, hits)

        batch_id = batch_result.get("batch_id")
        batch_total_seconds = float(batch_result.get("batch_total_seconds", 0.0))
        batch_service_seconds = float(batch_result.get("batch_kdl_seconds", 0.0))
        overall_seconds = (
            light_bm25_seconds
            + batch_total_seconds
            + chunk_stage_seconds
            + retrieval_seconds
        )
        wall_clock_seconds = time.perf_counter() - query_started
        timing = {
            "pipeline": PIPELINE_VERSION,
            "light_preparation": "pdf_inspector page index (prepared once)",
            "light_retrieval": "BM25 page retrieval per query",
            "downstream": (
                "KDL + pdf-inspector parse -> fixed_overlap 512/128 -> "
                "text-embedding-3-small -> hybrid alpha=0.7"
            ),
            "query_id": query_id,
            "light_retrieval_seconds": round(light_bm25_seconds, 6),
            "parsing_seconds": round(batch_total_seconds, 6),
            "kdl_batch_id": batch_id,
            "kdl_batch_pages": int(batch_result.get("batch_pages", 0)),
            "kdl_batch_wait_seconds": round(
                float(batch_result.get("batch_wait_seconds", 0.0)), 6
            ),
            "kdl_batch_service_seconds": round(batch_service_seconds, 6),
            "chunk_embed_index_seconds": round(chunk_stage_seconds, 6),
            "index_seconds": round(index_seconds, 6),
            "retrieval_seconds": round(retrieval_seconds, 6),
            "overall_seconds": round(overall_seconds, 6),
            "wall_clock_seconds": round(wall_clock_seconds, 6),
            "hit_pages": len(hits),
            "cache_hit_pages": len(cached_ids),
            "parsed_pages_this_query": len(parsed_ids),
            "new_pages_chunked_this_query": new_page_count,
            "chunks_available": self.prepared_chunk_count,
            "context_chunks": len(ranked),
        }
        with self._timing_lock:
            self._query_timings[query_id] = timing
            self._phase_work["light_retrieval"] += light_bm25_seconds
            if batch_id and batch_id not in self._accounted_kdl_batches:
                self._phase_work["parsing"] += batch_service_seconds
                self._accounted_kdl_batches.add(batch_id)
            self._phase_work["chunk_embed_index"] += chunk_stage_seconds
            self._phase_work["retrieval"] += retrieval_seconds
            self._stage_ended_at = time.perf_counter()

        with self._state_lock:
            selected_pages = {
                path: sorted(indices)
                for path, indices in self._selected_pages.items()
            }
        return OnDemandQueryResult(
            query_id=query_id,
            query=query,
            hits=hits,
            selected_pages=selected_pages,
            ranked_chunks=ranked,
            timing=timing,
            parsed_page_ids=tuple(sorted(parsed_ids)),
            cached_page_ids=tuple(sorted(cached_ids)),
        )

    def run_queries(
        self,
        queries: Iterable[tuple[str, str]],
        *,
        max_workers: int | None = None,
    ) -> dict[str, OnDemandQueryResult]:
        """Run ``(query_id, query)`` pairs concurrently and preserve input order."""
        items = [(str(query_id), str(query)) for query_id, query in queries]
        if len({query_id for query_id, _query in items}) != len(items):
            raise ValueError("query ids must be unique")
        if not items:
            return {}
        workers = max(1, int(max_workers or self.query_workers))
        results: dict[str, OnDemandQueryResult] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.run_query, query, query_id=query_id): query_id
                for query_id, query in items
            }
            for future in as_completed(futures):
                query_id = futures[future]
                results[query_id] = future.result()
        return {query_id: results[query_id] for query_id, _query in items}

    def timing_summary(self, *, light_preparation_seconds: float = 0.0) -> dict[str, Any]:
        """Return per-query means and all-data work/wall-clock summaries."""
        timings = list(self.query_timings.values())
        query_count = max(1, len(timings))
        preparation = float(light_preparation_seconds)

        def mean(key: str, fallback: str | None = None) -> float:
            values = []
            for item in timings:
                value = item.get(key)
                if value is None and fallback:
                    value = item.get(fallback, 0.0)
                values.append(float(value or 0.0))
            return sum(values) / len(values) if values else 0.0

        with self._timing_lock:
            phase = dict(self._phase_work)
            started = self._stage_started_at
            ended = self._stage_ended_at
        phase_work = {
            "Light Retrieval": preparation + phase["light_retrieval"],
            "Parsing": phase["parsing"],
            "Chunk&Embed&Index": phase["chunk_embed_index"],
            "Retrieval": phase["retrieval"],
        }
        if started is not None and ended is not None:
            overall_wall = preparation + max(0.0, ended - started)
        else:
            overall_wall = preparation + sum(
                float(item.get("wall_clock_seconds", item.get("overall_seconds", 0.0)))
                for item in timings
            )
        latency = {
            "Light Retrieval": preparation / query_count + mean(
                "light_retrieval_seconds"
            ),
            "Parsing": mean("parsing_seconds"),
            "Chunk&Embed&Index": mean("chunk_embed_index_seconds"),
            "Retrieval": mean("retrieval_seconds"),
            "Overall": preparation / query_count + mean(
                "wall_clock_seconds", "overall_seconds"
            ),
        }
        total = {**phase_work, "Overall": overall_wall}
        return {
            "pipeline": PIPELINE_VERSION,
            "query_count": len(timings),
            "query_workers": self.query_workers,
            "microbatch_window_seconds": self.microbatch_window_seconds,
            "microbatch_max_pages": self.microbatch_max_pages,
            "total_runtime_seconds_all_data": {
                key: round(value, 3) for key, value in total.items()
            },
            "phase_work_seconds_all_data": {
                key: round(value, 3) for key, value in phase_work.items()
            },
            "online_latency_seconds_per_query": {
                key: round(value, 4) for key, value in latency.items()
            },
            "light_preparation_seconds_all_data": round(preparation, 3),
            "online_stage_wall_seconds": round(overall_wall, 3),
            "parsed_page_cache": len(self._records_by_page),
            "prepared_chunk_cache": len(self._chunks_by_id),
            "notes": (
                "Overall is wall-clock runtime including light preparation. "
                "Phase columns are aggregate work and may overlap under concurrent "
                "queries; each shared KDL micro-batch is counted once."
            ),
        }

    def _requested_page_ids(self, selected: dict[str, Sequence[int]]) -> set[str]:
        return {
            self._page_id_by_location[(str(Path(path).resolve()), int(page_index))]
            for path, indices in selected.items()
            for page_index in indices
            if (str(Path(path).resolve()), int(page_index)) in self._page_id_by_location
        }

    def _ensure_parsed(
        self, hits: Sequence[DiscoveryHit]
    ) -> tuple[set[str], set[str], dict[str, Any]]:
        needed: defaultdict[str, set[int]] = defaultdict(set)
        cached: set[str] = set()
        with self._state_lock:
            for hit in hits:
                self._selected_page_ids.add(hit.page_id)
                self._selected_pages.setdefault(hit.evidence.file_path, set()).add(
                    int(hit.evidence.page_index)
                )
                if not self.force_reparse and hit.page_id in self._records_by_page:
                    cached.add(hit.page_id)
                else:
                    needed[hit.evidence.file_path].add(int(hit.evidence.page_index))
        if not needed:
            return set(), cached, {
                "batch_id": None,
                "batch_pages": 0,
                "batch_wait_seconds": 0.0,
                "batch_kdl_seconds": 0.0,
                "batch_total_seconds": 0.0,
            }

        batch_result = self._batcher.submit(
            {path: sorted(indices) for path, indices in needed.items()}
        )
        requested = self._requested_page_ids(needed)
        parsed = set(batch_result.get("parsed_ids") or set()) & requested
        with self._state_lock:
            cached.update(
                page_id
                for page_id in requested - parsed
                if page_id in self._records_by_page
            )
        return parsed, cached, batch_result

    def _parse_kdl_batch(
        self, selected: dict[str, list[int]], batch_id: str
    ) -> dict[str, Any]:
        started = time.perf_counter()
        with self._state_lock:
            parseable = {
                path: [
                    index
                    for index in indices
                    if self.force_reparse
                    or self._page_id_by_location.get(
                        (str(Path(path).resolve()), int(index))
                    )
                    not in self._records_by_page
                ]
                for path, indices in selected.items()
            }
            parseable = {path: indices for path, indices in parseable.items() if indices}
        if not parseable:
            return {
                "parsed_ids": set(),
                "batch_id": batch_id,
                "batch_pages": len(self._requested_page_ids(selected)),
                "batch_wait_seconds": 0.0,
                "batch_kdl_seconds": 0.0,
                "batch_total_seconds": 0.0,
            }

        parser_work_dir = self.work_dir / "on-demand-kdl" / batch_id
        result = run_selected_pages(
            parseable,
            parser_config=self.parser_config,
            project_root=self.project_root,
            work_dir=parser_work_dir,
            one_page_inputs=True,
        )
        parsed_ids: set[str] = set()
        with self._state_lock:
            for record in result.enriched.enriched_data:
                record_data = _record_dict(record)
                page_id = self._record_page_id(record_data)
                if not page_id:
                    continue
                self._records_by_page[page_id] = record_data
                parsed_ids.add(page_id)
                text = _record_page_text(record_data)
                if text:
                    self._page_texts[page_id] = text
            self._save_parser_cache()
        elapsed = time.perf_counter() - started
        print(
            f"KDL micro-batch {batch_id}: requested="
            f"{len(self._requested_page_ids(selected))}; parsed={len(parsed_ids)}; "
            f"quarantined={len(result.ingestion.quarantined_documents)}; "
            f"time={elapsed:.3f}s",
            flush=True,
        )
        return {
            "parsed_ids": parsed_ids,
            "batch_id": batch_id,
            "batch_pages": len(self._requested_page_ids(selected)),
            "batch_wait_seconds": 0.0,
            "batch_kdl_seconds": elapsed,
            "batch_total_seconds": elapsed,
        }

    def _ensure_chunks_and_index(
        self, page_ids: set[str]
    ) -> tuple[float, int, float]:
        started = time.perf_counter()
        with self._chunk_lock:
            with self._state_lock:
                available_page_ids = set(self._records_by_page)
            new_page_ids = [
                page_id
                for page_id in sorted(page_ids)
                if page_id in available_page_ids
                and not any(
                    chunk.page_id == page_id for chunk in self._chunks_by_id.values()
                )
            ]
            if new_page_ids:
                with self._state_lock:
                    records = [
                        copy.deepcopy(self._records_by_page[page_id])
                        for page_id in new_page_ids
                    ]
                from src.chunking_embedding.stage import run as run_chunking_embedding

                output = run_chunking_embedding(records, self.chunking_config)
                new_chunks = self._prepared_chunks(output, records)
                for chunk in new_chunks:
                    self._chunks_by_id[chunk.record_id] = chunk
                self._save_chunk_cache()
                if not new_chunks:
                    print(
                        f"Warning: no usable text chunks produced for "
                        f"{len(new_page_ids)} parsed page(s).",
                        flush=True,
                    )
            index_seconds = 0.0
            if self._chunk_index is None or new_page_ids:
                if self._chunks_by_id:
                    index_seconds = self._rebuild_chunk_index()
        return time.perf_counter() - started, len(new_page_ids), index_seconds

    def _prepared_chunks(self, output: Any, records: list[dict[str, Any]]) -> list[PreparedChunk]:
        object_to_page = {
            str(record.get("source_object_id")): self._record_page_id(record)
            for record in records
            if self._record_page_id(record)
        }
        vectors = {
            str(item["record_id"]): np.asarray(item["embedding"], dtype=np.float32)
            for item in output.vector_records
            if isinstance(item, dict) and item.get("record_id") and item.get("embedding") is not None
        }
        chunks: list[PreparedChunk] = []
        for record in output.retrieval_records:
            if record.retrieval_type != "text_chunk":
                continue
            page_id = object_to_page.get(str(record.source_object_id), "")
            text = str((record.payload or {}).get("text") or "").strip()
            vector = vectors.get(str(record.record_id))
            if page_id and text and vector is not None:
                chunks.append(
                    PreparedChunk(
                        record_id=str(record.record_id),
                        page_id=page_id,
                        text=text,
                        vector=vector,
                    )
                )
        return chunks

    def _rebuild_chunk_index(self) -> float:
        started = time.perf_counter()
        chunks = list(self._chunks_by_id.values())
        payload = [
            {"chunk_id": chunk.record_id, "doc_id": chunk.page_id, "text": chunk.text}
            for chunk in chunks
        ]
        bm25 = BM25Index(analyzer_name="auto").build(payload)
        from src.chunking_embedding.embedders import create_embedder

        params = dict(self.chunking_config.get("embedder_params") or {})
        self._chunk_embedder = create_embedder(
            str(self.chunking_config.get("embedder") or "openrouter_te3s"),
            params,
        )
        self._chunk_index = LocalIndex(
            index_id="axiom.on_demand_per_query.baseline_legacy",
            records=[
                ChunkRecord(chunk.record_id, chunk.page_id, chunk.text)
                for chunk in chunks
            ],
            bm25=bm25,
            vectors=np.asarray([chunk.vector for chunk in chunks], dtype=np.float32),
            embedder=self._chunk_embedder,
            embeddings_model=str(params.get("model") or ""),
            metric="cosine",
        )
        return time.perf_counter() - started

    def _retrieve_chunks(
        self, query: str, hits: Sequence[DiscoveryHit]
    ) -> tuple[list[PreparedChunk], float]:
        with self._chunk_lock:
            local_index = self._chunk_index
            local_embedder = self._chunk_embedder
            local_chunks = dict(self._chunks_by_id)
        if local_index is None or local_embedder is None or not local_chunks:
            return [], 0.0

        started = time.perf_counter()
        query_vector = local_embedder.embed([query])[0]
        scope = {hit.page_id for hit in hits}
        allowed_positions = {
            position
            for position, record in enumerate(local_index.records)
            if record.doc_id in scope
        }
        sparse_hits = [
            (local_index.record_at(position).chunk_id, score)
            for position, score in local_index.bm25.search(
                query, self.depth, allowed_positions
            )
        ]
        vector = np.asarray(query_vector, dtype=np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores = local_index.vectors @ vector
        if allowed_positions:
            mask = np.zeros(len(scores), dtype=bool)
            mask[list(allowed_positions)] = True
            scores = np.where(mask, scores, -np.inf)
        else:
            scores = np.full(len(scores), -np.inf, dtype=np.float32)
        count = min(self.depth, int(np.isfinite(scores).sum()))
        dense_hits: list[tuple[str, float]] = []
        if count:
            positions = np.argpartition(-scores, count - 1)[:count]
            positions = positions[np.argsort(-scores[positions])]
            dense_hits = [
                (local_index.record_at(int(position)).chunk_id, float(scores[position]))
                for position in positions
            ]
        fused = alpha_fuse(dense_hits, sparse_hits, self.alpha, self.top_k_chunks)
        return (
            [local_chunks[chunk_id] for chunk_id, _score in fused if chunk_id in local_chunks],
            time.perf_counter() - started,
        )

    def _record_page_id(self, record: dict[str, Any]) -> str | None:
        metadata = record.get("metadata") or {}
        source_metadata = metadata.get("source_metadata") or metadata
        original_path = source_metadata.get("discovery_original_path")
        indices = source_metadata.get("discovery_page_indices")
        if not original_path or not isinstance(indices, list) or len(indices) != 1:
            return None
        return self._page_id_by_location.get(
            (str(Path(original_path).resolve()), int(indices[0]))
        )

    def _make_cache_key(self) -> str:
        payload = {
            "version": PIPELINE_VERSION,
            "pages": [page.page_id for page in self.index.pages],
            "parser_config": self.parser_config,
            "chunking_config": self.chunking_config,
            "top_k_pages": self.top_k_pages,
            "top_k_chunks": self.top_k_chunks,
            "depth": self.depth,
            "alpha": self.alpha,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:20]

    def _load_caches(self) -> None:
        parser_payload = _read_json(self.cache_dir / "parsed_pages.json")
        if parser_payload.get("cache_key") == self._cache_key:
            records = parser_payload.get("records") or {}
            if isinstance(records, dict):
                self._records_by_page = {
                    str(page_id): value
                    for page_id, value in records.items()
                    if isinstance(value, dict)
                }
                self._page_texts = {
                    page_id: text
                    for page_id, record in self._records_by_page.items()
                    if (text := _record_page_text(record))
                }

        chunk_payload = _read_json(self.cache_dir / "prepared_chunks.json")
        if chunk_payload.get("cache_key") == self._cache_key:
            for item in chunk_payload.get("chunks") or []:
                if not isinstance(item, dict) or not item.get("record_id"):
                    continue
                try:
                    self._chunks_by_id[str(item["record_id"])] = PreparedChunk(
                        record_id=str(item["record_id"]),
                        page_id=str(item["page_id"]),
                        text=str(item["text"]),
                        vector=np.asarray(item["vector"], dtype=np.float32),
                    )
                except (KeyError, TypeError, ValueError):
                    continue

    def _save_parser_cache(self) -> None:
        _atomic_write_json(
            self.cache_dir / "parsed_pages.json",
            {"cache_key": self._cache_key, "records": self._records_by_page},
        )

    def _save_chunk_cache(self) -> None:
        _atomic_write_json(
            self.cache_dir / "prepared_chunks.json",
            {
                "cache_key": self._cache_key,
                "pipeline": PIPELINE_VERSION,
                "chunks": [
                    {
                        "record_id": chunk.record_id,
                        "page_id": chunk.page_id,
                        "text": chunk.text,
                        "vector": chunk.vector.tolist(),
                    }
                    for chunk in self._chunks_by_id.values()
                ],
            },
        )


def _merge_selected(
    selections: Iterable[dict[str, Sequence[int]]]
) -> dict[str, list[int]]:
    merged: defaultdict[str, set[int]] = defaultdict(set)
    for selection in selections:
        for path, indices in selection.items():
            merged[str(path)].update(int(index) for index in indices)
    return {
        path: sorted(indices)
        for path, indices in sorted(merged.items())
        if indices
    }


def _record_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    return dict(record.__dict__)


def _record_page_text(record: dict[str, Any]) -> str:
    return "\n\n".join(
        str(item.get("text") or "").strip()
        for item in (record.get("rows") or [])
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ).strip()


def _validate_baseline_chunking(config: dict[str, Any]) -> None:
    chunker = str(config.get("chunker") or "")
    params = config.get("chunker_params") or {}
    model = str((config.get("embedder_params") or {}).get("model") or "")
    if chunker != "fixed_overlap" or int(params.get("n_words", 0)) != 512:
        raise ValueError(
            "on-demand-per-query baseline requires chunker=fixed_overlap "
            "with chunker_params.n_words=512"
        )
    if model != "openai/text-embedding-3-small":
        raise ValueError(
            "on-demand-per-query baseline requires "
            "embedder model openai/text-embedding-3-small"
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "OnDemandPerQueryRunner",
    "OnDemandQueryResult",
    "PIPELINE_VERSION",
    "PreparedChunk",
]
