"""Run the KDL-backed second retrieval on a generic benchmark light run.

The light runner deliberately stops at page discovery.  This adapter consumes
its slim top-page JSONL, sends the union of those pages through the existing
KDL + PDF-inspector ingestion pipeline, and evaluates two inexpensive second
retrieval arms:

``pages``
    BM25 over the accurately parsed page text, returning top pages.
``chunks``
    The existing fixed-overlap chunking/embedding and hybrid retrieval, with
    the returned chunks projected back to unique pages.

No qrels are used for parsing, chunking, indexing, or ranking.  They are read
only by the final evaluation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
import argparse
import hashlib
import json
import math
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.pipeline import (  # noqa: E402
    _restore_original_page_coordinates,
    run_selected_pages,
)
from research.data_discovery.raw_hierarchical import load_documents  # noqa: E402
from research.data_discovery.run_vidore_e2e import (  # noqa: E402
    PreparedChunk,
    _make_embedder,
    _retrieve_chunks,
)
from src.retrieval.sparse import BM25Index  # noqa: E402
from src.utils.config import load_config, resolve_parser_config  # noqa: E402


@dataclass(frozen=True)
class Query:
    qid: str
    query: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a checkpoint atomically so an interrupted run keeps the old one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    _write_jsonl(temporary, rows)
    temporary.replace(path)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one small progress event without rewriting the run log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def _page_manifest(
    selected: dict[str, list[int]],
    page_ids: dict[tuple[str, int], str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for source_path, indices in sorted(selected.items()):
        for page_index in sorted(int(value) for value in indices):
            page_id = page_ids.get((source_path, page_index))
            if not page_id:
                raise ValueError(f"Selected page has no canonical page id: {source_path}#{page_index}")
            entries.append({
                "page_id": page_id,
                "source_path": source_path,
                "page_index": page_index,
            })
    return entries


def _record_as_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    return dict(vars(record))


def _record_page_id(record: Any, page_ids: dict[tuple[str, int], str]) -> str | None:
    metadata = dict(
        record.get("metadata", {}) if isinstance(record, dict) else getattr(record, "metadata", {}) or {}
    )
    source_metadata = dict(metadata.get("source_metadata") or {})
    path = source_metadata.get("discovery_original_path") or metadata.get("discovery_original_path")
    indices = source_metadata.get("discovery_page_indices") or metadata.get("discovery_page_indices")
    if not path or not isinstance(indices, list) or len(indices) != 1:
        return None
    return page_ids.get((str(Path(path).resolve()), int(indices[0])))


def _record_text(record: Any) -> str:
    rows = record.get("rows", []) if isinstance(record, dict) else getattr(record, "rows", []) or []
    text = "\n\n".join(
        str(row.get("text") or "").strip()
        for row in rows
        if isinstance(row, dict) and str(row.get("text") or "").strip()
    ).strip()
    if not text:
        text = str(record.get("text", "") if isinstance(record, dict) else getattr(record, "text", "") or "").strip()
    return text


def _records_by_page(
    records: Iterable[Any],
    page_ids: dict[tuple[str, int], str],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        page_id = _record_page_id(record, page_ids)
        if page_id:
            output[page_id] = _record_as_dict(record)
    return output


def _load_record_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    output: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        page_id = str(row.get("page_id") or "")
        record = row.get("record")
        if page_id and isinstance(record, dict):
            output[page_id] = record
    return output


def _write_record_checkpoint(path: Path, records_by_page: dict[str, dict[str, Any]]) -> None:
    _write_jsonl_atomic(
        path,
        ({"page_id": page_id, "record": records_by_page[page_id]}
         for page_id in sorted(records_by_page)),
    )


def _load_legacy_text_checkpoint(
    path: Path,
    manifest_entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Migrate the pre-resume parsed_pages.jsonl into a minimal checkpoint.

    Older second runs persisted page text but not the enriched record needed
    by the chunks arm.  Reusing that text is still safe for the pages arm and
    gives the chunks arm a deterministic text-only fallback; new runs persist
    the full enriched record after every completed page.
    """
    if not path.is_file():
        return {}
    expected = {str(entry["page_id"]): entry for entry in manifest_entries}
    output: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        page_id = str(row.get("page_id") or "")
        text = str(row.get("text") or "").strip()
        entry = expected.get(page_id)
        if not entry or not text:
            continue
        output[page_id] = {
            "source_object_id": f"resume-page-{_stable_hash(page_id)[:24]}",
            "rows": [{"text": text}],
            "annotations": {},
            "profile": {},
            "metadata": {
                "discovery_original_path": entry["source_path"],
                "discovery_page_indices": [int(entry["page_index"])],
                "resume_source": "legacy_parsed_pages.jsonl",
            },
        }
    return output


def _load_raw_kdl_records(
    artifacts_dir: Path,
    manifest_entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Recover completed pages directly from KDL's per-page raw artifacts."""
    from src import cleaning, enrichment
    from src.ingestion import runner as ingestion_runner
    from src.ingestion.parsing import infer_initial_schema
    from src.ingestion.parsing.kdl import _build_extraction, _source_blocks
    from src.models import ParsedData
    from src.models import make_id

    output: dict[str, dict[str, Any]] = {}
    for entry in manifest_entries:
        page_id = str(entry["page_id"])
        original = Path(str(entry["source_path"])).resolve()
        page_index = int(entry["page_index"])
        source_uri = f"{original}#page={page_index + 1}"
        object_id = make_id("data-object", source_uri)
        raw_path = artifacts_dir / object_id / "result.json"
        if not raw_path.is_file():
            continue
        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            source_blocks = _source_blocks(raw.get("pages") or [])
            markdown = str(raw.get("markdown") or "")
            extraction = _build_extraction(markdown, source_blocks)
            input_metadata = {
                "file_name": original.name,
                "discovery_original_path": str(original),
                "discovery_page_indices": [page_index],
                "discovery_page_numbers": [page_index + 1],
            }
            metadata: dict[str, Any] = {
                "parser": "kdl_pdf_inspector",
                "method": "vllm",
                "page_count": len(raw.get("pages") or []),
                "raw_output_path": str(raw_path.with_suffix(".md")),
                "raw_metadata_path": str(raw_path),
                **input_metadata,
            }
            for key in ("usage", "_routing", "_global_scheduler"):
                value = raw.get(key)
                if key == "usage" and isinstance(value, dict):
                    metadata["kdl_usage"] = value
                elif key != "usage" and isinstance(value, dict):
                    metadata["kdl_global_scheduler" if key == "_global_scheduler" else key] = value
            parsed = ParsedData(
                object_id=object_id,
                source_uri=source_uri,
                source_format="pdf",
                rows=[{
                    "extraction": extraction,
                    "text": extraction["main_text"],
                    "source_blocks": source_blocks,
                    "reading_order": [block["component_id"] for block in source_blocks],
                }],
                text=markdown,
                metadata=metadata,
            )
            _restore_original_page_coordinates(parsed)
            schema = infer_initial_schema(parsed)
            cleaned = cleaning.run([parsed], [schema])
            enriched = enrichment.run(cleaned.cleaned_data, cleaned.cleaned_schemas)
            if enriched.enriched_data:
                output[page_id] = _record_as_dict(enriched.enriched_data[0])
        except Exception:
            # A partial/corrupt raw result must be reparsed rather than being
            # allowed to poison the resume checkpoint.
            continue
    return output


def _parse_with_resume(
    args: argparse.Namespace,
    selected: dict[str, list[int]],
    page_ids: dict[tuple[str, int], str],
    parser_config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Parse all pending pages in one global provider call with page checkpoints.

    The parser receives the complete pending page pool in one call.  The
    callback only persists completed pages; it does not partition the KDL
    queue or alter the configured request/render/model concurrency.  If the
    provider stops, a later ``--resume`` run sends only pages that were not
    checkpointed.
    """
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = _page_manifest(selected, page_ids)
    parser_signature = {
        "parser_config": parser_config,
        "pages": manifest_entries,
    }
    manifest = {
        "contract_version": "benchmark-kdl-second-resume-v1",
        "light_run": str(args.light_run.resolve()),
        "page_count": len(manifest_entries),
        "input_hash": _stable_hash(parser_signature),
        "pages": manifest_entries,
    }
    manifest_path = checkpoint_dir / "page_manifest.json"
    checkpoint_path = checkpoint_dir / "enriched_pages.jsonl"
    status_path = checkpoint_dir / "status.json"
    events_path = args.output_dir / "logs" / "events.jsonl"

    if args.resume and manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("input_hash") != manifest["input_hash"]:
            # The original manifest included absolute source paths.  Those
            # paths legitimately change when a Drive run is copied to local
            # storage, while canonical page IDs and zero-based page indices do
            # not.  Permit that relocation only when the selected page set is
            # identical; parser/config changes still produce a different page
            # set or must use a new output directory.
            old_pages = {
                (str(item.get("page_id") or ""), int(item.get("page_index", -1)))
                for item in existing_manifest.get("pages") or []
            }
            new_pages = {
                (str(item.get("page_id") or ""), int(item.get("page_index", -1)))
                for item in manifest["pages"]
            }
            if old_pages != new_pages:
                raise RuntimeError(
                    "Resume checkpoint does not match the current light run/parser page set. "
                    "Use a new --output-dir or omit --resume to start a new run."
                )
    elif args.resume and checkpoint_path.is_file():
        raise RuntimeError(
            f"Cannot safely resume {checkpoint_path}: page_manifest.json is missing. "
            "Use a new output directory or rerun without --resume."
        )
    elif args.resume:
        # This is the migration path for an older second run which only has
        # parsed_pages.jsonl.  Keep that file intact so it can seed the new
        # structured checkpoint below.
        _write_json(manifest_path, manifest)
        _write_record_checkpoint(checkpoint_path, {})
    else:
        _write_json(manifest_path, manifest)
        # A non-resume invocation explicitly starts a new parse checkpoint.
        # This prevents an old partial checkpoint/text file from being
        # mistaken for progress when the new run is later resumed.
        _write_record_checkpoint(checkpoint_path, {})
        _write_jsonl_atomic(args.output_dir / "parsed_pages.jsonl", [])

    records_by_page = _load_record_checkpoint(checkpoint_path) if args.resume else {}
    raw_kdl_reused = 0
    if args.resume:
        raw_records = _load_raw_kdl_records(
            args.work_dir / "parser-assets",
            manifest_entries,
        )
        for page_id, record in raw_records.items():
            if page_id not in records_by_page:
                records_by_page[page_id] = record
                raw_kdl_reused += 1
        if raw_kdl_reused:
            _write_record_checkpoint(checkpoint_path, records_by_page)

    legacy_reused = 0
    if args.resume:
        legacy_records = _load_legacy_text_checkpoint(
            args.output_dir / "parsed_pages.jsonl",
            manifest_entries,
        )
        if legacy_records:
            for page_id, record in legacy_records.items():
                if page_id not in records_by_page:
                    records_by_page[page_id] = record
                    legacy_reused += 1
        if legacy_reused:
            _write_record_checkpoint(checkpoint_path, records_by_page)
    expected_ids = {entry["page_id"] for entry in manifest_entries}
    records_by_page = {page_id: record for page_id, record in records_by_page.items() if page_id in expected_ids}
    pending = [entry for entry in manifest_entries if entry["page_id"] not in records_by_page]
    parse_started = time.perf_counter()
    stats: dict[str, Any] = {
        "resume_requested": bool(args.resume),
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "selected_pages": len(manifest_entries),
        "cached_pages": len(records_by_page),
        "raw_kdl_artifacts_reused": raw_kdl_reused,
        "legacy_parsed_pages_reused": legacy_reused,
        "pending_pages": len(pending),
        "global_parse_call": False,
        "checkpoint_updates": 0,
    }
    _append_jsonl(events_path, {
        "timestamp": time.time(),
        "stage": "kdl_parse",
        "status": "started",
        "resume": bool(args.resume),
        "cached_pages": len(records_by_page),
        "pending_pages": len(pending),
    })
    _write_json(status_path, {**stats, "status": "running"})

    try:
        if pending:
            pending_selected: dict[str, list[int]] = {}
            pending_page_ids: dict[tuple[str, int], str] = {}
            for entry in pending:
                source_path = str(entry["source_path"])
                page_index = int(entry["page_index"])
                pending_selected.setdefault(source_path, []).append(page_index)
                pending_page_ids[(source_path, page_index)] = str(entry["page_id"])

            from src import cleaning, enrichment

            def checkpoint_completed(partial: Any) -> None:
                # ``run_selected_pages`` has already restored discovery
                # metadata before invoking this callback.  We run the same
                # downstream stages used by the normal path so the durable
                # record is immediately usable by both retrieval arms.
                cleaned = cleaning.run(partial.parsed_data, partial.initial_schemas)
                enriched = enrichment.run(cleaned.cleaned_data, cleaned.cleaned_schemas)
                parsed = _records_by_page(enriched.enriched_data, pending_page_ids)
                if not parsed:
                    return
                records_by_page.update(parsed)
                _write_record_checkpoint(checkpoint_path, records_by_page)
                stats["cached_pages"] = len(records_by_page)
                stats["pending_pages"] = len(manifest_entries) - len(records_by_page)
                stats["checkpoint_updates"] += len(parsed)
                _write_json(status_path, {**stats, "status": "running"})
                _append_jsonl(events_path, {
                    "timestamp": time.time(),
                    "stage": "kdl_parse_page",
                    "status": "complete",
                    "page_ids": sorted(parsed),
                    "checkpoint_pages": len(records_by_page),
                })

            stats["global_parse_call"] = True
            _append_jsonl(events_path, {
                "timestamp": time.time(),
                "stage": "kdl_parse_global",
                "status": "started",
                "pages": len(pending),
                "kdl_request_workers": args.kdl_request_workers,
                "kdl_request_batch_size": args.kdl_request_batch_size,
                "kdl_max_model_sequences": args.kdl_max_model_sequences,
            })
            result = run_selected_pages(
                pending_selected,
                parser_config=parser_config,
                chunking_config=None,
                project_root=ROOT,
                work_dir=args.work_dir,
                one_page_inputs=True,
                on_document_complete=checkpoint_completed,
            )
            # The callback is the normal durable path.  This final merge also
            # covers parsers that return a successful result only at the end.
            final_records = _records_by_page(result.enriched.enriched_data, pending_page_ids)
            if final_records:
                records_by_page.update(final_records)
                _write_record_checkpoint(checkpoint_path, records_by_page)
                stats["cached_pages"] = len(records_by_page)
                stats["pending_pages"] = len(manifest_entries) - len(records_by_page)
            _append_jsonl(events_path, {
                "timestamp": time.time(),
                "stage": "kdl_parse_global",
                "status": "complete",
                "pages": len(pending),
                "checkpoint_pages": len(records_by_page),
            })
            missing = sorted(expected_ids - set(records_by_page))
            if missing:
                raise RuntimeError(
                    f"Global KDL parse returned no usable records for {len(missing)} pages: "
                    f"{missing[:3]}"
                )
    except Exception as exc:
        stats["status"] = "error"
        stats["error"] = repr(exc)
        stats["total_seconds"] = round(time.perf_counter() - parse_started, 3)
        _write_json(status_path, stats)
        _append_jsonl(events_path, {
            "timestamp": time.time(),
            "stage": "kdl_parse",
            "status": "error",
            "error": repr(exc),
            "checkpoint_pages": len(records_by_page),
        })
        raise

    stats["status"] = "complete"
    stats["pending_pages"] = len(manifest_entries) - len(records_by_page)
    stats["total_seconds"] = round(time.perf_counter() - parse_started, 3)
    _write_json(status_path, stats)
    _append_jsonl(events_path, {
        "timestamp": time.time(),
        "stage": "kdl_parse",
        "status": "complete",
        "seconds": stats["total_seconds"],
        "checkpoint_pages": len(records_by_page),
        "checkpoint_updates": stats["checkpoint_updates"],
    })
    return records_by_page, stats


def _load_light(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = _read_jsonl(path)
    by_qid = {str(row["query_id"]): row for row in rows}
    pages_by_qid: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        chunks = row.get("chunks") or []
        if not isinstance(chunks, list):
            raise ValueError(f"Light row {row.get('query_id')} has non-list chunks")
        pages_by_qid[str(row["query_id"])] = [dict(chunk) for chunk in chunks]
    if not by_qid:
        raise ValueError(f"Empty light run: {path}")
    return by_qid, pages_by_qid


def _load_queries(path: Path, selected: set[str]) -> list[Query]:
    rows = _read_jsonl(path)
    queries = [Query(str(row["query_id"]), str(row.get("query") or "")) for row in rows]
    queries = [query for query in queries if query.qid in selected]
    missing = selected - {query.qid for query in queries}
    if missing:
        raise ValueError(f"Light run contains qids missing from queries.jsonl: {sorted(missing)[:5]}")
    return queries


def _selected_pages(
    dataset_root: Path,
    light_pages: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[int]], dict[tuple[str, int], str]]:
    documents = load_documents(dataset_root)
    by_doc = {document.doc_id: document for document in documents}
    selected: dict[str, set[int]] = {}
    page_ids: dict[tuple[str, int], str] = {}
    for chunks in light_pages.values():
        for chunk in chunks:
            doc_id = str(chunk["doc_id"])
            document = by_doc.get(doc_id)
            if document is None:
                raise ValueError(f"Light run references unknown document: {doc_id}")
            page_index = int(chunk["page_index"])
            if document.page_count_hint is not None and page_index >= document.page_count_hint:
                raise ValueError(f"Page outside manifest count: {doc_id}#{page_index}")
            source_path = str(document.path.resolve())
            selected.setdefault(source_path, set()).add(page_index)
            page_ids[(source_path, page_index)] = str(chunk["page_id"])
    normalized = {path: sorted(indices) for path, indices in sorted(selected.items())}
    return normalized, page_ids


def _record_page_texts(records: Iterable[Any], page_ids: dict[tuple[str, int], str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for record in records:
        metadata = dict(getattr(record, "metadata", {}) or {})
        source_metadata = dict(metadata.get("source_metadata") or {})
        path = source_metadata.get("discovery_original_path") or metadata.get("discovery_original_path")
        indices = source_metadata.get("discovery_page_indices") or metadata.get("discovery_page_indices")
        if not path or not isinstance(indices, list) or len(indices) != 1:
            continue
        page_id = page_ids.get((str(Path(path).resolve()), int(indices[0])))
        if not page_id:
            continue
        rows = getattr(record, "rows", []) or []
        text = "\n\n".join(
            str(row.get("text") or "").strip()
            for row in rows
            if isinstance(row, dict) and str(row.get("text") or "").strip()
        ).strip()
        if not text:
            text = str(getattr(record, "text", "") or "").strip()
        if text:
            texts[page_id] = text
    return texts


def _page_bm25(
    queries: list[Query],
    candidate_pages: dict[str, list[dict[str, Any]]],
    page_texts: dict[str, str],
    top_k: int,
) -> dict[str, list[tuple[str, float]]]:
    rows = [
        {"chunk_id": page_id, "doc_id": page_id, "text": text}
        for page_id, text in sorted(page_texts.items())
    ]
    index = BM25Index(analyzer_name="auto").build(rows)
    position_by_id = {page_id: position for position, page_id in enumerate(index.chunk_ids)}
    result: dict[str, list[tuple[str, float]]] = {}
    for query in queries:
        allowed = {
            position_by_id[chunk["page_id"]]
            for chunk in candidate_pages[query.qid]
            if chunk.get("page_id") in position_by_id
        }
        hits = index.search(query.query, top_k=top_k, allowed=allowed)
        result[query.qid] = [(index.chunk_ids[position], float(score)) for position, score in hits]
    return result


def _build_chunks(
    pipeline: Any,
    page_texts: dict[str, str],
    page_ids: dict[tuple[str, int], str],
    chunking_config: dict[str, Any],
    page_for_record: dict[str, str] | None = None,
) -> list[PreparedChunk]:
    from src.chunking_embedding.stage import run as run_chunking_embedding

    output = run_chunking_embedding(
        [record.__dict__ for record in pipeline.enriched.enriched_data],
        chunking_config,
    )
    vectors = {
        str(item["record_id"]): item["embedding"]
        for item in output.vector_records
    }
    object_to_page: dict[str, str] = {}
    for record in pipeline.enriched.enriched_data:
        metadata = dict(getattr(record, "metadata", {}) or {})
        source_metadata = dict(metadata.get("source_metadata") or {})
        path = source_metadata.get("discovery_original_path") or metadata.get("discovery_original_path")
        indices = source_metadata.get("discovery_page_indices") or metadata.get("discovery_page_indices")
        if path and isinstance(indices, list) and len(indices) == 1:
            page_id = page_ids.get((str(Path(path).resolve()), int(indices[0])))
            if page_id and page_id in page_texts:
                object_to_page[str(record.source_object_id)] = page_id
    chunks: list[PreparedChunk] = []
    for record in output.retrieval_records:
        if record.retrieval_type != "text_chunk":
            continue
        page_id = (page_for_record or {}).get(str(record.source_object_id)) or object_to_page.get(str(record.source_object_id), "")
        text = str((record.payload or {}).get("text") or "").strip()
        vector = vectors.get(str(record.record_id))
        if page_id and text and vector is not None:
            chunks.append(PreparedChunk(str(record.record_id), page_id, text, vector))
    return chunks


def _normalise_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    for row in _read_jsonl(path):
        qrels.setdefault(str(row["query_id"]), {})[str(row["page_id"])] = float(row.get("relevance", 1))
    return qrels


def _ndcg(relevances: list[float], ideal: list[float], k: int) -> float:
    def dcg(values: list[float]) -> float:
        return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values[:k]))
    denominator = dcg(sorted(ideal, reverse=True))
    return dcg(relevances) / denominator if denominator else 0.0


def _evaluate(
    queries: list[Query],
    ranked: dict[str, list[str]],
    qrels: dict[str, dict[str, float]],
    k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    hits = 0
    recall_numerator = 0
    recall_denominator = 0
    ndcgs: list[float] = []
    for query in queries:
        gold = qrels.get(query.qid, {})
        ids = ranked.get(query.qid, [])[:k]
        found = sum(1 for page_id in set(ids) if page_id in gold)
        hit = bool(found)
        hits += int(hit)
        recall_numerator += found
        recall_denominator += len(gold)
        ndcgs.append(_ndcg([gold.get(page_id, 0.0) for page_id in ids], list(gold.values()), k))
        rows.append({
            "query_id": query.qid,
            "gold_pages": len(gold),
            "retrieved_pages": ids,
            "gold_pages_retrieved": found,
            "page_hit": hit,
            "page_recall": found / len(gold) if gold else None,
        })
    count = len(queries) or 1
    summary = {
        "queries": len(queries),
        "page_recall_at_k": 100 * recall_numerator / recall_denominator if recall_denominator else None,
        "page_hit_at_k": 100 * hits / count,
        "ndcg_at_k": 100 * sum(ndcgs) / count,
        "gold_pages_retrieved": recall_numerator,
        "gold_pages": recall_denominator,
        "k": k,
    }
    return summary, rows


def _parser_config(args: argparse.Namespace) -> dict[str, Any]:
    loaded = load_config(args.parser_config)
    config = resolve_parser_config(ROOT, loaded.get("parsing") or {}, args.work_dir / "parser-assets")
    kdl = dict(config.get("kdl") or {})
    if args.kdl_endpoint_url:
        kdl["endpoint_url"] = args.kdl_endpoint_url.rstrip("/")
    if args.kdl_model:
        kdl["model"] = args.kdl_model
    kdl.update({
        "max_workers": args.kdl_max_workers,
        "bbox_max_workers": args.kdl_bbox_max_workers,
        "render_processes": args.kdl_render_processes,
        "request_workers": args.kdl_request_workers,
        "request_batch_size": args.kdl_request_batch_size,
        "max_model_sequences": args.kdl_max_model_sequences,
    })
    config["kdl"] = kdl
    return config


def main(argv: list[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)
    (args.output_dir / "reports").mkdir(exist_ok=True)

    light_by_qid, candidate_pages = _load_light(args.light_run)
    queries = _load_queries(args.dataset_root / "queries.jsonl", set(light_by_qid))
    selected, page_ids = _selected_pages(args.dataset_root, candidate_pages)
    _write_json(args.output_dir / "selected_pages.json", selected)

    parser_config = _parser_config(args)
    enriched_by_page, parse_stats = _parse_with_resume(
        args,
        selected,
        page_ids,
        parser_config,
    )
    parser_seconds = float(parse_stats.get("total_seconds") or 0.0)
    enriched_records = [
        SimpleNamespace(**enriched_by_page[page_id])
        for page_id in sorted(enriched_by_page)
    ]
    pipeline = SimpleNamespace(
        enriched=SimpleNamespace(enriched_data=enriched_records),
    )
    page_texts = {
        page_id: text
        for page_id, record in enriched_by_page.items()
        if (text := _record_text(record))
    }
    if not page_texts:
        raise RuntimeError("KDL returned no usable page text")
    _write_jsonl_atomic(
        args.output_dir / "parsed_pages.jsonl",
        ({"page_id": page_id, "text": text} for page_id, text in sorted(page_texts.items())),
    )

    qrels_path = args.qrels_page or (args.light_run.parent.parent / "qrels_page_level.jsonl")
    if not qrels_path.is_file():
        raise FileNotFoundError(f"Page qrels not found: {qrels_path}")
    qrels = _normalise_qrels(qrels_path)
    report: dict[str, Any] = {
        "contract_version": "benchmark-kdl-second-v1",
        "light_run": str(args.light_run),
        "dataset_root": str(args.dataset_root),
        "queries": len(queries),
        "selected_unique_pages": sum(len(value) for value in selected.values()),
        "parsed_pages_with_text": len(page_texts),
        "resume": parse_stats,
        "kdl": {
            "endpoint_url": args.kdl_endpoint_url or os.environ.get("VLLM_API_BASE"),
            "max_workers": args.kdl_max_workers,
            "bbox_max_workers": args.kdl_bbox_max_workers,
            "render_processes": args.kdl_render_processes,
            "request_workers": args.kdl_request_workers,
            "request_batch_size": args.kdl_request_batch_size,
            "max_model_sequences": args.kdl_max_model_sequences,
        },
        "timing_seconds": {"kdl_parse_clean_enrich": round(parser_seconds, 3)},
        "arms": {},
    }

    page_ranked_scores = _page_bm25(queries, candidate_pages, page_texts, args.top_k_pages)
    page_ranked = {qid: [page_id for page_id, _score in scores] for qid, scores in page_ranked_scores.items()}
    _write_jsonl(
        args.output_dir / "runs" / f"legacy_pages_top{args.top_k_pages}.jsonl",
        ({"query_id": query.qid, "query": query.query, "pages": page_ranked[query.qid]} for query in queries),
    )
    page_summary, page_rows = _evaluate(queries, page_ranked, qrels, args.top_k_pages)
    _write_jsonl(args.output_dir / "reports" / "per_query_pages.jsonl", page_rows)
    report["arms"]["pages"] = page_summary

    if "chunks" in {arm.strip() for arm in args.arms.split(",") if arm.strip()}:
        chunk_started = time.perf_counter()
        chunk_config_loaded = load_config(args.chunking_config)
        chunk_config = dict(chunk_config_loaded.get("chunking_embedding") or {})
        if args.embedder:
            chunk_config["embedder"] = args.embedder
        page_for_record = {
            str(record.get("source_object_id")): page_id
            for page_id, record in enriched_by_page.items()
            if record.get("source_object_id")
        }
        chunks = _build_chunks(pipeline, page_texts, page_ids, chunk_config, page_for_record)
        chunking_seconds = time.perf_counter() - chunk_started
        if chunks:
            candidates = {
                qid: [SimpleNamespace(page_id=chunk["page_id"]) for chunk in candidate_pages[qid]]
                for qid in candidate_pages
            }
            query_hits = {query.qid: candidates[query.qid] for query in queries}
            chunk_ranked = _retrieve_chunks(
                chunks,
                query_hits,
                [SimpleNamespace(qid=query.qid, query=query.query) for query in queries],
                chunk_config,
                top_k=args.top_k_chunks,
                depth=args.retrieval_depth,
                alpha=args.alpha,
                batch_size=args.retrieval_batch_size,
            )
            chunk_pages: dict[str, list[str]] = {}
            chunk_rows: list[dict[str, Any]] = []
            for query in queries:
                selected_page_ids: list[str] = []
                chunk_units = []
                for rank, chunk in enumerate(chunk_ranked.get(query.qid, []), start=1):
                    chunk_units.append({"chunk_id": chunk.record_id, "page_id": chunk.page_id, "rank": rank})
                    # Keep the retrieval run slim (IDs only), but persist the
                    # exact ranked text separately for answer evaluation.  A
                    # chunk id includes the ephemeral source-object id used
                    # during ingestion, so reconstructing its text later from
                    # a copied KDL cache is not always possible.
                    if chunk.page_id not in selected_page_ids:
                        selected_page_ids.append(chunk.page_id)
                chunk_pages[query.qid] = selected_page_ids[: args.top_k_pages]
                chunk_rows.append({"query_id": query.qid, "chunks": chunk_units, "pages": chunk_pages[query.qid]})
            _write_jsonl(args.output_dir / "runs" / f"legacy_chunks_top{args.top_k_chunks}.jsonl", chunk_rows)
            _write_jsonl(
                args.output_dir / "runs" / f"legacy_chunks_top{args.top_k_chunks}_context.jsonl",
                (
                    {
                        "query_id": query.qid,
                        "chunks": [
                            {
                                "chunk_id": chunk.record_id,
                                "page_id": chunk.page_id,
                                "rank": rank,
                                "text": chunk.text,
                            }
                            for rank, chunk in enumerate(chunk_ranked.get(query.qid, []), start=1)
                        ],
                    }
                    for query in queries
                ),
            )
            chunk_summary, query_rows = _evaluate(queries, chunk_pages, qrels, args.top_k_pages)
            _write_jsonl(args.output_dir / "reports" / "per_query_chunks.jsonl", query_rows)
            chunk_summary["chunks_returned"] = args.top_k_chunks
            chunk_summary["chunk_count"] = len(chunks)
            chunk_summary["chunking_embedding_seconds"] = round(chunking_seconds, 3)
            report["arms"]["chunks"] = chunk_summary
        else:
            report["arms"]["chunks"] = {"error": "no_chunks", "chunking_embedding_seconds": round(chunking_seconds, 3)}

    report["timing_seconds"]["total"] = round(time.perf_counter() - started, 3)
    _write_json(args.output_dir / "config.json", {
        "dataset_root": str(args.dataset_root),
        "light_run": str(args.light_run),
        "top_k_pages": args.top_k_pages,
        "top_k_chunks": args.top_k_chunks,
        "retrieval_depth": args.retrieval_depth,
        "alpha": args.alpha,
        "arms": args.arms,
        "resume": args.resume,
        "kdl": report["kdl"],
    })
    _write_json(args.output_dir / "timing.json", report["timing_seconds"])
    _write_json(args.output_dir / "reports" / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--light-run", type=Path, required=True)
    parser.add_argument("--qrels-page", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--parser-config", type=Path, default=ROOT / "configs/pipeline.data-discovery.yaml")
    parser.add_argument("--chunking-config", type=Path, default=ROOT / "configs/pipeline.yaml")
    parser.add_argument("--kdl-endpoint-url", default=os.environ.get("VLLM_API_BASE", ""))
    parser.add_argument("--kdl-model", default=os.environ.get("VLLM_MODEL_NAME", "kdl-frontier-parser-nano"))
    parser.add_argument("--kdl-max-workers", type=int, default=512)
    parser.add_argument("--kdl-bbox-max-workers", type=int, default=256)
    parser.add_argument("--kdl-render-processes", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--kdl-request-workers", type=int, default=512)
    parser.add_argument("--kdl-request-batch-size", type=int, default=1)
    parser.add_argument("--kdl-max-model-sequences", type=int, default=1024)
    parser.add_argument("--top-k-pages", type=int, default=10)
    parser.add_argument("--top-k-chunks", type=int, default=10)
    parser.add_argument("--retrieval-depth", type=int, default=100)
    parser.add_argument("--retrieval-batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--arms", default="pages,chunks")
    parser.add_argument("--embedder", default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from output-dir/checkpoints/enriched_pages.jsonl and parse "
            "only pages missing from the latest durable checkpoint."
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
