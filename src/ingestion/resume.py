"""Reload a previous run's parsed documents so a broken run can continue.

Parsing is the expensive, network-dependent stage: a hosted VLM behind a tunnel
drops, a Colab session expires, and everything already parsed is thrown away
because ``run_pipeline`` mints a fresh run id and starts from page one.

Resume reuses the run directory and skips inputs already parsed *successfully*.
Quarantined documents are deliberately retried -- a page range that failed
because the endpoint died is not a property of the document, and treating it as
final would bake a transient outage into the corpus.

Reloaded documents re-enter the pipeline as ordinary results, so cleaning,
enrichment and chunking see the whole corpus rather than only the newly parsed
tail. Skipping the parse must not mean skipping the document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import logging

from ..models import DataObject, ParsedData
from .parsing.parser import infer_initial_schema

logger = logging.getLogger(__name__)


def _parsed_from(payload: dict[str, Any]) -> ParsedData | None:
    raw = payload.get("parsed")
    if not isinstance(raw, dict) or not raw.get("object_id"):
        return None
    return ParsedData(
        object_id=str(raw["object_id"]),
        source_uri=str(raw.get("source_uri") or ""),
        source_format=str(raw.get("source_format") or "unknown"),
        rows=list(raw.get("rows") or []),
        text=raw.get("text"),
        metadata=dict(raw.get("metadata") or {}),
    )


def _data_object_from(payload: dict[str, Any]) -> DataObject | None:
    raw = payload.get("source")
    if not isinstance(raw, dict) or not raw.get("object_id"):
        return None
    return DataObject(
        object_id=str(raw["object_id"]),
        uri=str(raw.get("uri") or ""),
        content_type=str(raw.get("content_type") or "unknown"),
        metadata=dict(raw.get("metadata") or {}),
    )


def load_completed(ingested_dir: str | Path) -> tuple[set[str], Any]:
    """Return (object_ids already parsed, an IngestionOutput holding them).

    A document whose file is unreadable or whose ``parsed`` block is missing is
    *not* counted as complete: it will be parsed again. Re-parsing costs GPU
    time; trusting a truncated file costs a silently incomplete corpus, and
    only one of those is recoverable.
    """
    from .runner import IngestionOutput  # local: runner imports this module

    output = IngestionOutput()
    done: set[str] = set()
    directory = Path(ingested_dir) / "documents"
    if not directory.is_dir():
        return done, output

    unreadable = 0
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unreadable += 1
            continue
        if payload.get("status") != "succeeded":
            continue
        data_object = _data_object_from(payload)
        parsed = _parsed_from(payload)
        if data_object is None or parsed is None:
            unreadable += 1
            continue
        output.data_objects.append(data_object)
        output.parsed_data.append(parsed)
        output.initial_schemas.append(infer_initial_schema(parsed))
        done.add(data_object.object_id)

    if unreadable:
        logger.warning(
            "%d persisted document(s) could not be reloaded and will be parsed "
            "again; a truncated file must not be mistaken for a finished one.",
            unreadable,
        )
    logger.info("Resume: %d document(s) already parsed in %s", len(done), ingested_dir)
    return done, output
