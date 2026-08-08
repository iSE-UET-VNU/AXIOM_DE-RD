"""Turn production pipeline output into a benchmark corpus.

    python -m src.evaluation.build_corpus --from-pipeline data/output/<run_id>

``build_corpus`` normally parses the lake itself with ``extract.py`` -- text
layers only, no OCR. The pipeline parses the same files with lift_api or
chandra2 and writes richer documents to ``data/output/<run_id>/documents/``.
Without this adapter those two never meet, so the parser under test can never
be the one the benchmark measures.

The output is the same ``corpus.jsonl`` shape ``build_corpus`` writes, with one
addition: every record carries ``parser``. A text-arm corpus and a VLM-arm
corpus are otherwise identical on disk, and two arms that cannot be told apart
are two arms that will eventually be confused for each other.

Blocks are emitted in ``reading_order`` when the pipeline provides one. Falling
back to file order would reorder evidence without changing any count -- the
failure ``test_reading_order_seam.py`` exists to catch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator
import json

# Pipeline block types -> the ``kind`` vocabulary extract.py emits, so records
# from either source are interchangeable downstream.
_KIND = {
    "text": "paragraph",
    "paragraph": "paragraph",
    "title": "heading",
    "sectionheader": "heading",
    "header": "heading",
    "listitem": "list_item",
    "table": "table",
    "figure": "figure",
    "caption": "caption",
    "formula": "formula",
}


class PipelineRunError(RuntimeError):
    """The run directory is missing, empty, or not a pipeline output."""


def documents(run_dir: Path) -> Iterator[dict[str, Any]]:
    docs = sorted((run_dir / "documents").glob("*.json"))
    if not docs:
        raise PipelineRunError(
            f"{run_dir}/documents/ holds no documents. Point --from-pipeline at a "
            "data/output/<run_id> directory produced by scripts/run_pipeline.py."
        )
    for path in docs:
        yield json.loads(path.read_text(encoding="utf-8"))


def parser_name(run_dir: Path) -> str:
    """Which parser produced this run, read from the ingestion metadata.

    Unknown rather than a guess: a corpus mislabelled with the wrong parser is
    worse than one that admits it does not know, because the label is what the
    whole parsing comparison rests on.
    """
    meta = run_dir.parent.parent / "ingested" / run_dir.name / "metadata.json"
    if not meta.exists():
        return "unknown"
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    return str((payload.get("parser") or {}).get("provider") or "unknown")


def blocks_of(document: dict[str, Any]) -> list[dict[str, Any]]:
    content = document.get("content") or {}
    blocks = [b for b in (content.get("blocks") or []) if isinstance(b, dict)]
    order = content.get("reading_order") or []
    if order:
        by_id = {b.get("component_id"): b for b in blocks}
        ordered = [by_id[cid] for cid in order if cid in by_id]
        # Blocks absent from reading_order still belong in the corpus; dropping
        # them would lose text silently. They go after, in file order.
        seen = {id(b) for b in ordered}
        ordered.extend(b for b in blocks if id(b) not in seen)
        blocks = ordered

    out: list[dict[str, Any]] = []
    for block in blocks:
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "text": text,
                "kind": _KIND.get(str(block.get("type") or "").lower(), "paragraph"),
                "page": block.get("page"),
                "section": None,
            }
        )
    return out


def record_of(document: dict[str, Any], parser: str) -> dict[str, Any] | None:
    """One corpus record, or None when the parse produced no text.

    Returning None rather than an empty record keeps the coverage number
    honest: a document that parsed but yielded nothing is a failure, and it is
    exactly the case (scanned PDFs) the VLM arm is meant to fix.
    """
    blocks = blocks_of(document)
    if not blocks:
        return None
    meta = document.get("document") or {}
    name = meta.get("file_name") or meta.get("document_id") or "unknown"
    suffix = Path(str(name)).suffix.lower()
    return {
        "doc_id": str(name),
        "title": Path(str(name)).stem,
        "modality": "table" if any(b["kind"] == "table" for b in blocks) else "text",
        "suffix": suffix,
        "parser": parser,
        "blocks": blocks,
    }
