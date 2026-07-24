"""Read pipeline artifacts for the Streamlit demonstration application.

The demo deliberately reads the persisted JSON contracts instead of rebuilding
pipeline state.  This keeps the UI useful after a process restart and lets it
display both current ``output-document-v4`` artifacts and older demo runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGES = ("ingested", "cleaned", "enriched", "embedded", "output")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class RunOverview:
    """Small, display-ready summary of one persisted pipeline run."""

    run_id: str
    created_at: str | None
    document_count: int
    status: str
    input_source: str | None
    available_stages: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DocumentOverview:
    """Document identity merged from stage metadata and output artifacts."""

    document_id: str
    file_name: str
    status: str
    content_type: str | None
    size_bytes: int | None
    source_uri: str | None
    title: str | None = None


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and return an empty mapping for a missing file."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def discover_runs(project_root: str | Path = PROJECT_ROOT) -> list[RunOverview]:
    """Discover runs across every persisted stage, newest first."""
    root = Path(project_root).resolve()
    run_ids: set[str] = set()
    for stage in STAGES:
        stage_root = root / "data" / stage
        if stage_root.is_dir():
            run_ids.update(path.name for path in stage_root.iterdir() if path.is_dir())

    runs: list[RunOverview] = []
    for run_id in run_ids:
        if not _SAFE_ID.fullmatch(run_id):
            continue
        available = tuple(
            stage
            for stage in STAGES
            if (root / "data" / stage / run_id / "metadata.json").is_file()
        )
        if not available:
            continue
        preferred_stage = "output" if "output" in available else available[-1]
        metadata = load_stage_metadata(run_id, preferred_stage, root)
        ingested = load_stage_metadata(run_id, "ingested", root)
        progress = ingested.get("progress", {}) if isinstance(ingested, dict) else {}
        errors = metadata.get("errors", [])
        status = str(progress.get("status") or "completed")
        if errors and status == "completed":
            status = "completed_with_errors"
        count = _int_value(
            metadata.get("document_count"),
            progress.get("expected_document_count"),
            len(metadata.get("documents", [])),
        )
        runs.append(
            RunOverview(
                run_id=run_id,
                created_at=_optional_text(metadata.get("created_at")),
                document_count=count,
                status=status,
                input_source=_optional_text(metadata.get("input_source")),
                available_stages=available,
                metadata=metadata,
            )
        )

    return sorted(runs, key=lambda item: (item.created_at or "", item.run_id), reverse=True)


def load_stage_metadata(
    run_id: str,
    stage: str,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    _validate_identifier(run_id, "run id")
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage: {stage}")
    return read_json(Path(project_root).resolve() / "data" / stage / run_id / "metadata.json")


def list_documents(
    run_id: str,
    project_root: str | Path = PROJECT_ROOT,
) -> list[DocumentOverview]:
    """List successful and quarantined documents from a run."""
    root = Path(project_root).resolve()
    output_metadata = load_stage_metadata(run_id, "output", root)
    ingested_metadata = load_stage_metadata(run_id, "ingested", root)
    rows: dict[str, dict[str, Any]] = {}

    for metadata in (output_metadata, ingested_metadata):
        documents = metadata.get("documents", [])
        if not isinstance(documents, list):
            continue
        for item in documents:
            if not isinstance(item, dict):
                continue
            document_id = _optional_text(item.get("document_id"))
            if document_id and _SAFE_ID.fullmatch(document_id):
                rows.setdefault(document_id, {}).update(
                    {key: value for key, value in item.items() if value is not None}
                )

    for stage in ("output", "ingested"):
        document_dir = root / "data" / stage / run_id / "documents"
        if document_dir.is_dir():
            for path in document_dir.glob("*.json"):
                if _SAFE_ID.fullmatch(path.stem):
                    rows.setdefault(path.stem, {"document_id": path.stem})

    result: list[DocumentOverview] = []
    for document_id, item in rows.items():
        payload = load_document(run_id, document_id, project_root=root)
        view = extract_document_view(payload)
        identity = view["identity"]
        file_name = str(
            item.get("file_name")
            or identity.get("file_name")
            or Path(str(item.get("source_uri") or document_id)).name
            or document_id
        )
        result.append(
            DocumentOverview(
                document_id=document_id,
                file_name=file_name,
                status=str(item.get("status") or view.get("status") or "succeeded"),
                content_type=_optional_text(item.get("content_type") or identity.get("content_type")),
                size_bytes=_optional_int(item.get("size_bytes") or identity.get("size_bytes")),
                source_uri=_optional_text(item.get("source_uri") or identity.get("source_uri")),
                title=_optional_text(identity.get("title")),
            )
        )
    return sorted(result, key=lambda item: item.file_name.casefold())


def load_document(
    run_id: str,
    document_id: str,
    *,
    stage: str | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Load a document, preferring consumer output over ingestion audit data."""
    _validate_identifier(run_id, "run id")
    _validate_identifier(document_id, "document id")
    root = Path(project_root).resolve()
    candidate_stages = (stage,) if stage else ("output", "ingested")
    for candidate_stage in candidate_stages:
        if candidate_stage not in STAGES:
            raise ValueError(f"Unsupported stage: {candidate_stage}")
        path = root / "data" / candidate_stage / run_id / "documents" / f"{document_id}.json"
        payload = read_json(path)
        if payload:
            return payload
    return {}


def persist_dataeng_response(
    response: dict[str, Any],
    project_root: str | Path = PROJECT_ROOT,
) -> str:
    """Persist only final normalized output so the demo explorer can render it."""
    metadata = response.get("metadata")
    documents = response.get("documents")
    if not isinstance(metadata, dict) or not isinstance(documents, list):
        raise ValueError("Data engineering response must contain metadata and documents.")
    run_id = str(metadata.get("run_id") or "").strip()
    _validate_identifier(run_id, "run id")

    root = Path(project_root).resolve()
    output_dir = root / "data" / "output" / run_id
    document_dir = output_dir / "documents"
    document_dir.mkdir(parents=True, exist_ok=True)
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Data engineering documents must be JSON objects.")
        identity = document.get("document")
        if not isinstance(identity, dict):
            raise ValueError("Output document is missing its document identity.")
        document_id = str(identity.get("document_id") or "").strip()
        _validate_identifier(document_id, "document id")
        _write_json(document_dir / f"{document_id}.json", document)
    _write_json(output_dir / "metadata.json", metadata)
    return run_id


def extract_document_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize v3, legacy v2, and ingestion artifacts for the UI."""
    if not payload:
        return _empty_document_view()

    contract = str(payload.get("contract_version") or "unknown")
    if isinstance(payload.get("document"), dict) and isinstance(payload.get("content"), dict):
        lineage = payload.get("lineage") if isinstance(payload.get("lineage"), dict) else {}
        return {
            "contract_version": contract,
            "identity": dict(payload["document"]),
            "content": dict(payload["content"]),
            "retrieval": _mapping(payload.get("retrieval")),
            "lineage": lineage,
            "status": str(lineage.get("status") or "succeeded"),
        }

    source = _mapping(payload.get("source"))
    source_metadata = _mapping(source.get("metadata"))
    parsed = _mapping(payload.get("parsed"))
    if not parsed:
        parsed = _mapping(_mapping(payload.get("ingestion")).get("parsed"))
    extraction = _first_extraction(parsed)
    identity = {
        "document_id": payload.get("document_id") or source.get("object_id"),
        "source_uri": source.get("uri") or parsed.get("source_uri"),
        "file_name": source_metadata.get("file_name"),
        "content_type": source.get("content_type"),
        "size_bytes": source_metadata.get("size_bytes"),
        "sha256": source_metadata.get("sha256"),
        "title": extraction.get("title"),
        "document_type": extraction.get("document_type"),
        "language": extraction.get("language"),
    }
    content = {
        "main_text": extraction.get("main_text") or parsed.get("text") or "",
        "tables": _list_value(extraction.get("tables")),
        "figures": _list_value(extraction.get("figures")),
        "formulas": _list_value(extraction.get("formulas")),
    }
    quarantine = _mapping(payload.get("quarantine"))
    status = str(payload.get("status") or quarantine.get("status") or "succeeded")
    return {
        "contract_version": contract,
        "identity": {key: value for key, value in identity.items() if value is not None},
        "content": content,
        "retrieval": _mapping(payload.get("retrieval")),
        "lineage": _mapping(payload.get("lineage")),
        "status": status,
    }


def find_source_file(
    run_id: str,
    document: DocumentOverview,
    project_root: str | Path = PROJECT_ROOT,
) -> Path | None:
    """Find a local source copy without allowing artifact paths to escape the repo."""
    root = Path(project_root).resolve()
    source_uri = document.source_uri
    if source_uri and not source_uri.startswith("s3://"):
        candidate = Path(source_uri)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if _is_within(candidate, root) and candidate.is_file():
            return candidate

    raw_run = root / "data" / "raw" / run_id
    if raw_run.is_dir():
        for path in raw_run.rglob("*"):
            if path.name != document.file_name or not path.is_file():
                continue
            candidate = path.resolve()
            if _is_within(candidate, root):
                return candidate
    return None


def build_run_zip(run_id: str, project_root: str | Path = PROJECT_ROOT) -> bytes:
    """Create a download containing JSON artifacts for all available stages."""
    _validate_identifier(run_id, "run id")
    root = Path(project_root).resolve()
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for stage in STAGES:
            stage_root = root / "data" / stage / run_id
            if not stage_root.is_dir():
                continue
            for path in sorted(stage_root.rglob("*.json")):
                archive.write(path, path.relative_to(root).as_posix())
    return buffer.getvalue()


def format_bytes(value: int | None) -> str:
    if value is None:
        return "—"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _first_extraction(parsed: dict[str, Any]) -> dict[str, Any]:
    rows = parsed.get("rows", [])
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("extraction"), dict):
                return row["extraction"]
    return {}


def _empty_document_view() -> dict[str, Any]:
    return {
        "contract_version": "unknown",
        "identity": {},
        "content": {"main_text": "", "tables": [], "figures": [], "formulas": []},
        "retrieval": {},
        "lineage": {},
        "status": "unknown",
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_value(*values: Any) -> int:
    for value in values:
        converted = _optional_int(value)
        if converted is not None:
            return converted
    return 0


def _validate_identifier(value: str, label: str) -> None:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"Invalid {label}: {value!r}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
