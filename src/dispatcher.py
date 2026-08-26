"""Route every AXIOM input source before document or spreadsheet processing."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from .models import PipelineState, make_id
from .local_reader import read_local_inputs
from .pipeline import run_pipeline
from .s3_reader import (
    S3InputConfig,
    download_all_presigned_s3_inventory,
    parse_s3_uri,
    read_s3_input,
)
from .table_agent import (
    TableAgentClient,
    TableAgentClientConfig,
    normalize_table_agent_response,
)
from .utils.config import load_config
from .utils.env import load_dotenv_file
from .utils.paths import portable_path
from .api.output import build_dataeng_output, public_dataeng_response
from .api.schemas import DataEngRequest, PresignedFileRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = "configs/pipeline.yaml"


@dataclass(frozen=True)
class DataEngDispatchResult:
    """Combined API-shaped result for any supported AXIOM input source."""

    response: dict[str, Any]
    pipeline_state: PipelineState | None
    table_document_count: int


LocalDataEngDispatchResult = DataEngDispatchResult


def dispatch_dataeng_request(
    request: DataEngRequest,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Return the existing DataEng response after API-level input routing."""
    return public_dataeng_response(
        dispatch_dataeng_inputs(
            config_path=config_path,
            presigned_inventory=request.to_inventory(),
        ).response
    )


def dispatch_dataeng_inputs(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    presigned_inventory: dict[str, Any] | None = None,
    s3_uri: str | None = None,
    s3_info_file: str | Path | None = None,
    s3_object_key: str | None = None,
    s3_all_objects: bool = False,
    local_raw: str | Path | None = None,
) -> DataEngDispatchResult:
    """Route every supported public input mode through one dispatch boundary."""
    load_dotenv_file(PROJECT_ROOT)
    config_file = _config_path(config_path)
    config = load_config(config_file)
    mode = _resolve_dispatch_mode(
        config,
        presigned_inventory=presigned_inventory,
        s3_uri=s3_uri,
        s3_info_file=s3_info_file,
        local_raw=local_raw,
    )

    if mode == "local_raw":
        local_value = local_raw
        if local_value is None:
            local_config = config.get("local_input")
            if not isinstance(local_config, dict):
                local_config = {}
            local_value = str(local_config.get("path") or "data/raw")
        paths, input_root = _discover_local_dispatch_files(
            local_value,
            config=config,
        )
        return dispatch_local_dataeng_files(
            paths,
            config_path=config_file,
            pipeline_input_root=input_root,
        )

    if mode == "presigned_info":
        if presigned_inventory is not None and s3_object_key:
            raise ValueError(
                "s3_object_key cannot be combined with an in-memory presigned inventory."
            )
        inventory = (
            presigned_inventory
            if presigned_inventory is not None
            else _presigned_inventory_from_file(
                s3_info_file,
                config=config,
                object_key=s3_object_key,
                all_objects=s3_all_objects,
            )
        )
        request = DataEngRequest.model_validate(inventory)
        return _dispatch_presigned_request(
            request,
            config_path=config_file,
        )

    if s3_all_objects:
        raise ValueError("s3_all_objects requires presigned_info input.")
    if s3_object_key:
        raise ValueError("s3_object_key can only be used with an S3 info file.")
    resolved_uri = _configured_s3_uri(s3_uri, config)
    _, object_key = parse_s3_uri(resolved_uri)
    table_client = _table_client(config)
    if not table_client.accepts(object_key):
        state = run_pipeline(
            config_file,
            s3_uri=resolved_uri,
        )
        return DataEngDispatchResult(
            response=build_dataeng_output(state),
            pipeline_state=state,
            table_document_count=0,
        )

    with tempfile.TemporaryDirectory(prefix="axiom-table-agent-") as temp_text:
        s3_mapping = config.get("s3_input")
        if not isinstance(s3_mapping, dict):
            s3_mapping = {}
        downloaded = read_s3_input(
            {
                **s3_mapping,
                "mode": "s3_uri",
                "file_name": Path(object_key).name,
            },
            Path(temp_text) / "objects",
            s3_uri=resolved_uri,
            project_root=PROJECT_ROOT,
        )
        run_id = make_id("api-table-run", time.time_ns())
        table_document = normalize_table_agent_response(
            table_client.process_workbook(downloaded.path),
            source_uri=downloaded.source_uri,
            source_metadata=downloaded.metadata,
            run_id=run_id,
        )
    documents = [table_document]
    return DataEngDispatchResult(
        response={
            "metadata": _merge_metadata(
                None,
                documents=documents,
                run_id=run_id,
                input_source=downloaded.source_uri,
            ),
            "documents": documents,
        },
        pipeline_state=None,
        table_document_count=1,
    )


def _dispatch_presigned_request(
    request: DataEngRequest,
    *,
    config_path: str | Path,
) -> DataEngDispatchResult:
    load_dotenv_file(PROJECT_ROOT)
    config_file = _config_path(config_path)
    config = load_config(config_file)
    table_client = _table_client(config)
    table_files: list[PresignedFileRequest] = []
    pipeline_files: list[PresignedFileRequest] = []
    for item in request.files:
        target = table_files if table_client.accepts(item.key) else pipeline_files
        target.append(item)

    if not table_files:
        state = run_pipeline(
            config_file,
            presigned_inventory=request.to_inventory(),
        )
        return DataEngDispatchResult(
            response=build_dataeng_output(state),
            pipeline_state=state,
            table_document_count=0,
        )

    pipeline_state: PipelineState | None = None
    pipeline_output: dict[str, Any] | None = None
    if pipeline_files:
        pipeline_state = run_pipeline(
            config_file,
            presigned_inventory=_inventory(request.bucket, pipeline_files)
        )
        pipeline_output = build_dataeng_output(pipeline_state)

    run_id = (
        str((pipeline_output or {}).get("metadata", {}).get("run_id") or "")
        or make_id("api-table-run", time.time_ns())
    )
    table_documents = _process_table_files(
        request.bucket,
        table_files,
        table_client=table_client,
        s3_config=S3InputConfig.from_mapping(config.get("s3_input")),
        run_id=run_id,
    )
    pipeline_documents = (
        list(pipeline_output.get("documents", []))
        if pipeline_output is not None
        else []
    )
    documents = _restore_request_order(
        request,
        [*pipeline_documents, *table_documents],
    )
    metadata = _merge_metadata(
        pipeline_output.get("metadata") if pipeline_output else None,
        documents=documents,
        run_id=run_id,
        input_source=f"s3://{request.bucket}",
    )
    return DataEngDispatchResult(
        response={"metadata": metadata, "documents": documents},
        pipeline_state=pipeline_state,
        table_document_count=len(table_documents),
    )


def dispatch_local_dataeng_files(
    files: list[str | Path],
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    pipeline_input_root: str | Path | None = None,
) -> DataEngDispatchResult:
    """Route local workbook uploads while preserving the normal local pipeline."""
    load_dotenv_file(PROJECT_ROOT)
    config_file = _config_path(config_path)
    config = load_config(config_file)
    table_client = TableAgentClient(
        TableAgentClientConfig.from_mapping(config.get("table_agent"))
    )
    paths = [Path(value).resolve() for value in files]
    if not paths:
        raise ValueError("At least one local input file is required.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Local input file does not exist: {missing[0]}")

    table_files = [path for path in paths if table_client.accepts(path)]
    pipeline_files = [path for path in paths if path not in table_files]

    pipeline_state: PipelineState | None = None
    pipeline_output: dict[str, Any] | None = None
    if pipeline_files:
        input_root = _local_input_root(pipeline_files, pipeline_input_root)
        filtered_config = _local_pipeline_config(config, pipeline_files)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="axiom-local-dispatch-",
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(filtered_config, handle, ensure_ascii=False, indent=2)
            filtered_config_path = Path(handle.name)
        try:
            pipeline_state = run_pipeline(
                filtered_config_path,
                local_raw=input_root,
            )
            pipeline_output = build_dataeng_output(pipeline_state)
        finally:
            filtered_config_path.unlink(missing_ok=True)

    if not table_files:
        if pipeline_output is None:
            raise RuntimeError("Local pipeline did not produce an output.")
        return DataEngDispatchResult(
            response=pipeline_output,
            pipeline_state=pipeline_state,
            table_document_count=0,
        )

    run_id = (
        str((pipeline_output or {}).get("metadata", {}).get("run_id") or "")
        or make_id("api-table-run", time.time_ns())
    )
    source_root = _local_input_root(paths, pipeline_input_root)
    table_documents = [
        normalize_table_agent_response(
            table_client.process_workbook(path),
            source_uri=portable_path(path, PROJECT_ROOT),
            source_metadata=_local_source_metadata(path, source_root),
            run_id=run_id,
        )
        for path in table_files
    ]
    pipeline_documents = (
        list(pipeline_output.get("documents", []))
        if pipeline_output is not None
        else []
    )
    documents = _restore_source_order(
        [portable_path(path, PROJECT_ROOT) for path in paths],
        [*pipeline_documents, *table_documents],
    )
    metadata = _merge_metadata(
        pipeline_output.get("metadata") if pipeline_output else None,
        documents=documents,
        run_id=run_id,
        input_source=portable_path(source_root, PROJECT_ROOT),
    )
    return DataEngDispatchResult(
        response={"metadata": metadata, "documents": documents},
        pipeline_state=pipeline_state,
        table_document_count=len(table_documents),
    )


def _process_table_files(
    bucket: str,
    files: list[PresignedFileRequest],
    *,
    table_client: TableAgentClient,
    s3_config: S3InputConfig,
    run_id: str,
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="axiom-table-agent-") as temp_text:
        downloaded = download_all_presigned_s3_inventory(
            inventory=_inventory(bucket, files),
            destination_dir=Path(temp_text) / "objects",
            file_name=None,
            max_size_bytes=s3_config.max_size_bytes,
            timeout_seconds=s3_config.request_timeout_seconds,
            source="AXIOM /v1/dataeng request",
        )
        documents: list[dict[str, Any]] = []
        for item in downloaded.objects:
            response = table_client.process_workbook(item.path)
            documents.append(
                normalize_table_agent_response(
                    response,
                    source_uri=item.source_uri,
                    source_metadata=item.metadata,
                    run_id=run_id,
                )
            )
        return documents


def _restore_request_order(
    request: DataEngRequest,
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_uris = [
        f"s3://{request.bucket}/{item.key}"
        for item in request.files
    ]
    return _restore_source_order(source_uris, documents)


def _restore_source_order(
    source_uris: list[str],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_uri: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for document in documents:
        identity = document.get("document")
        source_uri = identity.get("source_uri") if isinstance(identity, dict) else None
        if isinstance(source_uri, str):
            by_uri[source_uri].append(document)

    ordered: list[dict[str, Any]] = []
    for source_uri in source_uris:
        matches = by_uri.get(source_uri)
        if matches:
            ordered.append(matches.popleft())
    return ordered


def _merge_metadata(
    pipeline_metadata: dict[str, Any] | None,
    *,
    documents: list[dict[str, Any]],
    run_id: str,
    input_source: str,
) -> dict[str, Any]:
    metadata = (
        deepcopy(pipeline_metadata)
        if isinstance(pipeline_metadata, dict)
        else _empty_output_metadata(run_id, input_source)
    )
    summaries = [_document_summary(document) for document in documents]
    metadata["run_id"] = run_id
    metadata["input_source"] = input_source
    metadata["document_count"] = len(documents)
    metadata["documents"] = summaries

    table_documents = [
        document
        for document in documents
        if _document_processor(document) == "table_agent"
    ]
    summary = metadata.get("summary")
    if not isinstance(summary, dict):
        summary = {}
        metadata["summary"] = summary
    summary["table_agent"] = {
        "document_count": len(table_documents),
        "job_ids": [
            _document_processor_metadata(document).get("job_id")
            for document in table_documents
            if _document_processor_metadata(document).get("job_id")
        ],
    }
    completed = metadata.get("completed_modules")
    if not isinstance(completed, list):
        completed = []
        metadata["completed_modules"] = completed
    for module in ("table_agent", "normalization"):
        if module not in completed:
            completed.append(module)
    return metadata


def _empty_output_metadata(run_id: str, input_source: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": "output",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_source": input_source,
        "document_count": 0,
        "documents": [],
        "schema": {
            "document_file": {
                "document": "consumer-facing document identity and source summary",
                "ingest": "source object and parsed data",
                "clean": "cleaned data",
                "enrich": "enriched data",
                "retrieval": "retrieval items and embeddings",
            },
            "records": {},
        },
        "errors": [],
        "summary": {},
        "completed_modules": [],
        "stage_dirs": {},
    }


def _document_processor(document: dict[str, Any]) -> str | None:
    return str(_document_processor_metadata(document).get("processor") or "") or None


def _document_processor_metadata(document: dict[str, Any]) -> dict[str, Any]:
    enrich = document.get("enrich")
    data = enrich.get("data") if isinstance(enrich, dict) else None
    metadata = data.get("metadata") if isinstance(data, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def _document_summary(document: dict[str, Any]) -> dict[str, Any]:
    identity = document.get("document")
    if not isinstance(identity, dict):
        identity = {}
    return {
        key: value
        for key, value in {
            "document_id": identity.get("document_id"),
            "source_uri": identity.get("source_uri"),
            "content_type": identity.get("content_type"),
            "file_name": identity.get("file_name"),
            "size_bytes": identity.get("size_bytes"),
        }.items()
        if value is not None
    }


def _inventory(
    bucket: str,
    files: list[PresignedFileRequest],
) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "files": [
            item.model_dump(exclude_none=True)
            for item in files
        ],
    }


def _table_client(config: dict[str, Any]) -> TableAgentClient:
    return TableAgentClient(
        TableAgentClientConfig.from_mapping(config.get("table_agent"))
    )


def _resolve_dispatch_mode(
    config: dict[str, Any],
    *,
    presigned_inventory: dict[str, Any] | None,
    s3_uri: str | None,
    s3_info_file: str | Path | None,
    local_raw: str | Path | None,
) -> str:
    explicit_modes = [
        mode
        for mode, value in (
            ("presigned_info", presigned_inventory),
            ("s3_uri", s3_uri),
            ("presigned_info", s3_info_file),
            ("local_raw", local_raw),
        )
        if value is not None
    ]
    if len(explicit_modes) > 1:
        raise ValueError("Use only one input source per AXIOM run.")
    input_config = config.get("input")
    if not isinstance(input_config, dict):
        input_config = {}
    s3_config = config.get("s3_input")
    if not isinstance(s3_config, dict):
        s3_config = {}
    configured = input_config.get("mode", s3_config.get("mode", "s3_uri"))
    mode = explicit_modes[0] if explicit_modes else str(configured)
    mode = mode.strip().lower()
    if mode not in {"s3_uri", "presigned_info", "local_raw"}:
        raise ValueError(
            "input.mode must be 's3_uri', 'presigned_info', or 'local_raw'."
        )
    return mode


def _discover_local_dispatch_files(
    local_raw: str | Path,
    *,
    config: dict[str, Any],
) -> tuple[list[Path], Path]:
    local_config = config.get("local_input")
    if not isinstance(local_config, dict):
        local_config = {}
    discovery_config = deepcopy(local_config)
    table_config = TableAgentClientConfig.from_mapping(
        config.get("table_agent")
    )
    extensions = list(discovery_config.get("include_extensions") or [])
    if table_config.enabled:
        for extension in table_config.supported_extensions:
            if extension not in extensions:
                extensions.append(extension)
    discovery_config["include_extensions"] = extensions
    batch = read_local_inputs(
        discovery_config,
        local_raw=local_raw,
        project_root=PROJECT_ROOT,
    )
    root = Path(local_raw)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return [item.path.resolve() for item in batch.inputs], root.resolve()


def _presigned_inventory_from_file(
    value: str | Path | None,
    *,
    config: dict[str, Any],
    object_key: str | None,
    all_objects: bool,
) -> dict[str, Any]:
    s3_config = config.get("s3_input")
    if not isinstance(s3_config, dict):
        s3_config = {}
    configured_path = value or s3_config.get("info_file")
    if not configured_path:
        raise RuntimeError(
            "Missing S3 info file. Set s3_input.info_file or pass --s3-info-file."
        )
    path = Path(configured_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        inventory = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"S3 info file does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in S3 info file: {path}") from exc
    if not isinstance(inventory, dict):
        raise ValueError(f"S3 info file must contain a JSON object: {path}")
    entries = inventory.get("files")
    if not isinstance(entries, list):
        raise ValueError(f"S3 inventory is missing a files array: {path}")
    files = [
        item
        for item in entries
        if isinstance(item, dict)
        and str(item.get("key") or "").strip()
        and str(item.get("presigned_url") or "").strip()
        and not str(item.get("key")).endswith("/")
    ]
    if not files:
        raise ValueError(f"S3 inventory has no downloadable objects: {path}")

    process_all = all_objects or bool(s3_config.get("all_objects", False))
    selected_key = (
        str(object_key).strip()
        if object_key
        else str(
            os.getenv(str(s3_config.get("object_key_env", "S3_OBJECT_KEY")))
            or s3_config.get("object_key")
            or ""
        ).strip()
    )
    if process_all and selected_key:
        raise ValueError("Use either s3_all_objects or s3_object_key, not both.")
    if not process_all:
        if selected_key:
            files = [
                item
                for item in files
                if str(item.get("key")) == selected_key
            ]
            if not files:
                raise ValueError(
                    f"Object key {selected_key!r} was not found in S3 info file: {path}"
                )
        elif len(files) != 1:
            raise ValueError(
                f"S3 info file contains {len(files)} downloadable objects; "
                "select one with --s3-object-key or use --s3-all-objects."
            )
    return {
        "bucket": inventory.get("bucket"),
        "files": files,
    }


def _configured_s3_uri(
    explicit_uri: str | None,
    config: dict[str, Any],
) -> str:
    s3_config = config.get("s3_input")
    if not isinstance(s3_config, dict):
        s3_config = {}
    env_name = str(s3_config.get("uri_env", "S3_URI"))
    uri = str(explicit_uri or os.getenv(env_name) or "").strip()
    if not uri:
        raise RuntimeError(f"Missing S3 object URI. Set {env_name} or pass --s3-uri.")
    return uri


def _local_input_root(
    files: list[Path],
    configured_root: str | Path | None,
) -> Path:
    if configured_root is not None:
        root = Path(configured_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Local input root does not exist: {root}")
        if root.is_file():
            if len(files) != 1 or files[0] != root:
                raise ValueError(
                    "A file pipeline_input_root must match the only local input."
                )
            return root
        outside = [path for path in files if not path.is_relative_to(root)]
        if outside:
            raise ValueError(
                f"Local input file is outside pipeline_input_root: {outside[0]}"
            )
        return root
    if len(files) == 1:
        return files[0]
    common = Path(os.path.commonpath([str(path.parent) for path in files]))
    return common.resolve()


def _local_pipeline_config(
    config: dict[str, Any],
    pipeline_files: list[Path],
) -> dict[str, Any]:
    filtered = deepcopy(config)
    local_input = filtered.get("local_input")
    if not isinstance(local_input, dict):
        local_input = {}
        filtered["local_input"] = local_input
    local_input["include_extensions"] = sorted(
        {path.suffix.lower() for path in pipeline_files}
    )
    return filtered


def _local_source_metadata(path: Path, root: Path) -> dict[str, Any]:
    try:
        file_name = path.relative_to(root).as_posix() if root.is_dir() else path.name
    except ValueError:
        file_name = path.name
    return {
        "input_provider": "local_raw",
        "file_name": file_name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "response_content_type": (
            mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


__all__ = [
    "DataEngDispatchResult",
    "LocalDataEngDispatchResult",
    "dispatch_dataeng_request",
    "dispatch_dataeng_inputs",
    "dispatch_local_dataeng_files",
]
