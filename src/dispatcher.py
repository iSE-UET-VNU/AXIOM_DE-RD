"""Route every AXIOM input source into the document pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .models import PipelineState
from .local_reader import read_local_inputs
from .pipeline import run_pipeline
from .utils.config import default_pipeline_config_path, load_config
from .utils.env import load_dotenv_file
from .api.output import build_dataeng_output, public_dataeng_response
from .api.schemas import DataEngRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataEngDispatchResult:
    """Combined API-shaped result for any supported AXIOM input source."""

    response: dict[str, Any]
    pipeline_state: PipelineState


LocalDataEngDispatchResult = DataEngDispatchResult


def dispatch_dataeng_request(
    request: DataEngRequest,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the existing DataEng response after API-level input routing."""
    return public_dataeng_response(
        dispatch_dataeng_inputs(
            config_path=config_path,
            presigned_inventory=request.to_inventory(),
        ).response
    )


def dispatch_dataeng_inputs(
    config_path: str | Path | None = None,
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
    config_path = config_path or default_pipeline_config_path()
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
    state = run_pipeline(
        config_file,
        s3_uri=resolved_uri,
    )
    return DataEngDispatchResult(
        response=build_dataeng_output(state),
        pipeline_state=state,
    )


def _dispatch_presigned_request(
    request: DataEngRequest,
    *,
    config_path: str | Path,
) -> DataEngDispatchResult:
    load_dotenv_file(PROJECT_ROOT)
    config_file = _config_path(config_path)
    state = run_pipeline(
        config_file,
        presigned_inventory=request.to_inventory(),
    )
    return DataEngDispatchResult(
        response=build_dataeng_output(state),
        pipeline_state=state,
    )


def dispatch_local_dataeng_files(
    files: list[str | Path],
    *,
    config_path: str | Path | None = None,
    pipeline_input_root: str | Path | None = None,
) -> DataEngDispatchResult:
    """Run local files through the normal document pipeline."""
    load_dotenv_file(PROJECT_ROOT)
    config_path = config_path or default_pipeline_config_path()
    config_file = _config_path(config_path)
    config = load_config(config_file)
    paths = [Path(value).resolve() for value in files]
    if not paths:
        raise ValueError("At least one local input file is required.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Local input file does not exist: {missing[0]}")

    input_root = _local_input_root(paths, pipeline_input_root)
    filtered_config = _local_pipeline_config(config, paths)
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
    return DataEngDispatchResult(
        response=pipeline_output,
        pipeline_state=pipeline_state,
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
