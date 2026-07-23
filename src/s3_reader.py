"""Read one S3 object through boto3 or a presigned-URL inventory."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import mimetypes
import os
from pathlib import Path
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_MAX_SIZE_BYTES = 512 * 1024 * 1024
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class S3InputConfig:
    """Configuration for loading one S3 object into the pipeline."""

    mode: str = "s3_uri"
    uri_env: str = "S3_URI"
    profile_env: str = "AWS_PROFILE"
    region_env: str = "AWS_REGION"
    endpoint_url_env: str = "S3_ENDPOINT_URL"
    info_file: str | None = None
    object_key: str | None = None
    object_key_env: str = "S3_OBJECT_KEY"
    file_name: str | None = None
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES
    request_timeout_seconds: float = 60.0

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "S3InputConfig":
        value = value or {}
        return cls(
            mode=str(value.get("mode", "s3_uri")).strip().lower(),
            uri_env=str(value.get("uri_env", "S3_URI")),
            profile_env=str(value.get("profile_env", "AWS_PROFILE")),
            region_env=str(value.get("region_env", "AWS_REGION")),
            endpoint_url_env=str(value.get("endpoint_url_env", "S3_ENDPOINT_URL")),
            info_file=_optional_string(value.get("info_file")),
            object_key=_optional_string(value.get("object_key")),
            object_key_env=str(value.get("object_key_env", "S3_OBJECT_KEY")),
            file_name=_optional_string(value.get("file_name")),
            max_size_bytes=int(value.get("max_size_bytes", DEFAULT_MAX_SIZE_BYTES)),
            request_timeout_seconds=float(value.get("request_timeout_seconds", 60.0)),
        )


@dataclass(frozen=True)
class DownloadedObject:
    """A locally persisted raw object plus its S3 provenance."""

    path: Path
    source_uri: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DownloadedObjectBatch:
    """A bucket-level source and all objects downloaded from its inventory."""

    source: str
    objects: list[DownloadedObject]


def read_s3_input(
    config_value: dict[str, Any] | None,
    destination_dir: str | Path,
    *,
    s3_uri: str | None = None,
    s3_info_file: str | Path | None = None,
    s3_object_key: str | None = None,
    project_root: str | Path | None = None,
) -> DownloadedObject:
    """Resolve one of the supported S3 input modes and download one object."""
    config = S3InputConfig.from_mapping(config_value)
    if config.mode not in {"s3_uri", "presigned_info"}:
        raise ValueError(
            "s3_input.mode must be either 's3_uri' or 'presigned_info'."
        )
    if s3_uri and s3_info_file:
        raise ValueError("Use either s3_uri or s3_info_file, not both.")

    use_presigned_info = s3_info_file is not None or (
        s3_uri is None and config.mode == "presigned_info"
    )
    if use_presigned_info:
        configured_info_file = s3_info_file or config.info_file
        if not configured_info_file:
            raise RuntimeError(
                "Missing S3 info file. Set s3_input.info_file or pass --s3-info-file."
            )
        info_path = _resolve_info_file(configured_info_file, project_root)
        object_key = (
            _optional_string(s3_object_key)
            or _optional_env(config.object_key_env)
            or config.object_key
        )
        return download_presigned_s3_info(
            info_path=info_path,
            destination_dir=destination_dir,
            object_key=object_key,
            file_name=config.file_name,
            max_size_bytes=config.max_size_bytes,
            timeout_seconds=config.request_timeout_seconds,
        )

    if s3_object_key:
        raise ValueError("s3_object_key can only be used with an S3 info file.")
    resolved_uri = (s3_uri or os.getenv(config.uri_env) or "").strip()
    if not resolved_uri:
        raise RuntimeError(f"Missing S3 object URI. Set {config.uri_env} or pass --s3-uri.")
    bucket, key = parse_s3_uri(resolved_uri)
    return download_s3_object(
        bucket=bucket,
        key=key,
        destination_dir=destination_dir,
        file_name=config.file_name,
        max_size_bytes=config.max_size_bytes,
        profile=_optional_env(config.profile_env),
        region=_optional_env(config.region_env) or _optional_env("AWS_DEFAULT_REGION"),
        endpoint_url=_optional_env(config.endpoint_url_env),
    )


def download_presigned_s3_info(
    *,
    info_path: str | Path,
    destination_dir: str | Path,
    object_key: str | None = None,
    file_name: str | None = None,
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    timeout_seconds: float = 60.0,
) -> DownloadedObject:
    """Select and download one object from an S3 inventory with presigned URLs."""
    path = Path(info_path)
    inventory = _load_s3_info(path)
    bucket = _inventory_bucket(inventory, path)
    entries = _presigned_entries(inventory.get("files"), path)
    entry = _select_presigned_entry(entries, object_key, path)
    return _download_presigned_entry(
        bucket=bucket,
        entry=entry,
        destination_dir=destination_dir,
        file_name=file_name,
        max_size_bytes=max_size_bytes,
        timeout_seconds=timeout_seconds,
    )


def download_all_presigned_s3_info(
    *,
    info_path: str | Path,
    destination_dir: str | Path,
    file_name: str | None = None,
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    timeout_seconds: float = 60.0,
) -> DownloadedObjectBatch:
    """Download every valid presigned object from one S3 inventory."""
    path = Path(info_path)
    inventory = _load_s3_info(path)
    return download_all_presigned_s3_inventory(
        inventory=inventory,
        destination_dir=destination_dir,
        file_name=file_name,
        max_size_bytes=max_size_bytes,
        timeout_seconds=timeout_seconds,
        source=path,
    )


def download_all_presigned_s3_inventory(
    *,
    inventory: dict[str, Any],
    destination_dir: str | Path,
    file_name: str | None = None,
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    timeout_seconds: float = 60.0,
    source: str | Path = "request payload",
) -> DownloadedObjectBatch:
    """Download every object from an in-memory presigned S3 inventory."""
    if not isinstance(inventory, dict):
        raise ValueError(f"S3 inventory must contain a JSON object: {source}")
    bucket = _inventory_bucket(inventory, source)
    inventory_entries = _presigned_entries(inventory.get("files"), source)
    entries = [entry for entry in inventory_entries if not str(entry["key"]).endswith("/")]
    skipped_count = len(inventory_entries) - len(entries)
    if skipped_count:
        logger.info("Skipping %s S3 folder marker(s) from the inventory", skipped_count)
    if not entries:
        raise RuntimeError(f"No downloadable file objects found in S3 inventory: {source}")
    destination = Path(destination_dir)
    objects: list[DownloadedObject] = []
    for index, entry in enumerate(entries):
        logger.info(
            "Downloading presigned S3 object %s/%s: %s",
            index + 1,
            len(entries),
            entry["key"],
        )
        objects.append(
            _download_presigned_entry(
                bucket=bucket,
                entry=entry,
                destination_dir=destination / f"{index:06d}",
                file_name=file_name,
                max_size_bytes=max_size_bytes,
                timeout_seconds=timeout_seconds,
            )
        )
    return DownloadedObjectBatch(source=f"s3://{bucket}", objects=objects)


def read_all_presigned_s3_inputs(
    config_value: dict[str, Any] | None,
    destination_dir: str | Path,
    *,
    s3_info_file: str | Path | None = None,
    inventory: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> DownloadedObjectBatch:
    """Resolve a file-backed or in-memory inventory and download every object."""
    config = S3InputConfig.from_mapping(config_value)
    if inventory is not None:
        if s3_info_file is not None:
            raise ValueError("Use either inventory or s3_info_file, not both.")
        return download_all_presigned_s3_inventory(
            inventory=inventory,
            destination_dir=destination_dir,
            file_name=config.file_name,
            max_size_bytes=config.max_size_bytes,
            timeout_seconds=config.request_timeout_seconds,
        )
    configured_info_file = s3_info_file or config.info_file
    if not configured_info_file:
        raise RuntimeError(
            "Missing S3 info file. Set s3_input.info_file or pass --s3-info-file."
        )
    info_path = _resolve_info_file(configured_info_file, project_root)
    return download_all_presigned_s3_info(
        info_path=info_path,
        destination_dir=destination_dir,
        file_name=config.file_name,
        max_size_bytes=config.max_size_bytes,
        timeout_seconds=config.request_timeout_seconds,
    )


def _download_presigned_entry(
    *,
    bucket: str,
    entry: dict[str, Any],
    destination_dir: str | Path,
    file_name: str | None,
    max_size_bytes: int,
    timeout_seconds: float,
) -> DownloadedObject:
    if max_size_bytes <= 0:
        raise ValueError("s3_input.max_size_bytes must be greater than zero.")
    if timeout_seconds <= 0:
        raise ValueError("s3_input.request_timeout_seconds must be greater than zero.")

    key = str(entry["key"])
    presigned_url = str(entry["presigned_url"])
    parsed_url = urlsplit(presigned_url)
    if parsed_url.scheme.lower() not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"Invalid presigned URL for object {key!r}.")

    expected_size = _optional_non_negative_int(entry.get("size"), "size", key)
    if expected_size is not None and expected_size > max_size_bytes:
        raise RuntimeError(
            "S3 object exceeds s3_input.max_size_bytes "
            f"({expected_size} > {max_size_bytes})."
        )

    request = Request(
        presigned_url,
        headers={"User-Agent": "AXIOM-DE-RD/0.1"},
        method="GET",
    )
    try:
        response = urlopen(request, timeout=timeout_seconds)
    except HTTPError as exc:
        raise RuntimeError(
            f"Could not read presigned S3 object {key!r}: HTTP {exc.code}."
        ) from None
    except URLError:
        raise RuntimeError(f"Could not reach presigned S3 object {key!r}.") from None

    response_headers = response.headers
    content_length = _header_int(response_headers, "Content-Length")
    if content_length is not None and content_length > max_size_bytes:
        response.close()
        raise RuntimeError(
            "S3 object exceeds s3_input.max_size_bytes "
            f"({content_length} > {max_size_bytes})."
        )

    content_type = _content_type(_header_value(response_headers, "Content-Type"))
    resolved_name = _resolve_file_name(file_name, key, content_type)
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / resolved_name
    size_bytes, checksum = _write_stream(
        response,
        output_path,
        max_size_bytes=max_size_bytes,
    )
    if expected_size is not None and size_bytes != expected_size:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded size does not match S3 info for object {key!r} "
            f"({size_bytes} != {expected_size})."
        )

    return DownloadedObject(
        path=output_path,
        source_uri=f"s3://{bucket}/{key}",
        metadata={
            "input_provider": "presigned_s3_info",
            "file_name": resolved_name,
            "size_bytes": size_bytes,
            "sha256": checksum,
            "response_content_type": content_type,
            "s3_bucket": bucket,
            "s3_key": key,
            "s3_etag": _clean_etag(
                entry.get("etag") or _header_value(response_headers, "ETag")
            ),
            "s3_version_id": None,
        },
    )


def download_s3_object(
    *,
    bucket: str,
    key: str,
    destination_dir: str | Path,
    file_name: str | None = None,
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    profile: str | None = None,
    region: str | None = None,
    endpoint_url: str | None = None,
) -> DownloadedObject:
    """Stream one object through boto3 and return its persisted raw path."""
    if max_size_bytes <= 0:
        raise ValueError("s3_input.max_size_bytes must be greater than zero.")

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as exc:
        raise RuntimeError(
            "Missing boto3. Install project dependencies with: pip install -e ."
        ) from exc

    session = boto3.Session(profile_name=profile, region_name=region)
    client_kwargs: dict[str, Any] = {}
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
    client = session.client("s3", **client_kwargs)

    try:
        response = client.get_object(Bucket=bucket, Key=key)
        content_length = int(response.get("ContentLength") or 0)
        if content_length > max_size_bytes:
            response["Body"].close()
            raise RuntimeError(
                "S3 object exceeds s3_input.max_size_bytes "
                f"({content_length} > {max_size_bytes})."
            )

        content_type = _optional_string(response.get("ContentType"))
        resolved_name = _resolve_file_name(file_name, key, content_type)
        destination = Path(destination_dir)
        destination.mkdir(parents=True, exist_ok=True)
        output_path = destination / resolved_name
        size_bytes, checksum = _write_stream(
            response["Body"],
            output_path,
            max_size_bytes=max_size_bytes,
        )
    except (BotoCoreError, ClientError) as exc:
        error_code = "unknown"
        if isinstance(exc, ClientError):
            error_code = str(exc.response.get("Error", {}).get("Code") or error_code)
        raise RuntimeError(f"Could not read s3://{bucket}/{key}: {error_code}.") from None

    source_uri = f"s3://{bucket}/{key}"
    return DownloadedObject(
        path=output_path,
        source_uri=source_uri,
        metadata={
            "input_provider": "boto3_s3",
            "file_name": resolved_name,
            "size_bytes": size_bytes,
            "sha256": checksum,
            "response_content_type": content_type,
            "s3_bucket": bucket,
            "s3_key": key,
            "s3_etag": _clean_etag(response.get("ETag")),
            "s3_version_id": _optional_string(response.get("VersionId")),
        },
    )


def parse_s3_uri(value: str) -> tuple[str, str]:
    """Split an s3://bucket/key URI into boto3 parameters."""
    parsed = urlsplit(value)
    if parsed.scheme.lower() != "s3" or not parsed.netloc:
        raise ValueError("S3 input must use the form s3://bucket/key.")
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError("S3 URI must include an object key.")
    if parsed.query or parsed.fragment:
        raise ValueError("S3 URI must not include a query string or fragment.")
    return parsed.netloc, key


def _load_s3_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"S3 info file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in S3 info file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"S3 info file must contain a JSON object: {path}")
    return payload


def _select_presigned_entry(
    entries: list[dict[str, Any]],
    object_key: str | None,
    info_path: Path,
) -> dict[str, Any]:
    selected_key = _optional_string(object_key)
    if selected_key:
        matches = [entry for entry in entries if str(entry["key"]) == selected_key]
        if not matches:
            raise ValueError(
                f"Object key {selected_key!r} was not found in S3 info file: {info_path}"
            )
        return matches[0]
    if len(entries) == 1:
        return entries[0]
    raise ValueError(
        f"S3 info file contains {len(entries)} downloadable objects; "
        "select one with --s3-object-key or S3_OBJECT_KEY, or use --s3-all-objects."
    )


def _presigned_entries(
    value: Any,
    source: str | Path,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"S3 inventory is missing a files array: {source}")
    entries = [
        entry
        for entry in value
        if isinstance(entry, dict)
        and _optional_string(entry.get("key"))
        and _optional_string(entry.get("presigned_url"))
    ]
    if not entries:
        raise ValueError(f"S3 inventory has no downloadable objects: {source}")
    return entries


def _inventory_bucket(inventory: dict[str, Any], source: str | Path) -> str:
    bucket = _optional_string(inventory.get("bucket"))
    if not bucket:
        raise ValueError(f"S3 inventory is missing a bucket: {source}")
    return bucket


def _resolve_info_file(value: str | Path, project_root: str | Path | None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if project_root:
        return Path(project_root) / path
    return path


def _write_stream(body: Any, destination: Path, *, max_size_bytes: int) -> tuple[int, str]:
    temporary = destination.with_name(f".{destination.name}.part")
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with temporary.open("wb") as handle:
            while chunk := body.read(DOWNLOAD_CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > max_size_bytes:
                    raise RuntimeError(
                        "S3 object exceeds s3_input.max_size_bytes while downloading."
                    )
                handle.write(chunk)
                digest.update(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        body.close()
    return size_bytes, digest.hexdigest()


def _resolve_file_name(configured: str | None, key: str, content_type: str | None) -> str:
    candidate = configured or Path(key).name or "document"
    candidate = _safe_file_name(candidate)
    if not Path(candidate).suffix and content_type:
        candidate = f"{candidate}{mimetypes.guess_extension(content_type) or ''}"
    return candidate


def _header_value(headers: Any, name: str) -> str | None:
    try:
        value = headers.get(name)
    except (AttributeError, TypeError):
        return None
    return _optional_string(value)


def _header_int(headers: Any, name: str) -> int | None:
    value = _header_value(headers, name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _content_type(value: str | None) -> str | None:
    if value is None:
        return None
    return _optional_string(value.split(";", 1)[0])


def _optional_non_negative_int(value: Any, field_name: str, key: str) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {field_name} for object {key!r} in S3 info file.") from None
    if result < 0:
        raise ValueError(f"Invalid {field_name} for object {key!r} in S3 info file.")
    return result


def _safe_file_name(value: str) -> str:
    name = Path(value.replace("\\", "/")).name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return (name or "document")[:180]


def _clean_etag(value: Any) -> str | None:
    text = _optional_string(value)
    return text.strip('"') if text else None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_env(name: str) -> str | None:
    return _optional_string(os.getenv(name))
