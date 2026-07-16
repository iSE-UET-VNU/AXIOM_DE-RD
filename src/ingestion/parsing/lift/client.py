"""Datalab Lift API adapter for AXIOM ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
from pathlib import Path
import re
from typing import Any
import base64
import binascii
import json
import os
import time

from ....models import DataObject, ParsedData
from ....utils.paths import portable_path_value

SUPPORTED_EXTENSIONS = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"}
)
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "schemas/document_components.json"


@dataclass
class LiftAPIConfig:
    api_key_env: str = "DATALAB_API_KEY"
    mode: str = "balanced"
    schema_path: str | None = None
    output_dir: str | None = "data/work/datalab"
    extract_images: bool = True
    save_raw_outputs: bool = True
    project_root: str | None = None

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None) -> "LiftAPIConfig":
        config = config or {}
        return cls(
            api_key_env=str(config.get("api_key_env", "DATALAB_API_KEY")),
            mode=str(config.get("mode", "balanced")),
            schema_path=_optional_str(config.get("schema_path")),
            output_dir=_optional_str(config.get("output_dir", "data/work/datalab")),
            extract_images=bool(config.get("extract_images", True)),
            save_raw_outputs=bool(config.get("save_raw_outputs", True)),
            project_root=_optional_str(config.get("project_root")),
        )

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)


class LiftAPIParserClient:
    """Small wrapper around the Datalab hosted extraction API."""

    provider_name = "lift_api"
    supported_extensions = SUPPORTED_EXTENSIONS

    def __init__(self, config: LiftAPIConfig) -> None:
        self.config = config

    def parse_file(self, path: str | Path, data_object: DataObject) -> ParsedData:
        file_path = Path(path)
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise RuntimeError(f"Lift API does not support file type: {file_path.suffix}")

        if not self.config.api_key:
            raise RuntimeError(f"{self.config.api_key_env} is not set.")

        try:
            from datalab_sdk import ConvertOptions, DatalabClient, ExtractOptions
        except ImportError as exc:
            raise RuntimeError(
                "Missing Lift dependency. Install it with: pip install -e .[lift]"
            ) from exc

        schema = _load_schema(self.config.schema_path)
        client = DatalabClient()
        options = ExtractOptions(page_schema=json.dumps(schema), mode=self.config.mode)

        started = time.monotonic()
        result = client.extract(str(file_path), options=options)
        raw_json = _get_attr(result, "extraction_schema_json")
        extraction = _parse_extraction(raw_json)
        images = _normalize_images(_get_attr(result, "images", {}))
        image_source = "extract"
        conversion = None
        if self.config.extract_images and not images:
            conversion = client.convert(
                str(file_path),
                options=ConvertOptions(
                    mode=self.config.mode,
                    disable_image_extraction=False,
                    output_format="markdown",
                ),
            )
            images = _normalize_images(_get_attr(conversion, "images", {}))
            image_source = "convert"

        bundle_dir = _bundle_dir(
            self.config.output_dir,
            file_path,
            data_object.object_id,
        )
        image_files = _write_images(bundle_dir, images)
        raw_output_paths = _write_lift_raw_outputs(
            bundle_dir,
            extract_result=result,
            convert_result=conversion,
            image_files=image_files,
            enabled=self.config.save_raw_outputs,
        )

        payload = {
            "input": {
                "object_id": data_object.object_id,
                "uri": data_object.uri,
                "file_name": file_path.name,
                "content_type": data_object.content_type,
                "metadata": data_object.metadata,
            },
            "status": _get_attr(result, "status"),
            "page_count": _get_attr(result, "page_count"),
            "latency_seconds": round(time.monotonic() - started, 3),
            "extraction": extraction,
            "images": images,
            "image_files": image_files,
            "image_source": image_source,
            "raw_lift_outputs": raw_output_paths,
        }
        output_path = _write_output(bundle_dir, payload, self.config.project_root)

        text = _extraction_text(extraction)
        return ParsedData(
            object_id=data_object.object_id,
            source_uri=data_object.uri,
            source_format=str(data_object.metadata.get("format", file_path.suffix.lstrip("."))),
            rows=[{"extraction": extraction, "text": text}],
            text=text,
            metadata={
                "parser": "lift-api",
                "mode": self.config.mode,
                "status": payload["status"],
                "page_count": payload["page_count"],
                "latency_seconds": payload["latency_seconds"],
                "raw_output_path": str(output_path) if output_path else None,
                "image_count": len(image_files),
                "image_files": image_files,
                "image_source": image_source,
                "raw_lift_outputs": raw_output_paths,
            },
        )


def _load_schema(schema_path: str | None) -> dict[str, Any]:
    path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
    with path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise RuntimeError(f"Lift schema must be a JSON object: {path}")
    return schema


def _bundle_dir(output_dir: str | None, file_path: Path, object_id: str) -> Path | None:
    if not output_dir:
        return None
    slug = _safe_slug(file_path.stem)
    bundle_dir = Path(output_dir) / f"{slug}--{object_id}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    return bundle_dir


def _write_output(
    bundle_dir: Path | None,
    payload: dict[str, Any],
    project_root: str | Path | None = None,
) -> Path | None:
    if bundle_dir is None:
        return None
    output_path = bundle_dir / "result.json"
    output_path.write_text(
        json.dumps(portable_path_value(payload, project_root), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def _write_images(
    bundle_dir: Path | None, images: dict[str, str]
) -> list[dict[str, str]]:
    if bundle_dir is None or not images:
        return []

    image_dir = bundle_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_files: list[dict[str, str]] = []
    used_names: set[str] = set()

    for index, (name, encoded) in enumerate(sorted(images.items()), start=1):
        safe_name = _safe_image_name(name, index, encoded)
        safe_name = _deduplicate_image_name(safe_name, name, used_names)
        used_names.add(safe_name.casefold())
        output_path = image_dir / safe_name
        try:
            output_path.write_bytes(_decode_base64_image(encoded))
        except (binascii.Error, ValueError) as exc:
            image_files.append(
                {
                    "name": name,
                    "path": str(output_path),
                    "status": "decode_failed",
                    "error": str(exc),
                }
            )
            continue

        image_files.append({"name": name, "path": str(output_path), "status": "saved"})

    return image_files


def _write_lift_raw_outputs(
    bundle_dir: Path | None,
    *,
    extract_result: Any,
    convert_result: Any = None,
    image_files: list[dict[str, str]] | None = None,
    enabled: bool = True,
) -> dict[str, str]:
    if bundle_dir is None or not enabled:
        return {}

    asset_map = _image_asset_map(image_files or [])
    paths: dict[str, str] = {}
    paths.update(_write_result_artifacts(bundle_dir, "extract", extract_result, asset_map))
    if convert_result is not None:
        paths.update(_write_result_artifacts(bundle_dir, "convert", convert_result, asset_map))
    return paths


def _write_result_artifacts(
    bundle_dir: Path,
    prefix: str,
    result: Any,
    asset_map: dict[str, str] | None = None,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    result_data = _result_to_dict(result)

    raw_json_path = bundle_dir / f"{prefix}.raw.json"
    raw_json_path.write_text(
        json.dumps(result_data, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    paths[f"{prefix}_raw_json"] = str(raw_json_path)

    for field, suffix in (
        ("markdown", ".md"),
        ("html", ".html"),
        ("extraction_schema_json", ".extraction_schema.json"),
    ):
        value = _get_attr(result, field)
        if isinstance(value, str) and value:
            path = bundle_dir / f"{prefix}{suffix}"
            path.write_text(value, encoding="utf-8")
            paths[f"{prefix}_{field}"] = str(path)
            if field == "markdown":
                rendered_path = bundle_dir / f"{prefix}.rendered.md"
                if rendered_path.exists():
                    rendered_path.unlink()
                rendered = _render_markdown_with_assets(value, asset_map or {})
                if rendered is not None:
                    rendered_path.write_text(rendered, encoding="utf-8")
                    paths[f"{prefix}_rendered_markdown"] = str(rendered_path)

    for field, suffix in (("json", ".document.json"), ("chunks", ".chunks.json")):
        value = _get_attr(result, field)
        if value:
            path = bundle_dir / f"{prefix}{suffix}"
            path.write_text(
                json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
                encoding="utf-8",
            )
            paths[f"{prefix}_{field}"] = str(path)

    return paths


def _result_to_dict(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    return {"value": result}


def _json_default(value: Any) -> str:
    return str(value)


def _normalize_images(raw_images: Any) -> dict[str, str]:
    if not isinstance(raw_images, dict):
        return {}
    return {str(name): str(encoded) for name, encoded in raw_images.items() if encoded}


def _decode_base64_image(encoded: str) -> bytes:
    if "," in encoded and encoded.strip().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    encoded = "".join(encoded.split())
    return base64.b64decode(encoded, validate=True)


def _safe_image_name(name: str, index: int, encoded: str) -> str:
    path_name = Path(name.replace("\\", "/")).name.strip()
    if path_name and Path(path_name).suffix:
        return path_name

    suffix = _image_suffix_from_data_uri(encoded) or ".png"
    stem = Path(path_name).stem if path_name else f"image_{index:03d}"
    return f"{stem}{suffix}"


def _deduplicate_image_name(candidate: str, original_name: str, used_names: set[str]) -> str:
    if candidate.casefold() not in used_names:
        return candidate
    path = Path(candidate)
    digest = hashlib.sha1(original_name.replace("\\", "/").encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}--{digest}{path.suffix}"


def _safe_slug(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (slug or "document")[:max_length].rstrip("-._") or "document"


def _image_asset_map(image_files: list[dict[str, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    basename_targets: dict[str, set[str]] = {}
    for item in image_files:
        if item.get("status") != "saved" or not item.get("name") or not item.get("path"):
            continue
        original = str(item["name"]).replace("\\", "/")
        target = Path(str(item["path"])).name
        mapping[original] = target
        basename_targets.setdefault(Path(original).name, set()).add(target)
    for basename, targets in basename_targets.items():
        if len(targets) == 1:
            mapping.setdefault(basename, next(iter(targets)))
    return mapping


_MARKDOWN_IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^\s)]+)(\))")
_HTML_IMAGE_RE = re.compile(r"(<img\b[^>]*?\bsrc=[\"'])([^\"']+)([\"'])", re.IGNORECASE)


def _rewrite_markdown_asset_refs(markdown: str, asset_map: dict[str, str]) -> str:
    def replace_target(match: re.Match[str]) -> str:
        target = match.group(2).strip("<>").replace("\\", "/")
        replacement = _asset_target(target, asset_map)
        return f"{match.group(1)}{replacement or match.group(2)}{match.group(3)}"

    def replace_html_target(match: re.Match[str]) -> str:
        target = match.group(2).replace("\\", "/")
        replacement = _asset_target(target, asset_map)
        return f"{match.group(1)}{replacement or match.group(2)}{match.group(3)}"

    rendered = _MARKDOWN_IMAGE_RE.sub(replace_target, markdown)
    return _HTML_IMAGE_RE.sub(replace_html_target, rendered)


def _render_markdown_with_assets(markdown: str, asset_map: dict[str, str]) -> str | None:
    rendered = _rewrite_markdown_asset_refs(markdown, asset_map)
    if rendered == markdown:
        return None
    valid_targets = {f"images/{name}" for name in asset_map.values()}
    for pattern in (_MARKDOWN_IMAGE_RE, _HTML_IMAGE_RE):
        for match in pattern.finditer(rendered):
            target = match.group(2).strip("<>").replace("\\", "/")
            if _is_external_image_target(target):
                continue
            if target not in valid_targets:
                return None
    return rendered


def _asset_target(target: str, asset_map: dict[str, str]) -> str | None:
    if _is_external_image_target(target) or target.lower().startswith("images/"):
        return None
    image_name = asset_map.get(target)
    return f"images/{image_name}" if image_name else None


def _is_external_image_target(target: str) -> bool:
    return target.lower().startswith(("http://", "https://", "data:")) or target.startswith(("#", "/"))


def _image_suffix_from_data_uri(encoded: str) -> str | None:
    if not encoded.startswith("data:image/"):
        return None
    mime = encoded.split(";", 1)[0].removeprefix("data:image/")
    if mime == "jpeg":
        return ".jpg"
    if mime:
        return f".{mime}"
    return None


def _parse_extraction(raw_json: Any) -> Any:
    if raw_json is None:
        return None
    if isinstance(raw_json, (dict, list)):
        return raw_json
    try:
        return json.loads(str(raw_json))
    except json.JSONDecodeError:
        return raw_json


def _extraction_text(extraction: Any) -> str | None:
    if not isinstance(extraction, dict):
        return None
    for field in ("main_text", "markdown", "text", "content"):
        value = extraction.get(field)
        if value:
            return str(value)
    return None


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _optional_str(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)
