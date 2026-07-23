"""Run parser comparison experiments without changing production parser routing."""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import Counter
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import unicodedata
from urllib.parse import quote


REQUIRED_CONVERT_OUTPUTS = ("markdown", "html", "json", "chunks")
SECRET_KEYS = frozenset({"api_key", "authorization", "checkpoint_id"})
CONFIG = {
    "provider": "datalab",
    "mode": "accurate",
    "skip_cache": True,
    "output_format": "markdown,html,json,chunks",
    "save_checkpoint": True,
    "add_block_ids": True,
    "include_markdown_in_chunks": True,
    "disable_image_extraction": False,
    "extract_schema": "src/ingestion/parsing/lift/schemas/document_components.json",
    "extract_checkpoint_reuse": True,
}

IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+[^)]*)?\)")
LINK_PATTERN = re.compile(r"(?<!!)\[([^\]]+)\]\([^)]+\)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_TABLE_PATTERN = re.compile(r"<table\b", re.IGNORECASE)
HTML_ROW_PATTERN = re.compile(r"<tr\b", re.IGNORECASE)
PIPE_SEPARATOR_PATTERN = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
UNORDERED_LIST_PATTERN = re.compile(r"^\s*[-+*]\s+\S", re.MULTILINE)
ORDERED_LIST_PATTERN = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)
WORD_PATTERN = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
        + "\n",
        encoding="utf-8",
    )


def result_to_dict(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return dict(result)
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)
    return {"value": result}


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if str(key).casefold() in SECRET_KEYS else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    return value


def load_manifest(experiment_dir: Path) -> dict[str, Any]:
    manifest_path = experiment_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest.get("documents")
    if not isinstance(documents, list) or len(documents) != 5:
        raise RuntimeError("Datalab baseline requires exactly five manifest documents.")

    for document in documents:
        input_path = input_path_for(experiment_dir, document)
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        actual_sha256 = sha256_file(input_path)
        if actual_sha256 != document.get("sha256"):
            raise RuntimeError(
                f"Corpus checksum mismatch for {document.get('relative_path')}: "
                f"{actual_sha256} != {document.get('sha256')}"
            )
    return manifest


def input_path_for(experiment_dir: Path, document: dict[str, Any]) -> Path:
    return experiment_dir / "corpus" / document["document_id"] / document["filename"]


def config_hash(sdk_version: str, schema: dict[str, Any]) -> str:
    payload = {"config": CONFIG, "sdk_version": sdk_version, "schema": schema}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def get_attr(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def normalize_images(raw_images: Any) -> dict[str, str]:
    if not isinstance(raw_images, dict):
        return {}
    return {str(name): str(encoded) for name, encoded in raw_images.items() if encoded}


def decode_base64(encoded: str) -> bytes:
    if encoded.strip().startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    return base64.b64decode("".join(encoded.split()), validate=True)


def write_images(document_dir: Path, raw_images: Any) -> list[dict[str, Any]]:
    images = normalize_images(raw_images)
    if not images:
        return []
    output_dir = document_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    records: list[dict[str, Any]] = []
    for index, (original_name, encoded) in enumerate(sorted(images.items()), start=1):
        candidate = Path(original_name.replace("\\", "/")).name or f"image_{index:04d}.png"
        if not Path(candidate).suffix:
            candidate = f"{candidate}.png"
        if candidate.casefold() in used:
            path = Path(candidate)
            suffix = hashlib.sha1(original_name.encode("utf-8")).hexdigest()[:8]
            candidate = f"{path.stem}--{suffix}{path.suffix}"
        used.add(candidate.casefold())
        output_path = output_dir / candidate
        try:
            payload = decode_base64(encoded)
            output_path.write_bytes(payload)
            records.append(
                {
                    "name": original_name,
                    "path": output_path.relative_to(document_dir).as_posix(),
                    "size_bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                    "status": "saved",
                }
            )
        except (binascii.Error, ValueError) as exc:
            records.append(
                {"name": original_name, "path": None, "status": "decode_failed", "error": str(exc)}
            )
    return records


def write_convert_artifacts(document_dir: Path, result: Any) -> list[dict[str, Any]]:
    raw = redact_secrets(result_to_dict(result))
    write_json(document_dir / "convert.raw.json", raw)

    markdown = get_attr(result, "markdown")
    html = get_attr(result, "html")
    document_json = get_attr(result, "json")
    chunks = get_attr(result, "chunks")
    if isinstance(markdown, str):
        (document_dir / "convert.md").write_text(markdown, encoding="utf-8")
    if isinstance(html, str):
        (document_dir / "convert.html").write_text(html, encoding="utf-8")
    if document_json is not None:
        write_json(document_dir / "convert.document.json", document_json)
    if chunks is not None:
        write_json(document_dir / "convert.chunks.json", chunks)
    return write_images(document_dir, get_attr(result, "images", {}))


def parse_extraction(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        return None
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return raw


def write_extract_artifacts(document_dir: Path, result: Any) -> Any:
    write_json(document_dir / "extract.raw.json", redact_secrets(result_to_dict(result)))
    extraction = parse_extraction(get_attr(result, "extraction_schema_json"))
    write_json(document_dir / "extract.extraction.json", extraction)
    return extraction


def convert_missing_fields(result: Any) -> list[str]:
    missing = []
    for field in REQUIRED_CONVERT_OUTPUTS:
        if get_attr(result, field) is None:
            missing.append(field)
    if not get_attr(result, "checkpoint_id"):
        missing.append("checkpoint_id")
    return missing


def completed_marker_is_valid(
    document_dir: Path, document: dict[str, Any], expected_config_hash: str
) -> bool:
    marker_path = document_dir / "completion.json"
    required = (
        "convert.raw.json",
        "convert.md",
        "convert.html",
        "convert.document.json",
        "convert.chunks.json",
        "extract.raw.json",
        "extract.extraction.json",
        "metadata.json",
    )
    if not marker_path.is_file() or any(not (document_dir / name).is_file() for name in required):
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    return (
        marker.get("status") == "completed"
        and marker.get("input_sha256") == document["sha256"]
        and marker.get("config_hash") == expected_config_hash
    )


def archive_partial(document_dir: Path) -> None:
    if not document_dir.exists():
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = document_dir.with_name(f"{document_dir.name}.partial-{timestamp}")
    document_dir.rename(target)
    print(f"Archived partial artifact: {target}", flush=True)


def run_one_document(
    *,
    client: Any,
    ConvertOptions: Any,
    ExtractOptions: Any,
    experiment_dir: Path,
    document: dict[str, Any],
    schema: dict[str, Any],
    sdk_version: str,
    expected_config_hash: str,
) -> dict[str, Any]:
    input_path = input_path_for(experiment_dir, document)
    documents_root = experiment_dir / "datalab" / "documents"
    final_dir = documents_root / document["document_id"]
    work_dir = documents_root / f".{document['document_id']}.working-{int(time.time())}"
    work_dir.mkdir(parents=True, exist_ok=False)

    started = time.monotonic()
    convert_started = time.monotonic()
    convert_result = client.convert(
        file_path=input_path,
        options=ConvertOptions(
            mode="accurate",
            skip_cache=True,
            output_format="markdown,html,json,chunks",
            save_checkpoint=True,
            add_block_ids=True,
            include_markdown_in_chunks=True,
            disable_image_extraction=False,
        ),
        max_polls=600,
        poll_interval=1,
    )
    convert_latency = round(time.monotonic() - convert_started, 3)
    if not get_attr(convert_result, "success", False):
        raise RuntimeError(f"Datalab Convert failed: {get_attr(convert_result, 'error')}")
    missing = convert_missing_fields(convert_result)
    if missing:
        raise RuntimeError(f"Datalab Convert response missing requested fields: {missing}")

    checkpoint_id = str(get_attr(convert_result, "checkpoint_id"))
    checkpoint_hash = sha256_bytes(checkpoint_id.encode("utf-8"))
    image_files = write_convert_artifacts(work_dir, convert_result)

    extract_started = time.monotonic()
    extract_result = client.extract(
        options=ExtractOptions(
            page_schema=json.dumps(schema, ensure_ascii=False),
            checkpoint_id=checkpoint_id,
            mode="accurate",
            skip_cache=True,
        ),
        max_polls=600,
        poll_interval=1,
    )
    extract_latency = round(time.monotonic() - extract_started, 3)
    if not get_attr(extract_result, "success", False):
        raise RuntimeError(f"Datalab Extract failed: {get_attr(extract_result, 'error')}")
    extraction = write_extract_artifacts(work_dir, extract_result)
    if extraction is None:
        raise RuntimeError("Datalab Extract response did not contain extraction_schema_json.")

    completed_at = utc_now()
    metadata = {
        "status": "completed",
        "completed_at": completed_at,
        "provider": "datalab",
        "sdk_version": sdk_version,
        "experiment_id": json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))[
            "experiment_id"
        ],
        "document_id": document["document_id"],
        "filename": document["filename"],
        "relative_path": document["relative_path"],
        "mime_type": document["mime_type"],
        "input_size_bytes": document["size_bytes"],
        "input_sha256": document["sha256"],
        "page_count": get_attr(convert_result, "page_count"),
        "latency_seconds": round(time.monotonic() - started, 3),
        "convert_latency_seconds": convert_latency,
        "extract_latency_seconds": extract_latency,
        "parse_quality_score": get_attr(convert_result, "parse_quality_score"),
        "convert_runtime": get_attr(convert_result, "runtime"),
        "convert_cost_breakdown": get_attr(convert_result, "cost_breakdown"),
        "extract_cost_breakdown": get_attr(extract_result, "cost_breakdown"),
        "versions": get_attr(convert_result, "versions"),
        "checkpoint_reuse": True,
        "checkpoint_id_sha256": checkpoint_hash,
        "config_hash": expected_config_hash,
        "config": CONFIG,
        "image_count": sum(item.get("status") == "saved" for item in image_files),
        "images": image_files,
    }
    write_json(work_dir / "metadata.json", redact_secrets(metadata))
    write_json(
        work_dir / "completion.json",
        {
            "status": "completed",
            "completed_at": completed_at,
            "input_sha256": document["sha256"],
            "config_hash": expected_config_hash,
        },
    )

    if final_dir.exists():
        archive_partial(final_dir)
    work_dir.rename(final_dir)
    return metadata


def write_failure(
    experiment_dir: Path,
    document: dict[str, Any],
    expected_config_hash: str,
    exc: Exception,
) -> Path:
    failures_dir = experiment_dir / "datalab" / "failures"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = failures_dir / f"{document['document_id']}-{timestamp}.json"
    write_json(
        path,
        {
            "status": "failed",
            "failed_at": utc_now(),
            "document_id": document["document_id"],
            "relative_path": document["relative_path"],
            "input_sha256": document["sha256"],
            "config_hash": expected_config_hash,
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
    )
    return path


def run_datalab(args: argparse.Namespace) -> int:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("python-dotenv is required to read .env") from exc
    load_dotenv(args.env_file)
    api_key = os.getenv("DATALAB_API_KEY")
    if not api_key:
        raise RuntimeError("DATALAB_API_KEY is not set in the environment or .env file.")

    try:
        from datalab_sdk import ConvertOptions, DatalabClient, ExtractOptions
    except ImportError as exc:
        raise RuntimeError("Install the Lift extra before running Datalab.") from exc

    sdk_version = importlib.metadata.version("datalab-python-sdk")
    experiment_dir = args.experiment.resolve()
    manifest = load_manifest(experiment_dir)
    schema_path = Path(CONFIG["extract_schema"])
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    expected_config_hash = config_hash(sdk_version, schema)
    client = DatalabClient(api_key=api_key, timeout=300)
    documents_root = experiment_dir / "datalab" / "documents"
    documents_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, document in enumerate(manifest["documents"]):
        stop_after_preflight = False
        final_dir = documents_root / document["document_id"]
        if args.resume and completed_marker_is_valid(final_dir, document, expected_config_hash):
            print(f"SKIP {document['relative_path']} (valid completion marker)", flush=True)
            records.append({"document_id": document["document_id"], "status": "skipped"})
            continue

        print(
            f"RUN {index + 1}/5 {document['relative_path']} sha256={document['sha256'][:16]}...",
            flush=True,
        )
        try:
            metadata = run_one_document(
                client=client,
                ConvertOptions=ConvertOptions,
                ExtractOptions=ExtractOptions,
                experiment_dir=experiment_dir,
                document=document,
                schema=schema,
                sdk_version=sdk_version,
                expected_config_hash=expected_config_hash,
            )
            records.append(
                {
                    "document_id": document["document_id"],
                    "status": "completed",
                    "page_count": metadata["page_count"],
                    "latency_seconds": metadata["latency_seconds"],
                }
            )
            print(
                f"DONE {document['relative_path']} pages={metadata['page_count']} "
                f"latency={metadata['latency_seconds']}s",
                flush=True,
            )
        except Exception as exc:
            failure_path = write_failure(
                experiment_dir, document, expected_config_hash, exc
            )
            records.append(
                {
                    "document_id": document["document_id"],
                    "status": "failed",
                    "error": str(exc),
                    "failure_path": str(failure_path),
                }
            )
            print(
                f"FAILED {document['relative_path']}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if index == 0:
                print("Preflight failed; stopping before the remaining four documents.", flush=True)
                stop_after_preflight = True

        write_json(
            experiment_dir / "datalab" / "run_summary.json",
            {
                "experiment_id": manifest["experiment_id"],
                "updated_at": utc_now(),
                "sdk_version": sdk_version,
                "config_hash": expected_config_hash,
                "records": records,
            },
        )
        if stop_after_preflight:
            break

    failed = [record for record in records if record["status"] == "failed"]
    return 1 if failed else 0


def normalize_visible_markdown(markdown: str) -> str:
    """Return a comparison-only plain-text view of Markdown.

    This is intentionally not treated as ground truth. It is only used to
    measure how strongly the two providers agree with one another.
    """

    value = IMAGE_PATTERN.sub(lambda match: f" {match.group(1)} ", markdown)
    value = LINK_PATTERN.sub(lambda match: f" {match.group(1)} ", value)
    value = HTML_TAG_PATTERN.sub(" ", value)
    value = html.unescape(value)
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[`*_#>|~]", " ", value)
    return " ".join(value.split())


def markdown_table_metrics(lines: list[str], markdown: str) -> dict[str, int]:
    pipe_separator_indexes = {
        index for index, line in enumerate(lines) if PIPE_SEPARATOR_PATTERN.match(line)
    }
    pipe_table_count = len(pipe_separator_indexes)
    pipe_row_indexes: set[int] = set()
    for separator_index in pipe_separator_indexes:
        index = separator_index - 1
        while index >= 0 and "|" in lines[index] and lines[index].strip():
            pipe_row_indexes.add(index)
            index -= 1
        index = separator_index + 1
        while index < len(lines) and "|" in lines[index] and lines[index].strip():
            pipe_row_indexes.add(index)
            index += 1
    return {
        "html_table_count": len(HTML_TABLE_PATTERN.findall(markdown)),
        "html_table_row_count": len(HTML_ROW_PATTERN.findall(markdown)),
        "pipe_table_count": pipe_table_count,
        "pipe_table_row_count": len(pipe_row_indexes),
    }


def repeated_nonempty_line_count(lines: list[str]) -> int:
    normalized = [
        " ".join(unicodedata.normalize("NFKC", line).casefold().split())
        for line in lines
        if len(line.strip()) >= 12
    ]
    return sum(count - 1 for count in Counter(normalized).values() if count > 1)


def markdown_metrics(markdown_path: Path) -> dict[str, Any]:
    markdown = markdown_path.read_text(encoding="utf-8")
    lines = markdown.splitlines()
    visible_text = normalize_visible_markdown(markdown)
    words = WORD_PATTERN.findall(visible_text)
    headings = HEADING_PATTERN.findall(markdown)
    images = IMAGE_PATTERN.findall(markdown)
    without_images = IMAGE_PATTERN.sub(" ", markdown)
    normalized_body = normalize_visible_markdown(without_images)

    direct_asset_count = 0
    basename_asset_count = 0
    missing_asset_references: list[str] = []
    for _, target in images:
        if "://" in target or target.startswith("data:"):
            continue
        unescaped_target = target.replace("%20", " ")
        direct_path = markdown_path.parent / Path(unescaped_target.replace("/", os.sep))
        if direct_path.is_file():
            direct_asset_count += 1
            basename_asset_count += 1
            continue
        basename = Path(unescaped_target).name
        if basename and any(markdown_path.parent.rglob(basename)):
            basename_asset_count += 1
        else:
            missing_asset_references.append(target)

    heading_levels = Counter(len(prefix) for prefix, _ in headings)
    repeated_alt_count = sum(
        bool(normalized_alt) and normalized_alt in normalized_body
        for alt, _ in images
        if (normalized_alt := normalize_visible_markdown(alt))
    )
    metrics: dict[str, Any] = {
        "char_count": len(markdown),
        "word_count": len(words),
        "nonempty_line_count": sum(bool(line.strip()) for line in lines),
        "heading_count": len(headings),
        "heading_levels": {
            str(level): heading_levels.get(level, 0) for level in range(1, 7)
        },
        "max_heading_char_count": max((len(text) for _, text in headings), default=0),
        "image_reference_count": len(images),
        "directly_resolved_image_reference_count": direct_asset_count,
        "basename_resolved_image_reference_count": basename_asset_count,
        "missing_image_references": missing_asset_references,
        "image_alt_repeated_in_body_count": repeated_alt_count,
        "unordered_list_item_count": len(UNORDERED_LIST_PATTERN.findall(markdown)),
        "ordered_list_item_count": len(ORDERED_LIST_PATTERN.findall(markdown)),
        "raw_html_tag_count": len(HTML_TAG_PATTERN.findall(markdown)),
        "repeated_nonempty_line_count": repeated_nonempty_line_count(lines),
    }
    metrics.update(markdown_table_metrics(lines, markdown))
    return metrics


def markdown_agreement(chandra_markdown: str, datalab_markdown: str) -> dict[str, float]:
    chandra_text = normalize_visible_markdown(chandra_markdown)
    datalab_text = normalize_visible_markdown(datalab_markdown)
    chandra_tokens = set(WORD_PATTERN.findall(chandra_text))
    datalab_tokens = set(WORD_PATTERN.findall(datalab_text))
    union = chandra_tokens | datalab_tokens
    intersection = chandra_tokens & datalab_tokens
    return {
        "normalized_sequence_ratio": round(
            SequenceMatcher(None, chandra_text, datalab_text, autojunk=False).ratio(), 4
        ),
        "unique_token_jaccard": round(len(intersection) / len(union), 4) if union else 1.0,
    }


def validate_provider_input(
    experiment_dir: Path, provider: str, document: dict[str, Any]
) -> dict[str, Any]:
    document_dir = experiment_dir / provider / "documents" / document["document_id"]
    metadata_path = document_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("input_sha256") != document["sha256"]:
        raise RuntimeError(
            f"{provider} checksum mismatch for {document['relative_path']}: "
            f"{metadata.get('input_sha256')} != {document['sha256']}"
        )
    return metadata


def write_markdown_metrics_csv(path: Path, documents: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for document in documents:
        for provider, metrics in document["providers"].items():
            row = {
                "document_id": document["document_id"],
                "relative_path": document["relative_path"],
                "provider": provider,
                "page_count": metrics["page_count"],
                "char_count": metrics["char_count"],
                "word_count": metrics["word_count"],
                "nonempty_line_count": metrics["nonempty_line_count"],
                "heading_count": metrics["heading_count"],
                "max_heading_char_count": metrics["max_heading_char_count"],
                "image_reference_count": metrics["image_reference_count"],
                "directly_resolved_image_reference_count": metrics[
                    "directly_resolved_image_reference_count"
                ],
                "basename_resolved_image_reference_count": metrics[
                    "basename_resolved_image_reference_count"
                ],
                "image_alt_repeated_in_body_count": metrics[
                    "image_alt_repeated_in_body_count"
                ],
                "html_table_count": metrics["html_table_count"],
                "pipe_table_count": metrics["pipe_table_count"],
                "unordered_list_item_count": metrics["unordered_list_item_count"],
                "ordered_list_item_count": metrics["ordered_list_item_count"],
                "raw_html_tag_count": metrics["raw_html_tag_count"],
                "normalized_sequence_ratio": document["agreement"][
                    "normalized_sequence_ratio"
                ],
                "unique_token_jaccard": document["agreement"]["unique_token_jaccard"],
            }
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_side_by_side(
    path: Path, experiment_dir: Path, documents: list[dict[str, Any]]
) -> None:
    sections: list[str] = []
    for document in documents:
        columns: list[str] = []
        for provider in ("chandra2", "datalab"):
            relative_markdown_path = document["providers"][provider]["markdown_path"]
            markdown = (experiment_dir / relative_markdown_path).read_text(encoding="utf-8")
            columns.append(
                f'<article><h3>{html.escape(provider)}</h3>'
                f'<a href="../{quote(relative_markdown_path, safe="/")}">Open Markdown</a>'
                f'<pre>{html.escape(markdown)}</pre></article>'
            )
        sections.append(
            f'<section><h2>{html.escape(document["relative_path"])}</h2>'
            f'<p><code>{document["document_id"]}</code> · '
            f'<a href="../{quote(document["source_path"], safe="/")}">Open source</a></p>'
            f'<div class="columns">{"".join(columns)}</div></section>'
        )

    output = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Markdown comparison</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #172033; }}
section {{ border-top: 1px solid #cad2df; padding: 24px 0; }}
.columns {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
article {{ min-width: 0; }}
pre {{ background: #f5f7fa; border: 1px solid #d9e0ea; border-radius: 8px; padding: 16px;
       white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.45; }}
@media (max-width: 900px) {{ .columns {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Chandra2 vs Datalab: raw Markdown side by side</h1>
<p>This viewer shows raw Markdown, not rendered Markdown. Compare each column with its source.</p>
{"".join(sections)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")


def compare_markdown(args: argparse.Namespace) -> int:
    experiment_dir = args.experiment.resolve()
    manifest = load_manifest(experiment_dir)
    records: list[dict[str, Any]] = []

    for document in manifest["documents"]:
        provider_records: dict[str, dict[str, Any]] = {}
        markdown_text: dict[str, str] = {}
        for provider, filename in (("chandra2", "result.md"), ("datalab", "convert.md")):
            metadata = validate_provider_input(experiment_dir, provider, document)
            markdown_path = (
                experiment_dir / provider / "documents" / document["document_id"] / filename
            )
            if not markdown_path.is_file():
                raise FileNotFoundError(markdown_path)
            metrics = markdown_metrics(markdown_path)
            metrics.update(
                {
                    "markdown_path": markdown_path.relative_to(experiment_dir).as_posix(),
                    "page_count": metadata.get("page_count"),
                }
            )
            provider_records[provider] = metrics
            markdown_text[provider] = markdown_path.read_text(encoding="utf-8")

        records.append(
            {
                "document_id": document["document_id"],
                "relative_path": document["relative_path"],
                "source_path": input_path_for(experiment_dir, document)
                .relative_to(experiment_dir)
                .as_posix(),
                "mime_type": document["mime_type"],
                "sha256": document["sha256"],
                "providers": provider_records,
                "agreement": markdown_agreement(
                    markdown_text["chandra2"], markdown_text["datalab"]
                ),
            }
        )

    output_dir = args.output_dir or experiment_dir / "comparison"
    output_dir = output_dir.resolve()
    payload = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "scope": "markdown_only",
        "generated_at": utc_now(),
        "corpus_fingerprint": manifest["corpus_fingerprint"],
        "document_count": len(records),
        "page_count": sum(
            int(record["providers"]["chandra2"].get("page_count") or 0)
            for record in records
        ),
        "agreement_warning": (
            "Inter-provider agreement measures similarity, not OCR accuracy. "
            "Accuracy requires comparison with the source or ground truth."
        ),
        "documents": records,
    }
    write_json(output_dir / "markdown_metrics.json", payload)
    write_markdown_metrics_csv(output_dir / "markdown_metrics.csv", records)
    write_markdown_side_by_side(
        output_dir / "markdown_side_by_side.html", experiment_dir, records
    )
    print(f"Wrote {output_dir / 'markdown_metrics.json'}", flush=True)
    print(f"Wrote {output_dir / 'markdown_metrics.csv'}", flush=True)
    print(f"Wrote {output_dir / 'markdown_side_by_side.html'}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    datalab = subparsers.add_parser("run-datalab", help="Run Datalab Convert + Extract")
    datalab.add_argument("--experiment", type=Path, required=True)
    datalab.add_argument("--env-file", type=Path, default=Path(".env"))
    datalab.add_argument("--resume", action="store_true")
    datalab.set_defaults(func=run_datalab)
    markdown = subparsers.add_parser(
        "compare-markdown", help="Validate paired inputs and measure Markdown artifacts"
    )
    markdown.add_argument("--experiment", type=Path, required=True)
    markdown.add_argument("--output-dir", type=Path)
    markdown.set_defaults(func=compare_markdown)
    return parser


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
