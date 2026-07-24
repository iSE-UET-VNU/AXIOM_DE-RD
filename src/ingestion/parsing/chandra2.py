"""Chandra2 document provider backed by self-hosted inference."""

from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ...models import DataObject, ParsedData
from ...utils.paths import portable_path

CHANDRA2_EXTENSIONS = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff"}
)

TABLE_OCR_PROMPT = """
OCR this image as exactly one HTML table.
Return only one complete <table>...</table>; no prose, Markdown, JSON, or outer div.
Reconstruct the complete logical row and column grid from the visible borders.
A missing internal horizontal border means a cell spans rows: use rowspan.
A missing internal vertical border means a cell spans columns: use colspan.
Place each merged cell at its upper-left grid position and emit its content once.
Do not emit td/th cells for positions covered by an earlier rowspan or colspan.
Preserve visible text and genuinely empty cells.
Use only table, thead, tbody, tr, th, td, b, i, br, span, sup, sub, and math.
""".strip()


@dataclass(frozen=True)
class Chandra2Config:
    """Runtime and artifact settings for Chandra2 inference."""

    method: str = "vllm"
    batch_size: int = 28
    max_workers: int = 4
    max_output_tokens: int = 12384
    max_retries: int = 6
    include_images: bool = True
    include_headers_footers: bool = False
    save_raw_outputs: bool = True
    refine_tables: bool = False
    table_prompt: str = TABLE_OCR_PROMPT
    table_max_output_tokens: int = 4096
    table_crop_margin_ratio: float = 0.02
    table_crop_min_short_side: int = 1536
    table_crop_max_long_side: int = 3072
    table_crop_max_pixels: int = 3072 * 2048
    output_dir: str | None = "data/work/chandra2"
    project_root: str | None = None

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None) -> "Chandra2Config":
        config = config or {}
        method = _inference_method(config.get("method", "vllm"))
        return cls(
            method=method,
            batch_size=_positive_int(
                config,
                "batch_size",
                1 if method == "hf" else 28,
            ),
            max_workers=_positive_int(config, "max_workers", 4),
            max_output_tokens=_positive_int(config, "max_output_tokens", 12384),
            max_retries=_non_negative_int(config, "max_retries", 6),
            include_images=bool(config.get("include_images", True)),
            include_headers_footers=bool(
                config.get("include_headers_footers", False)
            ),
            save_raw_outputs=bool(config.get("save_raw_outputs", True)),
            refine_tables=bool(config.get("refine_tables", False)),
            table_prompt=_non_empty_str(
                config.get("table_prompt", TABLE_OCR_PROMPT),
                "table_prompt",
            ),
            table_max_output_tokens=_positive_int(
                config, "table_max_output_tokens", 4096
            ),
            table_crop_margin_ratio=_non_negative_float(
                config, "table_crop_margin_ratio", 0.02
            ),
            table_crop_min_short_side=_positive_int(
                config, "table_crop_min_short_side", 1536
            ),
            table_crop_max_long_side=_positive_int(
                config, "table_crop_max_long_side", 3072
            ),
            table_crop_max_pixels=_positive_int(
                config, "table_crop_max_pixels", 3072 * 2048
            ),
            output_dir=_optional_str(config.get("output_dir", "data/work/chandra2")),
            project_root=_optional_str(config.get("project_root")),
        )


@dataclass(frozen=True)
class _ChandraRuntime:
    load_file: Callable[[str, dict[str, Any]], list[Any]]
    inference_manager: Callable[..., Any]
    batch_input_item: Callable[..., Any]


class Chandra2Provider:
    """Convert PDF and image pages to ordered Markdown with Chandra2."""

    provider_name = "chandra2"
    supported_extensions = CHANDRA2_EXTENSIONS

    def __init__(
        self,
        config: Chandra2Config,
        *,
        _runtime_loader: Callable[[], _ChandraRuntime] | None = None,
    ) -> None:
        self.config = config
        self._runtime_loader = _runtime_loader or _load_runtime
        self._runtime: _ChandraRuntime | None = None
        self._manager: Any = None

    def parse_file(self, path: str | Path, data_object: DataObject) -> ParsedData:
        file_path = Path(path)
        if file_path.suffix.lower() not in self.supported_extensions:
            raise RuntimeError(
                f"Chandra2 does not support file type: {file_path.suffix}"
            )

        started = time.monotonic()
        runtime = self._get_runtime()
        pages = runtime.load_file(str(file_path), {})
        if not pages:
            raise RuntimeError("Chandra2 could not load any pages from the document.")

        results: list[Any] = []
        for start in range(0, len(pages), self.config.batch_size):
            page_batch = pages[start : start + self.config.batch_size]
            batch = [
                runtime.batch_input_item(image=image, prompt_type="ocr_layout")
                for image in page_batch
            ]
            generate_options: dict[str, Any] = {
                "include_images": self.config.include_images,
                "include_headers_footers": self.config.include_headers_footers,
                "max_output_tokens": self.config.max_output_tokens,
            }
            if self.config.method == "vllm":
                generate_options.update(
                    max_workers=self.config.max_workers,
                    max_retries=self.config.max_retries,
                )
            generated = list(self._get_manager().generate(batch, **generate_options))
            if len(generated) != len(batch):
                raise RuntimeError(
                    "Chandra2 returned a different number of results than input pages."
                )
            failed_pages = [
                start + index + 1
                for index, result in enumerate(generated)
                if bool(getattr(result, "error", False))
            ]
            if failed_pages:
                pages_text = ", ".join(str(page) for page in failed_pages)
                raise RuntimeError(f"Chandra2 failed to parse page(s): {pages_text}")
            results.extend(generated)

        page_payloads, raw_block_count = _page_payloads(results)
        page_markdown = [
            str(getattr(result, "markdown", "") or "") for result in results
        ]
        page_clean_html = [
            str(getattr(result, "html", "") or "") for result in results
        ]
        bundle_dir = _output_bundle_dir(
            self.config,
            file_path,
            data_object,
            results,
        )
        table_refinement = _refine_table_blocks(
            self.config,
            runtime,
            self._get_manager(),
            pages,
            page_payloads,
            page_markdown,
            page_clean_html,
            bundle_dir,
        )
        markdown = "\n\n".join(page_markdown)
        latency_seconds = round(time.monotonic() - started, 3)
        first_pass_token_count = sum(
            int(getattr(result, "token_count", 0) or 0) for result in results
        )
        token_count = first_pass_token_count + int(
            table_refinement["token_count"]
        )
        source_blocks = _source_blocks_from_pages(page_payloads)
        reading_order_complete = bool(source_blocks) and (
            len(source_blocks) == raw_block_count
            and all(page.get("blocks") for page in page_payloads)
        )
        extraction = _build_extraction(markdown, source_blocks)

        saved_images = _write_images(
            self.config,
            bundle_dir,
            results,
        )
        image_files = _align_image_files(
            extraction.get("figures", []),
            source_blocks,
            results,
            saved_images,
        )
        for page_index, page in enumerate(page_payloads):
            page["image_files"] = [
                image["name"]
                for image in saved_images
                if image["page"] == page_index and image["status"] == "saved"
            ]
        raw_output_paths = _write_raw_outputs(
            self.config,
            bundle_dir,
            file_path,
            data_object,
            markdown,
            results,
            page_clean_html,
            page_payloads,
            source_blocks,
            extraction,
            image_files,
            reading_order_complete,
            latency_seconds,
            token_count,
            table_refinement,
        )
        label_counts = Counter(
            str(block.get("raw_label") or block.get("type") or "Block")
            for block in source_blocks
        )

        return ParsedData(
            object_id=data_object.object_id,
            source_uri=data_object.uri,
            source_format=str(
                data_object.metadata.get("format", file_path.suffix.lstrip("."))
            ),
            rows=[
                {
                    "extraction": extraction,
                    "text": extraction["main_text"],
                    "source_blocks": source_blocks,
                    "reading_order": [
                        block["component_id"] for block in source_blocks
                    ],
                }
            ],
            text=markdown,
            metadata={
                "parser": "chandra2",
                "method": self.config.method,
                "model_name": _model_name(self.config.method),
                "page_count": len(results),
                "token_count": token_count,
                "first_pass_token_count": first_pass_token_count,
                "table_refinement": table_refinement,
                "latency_seconds": latency_seconds,
                "raw_output_path": raw_output_paths.get("result_markdown"),
                "raw_metadata_path": raw_output_paths.get("metadata"),
                "raw_chandra_outputs": raw_output_paths,
                "label_counts": dict(sorted(label_counts.items())),
                "source_block_count": len(source_blocks),
                "table_count": len(extraction["tables"]),
                "figure_count": len(extraction["figures"]),
                "formula_count": len(extraction["formulas"]),
                "image_count": sum(
                    item.get("status") == "saved" for item in image_files
                ),
                "image_files": image_files,
                "reading_order_source": (
                    "chandra2_layout" if source_blocks else "unavailable"
                ),
                "reading_order_complete": reading_order_complete,
            },
        )

    def _get_runtime(self) -> _ChandraRuntime:
        if self._runtime is None:
            self._runtime = self._runtime_loader()
        return self._runtime

    def _get_manager(self) -> Any:
        if self._manager is None:
            try:
                self._manager = self._get_runtime().inference_manager(
                    method=self.config.method
                )
            except ImportError as exc:
                if self.config.method == "hf":
                    raise RuntimeError(
                        "Chandra2 local inference requires the HuggingFace extras. "
                        'Install them with: pip install -e ".[chandra2-local]"'
                    ) from exc
                raise
        return self._manager


def _load_runtime() -> _ChandraRuntime:
    try:
        from chandra.input import load_file
        from chandra.model import InferenceManager
        from chandra.model.schema import BatchInputItem
    except ImportError as exc:
        raise RuntimeError(
            "Missing Chandra2 dependency. Install it with: pip install -e .[chandra2]"
        ) from exc
    return _ChandraRuntime(load_file, InferenceManager, BatchInputItem)


def _refine_table_blocks(
    config: Chandra2Config,
    runtime: _ChandraRuntime,
    manager: Any,
    page_images: list[Any],
    page_payloads: list[dict[str, Any]],
    page_markdown: list[str],
    page_clean_html: list[str],
    bundle_dir: Path | None,
) -> dict[str, Any]:
    """Run a focused OCR pass for each layout block labelled as a table."""

    summary: dict[str, Any] = {
        "enabled": config.refine_tables,
        "prompt_sha256": hashlib.sha256(
            config.table_prompt.encode("utf-8")
        ).hexdigest(),
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "token_count": 0,
        "records": [],
    }
    if not config.refine_tables:
        return summary

    artifact_dir = (
        bundle_dir / "table_refinement"
        if bundle_dir is not None and config.save_raw_outputs
        else None
    )
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    for page_index, (page_image, page_payload) in enumerate(
        zip(page_images, page_payloads)
    ):
        replacements: list[tuple[int, str]] = []
        page_box = _numeric_box(page_payload.get("page_box"))
        blocks = page_payload.get("blocks")
        if not isinstance(blocks, list):
            continue

        table_number = 0
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            if _normalize_label(str(block.get("label") or "")) != "Table":
                continue

            table_number += 1
            summary["attempted"] += 1
            bbox = _numeric_box(block.get("bbox"))
            record: dict[str, Any] = {
                "page": page_index,
                "block_index": block_index,
                "bbox": bbox,
                "status": "failed",
            }
            try:
                if not bbox or not page_box:
                    raise ValueError("table bbox or page_box is unavailable")
                crop_box = _crop_box_from_bbox(
                    page_image,
                    bbox,
                    page_box,
                    config.table_crop_margin_ratio,
                )
                crop = page_image.crop(crop_box)
                model_crop = _resize_table_crop(crop, config)
                record["crop_box_pixels"] = list(crop_box)
                record["model_image_size"] = list(model_crop.size)

                prefix = f"page_{page_index + 1:04d}_table_{table_number:04d}"
                if artifact_dir is not None:
                    crop_path = artifact_dir / f"{prefix}.crop.png"
                    model_crop.save(crop_path)
                    record["crop_path"] = portable_path(
                        crop_path, config.project_root
                    )

                batch = [
                    runtime.batch_input_item(
                        image=model_crop,
                        prompt=config.table_prompt,
                    )
                ]
                options: dict[str, Any] = {
                    "include_images": False,
                    "include_headers_footers": False,
                    "max_output_tokens": config.table_max_output_tokens,
                }
                if config.method == "vllm":
                    options.update(
                        max_workers=config.max_workers,
                        max_retries=config.max_retries,
                    )
                generated = list(manager.generate(batch, **options))
                if len(generated) != 1:
                    raise RuntimeError("table OCR returned no single result")
                result = generated[0]
                if bool(getattr(result, "error", False)):
                    raise RuntimeError("table OCR result is marked as failed")

                raw_response = str(getattr(result, "raw", "") or "")
                response = (
                    raw_response
                    or str(getattr(result, "html", "") or "")
                    or str(getattr(result, "markdown", "") or "")
                )
                refined_table = _extract_first_table(response)
                if not refined_table:
                    raise ValueError("table OCR did not return a complete HTML table")

                block["content"] = refined_table
                replacements.append((table_number - 1, refined_table))
                result_tokens = int(getattr(result, "token_count", 0) or 0)
                summary["token_count"] += result_tokens
                summary["succeeded"] += 1
                record.update(
                    status="succeeded",
                    token_count=result_tokens,
                    rowspan_count=_span_count(refined_table, "rowspan"),
                    colspan_count=_span_count(refined_table, "colspan"),
                )
                if artifact_dir is not None:
                    raw_path = artifact_dir / f"{prefix}.raw.html"
                    table_path = artifact_dir / f"{prefix}.table.html"
                    raw_path.write_text(response, encoding="utf-8")
                    table_path.write_text(refined_table, encoding="utf-8")
                    record["raw_path"] = portable_path(
                        raw_path, config.project_root
                    )
                    record["table_path"] = portable_path(
                        table_path, config.project_root
                    )
            except Exception as exc:
                summary["failed"] += 1
                record["error"] = f"{type(exc).__name__}: {exc}"
            summary["records"].append(record)

        if replacements:
            page_markdown[page_index] = _replace_tables(
                page_markdown[page_index], replacements
            )
            page_clean_html[page_index] = _replace_tables(
                page_clean_html[page_index], replacements
            )

    summary["complete"] = summary["failed"] == 0
    if artifact_dir is not None:
        summary["artifact_dir"] = portable_path(
            artifact_dir, config.project_root
        )
    return summary


def _crop_box_from_bbox(
    image: Any,
    bbox: list[int | float],
    page_box: list[int | float],
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    image_width, image_height = image.size
    page_x0, page_y0, page_x1, page_y1 = page_box
    page_width = float(page_x1) - float(page_x0)
    page_height = float(page_y1) - float(page_y0)
    if page_width <= 0 or page_height <= 0:
        raise ValueError("invalid page_box dimensions")

    x0, y0, x1, y1 = bbox
    left = (float(x0) - float(page_x0)) * image_width / page_width
    top = (float(y0) - float(page_y0)) * image_height / page_height
    right = (float(x1) - float(page_x0)) * image_width / page_width
    bottom = (float(y1) - float(page_y0)) * image_height / page_height
    margin = max(right - left, bottom - top) * margin_ratio
    crop_box = (
        max(0, math.floor(left - margin)),
        max(0, math.floor(top - margin)),
        min(image_width, math.ceil(right + margin)),
        min(image_height, math.ceil(bottom + margin)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError("table bbox maps to an empty crop")
    return crop_box


def _resize_table_crop(image: Any, config: Chandra2Config) -> Any:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("empty table crop")
    scale = min(
        config.table_crop_min_short_side / min(width, height),
        config.table_crop_max_long_side / max(width, height),
        math.sqrt(config.table_crop_max_pixels / (width * height)),
    )
    if abs(scale - 1.0) < 0.01:
        return image.copy()
    size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    try:
        from PIL import Image

        return image.resize(size, Image.Resampling.LANCZOS)
    except (ImportError, AttributeError):
        return image.resize(size)


def _extract_first_table(value: str) -> str | None:
    match = re.search(r"<table\b[^>]*>.*?</table\s*>", value, re.I | re.S)
    return match.group(0).strip() if match else None


def _replace_tables(value: str, replacements: list[tuple[int, str]]) -> str:
    """Replace tables by their page-local layout order."""

    by_index = dict(replacements)
    index = 0

    def replace_at_index(match: re.Match[str]) -> str:
        nonlocal index
        replacement = by_index.get(index, match.group(0))
        index += 1
        return replacement

    return re.sub(
        r"<table\b[^>]*>.*?</table\s*>",
        replace_at_index,
        value,
        flags=re.I | re.S,
    )


def _span_count(value: str, attribute: str) -> int:
    return len(re.findall(rf"\b{re.escape(attribute)}\s*=", value, re.I))


def _write_raw_outputs(
    config: Chandra2Config,
    bundle_dir: Path | None,
    file_path: Path,
    data_object: DataObject,
    markdown: str,
    results: list[Any],
    page_clean_html: list[str],
    page_payloads: list[dict[str, Any]],
    source_blocks: list[dict[str, Any]],
    extraction: dict[str, Any],
    image_files: list[dict[str, Any]],
    reading_order_complete: bool,
    latency_seconds: float,
    token_count: int,
    table_refinement: dict[str, Any],
) -> dict[str, str]:
    if not config.save_raw_outputs or bundle_dir is None:
        return {}

    pages_dir = bundle_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = bundle_dir / "result.md"
    clean_html_path = bundle_dir / "result.html"
    raw_html_path = bundle_dir / "raw.html"
    chunks_path = bundle_dir / "chunks.json"
    metadata_path = bundle_dir / "metadata.json"

    raw_sections: list[str] = []
    clean_sections: list[str] = []
    for page_index, (result, clean_html, page_payload) in enumerate(
        zip(results, page_clean_html, page_payloads),
        start=1,
    ):
        prefix = f"page_{page_index:04d}"
        raw_html = str(getattr(result, "raw", "") or "")
        raw_sections.append(
            f'<section data-page-number="{page_index}">\n{raw_html}\n</section>'
        )
        clean_sections.append(
            f'<section data-page-number="{page_index}">\n{clean_html}\n</section>'
        )
        (pages_dir / f"{prefix}.raw.html").write_text(
            raw_html,
            encoding="utf-8",
        )
        (pages_dir / f"{prefix}.clean.html").write_text(
            clean_html,
            encoding="utf-8",
        )
        (pages_dir / f"{prefix}.chunks.json").write_text(
            _json_text(page_payload),
            encoding="utf-8",
        )

    markdown_path.write_text(markdown, encoding="utf-8")
    clean_html_path.write_text("\n\n".join(clean_sections), encoding="utf-8")
    raw_html_path.write_text("\n\n".join(raw_sections), encoding="utf-8")
    chunks_path.write_text(
        _json_text(
            {
                "source_object_id": data_object.object_id,
                "input_path": str(file_path),
                "pages": page_payloads,
            }
        ),
        encoding="utf-8",
    )

    paths = {
        "result_markdown": portable_path(markdown_path, config.project_root),
        "result_html": portable_path(clean_html_path, config.project_root),
        "raw_html": portable_path(raw_html_path, config.project_root),
        "chunks": portable_path(chunks_path, config.project_root),
        "pages": portable_path(pages_dir, config.project_root),
    }
    label_counts = Counter(
        str(block.get("raw_label") or block.get("type") or "Block")
        for block in source_blocks
    )
    metadata_path.write_text(
        _json_text(
            {
                "source_object_id": data_object.object_id,
                "file_name": file_path.name,
                "method": config.method,
                "model_name": _model_name(config.method),
                "page_count": len(results),
                "token_count": token_count,
                "latency_seconds": latency_seconds,
                "label_counts": dict(sorted(label_counts.items())),
                "source_block_count": len(source_blocks),
                "table_count": len(extraction.get("tables", [])),
                "figure_count": len(extraction.get("figures", [])),
                "formula_count": len(extraction.get("formulas", [])),
                "image_files": image_files,
                "reading_order_source": (
                    "chandra2_layout" if source_blocks else "unavailable"
                ),
                "reading_order_complete": reading_order_complete,
                "table_refinement": table_refinement,
                "raw_chandra_outputs": paths,
            }
        ),
        encoding="utf-8",
    )
    paths["metadata"] = portable_path(
        metadata_path,
        config.project_root,
    )
    return paths


class _HTMLInspector(HTMLParser):
    """Collect readable text, image references, and math from model HTML."""

    _SEPARATORS = set(
        "br caption div h1 h2 h3 h4 h5 li ol p pre table td th tr ul".split()
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.images: list[dict[str, str]] = []
        self.math: list[str] = []
        self._in_math = False
        self._math_parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.casefold()
        if tag in self._SEPARATORS:
            self.parts.append(" ")
        if tag == "img":
            attributes = {key.casefold(): str(value or "") for key, value in attrs}
            alt = attributes.get("alt", "").strip()
            src = attributes.get("src", "").strip()
            self.images.append({"alt": alt, "src": src})
            if alt:
                self.parts.append(f" {alt} ")
        if tag == "math":
            self._in_math = True
            self._math_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "math" or not self._in_math:
            return
        value = _normalize_text("".join(self._math_parts))
        if value:
            self.math.append(value)
        self._in_math = False
        self._math_parts = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)
            if self._in_math:
                self._math_parts.append(data)


def _page_payloads(
    results: list[Any],
) -> tuple[list[dict[str, Any]], int]:
    pages: list[dict[str, Any]] = []
    raw_block_count = 0
    for page_number, result in enumerate(results, start=1):
        raw_blocks = list(getattr(result, "chunks", None) or [])
        raw_block_count += len(raw_blocks)
        blocks = [dict(block) for block in raw_blocks if isinstance(block, dict)]
        pages.append(
            {
                "page_number": page_number,
                "page_box": _numeric_box(getattr(result, "page_box", None)) or [],
                "token_count": int(getattr(result, "token_count", 0) or 0),
                "blocks": blocks,
                "image_files": [],
            }
        )
    return pages, raw_block_count


def _source_blocks_from_pages(
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_blocks: list[dict[str, Any]] = []
    for page in pages:
        page_index = int(page["page_number"]) - 1
        page_box = _numeric_box(page.get("page_box"))
        blocks = page.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block_index, raw_block in enumerate(blocks):
            if not isinstance(raw_block, dict):
                continue
            raw_label = str(raw_block.get("label") or "Block").strip() or "Block"
            block_type = _normalize_label(raw_label)
            html = str(raw_block.get("content") or "")
            inspected = _inspect_html(html)
            block: dict[str, Any] = {
                "component_id": (
                    f"/page/{page_index}/{block_type}/{block_index}"
                ),
                "page": page_index,
                "block_index": block_index,
                "type": block_type,
                "raw_label": raw_label,
                "text": _normalize_text("".join(inspected.parts)),
                "source": "chandra2_layout",
                "html": html,
                "section_hierarchy": {},
            }
            bbox = _numeric_box(raw_block.get("bbox"))
            if bbox:
                block["bbox"] = bbox
                block["polygon"] = _polygon_from_box(bbox)
            if page_box:
                block["page_box"] = page_box
            source_blocks.append(block)
    return source_blocks


def _build_extraction(
    markdown: str,
    source_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    title_block = next(
        (
            block
            for block in source_blocks
            if block.get("type") == "SectionHeader"
            and str(block.get("text") or "").strip()
        ),
        None,
    )
    title = str(title_block.get("text")) if title_block else None
    title_citations = [title_block["component_id"]] if title_block else []
    text = markdown.strip() or "\n\n".join(
        str(block.get("text") or "").strip()
        for block in source_blocks
        if str(block.get("text") or "").strip()
    )
    text_citations = [
        block["component_id"]
        for block in source_blocks
        if str(block.get("text") or "").strip()
    ]

    tables: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    formulas: list[str] = []
    formula_citations: list[str] = []
    used_caption_ids: set[str] = set()

    for index, block in enumerate(source_blocks):
        block_type = str(block.get("type") or "")
        html = str(block.get("html") or "")
        inspected = _inspect_html(html)
        component_id = str(block["component_id"])

        if block_type == "Table":
            caption, caption_citations = _adjacent_caption(
                source_blocks,
                index,
                prefer_next=False,
                used_caption_ids=used_caption_ids,
            )
            tables.append(
                {
                    "caption": caption,
                    "caption_citations": caption_citations,
                    "content": html or str(block.get("text") or ""),
                    "content_citations": [component_id],
                    "content_format": "html" if html else "text",
                }
            )

        is_figure = block_type in {"Image", "Figure", "Diagram"} or bool(
            inspected.images
        )
        if is_figure:
            caption, caption_citations = _adjacent_caption(
                source_blocks,
                index,
                prefer_next=True,
                used_caption_ids=used_caption_ids,
            )
            alt_text = " ".join(
                dict.fromkeys(
                    _normalize_text(image.get("alt", ""))
                    for image in inspected.images
                    if image.get("alt")
                )
            )
            description = alt_text or str(block.get("text") or "").strip()
            figures.append(
                {
                    "caption": caption,
                    "caption_citations": caption_citations,
                    "description": description,
                    "description_citations": [component_id],
                }
            )

        block_formulas = list(inspected.math)
        if block_type == "EquationBlock" and not block_formulas:
            formula_text = str(block.get("text") or "").strip()
            if formula_text:
                block_formulas.append(formula_text)
        for formula in block_formulas:
            if formula and formula not in formulas:
                formulas.append(formula)
                formula_citations.append(component_id)

    return {
        "document_type": None,
        "language": None,
        "title": title,
        "title_citations": title_citations,
        "main_text": text,
        "main_text_citations": text_citations,
        "tables": tables,
        "figures": figures,
        "formulas": formulas,
        "formulas_citations": formula_citations,
    }


def _adjacent_caption(
    blocks: list[dict[str, Any]],
    index: int,
    *,
    prefer_next: bool,
    used_caption_ids: set[str],
) -> tuple[str, list[str]]:
    candidates = (index + 1, index - 1) if prefer_next else (index - 1, index + 1)
    page = blocks[index].get("page")
    for candidate_index in candidates:
        if not 0 <= candidate_index < len(blocks):
            continue
        candidate = blocks[candidate_index]
        if candidate.get("page") != page or candidate.get("type") != "Caption":
            continue
        component_id = str(candidate.get("component_id") or "")
        if component_id in used_caption_ids:
            continue
        caption = str(candidate.get("text") or "").strip()
        if caption:
            used_caption_ids.add(component_id)
            return caption, [component_id]
    return "", []


def _output_bundle_dir(
    config: Chandra2Config,
    file_path: Path,
    data_object: DataObject,
    results: list[Any],
) -> Path | None:
    if not config.output_dir:
        return None
    has_images = config.include_images and any(
        bool(getattr(result, "images", None)) for result in results
    )
    if not config.save_raw_outputs and not has_images:
        return None
    bundle_dir = Path(config.output_dir) / (
        f"{_safe_slug(file_path.stem)}--{data_object.object_id}"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    return bundle_dir


def _write_images(
    config: Chandra2Config,
    bundle_dir: Path | None,
    results: list[Any],
) -> list[dict[str, Any]]:
    if not config.include_images:
        return []

    records: list[dict[str, Any]] = []
    used_names: set[str] = set()
    image_dir = bundle_dir / "images" if bundle_dir else None
    if image_dir:
        image_dir.mkdir(parents=True, exist_ok=True)

    for page_number, result in enumerate(results, start=1):
        for image_index, (original_name, image) in enumerate(
            dict(getattr(result, "images", None) or {}).items(),
            start=1,
        ):
            file_name = Path(str(original_name)).name
            if not file_name:
                file_name = f"page-{page_number}-image-{image_index}.webp"
            if file_name.casefold() in used_names:
                file_name = f"page-{page_number}-{image_index}-{file_name}"
            used_names.add(file_name.casefold())
            record: dict[str, Any] = {
                "page": page_number - 1,
                "original_name": str(original_name),
                "name": file_name,
                "path": None,
                "status": "not_saved",
            }
            if image_dir is not None:
                output_path = image_dir / file_name
                try:
                    image.save(output_path)
                except Exception as exc:
                    record["status"] = "save_failed"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                else:
                    record["status"] = "saved"
                    record["path"] = portable_path(
                        output_path,
                        config.project_root,
                    )
            records.append(record)
    return records


def _align_image_files(
    figures: Any,
    source_blocks: list[dict[str, Any]],
    results: list[Any],
    saved_images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(figures, list):
        return []
    blocks_by_id = {block["component_id"]: block for block in source_blocks}
    clean_refs_by_page = {
        page: _inspect_html(str(getattr(result, "html", "") or "")).images
        for page, result in enumerate(results)
    }
    image_names = {
        index: {
            Path(str(record.get("original_name") or "")).name.casefold(),
            Path(str(record.get("name") or "")).name.casefold(),
        }
        for index, record in enumerate(saved_images)
    }
    used: set[int] = set()
    aligned: list[dict[str, Any]] = []

    for figure_index, figure in enumerate(figures):
        citations = figure.get("description_citations", [])
        component_id = str(citations[0]) if citations else ""
        block = blocks_by_id.get(component_id, {})
        page = int(block.get("page", 0))
        raw_refs = _inspect_html(str(block.get("html") or "")).images
        raw_alts = {
            _normalize_text(ref["alt"]).casefold()
            for ref in raw_refs
            if ref["alt"]
        }
        candidates = {
            Path(ref["src"]).name.casefold()
            for ref in raw_refs
            if ref["src"]
        }
        candidates.update(
            Path(ref["src"]).name.casefold()
            for ref in clean_refs_by_page.get(page, [])
            if ref["src"]
            and _normalize_text(ref["alt"]).casefold() in raw_alts
        )
        if not candidates and block.get("type") in {"Image", "Figure", "Diagram"}:
            candidates.update(
                Path(ref["src"]).name.casefold()
                for ref in clean_refs_by_page.get(page, [])
                if ref["src"]
            )

        match = next(
            (
                index
                for index, record in enumerate(saved_images)
                if index not in used
                and record["page"] == page
                and candidates & image_names[index]
            ),
            None,
        )
        if match is None:
            aligned.append(
                {
                    "name": None,
                    "path": None,
                    "status": "unavailable",
                    "source_ref": component_id or f"figure:{figure_index}",
                }
            )
            continue

        used.add(match)
        record = saved_images[match]
        aligned.append(
            {
                "name": record["name"],
                "path": record["path"],
                "status": record["status"],
                "source_ref": component_id,
            }
        )

    aligned.extend(
        {
            "name": record["name"],
            "path": record["path"],
            "status": record["status"],
            "source_ref": None,
        }
        for index, record in enumerate(saved_images)
        if index not in used
    )
    return aligned


def _inspect_html(value: str) -> _HTMLInspector:
    inspector = _HTMLInspector()
    inspector.feed(value)
    inspector.close()
    return inspector


def _normalize_label(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value)
    return "".join(part[:1].upper() + part[1:] for part in parts) or "Block"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _numeric_box(value: Any) -> list[int | float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    return list(value)


def _polygon_from_box(
    box: list[int | float],
) -> list[list[int | float]]:
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _positive_int(config: dict[str, Any], name: str, default: int) -> int:
    value = int(config.get(name, default))
    if value <= 0:
        raise ValueError(f"chandra2.{name} must be greater than zero")
    return value


def _inference_method(value: Any) -> str:
    method = str(value).strip().lower().replace("-", "_")
    if method == "vllm":
        return "vllm"
    if method in {"hf", "local", "huggingface", "hugging_face"}:
        return "hf"
    raise ValueError(
        "chandra2.method must be 'vllm' or 'hf' (alias: 'local')"
    )


def _model_name(method: str) -> str:
    if method == "hf":
        return os.getenv("MODEL_CHECKPOINT", "datalab-to/chandra-ocr-2")
    return os.getenv("VLLM_MODEL_NAME", "chandra")


def _non_negative_int(config: dict[str, Any], name: str, default: int) -> int:
    value = int(config.get(name, default))
    if value < 0:
        raise ValueError(f"chandra2.{name} must be zero or greater")
    return value


def _non_negative_float(
    config: dict[str, Any],
    name: str,
    default: float,
) -> float:
    value = float(config.get(name, default))
    if value < 0:
        raise ValueError(f"chandra2.{name} must be zero or greater")
    return value


def _non_empty_str(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"chandra2.{name} must not be empty")
    return text


def _optional_str(value: Any) -> str | None:
    return str(value) if value not in {None, ""} else None


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._") or "document"
