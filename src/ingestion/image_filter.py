"""Conservative image filtering applied to provider-neutral parsed data."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re
import shutil
from typing import Any

from ..models import ParsedData


MIN_PAGE_AREA_RATIO = 0.01
MAX_OCR_LOGO_DESCRIPTION_CHARS = 120
FILTERED_IMAGE_DIRNAME = "filtered_images"
OCR_SOURCE_FORMATS = frozenset(
    {"pdf", "png", "jpg", "jpeg", "gif", "webp", "tif", "tiff"}
)

_FILTER_VERSION = "image-filter-v1"
_SHORT_DESCRIPTION_PATTERNS = {
    keyword: re.compile(rf"\b{keyword}\b", re.IGNORECASE)
    for keyword in ("logo", "icon")
}
_COMPONENT_PAGE_PATTERN = re.compile(r"^/page/(\d+)/")
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def apply_image_filters(parsed: ParsedData) -> dict[str, Any]:
    """Remove low-value figures while retaining source assets for audit.

    The function updates ``rows[].extraction.figures`` and the aligned
    ``metadata.image_files`` list in place. Parser-owned raw artifacts remain
    untouched in ``images``; retained image files are copied to the sibling
    ``filtered_images`` directory for downstream use.
    """

    source_format = str(parsed.source_format or "").strip().lower().lstrip(".")
    is_ocr_input = source_format in OCR_SOURCE_FORMATS
    original_assets = _image_assets(parsed.metadata.get("image_files"))
    expected_figure_count = _figure_count(parsed)
    positional_assets_aligned = len(original_assets) == expected_figure_count
    assets_by_source_ref = _assets_by_source_ref(original_assets)
    used_asset_indexes: set[int] = set()

    kept_assets: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    geometry_unavailable: list[dict[str, Any]] = []
    original_figure_count = 0
    kept_figure_count = 0

    for row_index, row in enumerate(parsed.rows):
        if not isinstance(row, dict):
            continue
        extraction = row.get("extraction")
        if not isinstance(extraction, dict):
            continue
        figures = extraction.get("figures")
        if not isinstance(figures, list):
            continue

        source_blocks = [
            block
            for block in row.get("source_blocks", [])
            if isinstance(block, dict)
        ] if isinstance(row.get("source_blocks"), list) else []
        blocks_by_id = {
            str(block["component_id"]): block
            for block in source_blocks
            if block.get("component_id")
        }
        page_bounds = _page_bounds(source_blocks)
        kept_figures: list[Any] = []

        for figure_index, figure in enumerate(figures):
            original_index = original_figure_count
            original_figure_count += 1
            figure_mapping = figure if isinstance(figure, dict) else {}
            source_ref = _figure_source_ref(figure_mapping)
            block = blocks_by_id.get(source_ref, {})
            area_ratio, geometry_reason, page_area_source = _area_ratio(
                figure_mapping,
                block,
                source_ref,
                page_bounds,
            )
            description = str(
                figure_mapping.get("description")
                if figure_mapping.get("description") is not None
                else figure if not isinstance(figure, dict) else ""
            ).strip()
            asset_index, asset = _resolve_asset(
                source_ref,
                original_index,
                original_assets,
                assets_by_source_ref,
                used_asset_indexes,
                allow_positional=positional_assets_aligned,
            )
            if asset_index is not None:
                used_asset_indexes.add(asset_index)

            reasons: list[str] = []
            if area_ratio is not None and area_ratio < MIN_PAGE_AREA_RATIO:
                reasons.append("below_min_page_area_ratio")
            if is_ocr_input and len(description) <= MAX_OCR_LOGO_DESCRIPTION_CHARS:
                reasons.extend(
                    f"short_ocr_{keyword}_description"
                    for keyword, pattern in _SHORT_DESCRIPTION_PATTERNS.items()
                    if pattern.search(description)
                )

            audit_item = {
                "original_index": original_index,
                "row_index": row_index,
                "figure_index": figure_index,
                "source_ref": source_ref or None,
                "area_ratio": area_ratio,
                "page_area_source": page_area_source,
                "description_length": len(description),
            }
            if geometry_reason is not None:
                geometry_unavailable.append(
                    {
                        **audit_item,
                        "reason": geometry_reason,
                    }
                )

            if reasons:
                dropped.append(
                    {
                        **audit_item,
                        "reasons": reasons,
                        "asset": dict(asset) if asset is not None else None,
                    }
                )
                continue

            kept_figures.append(figure)
            kept_figure_count += 1
            kept_assets.append(
                _copy_kept_asset(
                    _kept_asset(asset, source_ref, original_index)
                )
            )

        extraction["figures"] = kept_figures

    report = {
        "version": _FILTER_VERSION,
        "source_format": source_format,
        "ocr_input": is_ocr_input,
        "rules": {
            "min_page_area_ratio": MIN_PAGE_AREA_RATIO,
            "max_ocr_logo_description_chars": MAX_OCR_LOGO_DESCRIPTION_CHARS,
            "short_description_keywords": list(_SHORT_DESCRIPTION_PATTERNS),
            "keyword_match": "standalone_case_insensitive_word",
            "filtered_image_dirname": FILTERED_IMAGE_DIRNAME,
        },
        "before_count": original_figure_count,
        "kept_count": kept_figure_count,
        "dropped_count": len(dropped),
        "copied_count": sum(
            asset.get("filter_copy_status") == "copied"
            for asset in kept_assets
        ),
        "copy_failed_count": sum(
            asset.get("filter_copy_status") == "failed"
            for asset in kept_assets
        ),
        "geometry_unavailable_count": len(geometry_unavailable),
        "dropped": dropped,
        "geometry_unavailable": geometry_unavailable,
    }
    parsed.metadata["image_files"] = kept_assets
    parsed.metadata["figure_count"] = kept_figure_count
    parsed.metadata["image_count"] = sum(
        asset.get("status") == "saved" for asset in kept_assets
    )
    parsed.metadata["image_filtering"] = report
    return report


def _copy_kept_asset(asset: dict[str, Any]) -> dict[str, Any]:
    copied = dict(asset)
    if copied.get("status") != "saved" or not copied.get("path"):
        copied["filter_copy_status"] = "not_saved"
        return copied

    source_path = Path(str(copied["path"]))
    if source_path.parent.name.casefold() != "images":
        copied["filter_copy_status"] = "unsupported_source_directory"
        return copied
    if not source_path.is_file():
        copied["filter_copy_status"] = "source_missing"
        return copied

    try:
        filtered_dir = source_path.parent.parent / FILTERED_IMAGE_DIRNAME
        target_path = filtered_dir / source_path.name
        filtered_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    except OSError as exc:
        copied["filter_copy_status"] = "failed"
        copied["filter_copy_error"] = str(exc)
        return copied

    copied["original_path"] = str(source_path)
    copied["path"] = str(target_path)
    copied["filter_copy_status"] = "copied"
    return copied


def _image_assets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _figure_count(parsed: ParsedData) -> int:
    count = 0
    for row in parsed.rows:
        if not isinstance(row, dict):
            continue
        extraction = row.get("extraction")
        figures = extraction.get("figures") if isinstance(extraction, dict) else None
        if isinstance(figures, list):
            count += len(figures)
    return count


def _assets_by_source_ref(
    assets: list[dict[str, Any]],
) -> dict[str, list[int]]:
    by_source_ref: dict[str, list[int]] = {}
    for index, asset in enumerate(assets):
        source_ref = str(asset.get("source_ref") or "").strip()
        if source_ref:
            by_source_ref.setdefault(source_ref, []).append(index)
    return by_source_ref


def _resolve_asset(
    source_ref: str,
    original_index: int,
    assets: list[dict[str, Any]],
    assets_by_source_ref: dict[str, list[int]],
    used_indexes: set[int],
    *,
    allow_positional: bool,
) -> tuple[int | None, dict[str, Any] | None]:
    for index in assets_by_source_ref.get(source_ref, []):
        if index not in used_indexes:
            return index, assets[index]
    if (
        allow_positional
        and 0 <= original_index < len(assets)
        and original_index not in used_indexes
    ):
        return original_index, assets[original_index]
    return None, None


def _kept_asset(
    asset: dict[str, Any] | None,
    source_ref: str,
    original_index: int,
) -> dict[str, Any]:
    if asset is None:
        return {
            "name": None,
            "path": None,
            "status": "unavailable",
            "source_ref": source_ref or f"figure:{original_index}",
        }
    kept = dict(asset)
    if source_ref:
        kept.setdefault("source_ref", source_ref)
    return kept


def _figure_source_ref(figure: dict[str, Any]) -> str:
    for field in ("description_citations", "caption_citations"):
        citations = figure.get(field)
        if isinstance(citations, list):
            for citation in citations:
                if isinstance(citation, str) and citation.strip():
                    return citation.strip()

    source_ref = figure.get("source_ref")
    if isinstance(source_ref, str) and source_ref.strip():
        return source_ref.strip()

    source_refs = figure.get("source_refs")
    if isinstance(source_refs, list):
        for item in source_refs:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                component_id = item.get("component_id")
                if isinstance(component_id, str) and component_id.strip():
                    return component_id.strip()
    return ""


def _area_ratio(
    figure: dict[str, Any],
    block: dict[str, Any],
    source_ref: str,
    page_bounds: dict[int, tuple[tuple[float, float, float, float], str]],
) -> tuple[float | None, str | None, str | None]:
    geometry = block or figure
    bbox = _numeric_box(geometry.get("bbox"))
    if bbox is None:
        return None, "missing_or_invalid_bbox", None

    page_box = _numeric_box(geometry.get("page_box"))
    page_area_source = "page_box"
    if page_box is None:
        page = _page_number(geometry, source_ref)
        if page is None and len(page_bounds) == 1:
            page = next(iter(page_bounds))
        page_entry = page_bounds.get(page) if page is not None else None
        if page_entry is None:
            return None, "missing_or_invalid_page_area", None
        page_box, page_area_source = page_entry

    figure_area = _box_area(bbox)
    page_area = _box_area(page_box)
    if figure_area is None:
        return None, "missing_or_invalid_bbox", page_area_source
    if page_area is None:
        return None, "missing_or_invalid_page_area", page_area_source
    return figure_area / page_area, None, page_area_source


def _page_bounds(
    blocks: Iterable[dict[str, Any]],
) -> dict[int, tuple[tuple[float, float, float, float], str]]:
    explicit: dict[int, list[tuple[float, float, float, float]]] = {}
    inferred: dict[int, list[tuple[float, float, float, float]]] = {}
    for block in blocks:
        page = _page_number(block, str(block.get("component_id") or ""))
        if page is None:
            continue
        page_box = _numeric_box(block.get("page_box"))
        if page_box is not None:
            explicit.setdefault(page, []).append(page_box)
        bbox = _numeric_box(block.get("bbox"))
        if bbox is not None:
            inferred.setdefault(page, []).append(bbox)

    result: dict[int, tuple[tuple[float, float, float, float], str]] = {}
    for page in set(explicit) | set(inferred):
        if explicit.get(page):
            result[page] = (_union_boxes(explicit[page]), "page_box")
        elif inferred.get(page):
            result[page] = (_union_boxes(inferred[page]), "inferred_block_union")
    return result


def _page_number(block: dict[str, Any], source_ref: str) -> int | None:
    value = block.get("page")
    if isinstance(value, bool):
        value = None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    match = _COMPONENT_PAGE_PATTERN.match(source_ref)
    return int(match.group(1)) if match else None


def _numeric_box(value: Any) -> tuple[float, float, float, float] | None:
    parts: list[Any]
    if isinstance(value, (list, tuple)):
        parts = list(value)
    elif isinstance(value, str):
        parts = _NUMBER_PATTERN.findall(value)
    else:
        return None
    if len(parts) != 4:
        return None
    try:
        box = tuple(float(part) for part in parts)
    except (TypeError, ValueError):
        return None
    if _box_area(box) is None:
        return None
    return box


def _box_area(
    box: tuple[float, float, float, float],
) -> float | None:
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return None
    return width * height


def _union_boxes(
    boxes: Iterable[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    values = list(boxes)
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )
