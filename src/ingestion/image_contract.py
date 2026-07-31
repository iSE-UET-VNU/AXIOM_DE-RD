"""Normalize Markdown image assets into the ingestion figure contract."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_IMAGE_BLOCK_TYPES = {"image", "picture", "figure", "diagram"}


def normalize_markdown_images(
    extraction: Any,
    markdown: str,
    image_files: list[dict[str, Any]],
    source_blocks: list[dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    """Return an extraction mapping whose image assets are normalized figures."""

    normalized = dict(extraction) if isinstance(extraction, dict) else {}
    figures = normalized.get("figures")
    if isinstance(figures, list) and figures:
        if len(figures) == len(image_files):
            for figure, asset in zip(figures, image_files, strict=True):
                source_ref = _figure_source_ref(figure)
                if source_ref:
                    asset.setdefault("source_ref", source_ref)
        return normalized, 0

    candidates = _MARKDOWN_IMAGE.findall(markdown or "")
    used_assets: set[int] = set()
    used_blocks: set[str] = set()
    normalized_figures: list[dict[str, Any]] = []
    for alt_text, target in candidates:
        asset_index = _match_asset(target, image_files, used_assets)
        if asset_index is None:
            continue
        used_assets.add(asset_index)
        asset = image_files[asset_index]
        source_ref = _match_source_block(
            alt_text,
            target,
            source_blocks,
            used_blocks,
        )
        if source_ref:
            used_blocks.add(source_ref)
        else:
            source_ref = f"asset:{asset.get('name') or Path(target).name}"
        asset["source_ref"] = source_ref
        normalized_figures.append(
            {
                "caption": "",
                "caption_citations": [],
                "description": alt_text.strip(),
                "description_citations": [source_ref],
            }
        )

    normalized.setdefault("document_type", None)
    normalized.setdefault("language", None)
    normalized.setdefault("title", None)
    normalized.setdefault("title_citations", [])
    normalized.setdefault("tables", [])
    normalized.setdefault("formulas", [])
    normalized.setdefault("formulas_citations", [])
    if not normalized.get("main_text") and markdown:
        normalized["main_text"] = markdown
        normalized.setdefault("main_text_citations", [])
    normalized["figures"] = normalized_figures
    return normalized, len(normalized_figures)


def _match_asset(
    target: str,
    image_files: list[dict[str, Any]],
    used_assets: set[int],
) -> int | None:
    target_name = Path(target.replace("\\", "/")).name.casefold()
    for index, asset in enumerate(image_files):
        if index in used_assets:
            continue
        names = {
            Path(str(asset.get(field) or "").replace("\\", "/")).name.casefold()
            for field in ("name", "path")
        }
        if target_name in names:
            return index
    return None


def _match_source_block(
    alt_text: str,
    target: str,
    source_blocks: list[dict[str, Any]],
    used_blocks: set[str],
) -> str:
    target_name = Path(target.replace("\\", "/")).name.casefold()
    alt = alt_text.strip().casefold()
    for block in source_blocks:
        component_id = str(block.get("component_id") or "")
        if (
            not component_id
            or component_id in used_blocks
            or str(block.get("type") or "").casefold() not in _IMAGE_BLOCK_TYPES
        ):
            continue
        searchable = " ".join(
            str(block.get(field) or "") for field in ("html", "text")
        ).casefold()
        if target_name in searchable or (alt and alt in searchable):
            return component_id
    return ""


def _figure_source_ref(figure: Any) -> str:
    if not isinstance(figure, dict):
        return ""
    for field in ("description_citations", "caption_citations"):
        citations = figure.get(field)
        if isinstance(citations, list):
            for citation in citations:
                if isinstance(citation, str) and citation.strip():
                    return citation.strip()
    return str(figure.get("source_ref") or "").strip()
