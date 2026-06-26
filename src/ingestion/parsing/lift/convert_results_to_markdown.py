"""Convert Lift/Datalab result JSON files into OmniDocBench markdown files."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

META_SUFFIXES = ("_meta", "_citations")


def main() -> int:
    args = parse_args()
    args.output_md_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in sorted(args.results_dir.glob("*.json")):
        if path.name in {"manifest.json", "schema.json"}:
            continue

        payload = load_json(path)
        if not isinstance(payload, dict):
            continue

        input_info = payload.get("input") or {}
        file_name = input_info.get("file_name")
        extraction = payload.get("extraction")
        if not file_name:
            continue

        md_path = args.output_md_dir / Path(file_name).with_suffix(".md").name
        markdown = extraction_to_markdown(extraction) if isinstance(extraction, dict) else "\n"
        md_path.write_text(markdown, encoding="utf-8")
        written += 1

    print(f"Wrote {written} markdown files to {args.output_md_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Lift result JSON to markdown files.")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_md_dir", type=Path)
    return parser.parse_args()


def extraction_to_markdown(extraction: dict[str, Any]) -> str:
    markdown = clean_text(extraction.get("markdown"))
    if markdown:
        return markdown.rstrip() + "\n"

    parts: list[str] = []
    main_text = clean_text(extraction.get("main_text"))
    title = clean_text(extraction.get("title"))
    if title and not normalize_for_prefix(main_text).startswith(normalize_for_prefix(title)):
        parts.append(f"# {title.lstrip('#').strip()}")

    if main_text:
        parts.append(main_text)

    for table in extraction.get("tables") or []:
        text = table_to_markdown(table)
        if text:
            parts.append(text)

    for figure in extraction.get("figures") or []:
        text = clean_text(figure.get("caption")) if isinstance(figure, dict) else clean_text(figure)
        if text:
            parts.append(text)

    for formula in extraction.get("formulas") or []:
        text = clean_text(formula)
        if text:
            parts.append(text)

    if not parts:
        for key, value in extraction.items():
            if key.endswith(META_SUFFIXES):
                continue
            text = clean_text(value)
            if text:
                parts.append(text)

    return "\n\n".join(parts).strip() + "\n"


def table_to_markdown(table: Any) -> str:
    if isinstance(table, str):
        return table.strip()
    if not isinstance(table, dict):
        return clean_text(table)

    caption = clean_text(table.get("caption"))
    content = clean_text(table.get("content"))
    parts = []
    if caption:
        parts.append(f"Table: {caption}")
    if content:
        parts.append(content)
    return "\n\n".join(parts)


def normalize_for_prefix(value: str) -> str:
    return " ".join(clean_text(value).lstrip("#").split()).lower()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
