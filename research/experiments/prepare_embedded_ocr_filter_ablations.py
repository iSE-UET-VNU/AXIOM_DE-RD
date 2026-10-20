"""Prepare three independent embedded-image OCR filtering ablations.

Each output remains a single page text field: accepted OCR is inserted directly
after its original ``[Image: ...]`` marker.  The policies are fixed, qrel-free,
and independent: image geometry only, token cleaning only, or whole-result
text-quality gating only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.ocr_benchmark_embedded_images import IMAGE_MARKER, norm_label  # noqa: E402


TOKEN = re.compile(r"[^\W_]+(?:[._/+:-][^\W_]+)*", re.UNICODE)
SHORT_KEEP = {
    # Common English/French function words.
    "a", "i", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it", "me", "no", "of", "on", "or", "so", "to", "up", "us", "we",
    "à", "au", "ce", "de", "du", "en", "et", "il", "je", "la", "le", "ne", "ou", "se", "un", "y",
    # Frequent scientific/industrial units. Matching is lower-cased.
    "a", "v", "w", "n", "j", "k", "m", "s", "g", "pa", "hz", "kg", "cm", "mm", "nm", "kv", "kw", "mw", "ma", "db",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def geometry_by_occurrence(
    pages: list[dict[str, Any]], dataset_root: Path,
) -> tuple[dict[tuple[str, str], dict[str, float]], Counter[str]]:
    """Return max displayed bbox/page-area for each (page, resource label)."""
    import pymupdf

    pages_by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        if IMAGE_MARKER.search(str(page.get("text") or "")):
            pages_by_path[(dataset_root / str(page["relative_path"])).resolve()].append(page)
    output: dict[tuple[str, str], dict[str, float]] = {}
    stats: Counter[str] = Counter()
    for source_path, document_pages in pages_by_path.items():
        try:
            with pymupdf.open(source_path) as document:
                for row in document_pages:
                    try:
                        page = document[int(row["page_index"])]
                        page_area = max(float(page.rect.width * page.rect.height), 1.0)
                        for image in page.get_images(full=True):
                            label = norm_label(str(image[7]))
                            rects = page.get_image_rects(int(image[0]))
                            max_area = max((max(0.0, float(rect.width * rect.height)) for rect in rects), default=0.0)
                            key = (str(row["page_id"]), label)
                            current = output.get(key, {"area_ratio": 0.0})
                            if max_area / page_area > current["area_ratio"]:
                                output[key] = {"area_ratio": max_area / page_area, "page_area": page_area, "display_area": max_area}
                        stats["pages_ok"] += 1
                    except Exception:
                        stats["page_errors"] += 1
        except Exception:
            stats["document_errors"] += 1
    stats["mapped_page_labels"] = len(output)
    return output, stats


def clean_short_tokens(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        core = value.strip("._/+:-")
        if not core:
            return ""
        if any(character.isdigit() for character in core):
            return value
        letters = "".join(character for character in core if character.isalpha())
        if len(letters) >= 3:
            return value
        if letters.lower() in SHORT_KEEP:
            return value
        if letters.isupper():
            return value
        return ""

    cleaned = TOKEN.sub(replace, text)
    lines = []
    for line in cleaned.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        # Do not retain lines reduced to punctuation by token deletion.
        if any(character.isalnum() for character in line):
            lines.append(line)
    return "\n".join(lines).strip()


def quality_features(text: str) -> dict[str, float | int]:
    visible = [character for character in text if not character.isspace()]
    tokens = TOKEN.findall(text)
    lexical = []
    technical = []
    noisy_short = []
    for token in tokens:
        core = token.strip("._/+:-")
        letters = "".join(character for character in core if character.isalpha())
        if len(letters) >= 3:
            lexical.append(token)
        if any(character.isdigit() for character in core) or letters.lower() in SHORT_KEEP or letters.isupper():
            technical.append(token)
        if letters and len(letters) <= 2 and token not in technical:
            noisy_short.append(token)
    alnum = sum(character.isalnum() for character in visible)
    return {
        "visible_chars": len(visible),
        "tokens": len(tokens),
        "lexical_tokens": len(lexical),
        "technical_tokens": len(technical),
        "noisy_short_tokens": len(noisy_short),
        "alnum_ratio": alnum / len(visible) if visible else 0.0,
        "noisy_short_ratio": len(noisy_short) / len(tokens) if tokens else 1.0,
    }


def quality_accept(text: str) -> bool:
    features = quality_features(text)
    enough_content = (
        int(features["lexical_tokens"]) >= 3
        or int(features["technical_tokens"]) >= 2
    )
    return (
        int(features["visible_chars"]) >= 8
        and float(features["alnum_ratio"]) >= 0.65
        and float(features["noisy_short_ratio"]) <= 0.60
        and enough_content
    )


def merge_policy(
    pages: list[dict[str, Any]], results: list[dict[str, Any]],
    transform: Callable[[dict[str, Any]], str], policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queues: dict[str, dict[str, deque[str]]] = defaultdict(lambda: defaultdict(deque))
    stats: Counter[str] = Counter()
    for row in sorted(results, key=lambda item: (str(item.get("page_id")), str(item.get("image_label")), int(item.get("occurrence_on_page", 0)))):
        raw = str(row.get("text") or "").strip()
        if row.get("status") != "ok" or not raw:
            continue
        accepted = transform(row).strip()
        stats["ocr_nonempty_input"] += 1
        stats["raw_words"] += len(raw.split())
        if accepted:
            stats["accepted_images"] += 1
            stats["accepted_words"] += len(accepted.split())
        else:
            stats["rejected_images"] += 1
        queues[str(row["page_id"])][norm_label(str(row["image_label"]))].append(accepted)

    output: list[dict[str, Any]] = []
    changed_pages = 0
    for page in pages:
        page_queues = queues.get(str(page["page_id"]), {})
        def replace(match: re.Match[str]) -> str:
            queue = page_queues.get(norm_label(match.group(1).strip()))
            if not queue:
                return match.group(0)
            accepted = queue.popleft()
            return f"{match.group(0)}\n{accepted}" if accepted else match.group(0)
        updated = dict(page)
        updated["text"] = IMAGE_MARKER.sub(replace, str(page.get("text") or ""))
        updated["embedded_ocr_policy"] = policy
        if updated["text"] != page.get("text"):
            changed_pages += 1
        output.append(updated)
    return output, {**dict(stats), "changed_pages": changed_pages, "output_pages": len(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-pages", type=Path, required=True)
    parser.add_argument("--ocr-results", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-area-ratio", type=float, default=0.005)
    parser.add_argument("--min-pixel-area", type=int, default=1024)
    parser.add_argument("--min-pixel-side", type=int, default=16)
    args = parser.parse_args()

    started = perf_counter()
    pages = read_jsonl(args.native_pages)
    results = read_jsonl(args.ocr_results)
    geometry, geometry_stats = geometry_by_occurrence(pages, args.dataset_root)

    def size_only(row: dict[str, Any]) -> str:
        key = (str(row["page_id"]), norm_label(str(row["image_label"])))
        item = geometry.get(key)
        # Conservative fallback: missing geometry is retained, not rejected.
        if item is None:
            return str(row.get("text") or "")
        width, height = int(row.get("width") or 0), int(row.get("height") or 0)
        too_small_display = float(item["area_ratio"]) < args.min_area_ratio
        too_low_resolution = width * height < args.min_pixel_area or min(width, height) < args.min_pixel_side
        return "" if too_small_display or too_low_resolution else str(row.get("text") or "")

    policies: dict[str, Callable[[dict[str, Any]], str]] = {
        "size_filter_only": size_only,
        "token_cleaning_only": lambda row: clean_short_tokens(str(row.get("text") or "")),
        "quality_gate_only": lambda row: str(row.get("text") or "") if quality_accept(str(row.get("text") or "")) else "",
    }
    summaries: dict[str, Any] = {}
    for name, transform in policies.items():
        prepared, summary = merge_policy(pages, results, transform, name)
        target = args.output_dir / name / "pages.jsonl"
        write_jsonl(target, prepared)
        summaries[name] = {**summary, "pages_path": str(target.resolve())}
    report = {
        "input": {"native_pages": str(args.native_pages.resolve()), "ocr_results": str(args.ocr_results.resolve()), "pages": len(pages), "ocr_records": len(results)},
        "fixed_rules": {"size_filter_only": {"min_display_area_ratio": args.min_area_ratio, "min_pixel_area": args.min_pixel_area, "min_pixel_side": args.min_pixel_side, "unknown_geometry": "keep"}, "token_cleaning_only": "drop non-whitelisted alphabetic tokens of length <=2; keep numeric/alphanumeric, common function words, units, uppercase abbreviations", "quality_gate_only": {"min_visible_chars": 8, "min_alnum_ratio": 0.65, "max_noisy_short_ratio": 0.60, "content": ">=3 lexical tokens OR >=2 technical tokens"}},
        "geometry": dict(geometry_stats), "policies": summaries, "seconds": perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
