"""Run AXIOM's OmniDocBench subset through the Datalab hosted Lift API.

This mirrors the sibling lift repo flow:
1. read OmniDocBench subset images, usually from a manifest;
2. write manifest.json and schema.json into the run output directory;
3. call Datalab extraction for each image;
4. write one raw JSON result per image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import csv
import json
import os
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data/raw/omnidocbench_subset"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "schemas/omnidoc_markdown.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/lift_outputs/omnidocbench_subset_markdown_schema_100"


@dataclass(frozen=True)
class SelectedImage:
    index: int
    path: Path
    metadata: dict[str, Any]

    def to_manifest_item(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "file_name": self.path.name,
            "path": str(self.path),
            "manifest_metadata": self.metadata,
        }


def main() -> int:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    output_dir = resolve_path(args.output_dir)
    schema_path = resolve_path(args.schema)
    manifest_path = resolve_path(args.manifest) if args.manifest else dataset_root / "manifest.csv"
    if not manifest_path.exists():
        manifest_path = None

    images_dir = dataset_root / "images"
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    images = list_images(images_dir)
    selected = select_images(
        images=images,
        manifest_path=manifest_path,
        file_names=args.file_name,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    schema = load_json(schema_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "manifest.json", [item.to_manifest_item() for item in selected])
    dump_json(output_dir / "schema.json", schema)

    print(f"Selected {len(selected)} / {len(images)} OmniDocBench images")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"Schema: {output_dir / 'schema.json'}")

    if args.dry_run:
        for item in selected[:10]:
            print(f"{item.index:03d} {item.path.name}")
        if len(selected) > 10:
            print(f"... {len(selected) - 10} more")
        return 0

    if not os.getenv(args.api_key_env):
        raise SystemExit(f"{args.api_key_env} is not set.")

    try:
        from datalab_sdk import DatalabClient, ExtractOptions
    except ImportError as exc:
        raise SystemExit(
            "Missing datalab-python-sdk. Install it with: pip install datalab-python-sdk"
        ) from exc

    client = DatalabClient()
    options = ExtractOptions(page_schema=json.dumps(schema), mode=args.mode)

    for item in selected:
        output_path = output_dir / f"{item.index:03d}_{item.path.stem}.json"
        if output_path.exists() and not args.overwrite:
            print(f"[{item.index:03d}/{len(selected)}] skip {item.path.name}")
            continue

        print(f"[{item.index:03d}/{len(selected)}] extract {item.path.name}")
        payload = extract_image(client, options, item)
        dump_json(output_path, payload)

        if args.sleep > 0:
            time.sleep(args.sleep)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lift API on AXIOM OmniDocBench subset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest CSV/JSON. Defaults to <dataset-root>/manifest.csv when present.",
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--file-name",
        action="append",
        default=[],
        help="Specific image filename to process. Can be passed multiple times.",
    )
    parser.add_argument("--mode", default="balanced")
    parser.add_argument("--api-key-env", default="DATALAB_API_KEY")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_images(
    images: list[Path],
    manifest_path: Path | None,
    file_names: list[str],
    sample_size: int,
    seed: int,
) -> list[SelectedImage]:
    if manifest_path and manifest_path.exists():
        manifest_rows = read_manifest(manifest_path)
        return choose_manifest_files(images, manifest_rows)

    if file_names:
        manifest_rows = [{"file_name": name} for name in file_names]
        return choose_manifest_files(images, manifest_rows)

    sample = choose_sample(images, sample_size, seed)
    return [SelectedImage(index=idx, path=path, metadata={}) for idx, path in enumerate(sample, start=1)]


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise SystemExit(f"Manifest CSV has no header: {path}")
            if "basename" not in reader.fieldnames and "file_name" not in reader.fieldnames:
                raise SystemExit(f"Manifest CSV must contain basename or file_name: {path}")
            rows = []
            for idx, row in enumerate(reader, start=1):
                file_name = (row.get("basename") or row.get("file_name") or "").strip()
                if not file_name:
                    raise SystemExit(f"Manifest CSV row #{idx} is missing basename/file_name")
                rows.append({**row, "file_name": file_name})
            return rows

    payload = load_json(path)
    if not isinstance(payload, list):
        raise SystemExit(f"Manifest JSON must be a list: {path}")
    rows = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"Manifest item #{idx} must be an object")
        file_name = str(item.get("file_name") or item.get("basename") or "").strip()
        if not file_name:
            raise SystemExit(f"Manifest item #{idx} is missing file_name")
        rows.append({**item, "file_name": file_name})
    return rows


def choose_manifest_files(images: list[Path], manifest_rows: list[dict[str, Any]]) -> list[SelectedImage]:
    by_name = {image.name: image for image in images}
    selected = []
    missing = []
    for idx, row in enumerate(manifest_rows, start=1):
        file_name = str(row["file_name"])
        image = by_name.get(file_name)
        if image is None:
            missing.append(file_name)
            continue
        selected.append(SelectedImage(index=idx, path=image, metadata=row))

    if missing:
        missing_text = "\n  ".join(missing)
        raise SystemExit(f"Manifest file(s) not found in dataset images:\n  {missing_text}")
    return selected


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def choose_sample(images: list[Path], sample_size: int, seed: int) -> list[Path]:
    if sample_size >= len(images):
        return images
    rng = random.Random(seed)
    return sorted(rng.sample(images, sample_size))


def extract_image(client: Any, options: Any, item: SelectedImage) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = client.extract(str(item.path), options=options)
        raw_json = get_attr(result, "extraction_schema_json")
        extraction = parse_extraction(raw_json)
        return {
            "input": item.to_manifest_item(),
            "status": get_attr(result, "status"),
            "page_count": get_attr(result, "page_count"),
            "latency_seconds": round(time.monotonic() - started, 3),
            "extraction": extraction,
        }
    except Exception as exc:
        return {
            "input": item.to_manifest_item(),
            "status": "error",
            "latency_seconds": round(time.monotonic() - started, 3),
            "error": str(exc),
            "extraction": None,
        }


def parse_extraction(raw_json: Any) -> Any:
    if raw_json is None:
        return None
    if isinstance(raw_json, (dict, list)):
        return raw_json
    try:
        return json.loads(str(raw_json))
    except json.JSONDecodeError:
        return raw_json


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
