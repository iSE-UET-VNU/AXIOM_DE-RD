"""Build the upload bundle consumed by ``webAI_ColVec_visual_arm_physics.ipynb``.

The bundle contains no credentials, model weights, or generated page images.
Colab downloads the model and renders the images after upload.  The benchmark
parquet files are the compact local Physics export; the original three raw
image-bearing parquet shards are intentionally not copied because the notebook
does not read them and they are over 1 GB.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/work/webai_colvec_physics_bundle.zip"
NOTEBOOK = ROOT / "webAI_ColVec_visual_arm_physics.ipynb"
PDF_SOURCE = ROOT / "data/raw/benchmarks/vidore_v3/vidore_v3_physics/pdfs"
BENCHMARK_SOURCE = ROOT / "data/benchmark/vidore_v3/physics"
BM25_SOURCE = (
    ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"
    / "bm25_french.jsonl"
)
SCOPE_SOURCE = (
    ROOT / "data/benchmark/vidore_v3/exports"
    / "physics_file_retrieval_for_colembed"
)
RENDERER_SOURCE = ROOT / "research/experiments/render_page_images.py"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _copy_files(source_dir: Path, destination: Path, pattern: str) -> list[Path]:
    files = sorted(source_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files in {source_dir}")
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        shutil.copy2(source, destination / source.name)
    return files


def _archive_tracked_repo(staging: Path) -> None:
    archive_path = staging.parent / "tracked_repo.tar"
    with archive_path.open("wb") as archive_file:
        subprocess.run(
            ["git", "archive", "--format=tar", "HEAD"],
            cwd=ROOT,
            stdout=archive_file,
            check=True,
        )
    with tarfile.open(archive_path, "r") as archive:
        archive.extractall(staging, filter="data")
    archive_path.unlink()


def _write_kdl_pool(destination: Path) -> None:
    rows = _read_jsonl(BM25_SOURCE)
    if len(rows) != 302:
        raise ValueError(f"Expected 302 BM25 rows, found {len(rows)}")
    queries: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row["qid"])
        candidates = [str(chunk["chunk_id"]) for chunk in row.get("chunks", [])]
        if len(candidates) < 100:
            raise ValueError(f"KDL/BM25 pool for {qid} has fewer than 100 pages")
        queries[qid] = {"candidates": candidates[:100]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "dataset": "vidore_v3/physics",
                "language": "french",
                "source": "cached bm25_french.jsonl; diagnostic only",
                "queries": queries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_bundle_manifest(staging: Path, pdf_count: int) -> None:
    manifest = {
        "format": "axiom.webai_colvec_physics_colab_bundle.v1",
        "dataset": "vidore_v3/physics",
        "corpus": {"queries": 302, "files": pdf_count, "pages": 1674},
        "contains": {
            "source_code": True,
            "notebook": NOTEBOOK.name,
            "physics_pdfs": pdf_count,
            "compact_benchmark_parquet": True,
            "page_renderer": "research/experiments/render_page_images.py",
            "diagnostic_kdl_pool": "data/benchmark/vidore_v3/results/physics_KDL_pool.json",
            "file_scope_exports": "data/benchmark/vidore_v3/exports/physics_file_retrieval_for_colembed",
            "model_weights": False,
            "credentials": False,
            "pre_rendered_images": False,
        },
        "after_upload": [
            "Open webAI_ColVec_visual_arm_physics.ipynb in Colab.",
            "Run cells 1-24; cell 8 renders 1,674 page images.",
            "Download physics_colvec_export.zip from cell 24.",
            "Use the file-scope JSONL exports to filter ColVec scores by Kf.",
        ],
        "notes": [
            "The compact benchmark parquet is used by src.evaluation.benchmarks.ViDoreV3.",
            "The KDL pool is only used by notebook cell 21 for a diagnostic comparison.",
            "Cells 25 and later require a separate local slate evaluator and are not needed to produce ColVec scores.",
        ],
    }
    (staging / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build(output: Path) -> Path:
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(NOTEBOOK)
    pdfs = sorted(PDF_SOURCE.glob("*.pdf"))
    if len(pdfs) != 42:
        raise ValueError(f"Expected 42 Physics PDFs, found {len(pdfs)}")
    for required in (BENCHMARK_SOURCE, BM25_SOURCE, SCOPE_SOURCE, RENDERER_SOURCE):
        if not required.exists():
            raise FileNotFoundError(required)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="webai_colvec_bundle_") as temp:
        staging = Path(temp) / "bundle"
        staging.mkdir()
        _archive_tracked_repo(staging)

        shutil.copy2(NOTEBOOK, staging / NOTEBOOK.name)
        shutil.copy2(RENDERER_SOURCE, staging / "research/experiments/render_page_images.py")
        # HEAD can lag behind the working-tree parquet compatibility fix.  The
        # notebook's benchmark loader must carry the tested fallback as well.
        current_base = ROOT / "src/evaluation/benchmarks/base.py"
        shutil.copy2(current_base, staging / "src/evaluation/benchmarks/base.py")

        pdf_destination = staging / "data/raw/benchmarks/vidore_v3_physics"
        _copy_files(PDF_SOURCE, pdf_destination, "*.pdf")
        metadata = PDF_SOURCE / "metadata.csv"
        if metadata.is_file():
            shutil.copy2(metadata, pdf_destination / metadata.name)

        _copy_tree(BENCHMARK_SOURCE, staging / "data/benchmark/vidore_v3/physics")
        _write_kdl_pool(
            staging / "data/benchmark/vidore_v3/results/physics_KDL_pool.json"
        )
        _copy_tree(
            SCOPE_SOURCE,
            staging / "data/benchmark/vidore_v3/exports/physics_file_retrieval_for_colembed",
        )
        _write_bundle_manifest(staging, len(pdfs))

        with zipfile.ZipFile(
            output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as bundle:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(staging).as_posix())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = build(args.output)
    print(f"created {path}")
    print(f"size_bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
