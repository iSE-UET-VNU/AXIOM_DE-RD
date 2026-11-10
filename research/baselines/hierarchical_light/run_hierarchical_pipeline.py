"""Run the light preparation + hierarchical BM25 retrieval pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ondemand.bench import BENCH, fingerprint, work
from ondemand.hier_replica import run as run_hierarchical
from ondemand.light_ocr import merge as merge_ppocr
from ondemand.light_prep import run as run_light_prep


ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    prep = config["preparation"]
    retrieval = config["retrieval"]
    if retrieval["method"] != "hierarchical_bm25":
        raise ValueError("This runner only implements retrieval.method=hierarchical_bm25")
    if retrieval["file_k"] != 3:
        raise ValueError("The reproduction config must use file_k=3")
    if retrieval["page_k"] != 20:
        raise ValueError("The reproduction config must use page_k=20")
    if prep["ocr"]["language"] != "fra+eng" or prep["ocr"]["dpi"] != 200 or prep["ocr"]["psm"] != 3:
        raise ValueError("The current light-preparation reproduction uses fra+eng, 200 DPI, PSM 3")
    return config


def run(config_path: Path, skip_prep: bool = False) -> dict:
    config = load_config(config_path)
    prep = config["preparation"]
    retrieval = config["retrieval"]
    tag = config["run"]["tag"]
    prep_dir = work("light_prep", bench=BENCH)
    pages_path = prep_dir / prep["pages_source"]
    ppocr = prep.get("ppocr", {"enabled": False})

    if ppocr.get("enabled", False):
        base_pages = prep_dir / "pages_ocr_sparse.jsonl"
        if not skip_prep and prep["run"] and not (prep["reuse_existing"] and base_pages.exists()):
            run_light_prep(bench=BENCH, workers=int(prep["workers"]))
        if not (prep["reuse_existing"] and pages_path.exists()):
            zip_path = ROOT / ppocr["zip"]
            if not zip_path.exists():
                raise FileNotFoundError(f"Missing PP-OCRv5 result bundle: {zip_path}")
            pages_path = merge_ppocr(bench=BENCH, results=zip_path)
    elif not skip_prep and prep["run"]:
        if not (prep["reuse_existing"] and pages_path.exists()):
            pages_path = run_light_prep(bench=BENCH, workers=int(prep["workers"]))
    if not pages_path.exists():
        raise FileNotFoundError(f"Missing preparation artifact: {pages_path}")

    payload = run_hierarchical(
        bench=BENCH,
        pages_name=pages_path.name,
        tag=tag,
        file_k=int(retrieval["file_k"]),
        page_k=int(retrieval["page_k"]),
        bm25_weight=float(retrieval["page_bm25_weight"]),
        file_direct=float(retrieval["file_direct_weight"]),
        parent_weight=float(retrieval["parent_file_weight"]),
    )
    result_dir = work("results", tag, bench=BENCH)
    manifest = {
        "config": config,
        "config_path": str(config_path),
        "benchmark_root": str(BENCH),
        "bundle_fingerprint": fingerprint(BENCH),
        "pages_path": str(pages_path),
        "result_dir": str(result_dir),
        "retrieval": payload,
    }
    (result_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "hierarchical_light_config.json")
    parser.add_argument("--skip-prep", action="store_true", help="Reuse the existing preparation artifact")
    args = parser.parse_args()
    manifest = run(args.config, skip_prep=args.skip_prep)
    metrics = manifest["retrieval"]["metrics"]
    print(json.dumps({"result_dir": manifest["result_dir"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
