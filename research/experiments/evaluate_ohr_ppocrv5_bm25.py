"""Evaluate PP-OCRv5 replacement text with the frozen merged BM25 protocol."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
OCR_PAGES = ROOT / "data/work/ohr_replacement_20260920/ppocrv5_mobile/pages.jsonl"
OUT = ROOT / "data/work/ohr_replacement_20260920/ppocrv5_mobile/retrieval_bm25"
RUNNER_PATH = ROOT / "research/experiments/run_ohr_replacement_all_arms.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("merged_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load merged benchmark runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not OCR_PAGES.exists():
        raise FileNotFoundError(f"Missing completed PP-OCRv5 pages: {OCR_PAGES}")
    runner = load_runner()
    started = time.perf_counter()
    queries, qrels, languages = runner.load_inputs()
    page_texts, _ = runner.load_page_corpora()
    pp_rows = runner.read_jsonl(OCR_PAGES)
    pp_text = {str(row["page_id"]): str(row.get("text") or "") for row in pp_rows}
    if len(pp_text) != 409:
        raise RuntimeError(f"Expected 409 OHR pages, got {len(pp_text)}")
    # Exact ablation: old Physics/Industrial text is frozen; every OHR page
    # uses PDF-inspector native text or PP-OCRv5 full-page text in place of
    # the previous selective Tesseract text. Embedded-image OCR is excluded
    # so the delta is attributable to the full-page OCR replacement alone.
    base = dict(page_texts["baseline_bm25"])
    base.update(pp_text)
    rows = runner.build_bm25_arm(base, queries, languages)
    ohr_rows = [row for row in rows if str(row["query_id"]).startswith("ohrbench::")]
    original = [row for row in runner.read_jsonl(runner.OUTPUT_DEFAULT / "runs/baseline_bm25.jsonl") if str(row["query_id"]).startswith("ohrbench::")]
    report = {
        "protocol": {"corpus": "frozen 5,091 merged pages", "queries": "65 OHR", "retrieval": "same hierarchical BM25/Kf=3/up-to-200/top-20 as official baseline", "ablation": "replace selective full-page Tesseract with PP-OCRv5 mobile; no embedded-image OCR"},
        "ppocrv5_mobile": {**runner.metric_rows(ohr_rows, qrels, 10), **runner.metric_rows(ohr_rows, qrels, 20)},
        "tesseract_baseline": {**runner.metric_rows(original, qrels, 10), **runner.metric_rows(original, qrels, 20)},
        "wall_seconds": time.perf_counter() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    runner.write_jsonl(OUT / "run.jsonl", ohr_rows)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# OHR PP-OCRv5 mobile replacement: BM25", "", "| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 |", "|---|---:|---:|---:|---:|---:|"]
    for name, value in (("Tesseract baseline", report["tesseract_baseline"]), ("PP-OCRv5 mobile", report["ppocrv5_mobile"])):
        lines.append(f"| {name} | {100*value['page_recall@10']:.2f}% | {value['ndcg@10']:.2f} | {100*value['page_recall@20']:.2f}% | {value['ndcg@20']:.2f} | {100*value['file_recall@3']:.2f}% |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
