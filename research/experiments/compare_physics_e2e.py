"""Validate and compare the Physics end-to-end answer runs.

The answer runs are produced by ``src.evaluation.run_answer``. This script keeps
the comparison reproducible by validating the two retrieval JSONL inputs, then
joining their answer reports with the page/file retrieval report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_FILE_REPORT = (
    ROOT / "data/benchmark/vidore_v3/results/physics_baseline_comparison/file_level_report.json"
)
DEFAULT_ANSWER_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_e2e_bm25_vsplade"
DEFAULT_OUTPUT = DEFAULT_ANSWER_DIR / "comparison.json"

ARMS = {
    "PDF-inspector + BM25": {
        "run": ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl",
        "answer": "pdf_inspector_bm25.json",
        "file_report": "PDF-inspector + BM25",
    },
    "V-SPLADE (English query)": {
        "run": ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/vsplade_french_bm25-french_vs-english.jsonl",
        "answer": "vsplade_english_query.json",
        "file_report": "V-SPLADE (English query)",
    },
}


def _read_run(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Retrieval run not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_run(path: Path, expected_qids: set[str]) -> dict[str, Any]:
    rows = _read_run(path)
    qids = {str(row.get("qid")) for row in rows}
    missing = sorted(expected_qids - qids)
    extra = sorted(qids - expected_qids)
    if missing or extra:
        raise RuntimeError(
            f"{path}: qid mismatch; missing={missing[:3]} extra={extra[:3]}"
        )
    short = [
        str(row["qid"])
        for row in rows
        if len(row.get("chunks") or []) < 100
    ]
    if short:
        raise RuntimeError(f"{path}: fewer than 100 candidates for {short[:3]}")
    empty_context = sum(
        1
        for row in rows
        if any(not str(chunk.get("text") or "").strip() for chunk in (row.get("chunks") or [])[:10])
    )
    return {
        "records": len(rows),
        "unique_qids": len(qids),
        "candidate_count": 100,
        "rows_with_empty_top10_context": empty_context,
    }


def _percent(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-report", type=Path, default=DEFAULT_FILE_REPORT)
    parser.add_argument("--answer-dir", type=Path, default=DEFAULT_ANSWER_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    file_report = json.loads(args.file_report.read_text(encoding="utf-8"))
    methods = file_report["methods"]
    expected_qids = {
        str(row["qid"])
        for row in next(iter(methods.values()))["per_query"]
    }
    if len(expected_qids) != 302:
        raise RuntimeError(f"Expected 302 qids in file report, got {len(expected_qids)}")

    arms: dict[str, dict[str, Any]] = {}
    for name, config in ARMS.items():
        run_check = _validate_run(config["run"], expected_qids)
        answer_path = args.answer_dir / config["answer"]
        if not answer_path.is_file():
            raise FileNotFoundError(
                f"Answer report not found: {answer_path}. Run src.evaluation.run_answer first."
            )
        answer = json.loads(answer_path.read_text(encoding="utf-8"))
        if int(answer.get("scored", 0)) != 302:
            raise RuntimeError(f"{answer_path}: expected 302 scored answers")
        arms[name] = {
            "retrieval_run": str(config["run"]),
            "answer_report": str(answer_path),
            "run_validation": run_check,
            "page_file_retrieval": {
                key: value
                for key, value in methods[config["file_report"]].items()
                if key not in {"per_query", "definitions"}
            },
            "answer": answer,
        }

    comparison = {
        "dataset": "vidore_v3/physics",
        "language": "french",
        "queries": 302,
        "context_protocol": {
            "retrieval_depth": 10,
            "context_source": "PDF-inspector page text",
            "shared_across_arms": True,
            "max_context_chars": 12000,
            "generator": "deepseek/deepseek-v4-flash",
            "judge": "openai/gpt-4o",
        },
        "file_report": str(args.file_report),
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Physics end-to-end QA: PDF-inspector/BM25 vs V-SPLADE",
        "",
        "Both arms use the top 10 retrieved pages and the same PDF-inspector page "
        "text, generator, judge and 12,000-character context budget.",
        "",
        "| Arm | Correct | Correct + Partial | Context hit | Context recall | "
        "QA wall time (s) | Generation sum (s) | Judge sum (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in arms.items():
        answer = payload["answer"]
        lines.append(
            f"| {name} | {_percent(answer['accuracy'])} | "
            f"{_percent(answer['accuracy_credited'])} | "
            f"{_percent(answer['context_hit'])} | "
            f"{_percent(answer['context_recall'])} | "
            f"{answer['seconds']:.1f} | {answer['generation_seconds_sum']:.1f} | "
            f"{answer['judge_seconds_sum']:.1f} |"
        )
    lines += [
        "",
        "The JSON report also contains page/file retrieval metrics, per-arm "
        "validation, modality splits, multi-gold results and per-question answer "
        "records through the linked answer reports.",
    ]
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: payload["answer"] for name, payload in arms.items()}, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
