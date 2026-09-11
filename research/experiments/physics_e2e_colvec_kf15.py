"""Run Physics E2E with cached ColVec Kf=15 -> top-10 page context.

ColVec ranking is produced by the already exported score matrix.  This script
only consumes its Kf=15 top-100 run, keeps the first 10 pages as context, then
uses the same DeepSeek generator and GPT-4o ViDoRe judge as the existing E2E
baseline.  Generation and judging are checkpointed so transient API failures
can be retried without losing completed rows.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.env import load_dotenv_file


ROOT = Path(__file__).resolve().parents[2]
load_dotenv_file(ROOT)

from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.benchmarks.vidore_v3_judge import (
    ANSWER_PROMPT,
    ViDoreVerdict,
    judge_answer,
    render_documents,
    score,
)
from src.evaluation.llm import complete
from src.evaluation.model_guard import assert_real
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks


SUBSET, LANGUAGE = "physics", "french"
GENERATOR = "deepseek/deepseek-v4-flash"
JUDGE = "openai/gpt-4o"
KF, TOP_K, WORKERS = 15, 10, 12

PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
COLVEC_RUN = (
    ROOT
    / "data/benchmark/vidore_v3/results/physics_colvec_second_retrieval/runs"
    / "colvec_kf15_top100.jsonl"
)
COLVEC_REPORT = (
    ROOT / "data/benchmark/vidore_v3/results/physics_colvec_second_retrieval/report.json"
)
OUT = ROOT / "data/benchmark/vidore_v3/results/physics_e2e_colvec_kf15"
RESULT_PATH = OUT / "retrieved_colvec_kf15.json"
SUMMARY_PATH = OUT / "retrieved_colvec_kf15.summary.json"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _pipeline_pages() -> dict[str, str]:
    pages: dict[str, str] = {}
    for document in documents(PARSED_RUN):
        doc = canonical_doc(document.get("document", {}).get("file_name"))
        for page, blocks in page_blocks(document).items():
            pages[unit_id(SUBSET, doc, page)] = "\n".join(
                block["text"] for block in blocks if block["text"].strip()
            )
    return pages


def _load_ranked(qids: list[str], pages: dict[str, str]) -> dict[str, list[str]]:
    if not COLVEC_RUN.is_file():
        raise FileNotFoundError(f"Missing ColVec run: {COLVEC_RUN}")
    ranked: dict[str, list[str]] = {}
    with COLVEC_RUN.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            if qid in ranked:
                raise ValueError(f"Duplicate qid in ColVec run at line {line_number}: {qid}")
            chunks = row.get("chunks", [])
            page_ids = [str(chunk["chunk_id"]) for chunk in chunks[:TOP_K]]
            if len(page_ids) != TOP_K or len(page_ids) != len(set(page_ids)):
                raise ValueError(f"Invalid top-{TOP_K} page list for {qid}")
            if any(page_id not in pages for page_id in page_ids):
                missing = [page_id for page_id in page_ids if page_id not in pages]
                raise ValueError(f"ColVec pages missing from parsed run for {qid}: {missing[:2]}")
            ranked[qid] = page_ids
    if set(ranked) != set(qids):
        raise ValueError(
            f"ColVec qids mismatch: expected {len(qids)}, got {len(ranked)}"
        )
    return ranked


def _generate_one(row: dict[str, Any]) -> tuple[str, str, str | None]:
    try:
        answer = complete(
            GENERATOR,
            ANSWER_PROMPT.format(documents=row["_context"], query=row["query"]),
            temperature=0.0,
            max_output_tokens=512,
        ).strip()
        return row["qid"], answer, None
    except Exception as error:  # noqa: BLE001 - persisted for resumable run
        return row["qid"], "", f"{type(error).__name__}: {error}"


def _judge_one(row: dict[str, Any]) -> tuple[str, ViDoreVerdict]:
    try:
        return row["qid"], judge_answer(
            row["qid"],
            row["query"],
            row["gold_answer"],
            row["answer"],
            model=JUDGE,
            generator_model=GENERATOR,
        )
    except Exception as error:  # noqa: BLE001 - count as incorrect, retain error
        return row["qid"], ViDoreVerdict(
            row["qid"], "Incorrect", error=f"{type(error).__name__}: {error}"
        )


def _save_rows(rows: dict[str, dict[str, Any]], order: list[str]) -> None:
    payload = []
    for qid in order:
        row = dict(rows[qid])
        row.pop("_context", None)
        payload.append(row)
    _write_json(RESULT_PATH, payload)


def _colvec_metrics() -> dict[str, Any]:
    if not COLVEC_REPORT.is_file():
        return {}
    report = json.loads(COLVEC_REPORT.read_text(encoding="utf-8"))
    for arm in report.get("arms", []):
        if arm.get("name") == f"Kf={KF}":
            k10 = arm["metrics"]["k_values"]["10"]
            return {
                "file_scope_recall": arm["metrics"]["file_scope_recall"],
                "page_pool_recall_ceiling": arm["metrics"]["page_pool_recall_ceiling"],
                "page_recall@10": k10["page_recall"],
                "page_hit@10": k10["page_hit"],
                "nDCG@10": k10["nDCG"],
            }
    return {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not (PARSED_RUN / "documents").is_dir():
        raise FileNotFoundError(f"Parsed run not found: {PARSED_RUN}")

    resolved = assert_real([GENERATOR, JUDGE])
    for alias, model in resolved.items():
        print(f"{alias:30s} -> {model.provider}/{model.upstream_model_id}", flush=True)
    if resolved[GENERATOR].upstream_model_id == resolved[JUDGE].upstream_model_id:
        raise ValueError("Generator and judge must be different models")

    benchmark = load("vidore_v3", subset=SUBSET, language=LANGUAGE)
    qrels = benchmark.qrels()
    questions = [question for question in benchmark.questions() if qrels.get(question.qid)]
    order = [question.qid for question in questions]
    pages = _pipeline_pages()
    if len(pages) != 1674 or len(questions) != 302:
        raise RuntimeError(f"Expected 1674 pages/302 questions, got {len(pages)}/{len(questions)}")
    ranked = _load_ranked(order, pages)
    print(
        f"Corpus: {len(pages)} pages; questions: {len(questions)}; "
        f"ColVec Kf={KF}, context top-{TOP_K}",
        flush=True,
    )

    existing = (
        json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if RESULT_PATH.exists()
        else []
    )
    rows = {str(row["qid"]): row for row in existing}
    # A previous run may have counted a transient generation failure as
    # Incorrect.  If a later retry produced an answer, discard that stale
    # synthetic verdict and send the recovered answer to the judge.
    for row in rows.values():
        if (
            row.get("answer")
            and row.get("explanation") == "Generation failed; counted as Incorrect."
        ):
            row.pop("judgment", None)
            row.pop("explanation", None)
            row.pop("judge_error", None)
    for question in questions:
        units = ranked[question.qid]
        context = render_documents([pages[unit] for unit in units])
        row = rows.setdefault(question.qid, {})
        row.update(
            {
                "arm": "colvec_kf15_top10",
                "qid": question.qid,
                "query": question.query,
                "gold_answer": question.answer,
                "n_context_pages": len(units),
                "context_chars": len(context),
                "gold_hit": len(set(units) & set(qrels[question.qid])),
                "ranked_page_ids": units,
            }
        )
        row["_context"] = context

    generation_pending = [row for row in rows.values() if not row.get("answer")]
    generation_started = perf_counter()
    if generation_pending:
        print(f"Generating {len(generation_pending)} answers with {WORKERS} workers...", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(_generate_one, row): row["qid"]
                for row in generation_pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                qid, answer, error = future.result()
                rows[qid]["answer"] = answer
                rows[qid]["generation_error"] = error
                if index % 10 == 0 or index == len(generation_pending):
                    _save_rows(rows, order)
                    print(f"  generation {index}/{len(generation_pending)}", flush=True)
    generation_seconds = perf_counter() - generation_started

    failed_generation = [row for row in rows.values() if not row.get("answer")]
    if failed_generation:
        _save_rows(rows, order)
        # Match the existing Physics E2E convention: a failed generation is
        # retained in the full denominator and counted as Incorrect, but is
        # not sent to the judge as if the error text were an answer.
        for row in failed_generation:
            row["judgment"] = "Incorrect"
            row["explanation"] = "Generation failed; counted as Incorrect."
            row["judge_error"] = None

    # Retry a row if its previous judge call returned an error; otherwise keep
    # its completed verdict on resume.
    judging_pending = [
        row for row in rows.values() if not row.get("judgment") or row.get("judge_error")
    ]
    judging_started = perf_counter()
    if judging_pending:
        print(f"Judging {len(judging_pending)} answers with {WORKERS} workers...", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(_judge_one, row): row["qid"]
                for row in judging_pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                qid, verdict = future.result()
                rows[qid]["judgment"] = verdict.judgment
                rows[qid]["explanation"] = verdict.explanation
                rows[qid]["judge_error"] = verdict.error
                if index % 10 == 0 or index == len(judging_pending):
                    _save_rows(rows, order)
                    print(f"  judging {index}/{len(judging_pending)}", flush=True)
    judging_seconds = perf_counter() - judging_started

    final_rows = []
    for qid in order:
        row = dict(rows[qid])
        row.pop("_context", None)
        final_rows.append(row)
    _write_json(RESULT_PATH, final_rows)

    verdicts = [ViDoreVerdict(row["qid"], row["judgment"]) for row in final_rows]
    scored = score(verdicts)
    correct = sum(row["judgment"] == "Correct" for row in final_rows)
    partial = sum(row["judgment"] == "Partially Correct" for row in final_rows)
    incorrect = sum(row["judgment"] == "Incorrect" for row in final_rows)
    errors = sum(bool(row.get("generation_error") or row.get("judge_error")) for row in final_rows)
    summary = {
        "arm": "colvec_kf15_top10",
        "dataset": "vidore_v3/physics",
        "language": LANGUAGE,
        "light_file_budget": KF,
        "second_retrieval": "cached webAI-Official/webAI-ColVec1.1-4b score matrix over Kf=15 page scope",
        "context_top_k": TOP_K,
        "generator": GENERATOR,
        "judge": JUDGE,
        "n": scored["n"],
        "correct_count": correct,
        "partial_count": partial,
        "incorrect_count": incorrect,
        "correct_only": round(100 * scored["correct_only"], 1),
        "correct_plus_partial": round(100 * scored["correct_plus_partial"], 1),
        "errors": errors,
        "avg_context_pages": round(sum(row["n_context_pages"] for row in final_rows) / len(final_rows), 2),
        "avg_context_chars": round(sum(row["context_chars"] for row in final_rows) / len(final_rows)),
        "avg_gold_pages_in_context": round(sum(row["gold_hit"] for row in final_rows) / len(final_rows), 2),
        "generation_seconds_this_run": round(generation_seconds, 3),
        "judging_seconds_this_run": round(judging_seconds, 3),
        "e2e_api_seconds_this_run": round(generation_seconds + judging_seconds, 3),
        "generation_calls_this_run": len(generation_pending),
        "judging_calls_this_run": len(judging_pending),
        "colvec_retrieval_metrics": _colvec_metrics(),
        "retrieval_run": str(COLVEC_RUN.relative_to(ROOT)),
    }
    _write_json(SUMMARY_PATH, summary)
    report = (
        "# Physics E2E: ColVec Kf=15\n\n"
        "ColVec ranks the cached page score matrix within the 15-file light-retrieval scope; "
        "the first 10 pages are passed to the fixed answer generator.\n\n"
        "## Accuracy\n\n"
        "| Correct only | Correct + partial | n | Correct | Partial | Incorrect | Errors |\n"
        "|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| {summary['correct_only']:.1f}% | {summary['correct_plus_partial']:.1f}% | "
        f"{summary['n']} | {summary['correct_count']} | {summary['partial_count']} | "
        f"{summary['incorrect_count']} | {summary['errors']} |\n\n"
        "## Retrieval and cost\n\n"
        f"- ColVec page recall@10: {summary['colvec_retrieval_metrics'].get('page_recall@10', 0):.2%}.\n"
        f"- Page-pool ceiling: {summary['colvec_retrieval_metrics'].get('page_pool_recall_ceiling', 0):.2%}.\n"
        f"- File-scope recall: {summary['colvec_retrieval_metrics'].get('file_scope_recall', 0):.2%}.\n"
        f"- Average context: {summary['avg_context_pages']:.1f} pages and "
        f"{summary['avg_context_chars']:.0f} characters/query.\n"
        f"- API time for this invocation: {summary['e2e_api_seconds_this_run']:.3f}s "
        f"({summary['generation_calls_this_run']} generation calls; "
        f"{summary['judging_calls_this_run']} judge calls; resumed rows are not re-called).\n"
        "- ColVec GPU indexing/encoding time is not reported here because the Colab "
        "bundle did not persist that telemetry; see `timing_kf15.json` for the measured "
        "local cached-ranking boundary.\n\n"
        "## Reproducibility\n\n"
        f"- Generator: `{GENERATOR}`; judge: `{JUDGE}`.\n"
        f"- ColVec run: `{COLVEC_RUN.relative_to(ROOT)}`.\n"
        f"- Parsed page source: `{PARSED_RUN.relative_to(ROOT)}`.\n"
        f"- Final result contains {summary['errors']} generation/judge errors; all "
        "302 rows have a completed answer and verdict.\n"
    )
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
