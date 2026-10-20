"""Evaluate an oracle QA arm using every gold page's cached KDL text.

This is deliberately *not* a retrieval metric.  The page qrels are used to
construct the context, so the result is an answer-quality upper bound for the
current KDL page text and generator.  It is useful for separating retrieval
loss from generation/parsing loss.

The script uses the current merged benchmark (80 Physics, 75 Industrial and
65 OHR queries), the cached KDL page export, and the same ViDoRe generator and
three-way judge used by the other merged QA experiments.  Every query is
checkpointed after generation and judgment, so an interrupted API run can be
resumed without repeating successful calls.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmarks.vidore_v3_judge import (  # noqa: E402
    ViDoreVerdict,
    ViDoreStyle,
    judge_answer,
    score,
)
from src.evaluation.generate import ContextChunk, generate  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402


DEFAULT_DATASET = ROOT / "data/raw/0.1. BENCHMARK"
DEFAULT_OLD_QRELS = (
    ROOT
    / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20"
    / "bundle_debug_light-retrieval/input/qrels_page_level.jsonl"
)
DEFAULT_OHR_QRELS = ROOT / "data/work/ohr_replacement_20260920/inputs/qrels.jsonl"
DEFAULT_KDL_PAGES = (
    ROOT
    / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803"
    / "kdl/kdl_pages.jsonl"
)
DEFAULT_OUTPUT = ROOT / "data/work/ohr_replacement_20260920/e2e_oracle_kdl_gold"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def load_queries(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        answers = [str(value) for value in row.get("answers") or []]
        if not answers:
            raise RuntimeError(f"Query {row.get('query_id')} has no answer")
        qid = str(row["query_id"])
        result[qid] = {
            "query_id": qid,
            "source": str(row.get("source") or qid.split("::", 1)[0]),
            "query": str(row.get("query") or ""),
            "answers": answers,
        }
    return result


def load_gold_pages(old_qrels: Path, ohr_qrels: Path) -> dict[str, list[dict[str, Any]]]:
    """Merge the canonical old Physics/Industrial and OHR page qrels."""
    result: dict[str, list[dict[str, Any]]] = {}

    for row in read_jsonl(old_qrels):
        qid = str(row["query_id"])
        source = qid.split("::", 1)[0]
        if source not in {"vidore_physics", "vidore_industrial"}:
            continue
        page_id = str(row["page_id"])
        result.setdefault(qid, []).append(
            {
                "page_id": page_id,
                "doc_id": str(row.get("doc_id") or page_id.split("#page=", 1)[0]),
                "page_index": int(row.get("page_index", page_id.rsplit("=", 1)[-1])),
                "relevance": int(row.get("relevance", 1)),
                "source_evidence_id": row.get("source_evidence_id"),
            }
        )

    for row in read_jsonl(ohr_qrels):
        qid = str(row["query_id"])
        doc_id = str(row["doc_id"])
        page_index = int(row["page_index"])
        result.setdefault(qid, []).append(
            {
                "page_id": f"{doc_id}#page={page_index}",
                "doc_id": doc_id,
                "page_index": page_index,
                "relevance": int(row.get("relevance", 1)),
                "source_evidence_id": row.get("source_evidence_id"),
            }
        )

    # A page can occur in multiple qrel records.  Preserve the first order,
    # while keeping the strongest relevance label for auditability.
    deduped: dict[str, list[dict[str, Any]]] = {}
    for qid, rows in result.items():
        by_page: dict[str, dict[str, Any]] = {}
        for row in rows:
            page_id = row["page_id"]
            if page_id not in by_page or row["relevance"] > by_page[page_id]["relevance"]:
                by_page[page_id] = row
        deduped[qid] = list(by_page.values())
    return deduped


def load_kdl_pages(path: Path) -> dict[str, dict[str, Any]]:
    pages: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        page_id = str(row["page_id"])
        pages[page_id] = {
            "page_id": page_id,
            "doc_id": str(row.get("doc_id") or page_id.split("#page=", 1)[0]),
            "page_index": int(row.get("page_index", page_id.rsplit("=", 1)[-1])),
            "text": str(row.get("text") or ""),
            "kdl_status": row.get("kdl_status"),
            "provider": row.get("provider"),
        }
    return pages


def build_contexts(
    queries: dict[str, dict[str, Any]],
    gold: dict[str, list[dict[str, Any]]],
    pages: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if set(queries) != set(gold):
        raise RuntimeError(
            f"Query/qrel mismatch: queries={len(queries)}, qrel_qids={len(gold)}, "
            f"missing={sorted(set(queries) - set(gold))[:5]}, "
            f"extra={sorted(set(gold) - set(queries))[:5]}"
        )
    missing = [
        row["page_id"]
        for rows in gold.values()
        for row in rows
        if row["page_id"] not in pages
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} gold pages missing from KDL cache; first={missing[:5]}")

    contexts: dict[str, list[dict[str, Any]]] = {}
    for qid in queries:
        contexts[qid] = []
        for rank, row in enumerate(gold[qid], 1):
            page = pages[row["page_id"]]
            contexts[qid].append(
                {
                    "chunk_id": f"oracle_kdl:{row['page_id']}",
                    "page_id": row["page_id"],
                    "doc_id": row["doc_id"],
                    "page_index": row["page_index"],
                    "rank": rank,
                    "relevance": row["relevance"],
                    "text": page["text"],
                    "kdl_status": page["kdl_status"],
                    "provider": page["provider"],
                }
            )
    return contexts


def judge_against_all_answers(
    query: dict[str, Any], answer: str, judge_model: str, generator_model: str
) -> tuple[ViDoreVerdict, list[dict[str, Any]]]:
    labels = {"Incorrect": 0, "Partially Correct": 1, "Correct": 2}
    alternatives: list[dict[str, Any]] = []
    best: ViDoreVerdict | None = None
    for gold in query["answers"]:
        try:
            verdict = judge_answer(
                query["query_id"],
                query["query"],
                gold,
                answer,
                model=judge_model,
                generator_model=generator_model,
            )
        except Exception as error:  # keep one malformed judge reply auditable
            verdict = ViDoreVerdict(query["query_id"], "Incorrect", error=str(error))
        alternatives.append(
            {
                "gold_answer": gold,
                "judgment": verdict.judgment,
                "explanation": verdict.explanation,
                "error": verdict.error,
            }
        )
        if best is None or labels.get(verdict.judgment, -1) > labels.get(best.judgment, -1):
            best = verdict
    assert best is not None
    return best, alternatives


def evaluate_one(
    query: dict[str, Any],
    context: list[dict[str, Any]],
    generator_model: str,
    judge_model: str,
    max_context_chars: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    chunks = [
        ContextChunk(
            str(item["chunk_id"]),
            str(item["page_id"]),
            str(item["text"]),
            0.0,
        )
        for item in context
    ]
    generation_started = time.perf_counter()
    generation = generate(
        query["query_id"],
        query["query"],
        chunks,
        model=generator_model,
        max_chars=max_context_chars,
        max_output_tokens=max_output_tokens,
        render_prompt=ViDoreStyle().render_prompt,
    )
    generation_seconds = time.perf_counter() - generation_started
    judge_started = time.perf_counter()
    alternatives: list[dict[str, Any]] = []
    if generation.error:
        verdict = ViDoreVerdict(query["query_id"], "Incorrect", error=generation.error)
    else:
        verdict, alternatives = judge_against_all_answers(
            query, generation.answer, judge_model, generator_model
        )
    return {
        "query_id": query["query_id"],
        "source": query["source"],
        "query": query["query"],
        "gold_answers": query["answers"],
        "gold_page_count": len(context),
        "gold_page_ids": [item["page_id"] for item in context],
        "answer": generation.answer,
        "generation_error": generation.error,
        "judgment": verdict.judgment,
        "explanation": verdict.explanation,
        "judge_error": verdict.error,
        "judge_alternatives": alternatives,
        "correct": verdict.judgment == "Correct" and not verdict.error,
        "credited": verdict.judgment in {"Correct", "Partially Correct"} and not verdict.error,
        "chunks_used": generation.chunks_used,
        "chars_used": generation.chars_used,
        "context_pages": generation.context_doc_ids,
        "generation_seconds": round(generation_seconds, 3),
        "judge_seconds": round(time.perf_counter() - judge_started, 3),
    }


def make_report(
    rows: list[dict[str, Any]],
    contexts: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    started: float,
) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("generation_error") and not row.get("judge_error")]
    verdicts = [ViDoreVerdict(row["query_id"], row["judgment"]) for row in valid]
    metrics = score(verdicts) if verdicts else {"n": 0, "correct_only": 0.0, "correct_plus_partial": 0.0}
    by_source: dict[str, dict[str, Any]] = {}
    for source in ("vidore_physics", "vidore_industrial", "ohrbench"):
        source_rows = [row for row in rows if row["source"] == source]
        group = [row for row in valid if row["source"] == source]
        group_verdicts = [ViDoreVerdict(row["query_id"], row["judgment"]) for row in group]
        group_score = score(group_verdicts) if group_verdicts else {"correct_only": 0.0, "correct_plus_partial": 0.0}
        by_source[source] = {
            "queries": len(source_rows),
            "valid": len(group),
            "errors": len(source_rows) - len(group),
            "correct_only": round(100 * group_score["correct_only"], 2),
            "correct_plus_partial": round(100 * group_score["correct_plus_partial"], 2),
            "judgments": {
                label: sum(row["judgment"] == label for row in group)
                for label in ("Correct", "Partially Correct", "Incorrect")
            },
        }
    total_queries = len(rows)
    total_errors = total_queries - len(valid)
    return {
        "contract_version": "merged-oracle-kdl-gold-qa-v1",
        "arm": "oracle_all_gold_pages_kdl_text",
        "oracle_warning": "qrels select the context; this is an answer-quality upper bound, not a retrieval result",
        "dataset_root": str(args.dataset_root),
        "old_qrels": str(args.old_qrels),
        "ohr_qrels": str(args.ohr_qrels),
        "kdl_pages": str(args.kdl_pages),
        "generator": args.generator,
        "judge": args.judge,
        "max_context_chars": args.max_context_chars,
        "max_output_tokens": args.max_output_tokens,
        "workers": args.workers,
        "queries": len(rows),
        "judge_valid": len(valid),
        "errors": total_errors,
        "correct_only": round(100 * metrics["correct_only"], 2),
        "correct_plus_partial": round(100 * metrics["correct_plus_partial"], 2),
        "correct_only_over_all": round(
            100 * sum(row["judgment"] == "Correct" for row in valid) / max(1, total_queries), 2
        ),
        "correct_plus_partial_over_all": round(
            100 * sum(row["judgment"] in {"Correct", "Partially Correct"} for row in valid)
            / max(1, total_queries),
            2,
        ),
        "judgments": {
            label: sum(row["judgment"] == label for row in valid)
            for label in ("Correct", "Partially Correct", "Incorrect")
        },
        "by_source": by_source,
        "gold_pages_total": sum(len(context) for context in contexts.values()),
        "gold_pages_mean": round(sum(len(context) for context in contexts.values()) / max(1, len(contexts)), 2),
        "context_mean_pages_used": round(sum(row["chunks_used"] for row in rows) / max(1, len(rows)), 2),
        "context_mean_chars_used": round(sum(row["chars_used"] for row in rows) / max(1, len(rows)), 2),
        "context_max_chars_used": max((row["chars_used"] for row in rows), default=0),
        "generation_seconds_sum": round(sum(row["generation_seconds"] for row in rows), 2),
        "generation_seconds_mean": round(sum(row["generation_seconds"] for row in rows) / max(1, len(rows)), 3),
        "judge_seconds_sum": round(sum(row["judge_seconds"] for row in rows), 2),
        "judge_seconds_mean": round(sum(row["judge_seconds"] for row in rows) / max(1, len(rows)), 3),
        "wall_seconds": round(time.perf_counter() - started, 2) if started else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--old-qrels", type=Path, default=DEFAULT_OLD_QRELS)
    parser.add_argument("--ohr-qrels", type=Path, default=DEFAULT_OHR_QRELS)
    parser.add_argument("--kdl-pages", type=Path, default=DEFAULT_KDL_PAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generator", default="deepseek/deepseek-v4.1-flash")
    parser.add_argument("--judge", default="openai/gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int, default=200_000)
    # V4.1 Flash may spend the first part of the completion budget on
    # reasoning.  512 can therefore end with an empty message.content.
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    load_dotenv_file(ROOT)

    started = time.perf_counter()
    queries = load_queries(args.dataset_root / "queries.jsonl")
    gold = load_gold_pages(args.old_qrels, args.ohr_qrels)
    pages = load_kdl_pages(args.kdl_pages)
    contexts = build_contexts(queries, gold, pages)
    context_rows = [
        {
            "query_id": qid,
            "source": queries[qid]["source"],
            "query": queries[qid]["query"],
            "gold_pages": context,
        }
        for qid, context in contexts.items()
    ]
    out = args.output_dir
    write_jsonl(out / "contexts" / "oracle_gold_kdl_pages.jsonl", context_rows)
    write_json(
        out / "oracle_input_manifest.json",
        {
            "queries": len(queries),
            "gold_qrel_page_refs": sum(len(value) for value in contexts.values()),
            "kdl_pages": len(pages),
            "kdl_empty_pages": sum(not page["text"].strip() for page in pages.values()),
            "all_gold_pages_mapped": True,
            "context_policy": "all page-level qrel evidence pages; KDL cached text; no retrieval ranking",
        },
    )

    rows_path = out / "qa" / "oracle_gold_kdl_pages.jsonl"
    existing = {str(row["query_id"]): row for row in read_jsonl(rows_path)} if rows_path.is_file() else {}
    order = list(queries)
    pending = [
        qid
        for qid in order
        if qid not in existing
        or (args.retry_errors and (existing[qid].get("generation_error") or existing[qid].get("judge_error")))
    ]
    print(
        f"Oracle QA: {len(order)} queries, {len(pending)} pending, "
        f"{sum(len(value) for value in contexts.values())} gold pages, workers={args.workers}",
        flush=True,
    )
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    evaluate_one,
                    queries[qid],
                    contexts[qid],
                    args.generator,
                    args.judge,
                    args.max_context_chars,
                    args.max_output_tokens,
                ): qid
                for qid in pending
            }
            for number, future in enumerate(as_completed(futures), 1):
                qid = futures[future]
                try:
                    existing[qid] = future.result()
                except Exception as error:  # checkpoint a failed query explicitly
                    existing[qid] = {
                        "query_id": qid,
                        "source": queries[qid]["source"],
                        "query": queries[qid]["query"],
                        "gold_answers": queries[qid]["answers"],
                        "gold_page_count": len(contexts[qid]),
                        "gold_page_ids": [item["page_id"] for item in contexts[qid]],
                        "answer": "",
                        "generation_error": f"worker exception: {error}",
                        "judge_error": None,
                        "judgment": "Incorrect",
                        "explanation": "",
                        "judge_alternatives": [],
                        "correct": False,
                        "credited": False,
                        "chunks_used": 0,
                        "chars_used": 0,
                        "context_pages": [],
                        "generation_seconds": 0.0,
                        "judge_seconds": 0.0,
                    }
                write_jsonl(out / "qa" / "oracle_gold_kdl_pages.jsonl", [existing[item] for item in order if item in existing])
                print(f"Oracle QA progress: {number}/{len(pending)} ({qid})", flush=True)

    rows = [existing[qid] for qid in order if qid in existing]
    report = make_report(rows, contexts, args, started)
    write_json(out / "qa" / "oracle_gold_kdl_pages.report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(rows) == len(order) else 2


if __name__ == "__main__":
    raise SystemExit(main())
