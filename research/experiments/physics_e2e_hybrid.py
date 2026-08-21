"""Run the ViDoRe-v3 physics E2E row for KDL + pdf-inspector.

This intentionally mirrors ``physics_e2e.py``: French questions, page-level
retrieval, plain BM25 + text-embedding-3-small, alpha=0.7, depth=100, top-10
full-page context, DeepSeek V4 Flash generation, and GPT-4o judging.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytrec_eval

from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
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
from src.evaluation.retrieval import alpha_fuse
from src.retrieval.sparse import BM25Index
from src.utils.env import load_dotenv_file


ROOT = Path(__file__).resolve().parents[2]
load_dotenv_file(ROOT)

RUN = (
    ROOT
    / "data/output/vidore-v3-physics-kdl-pdf-inspector"
    / "32c32a45a92c45bb"
)
OUT = ROOT / "data/benchmark/vidore_v3/results/physics_e2e"
ARM = "retrieved_kdl_pdf_inspector"
SUBSET, LANGUAGE = "physics", "french"
GENERATOR = "deepseek/deepseek-v4-flash"
JUDGE = "openai/gpt-4o"
TOP_K, DEPTH, ALPHA, WORKERS = 10, 100, 0.7, 12

RETRIEVAL_PATH = OUT / f"{ARM}.retrieval.json"
RESULT_PATH = OUT / f"{ARM}.json"
SUMMARY_PATH = OUT / f"{ARM}.summary.json"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _pipeline_pages() -> dict[str, str]:
    pages: dict[str, str] = {}
    for document in documents(RUN):
        doc = canonical_doc(document.get("document", {}).get("file_name"))
        for page, blocks in page_blocks(document).items():
            pages[unit_id(SUBSET, doc, page)] = "\n".join(
                block["text"] for block in blocks if block["text"].strip()
            )
    return pages


def _retrieve(
    pages: dict[str, str], questions: list[Any], qrels: dict[str, dict[str, int]]
) -> tuple[dict[str, list[str]], float, dict[str, Any]]:
    if RETRIEVAL_PATH.exists():
        payload = json.loads(RETRIEVAL_PATH.read_text(encoding="utf-8"))
        return payload["ranked"], float(payload["ndcg@10"]), payload

    started = perf_counter()
    units = [unit for unit, text in pages.items() if text.strip()]
    texts = [pages[unit] for unit in units]
    embedder = OpenRouterEmbedder(
        cache_dir=ROOT / "data/work/vidore_physics_emb", batch_size=64
    )

    print(f"Embedding {len(questions)} queries...", flush=True)
    query_vectors = np.asarray(
        embedder.embed([question.query for question in questions]), dtype=np.float32
    )
    query_vectors /= np.clip(
        np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12, None
    )

    print(f"Building BM25 and embedding {len(texts)} parsed pages...", flush=True)
    bm25 = BM25Index(analyzer_name="plain").build(
        [
            {"chunk_id": unit, "doc_id": unit, "text": text}
            for unit, text in zip(units, texts)
        ]
    )
    matrix = np.asarray(embedder.embed(texts), dtype=np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)

    ranked: dict[str, list[str]] = {}
    run: dict[str, dict[str, float]] = {}
    for question, vector in zip(questions, query_vectors):
        lexical = bm25.search(question.query, DEPTH)
        scores = matrix @ vector
        top = np.argpartition(-scores, DEPTH)[:DEPTH]
        dense = sorted(
            ((int(index), float(scores[index])) for index in top),
            key=lambda pair: -pair[1],
        )
        fused = alpha_fuse(lexical, dense, ALPHA, DEPTH)
        ranked[question.qid] = [units[position] for position, _ in fused[:TOP_K]]
        run[question.qid] = {
            units[position]: fused_score for position, fused_score in fused
        }

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})
    evaluated = evaluator.evaluate(run)
    ndcg = 100 * sum(row["ndcg_cut_10"] for row in evaluated.values()) / len(
        evaluated
    )
    payload = {
        "arm": ARM,
        "alpha": ALPHA,
        "depth": DEPTH,
        "top_k": TOP_K,
        "analyzer": "plain",
        "embedder": "openai/text-embedding-3-small",
        "ndcg@10": ndcg,
        "seconds": round(perf_counter() - started, 3),
        "embedder_stats": embedder.stats,
        "ranked": ranked,
    }
    _write_json(RETRIEVAL_PATH, payload)
    return ranked, ndcg, payload


def _generate_one(qid: str, query: str, context: str) -> tuple[str, str, str | None]:
    try:
        answer = complete(
            GENERATOR,
            ANSWER_PROMPT.format(documents=context, query=query),
            temperature=0.0,
            max_output_tokens=512,
        ).strip()
        return qid, answer, None
    except Exception as error:  # noqa: BLE001 - persisted for a resumable run
        return qid, "", f"{type(error).__name__}: {error}"


def _judge_one(row: dict[str, Any]) -> tuple[str, ViDoreVerdict]:
    verdict = judge_answer(
        row["qid"],
        row["query"],
        row["gold_answer"],
        row["answer"],
        model=JUDGE,
        generator_model=GENERATOR,
    )
    return row["qid"], verdict


def _save_rows(rows: dict[str, dict[str, Any]], order: list[str]) -> None:
    _write_json(RESULT_PATH, [rows[qid] for qid in order])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not RUN.joinpath("documents").is_dir():
        raise FileNotFoundError(f"Pipeline run not found: {RUN}")

    resolved = assert_real([GENERATOR, JUDGE])
    for alias, model in resolved.items():
        print(f"{alias:30s} -> {model.provider}/{model.upstream_model_id}")

    benchmark = load("vidore_v3", subset=SUBSET, language=LANGUAGE)
    qrels = benchmark.qrels()
    questions = [question for question in benchmark.questions() if qrels.get(question.qid)]
    order = [question.qid for question in questions]
    pages = _pipeline_pages()
    print(
        f"Corpus: {len(pages)} pages; questions: {len(questions)}; "
        f"gold reachable: {sum(bool(set(qrels[q.qid]) & set(pages)) for q in questions)}",
        flush=True,
    )
    if len(pages) != 1674:
        raise RuntimeError(f"Expected 1674 physics pages, got {len(pages)}")

    ranked, ndcg, retrieval = _retrieve(pages, questions, qrels)
    print(f"{ARM}: NDCG@10={ndcg:.2f}", flush=True)

    existing = (
        json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if RESULT_PATH.exists()
        else []
    )
    rows = {str(row["qid"]): row for row in existing}
    for question in questions:
        units = ranked[question.qid]
        context = render_documents(
            [pages[unit] for unit in units if pages.get(unit, "").strip()]
        )
        current = rows.setdefault(question.qid, {})
        current.update(
            {
                "arm": ARM,
                "qid": question.qid,
                "query": question.query,
                "gold_answer": question.answer,
                "n_context_pages": len(units),
                "context_chars": len(context),
                "gold_hit": len(set(units) & set(qrels[question.qid])),
            }
        )
        current["_context"] = context

    generation_pending = [
        row for row in rows.values() if not row.get("answer")
    ]
    if generation_pending:
        print(f"Generating {len(generation_pending)} answers with {WORKERS} workers...", flush=True)
        started = perf_counter()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(
                    _generate_one, row["qid"], row["query"], row["_context"]
                ): row["qid"]
                for row in generation_pending
            }
            completed = 0
            for future in as_completed(futures):
                qid, answer, error = future.result()
                rows[qid]["answer"] = answer
                rows[qid]["generation_error"] = error
                completed += 1
                if completed % 10 == 0 or completed == len(generation_pending):
                    _save_rows(rows, order)
                    print(
                        f"  generation {completed}/{len(generation_pending)} "
                        f"({perf_counter() - started:.1f}s)",
                        flush=True,
                    )

    failed_generation = [row for row in rows.values() if not row.get("answer")]
    if failed_generation:
        _save_rows(rows, order)
        raise RuntimeError(
            f"{len(failed_generation)} generation calls failed; rerun to resume"
        )

    judging_pending = [
        row for row in rows.values() if not row.get("judgment")
    ]
    if judging_pending:
        print(f"Judging {len(judging_pending)} answers with {WORKERS} workers...", flush=True)
        started = perf_counter()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(_judge_one, row): row["qid"] for row in judging_pending}
            completed = 0
            for future in as_completed(futures):
                qid, verdict = future.result()
                rows[qid]["judgment"] = verdict.judgment
                rows[qid]["judge_error"] = verdict.error
                completed += 1
                if completed % 10 == 0 or completed == len(judging_pending):
                    _save_rows(rows, order)
                    print(
                        f"  judging {completed}/{len(judging_pending)} "
                        f"({perf_counter() - started:.1f}s)",
                        flush=True,
                    )

    final_rows = []
    for qid in order:
        row = dict(rows[qid])
        row.pop("_context", None)
        final_rows.append(row)
    _write_json(RESULT_PATH, final_rows)

    judge_errors = [row for row in final_rows if row.get("judge_error")]
    if judge_errors:
        raise RuntimeError(f"{len(judge_errors)} judge calls failed; inspect {RESULT_PATH}")

    scored = score(
        ViDoreVerdict(row["qid"], row["judgment"]) for row in final_rows
    )
    summary = {
        "arm": ARM,
        "run": str(RUN.relative_to(ROOT)),
        "generator": GENERATOR,
        "judge": JUDGE,
        "alpha": ALPHA,
        "depth": DEPTH,
        "top_k": TOP_K,
        "ndcg@10": round(ndcg, 2),
        "n": scored["n"],
        "correct_only": round(100 * scored["correct_only"], 1),
        "correct_plus_partial": round(100 * scored["correct_plus_partial"], 1),
        "ctx_pages": round(
            sum(row["n_context_pages"] for row in final_rows) / len(final_rows), 1
        ),
        "ctx_chars": round(
            sum(row["context_chars"] for row in final_rows) / len(final_rows)
        ),
        "gold_in_ctx": round(
            sum(row["gold_hit"] for row in final_rows) / len(final_rows), 2
        ),
        "retrieval_seconds": retrieval.get("seconds"),
    }
    _write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
