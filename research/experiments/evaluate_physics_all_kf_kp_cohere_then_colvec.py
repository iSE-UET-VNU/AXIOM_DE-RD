"""Run all Physics Kf/Kp settings with a large light band and Cohere rerank.

Pipeline:

    fixed Kf file scope -> light top-R page retrieval (R=100)
    -> Cohere rerank the R pages -> keep top-Kp
    -> cached ColVec rank -> final top-10

Only IDs and scores are written.  Page text is held in memory for the API
request and is never persisted by this experiment.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import math
import os
import sys
import time
from time import perf_counter
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.env import load_dotenv_file

load_dotenv_file(ROOT)

from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks


SCORE_DIR = ROOT / "data/work/physics_colvec_export"
LIGHT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_topk_budget_ablation/runs"
SCOPE_DIR = ROOT / "data/benchmark/vidore_v3/exports/physics_file_retrieval_for_colembed"
EXISTING_KF3_RERANK = (
    ROOT
    / "data/benchmark/vidore_v3/results/physics_fielded_kf3_openrouter_cohere_rerank/runs/or_only_r100.jsonl"
)
PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_all_kf_kp_cohere_then_colvec"
RERANK_CACHE_DIR = OUTPUT_DIR / "rerank_cache"
KF_VALUES = (3, 5, 10, 15, 20)
KP_VALUES = (10, 20, 30, 40, 50, 100)
LIGHT_DEPTH = 100
RERANKER = "cohere/rerank-v3.5"
RERANK_DEPTH = 100
RERANK_ENDPOINT = "https://openrouter.ai/api/v1/rerank"
WORKERS = 4
TIMEOUT_SECONDS = 120.0
MAX_RETRIES = 2
COLVEC_TOP_K = 10


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _ndcg(ranked: list[str], qrel: dict[str, int]) -> float:
    def gain(value: int) -> float:
        return float(2 ** int(value) - 1)

    dcg = sum(
        gain(qrel.get(page, 0)) / math.log2(rank + 1)
        for rank, page in enumerate(ranked[:COLVEC_TOP_K], 1)
        if qrel.get(page, 0) > 0
    )
    ideal = sorted((int(value) for value in qrel.values()), reverse=True)[:COLVEC_TOP_K]
    idcg = sum(gain(value) / math.log2(rank + 1) for rank, value in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def _page_texts() -> dict[str, str]:
    pages: dict[str, str] = {}
    for document in documents(PARSED_RUN):
        doc = canonical_doc(document.get("document", {}).get("file_name"))
        for page, blocks in page_blocks(document).items():
            pages[unit_id("physics", doc, page)] = "\n".join(
                block["text"] for block in blocks if block["text"].strip()
            )
    if len(pages) != 1674:
        raise RuntimeError(f"Expected 1674 parsed pages, got {len(pages)}")
    return pages


def _load_light_runs(qids: list[str], page_set: set[str]) -> dict[int, dict[str, list[str]]]:
    output: dict[int, dict[str, list[str]]] = {}
    for kf in KF_VALUES:
        path = LIGHT_DIR / f"full_kf{kf}.jsonl"
        rows: dict[str, list[str]] = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row["qid"])
                page_ids = [str(item["chunk_id"]) for item in row["results"]]
                if qid in rows or len(page_ids) < min(KP_VALUES) or len(page_ids) != len(set(page_ids)):
                    raise ValueError(f"Invalid light run {path}:{line_number}")
                if any(page_id not in page_set for page_id in page_ids):
                    raise ValueError(f"Unknown page in light run {path}:{line_number}")
                rows[qid] = page_ids[:LIGHT_DEPTH]
        if set(rows) != set(qids):
            raise ValueError(f"QID mismatch in {path}")
        output[kf] = rows
    return output


def _load_scope_files(qids: list[str]) -> dict[int, dict[str, list[str]]]:
    output: dict[int, dict[str, list[str]]] = {}
    for kf in KF_VALUES:
        path = SCOPE_DIR / f"queries_kf{kf}.jsonl"
        rows: dict[str, list[str]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[str(row["qid"])] = [str(value) for value in row["selected_file_ids"]]
        if set(rows) != set(qids):
            raise ValueError(f"Scope QID mismatch in {path}")
        output[kf] = rows
    return output


def _load_existing_kf3(qids: list[str], candidate_pages: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    with EXISTING_KF3_RERANK.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                qid = str(item["qid"])
                chunks = item["chunks"]
                pages = [str(chunk["chunk_id"]) for chunk in chunks]
                expected = candidate_pages[qid]
                if set(pages) != set(expected) or len(pages) != len(expected):
                    raise ValueError(f"Existing Kf=3 rerank membership mismatch for {qid}")
                rows[qid] = [
                    {"page_id": page, "score": float(chunk.get("rerank_score", chunk.get("score", 0.0))), "rank": rank}
                    for rank, (page, chunk) in enumerate(zip(pages, chunks), 1)
                ]
    if set(rows) != set(qids):
        raise ValueError("Existing Kf=3 rerank qids do not match")
    return rows


def _api_key() -> str:
    value = os.environ.get("OPENROUTER_API_KEY")
    if value:
        return value
    dotenv = ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                if value:
                    return value
    raise RuntimeError("OPENROUTER_API_KEY is not set")


def _rerank_one(qid: str, query: str, page_ids: list[str], texts: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    import requests

    payload = {
        "model": RERANKER,
        "query": query,
        "documents": [texts[page_id] for page_id in page_ids],
        "top_n": len(page_ids),
        "return_documents": False,
    }
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "X-Title": "AXIOM Physics all-Kf light rerank",
    }
    last_error = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                RERANK_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=TIMEOUT_SECONDS,
            )
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                if attempt < MAX_RETRIES:
                    time.sleep(min(2.0 ** attempt, 8.0))
                    continue
            response.raise_for_status()
            result = response.json().get("results")
            if not isinstance(result, list) or len(result) != len(page_ids):
                raise RuntimeError(f"Expected {len(page_ids)} rerank results, got {len(result) if isinstance(result, list) else 'none'}")
            ranked = []
            seen: set[int] = set()
            for rank, item in enumerate(result, 1):
                index = int(item["index"])
                if index in seen or not 0 <= index < len(page_ids):
                    raise RuntimeError("Reranker returned duplicate or invalid document index")
                seen.add(index)
                ranked.append({"page_id": page_ids[index], "score": float(item["relevance_score"]), "rank": rank})
            if len(seen) != len(page_ids):
                raise RuntimeError("Reranker omitted a candidate page")
            return qid, ranked
        except Exception as error:  # noqa: BLE001 - retry then fail clearly
            last_error = f"{type(error).__name__}: {error}"
            if attempt < MAX_RETRIES:
                time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(f"Cohere rerank failed for {qid}: {last_error}")


def _load_or_build_rerank(
    kf: int,
    qids: list[str],
    queries: dict[str, str],
    candidates: dict[str, list[str]],
    texts: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    cache_path = RERANK_CACHE_DIR / f"cohere_r{LIGHT_DEPTH}_kf{kf}.jsonl"
    cached: dict[str, list[dict[str, Any]]] = {}
    if cache_path.is_file():
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    cached[str(row["qid"])] = row["ranked"]
        for qid in list(cached):
            page_ids = [str(item["page_id"]) for item in cached[qid]]
            if qid not in candidates or set(page_ids) != set(candidates[qid]):
                raise ValueError(f"Invalid cached rerank membership for Kf={kf}, qid={qid}")

    if kf == 3 and len(cached) < len(qids):
        existing = _load_existing_kf3(qids, candidates)
        cached.update(existing)
        _write_jsonl(cache_path, [{"qid": qid, "ranked": cached[qid]} for qid in qids if qid in cached])
        source = "existing Cohere R=100 run for Kf=3"
    else:
        source = "cache or OpenRouter Cohere API"

    pending = [qid for qid in qids if qid not in cached]
    if pending:
        print(f"Cohere Kf={kf}: {len(pending)} requests pending", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(_rerank_one, qid, queries[qid], candidates[qid], texts): qid
                for qid in pending
            }
            for completed, future in enumerate(as_completed(futures), 1):
                qid, ranked = future.result()
                cached[qid] = ranked
                if completed % 25 == 0 or completed == len(pending):
                    _write_jsonl(cache_path, [{"qid": item, "ranked": cached[item]} for item in qids if item in cached])
                    print(f"  Cohere Kf={kf}: {completed}/{len(pending)}", flush=True)

    if set(cached) != set(qids):
        raise RuntimeError(f"Cohere cache incomplete for Kf={kf}")
    return cached, {"cache": str(cache_path.relative_to(ROOT)), "source": source, "requests_this_run": len(pending)}


def main() -> None:
    started = perf_counter()
    score_matrix = np.load(SCORE_DIR / "physics_colvec_scores.npy", mmap_mode="r")
    keys = [str(value) for value in _read_json(SCORE_DIR / "physics_colvec_keys.json")]
    qids = [str(value) for value in _read_json(SCORE_DIR / "physics_colvec_qids.json")]
    if score_matrix.shape != (len(qids), len(keys)):
        raise ValueError(f"Invalid ColVec score shape {score_matrix.shape}")
    page_set = set(keys)
    page_to_col = {page_id: index for index, page_id in enumerate(keys)}
    qid_to_row = {qid: index for index, qid in enumerate(qids)}

    texts = _page_texts()
    light_runs = _load_light_runs(qids, page_set)
    candidates = {kf: {qid: light_runs[kf][qid][:LIGHT_DEPTH] for qid in qids} for kf in KF_VALUES}
    scope_files = _load_scope_files(qids)
    questions = {question.qid: question for question in load("vidore_v3", subset="physics", language="french").questions()}
    queries = {qid: questions[qid].query for qid in qids}

    reranked_by_kf: dict[int, dict[str, list[dict[str, Any]]]] = {}
    reranker_meta: dict[int, dict[str, Any]] = {}
    for kf in KF_VALUES:
        reranked_by_kf[kf], reranker_meta[kf] = _load_or_build_rerank(
            kf, qids, queries, candidates[kf], texts
        )

    benchmark = load("vidore_v3", subset="physics", language="french")
    qrels = benchmark.qrels()
    if set(qrels) != set(qids):
        raise ValueError("QID mismatch between ColVec and benchmark qrels")

    arms: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "runs").mkdir(parents=True, exist_ok=True)
    for kf in KF_VALUES:
        file_recall = _mean([
            len(set(scope_files[kf][qid]) & {_file_id(page) for page in qrels[qid]})
            / len({_file_id(page) for page in qrels[qid]})
            for qid in qids
        ])
        for kp in KP_VALUES:
            arm_started = perf_counter()
            arm_rows: list[dict[str, Any]] = []
            run_rows: list[dict[str, Any]] = []
            for qid in qids:
                base_pool = candidates[kf][qid][:kp]
                ranked_rerank = reranked_by_kf[kf][qid]
                reranked_pool = [str(item["page_id"]) for item in ranked_rerank[:kp]]
                columns = np.asarray([page_to_col[page] for page in reranked_pool], dtype=np.int64)
                values = np.asarray(score_matrix[qid_to_row[qid], columns], dtype=np.float32)
                order = np.argsort(-values, kind="stable")[:COLVEC_TOP_K]
                final_pages = [reranked_pool[int(position)] for position in order]
                gold = set(qrels[qid])
                base_found = set(base_pool) & gold
                rerank_found = set(reranked_pool) & gold
                final_found = set(final_pages) & gold
                run_rows.append({
                    "qid": qid,
                    "file_ids": scope_files[kf][qid],
                    "chunks": [
                        {"page_id": page, "file_id": _file_id(page), "rank": rank, "score": float(values[int(position)])}
                        for rank, position in enumerate(order, 1)
                        for page in [reranked_pool[int(position)]]
                    ],
                })
                arm_rows.append({
                    "qid": qid,
                    "base_light_pool_recall_ceiling": len(base_found) / len(gold),
                    "cohere_pool_recall_ceiling": len(rerank_found) / len(gold),
                    "cohere_page_recall@10": len(
                        {
                            str(item["page_id"])
                            for item in reranked_by_kf[kf][qid][:10]
                        }
                        & gold
                    ) / len(gold),
                    "colvec_page_recall@10": len(final_found) / len(gold),
                    "colvec_page_hit@10": bool(final_found),
                    "colvec_nDCG@10": _ndcg(final_pages, qrels[qid]),
                    "pages_sent_to_colvec": len(reranked_pool),
                    "final_page_ids": final_pages,
                })
            metrics = {
                "k_files": kf,
                "k_pages": kp,
                "light_retrieval_depth": LIGHT_DEPTH,
                "rerank_depth": RERANK_DEPTH,
                "file_recall@kf": file_recall,
                "base_light_pool_recall_ceiling": _mean([row["base_light_pool_recall_ceiling"] for row in arm_rows]),
                "cohere_pool_recall_ceiling": _mean([row["cohere_pool_recall_ceiling"] for row in arm_rows]),
                "cohere_page_recall@10": _mean([row["cohere_page_recall@10"] for row in arm_rows]),
                "colvec_page_recall@10": _mean([row["colvec_page_recall@10"] for row in arm_rows]),
                "colvec_page_hit@10": _mean([float(row["colvec_page_hit@10"]) for row in arm_rows]),
                "colvec_nDCG@10": _mean([row["colvec_nDCG@10"] for row in arm_rows]),
                "avg_pages_sent_to_colvec": _mean([float(row["pages_sent_to_colvec"]) for row in arm_rows]),
                "retrieval_seconds": perf_counter() - arm_started,
            }
            name = f"Kf={kf},Kp={kp}"
            arms.append({"name": name, "metrics": metrics})
            per_query.extend({"arm": name, **row} for row in arm_rows)
            _write_jsonl(OUTPUT_DIR / "runs" / f"cohere_r{LIGHT_DEPTH}_then_colvec_kf{kf}_kp{kp}_top10.jsonl", run_rows)

    report = {
        "dataset": "vidore_v3/physics",
        "language": "french",
        "method": "light top-R=100 -> Cohere rerank -> top-Kp -> cached ColVec top-10",
        "config": {
            "k_files": list(KF_VALUES),
            "k_pages": list(KP_VALUES),
            "light_retrieval_depth": LIGHT_DEPTH,
            "reranker": RERANKER,
            "rerank_depth": RERANK_DEPTH,
            "colvec_top_k": COLVEC_TOP_K,
            "qrels_used_for_ranking": False,
        },
        "inputs": {"score_shape": list(score_matrix.shape), "queries": len(qids), "pages": len(keys), "files": len({_file_id(page) for page in keys})},
        "artifacts_store_page_text": False,
        "reranker": reranker_meta,
        "timing_seconds": {"total": perf_counter() - started},
        "arms": arms,
    }
    _write_json(OUTPUT_DIR / "report.json", report)
    _write_jsonl(OUTPUT_DIR / "per_query.jsonl", per_query)
    lines = [
        "# Physics: all Kf/Kp — light R=100, Cohere rerank, then ColVec",
        "",
        "For every Kf, light retrieval produces a fixed top-100 page band. Cohere reranks that band and only its top-Kp pages are sent to ColVec. New artifacts contain IDs/scores only; page text is not persisted.",
        "",
        "| Kf | Kp | Light ceiling | Cohere pool ceiling | Cohere recall@10 | ColVec recall@10 | ColVec nDCG@10 | File recall | Pages sent |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        m = arm["metrics"]
        lines.append(
            f"| {m['k_files']} | {m['k_pages']} | {m['base_light_pool_recall_ceiling']:.2%} | "
            f"{m['cohere_pool_recall_ceiling']:.2%} | {m['cohere_page_recall@10']:.2%} | "
            f"{m['colvec_page_recall@10']:.2%} | {m['colvec_nDCG@10']:.2%} | "
            f"{m['file_recall@kf']:.2%} | {m['avg_pages_sent_to_colvec']:.2f} |"
        )
    lines += [
        "",
        "- R=100 is fixed; Kp controls only the post-Cohere pool sent to ColVec.",
        "- Cohere rerank can only reorder the light R=100 band; it cannot recover pages outside that band.",
        "- No page text is stored in the newly generated result files.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "arms": arms}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
