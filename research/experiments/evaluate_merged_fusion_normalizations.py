"""Screen score normalization and RRF on the cached merged 220-query run.

This is an offline ranking-only experiment. It reuses cached BM25/dense
matrices and preserves the current language route: BM25 contributes only to
files detected as English; other files use dense scores only.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/work/ohr_replacement_20260920/retrieval/combined_220_base"
OUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/fusion_normalization_screen"
OLD_BUNDLE = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval"
OHR_ROOT = ROOT / "data/work/ohr_replacement_20260920"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def max_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    peak = float(values.max()) if values.size else 0.0
    return values / peak if peak > 0 else np.zeros_like(values)


def minmax_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return np.zeros_like(values)
    low, high = float(values.min()), float(values.max())
    return (values - low) / (high - low) if high > low else np.zeros_like(values)


def rank_positions(scores: np.ndarray, allowed: np.ndarray | None = None) -> list[int]:
    positions = np.arange(scores.size, dtype=np.int64) if allowed is None else np.asarray(allowed, dtype=np.int64)
    return sorted((int(i) for i in positions), key=lambda i: (-float(scores[i]), i))


def ndcg(ranked: list[str], gold: set[str], depth: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 2.0) for rank, page in enumerate(ranked[:depth]) if page in gold)
    ideal = sum(1.0 / math.log2(rank + 2.0) for rank in range(min(len(gold), depth)))
    return dcg / ideal if ideal else 0.0


def load_language() -> dict[str, str]:
    languages: dict[str, str] = {}
    path = OLD_BUNDLE / "reports/language_routed_bm25/file_language_detection.jsonl"
    for row in read_jsonl(path):
        fid = str(row.get("file_id") or row.get("doc_id") or "")
        if fid:
            languages[fid] = str(row.get("detected_language") or "en")
    for row in read_jsonl(OHR_ROOT / "ohr_preparation/parsing/pages.jsonl"):
        languages[file_id(str(row["page_id"]))] = "en"
    return languages


def load_qrels() -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(OLD_BUNDLE / "input/qrels_page_level.jsonl"):
        qid = str(row["query_id"])
        if qid.startswith(("vidore_physics::", "vidore_industrial::")):
            qrels[qid].add(str(row["page_id"]))
    for row in read_jsonl(OHR_ROOT / "inputs/qrels.jsonl"):
        qrels[str(row["query_id"])].add(f"{row['doc_id']}#page={int(row['page_index'])}")
    return dict(qrels)


def run_score_based(
    *, page_bm: np.ndarray, file_bm: np.ndarray, dense: np.ndarray, page_ids: list[str], file_ids: list[str], languages: dict[str, str], mode: str,
) -> list[dict[str, Any]]:
    page_files = [file_id(pid) for pid in page_ids]
    page_count = len(page_ids)
    file_pos = {fid: i for i, fid in enumerate(file_ids)}
    page_file_pos = np.asarray([file_pos[fid] for fid in page_files], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    norm = max_norm if mode == "max_current" else minmax_norm
    for qi in range(dense.shape[0]):
        bm = norm(page_bm[qi])
        de = norm(dense[qi])
        direct = np.zeros(len(file_ids), dtype=np.float64)
        page_score = np.zeros(page_count, dtype=np.float64)
        for fi, fid in enumerate(file_ids):
            mask = page_file_pos == fi
            if languages.get(fid, "en") == "en":
                page_score[mask] = 0.70 * bm[mask] + 0.30 * de[mask]
                direct[fi] = 0.70 * float(norm(file_bm[qi])[fi]) + 0.30 * float(de[mask].max(initial=0.0))
            else:
                page_score[mask] = de[mask]
                direct[fi] = float(de[mask].max(initial=0.0))
        page_max = np.zeros(len(file_ids), dtype=np.float64)
        np.maximum.at(page_max, page_file_pos, page_score)
        file_score = 0.50 * direct + 0.50 * norm(page_max)
        selected = np.argsort(-file_score, kind="stable")[:3]
        selected_mask = np.isin(page_file_pos, selected)
        final = 0.85 * page_score + 0.15 * norm(file_score)[page_file_pos]
        positions = np.flatnonzero(selected_mask)
        positions = positions[np.argsort(-final[positions], kind="stable")]
        rows.append({
            "selected_files": [file_ids[int(i)] for i in selected],
            "candidate_pages": [page_ids[int(i)] for i in positions[:200]],
            "returned_pages": [page_ids[int(i)] for i in positions[:20]],
            "returned_scores": [float(final[int(i)]) for i in positions[:20]],
        })
    return rows


def run_rrf(*, page_bm: np.ndarray, file_bm: np.ndarray, dense: np.ndarray, page_ids: list[str], file_ids: list[str], languages: dict[str, str], k: int = 60) -> list[dict[str, Any]]:
    page_files = [file_id(pid) for pid in page_ids]
    file_pos = {fid: i for i, fid in enumerate(file_ids)}
    page_file_pos = np.asarray([file_pos[fid] for fid in page_files], dtype=np.int64)
    pages_by_file = {fi: np.flatnonzero(page_file_pos == fi) for fi in range(len(file_ids))}
    rows: list[dict[str, Any]] = []
    for qi in range(dense.shape[0]):
        dense_page_order = rank_positions(dense[qi])
        dense_page_rank = {pos: rank for rank, pos in enumerate(dense_page_order, 1)}
        dense_file_best = np.full(len(file_ids), -np.inf, dtype=np.float64)
        for pos in dense_page_order:
            dense_file_best[page_file_pos[pos]] = max(dense_file_best[page_file_pos[pos]], float(dense[qi, pos]))
        dense_file_order = rank_positions(dense_file_best)
        dense_file_rank = {fi: rank for rank, fi in enumerate(dense_file_order, 1)}

        eligible_files = [fi for fi, fid in enumerate(file_ids) if languages.get(fid, "en") == "en"]
        bm_file_order = rank_positions(file_bm[qi], eligible_files)
        bm_file_rank = {fi: rank for rank, fi in enumerate(bm_file_order, 1) if file_bm[qi, fi] > 0}
        file_rrf = np.asarray([1.0 / (k + dense_file_rank[fi]) + (1.0 / (k + bm_file_rank[fi]) if fi in bm_file_rank else 0.0) for fi in range(len(file_ids))])
        selected = np.argsort(-file_rrf, kind="stable")[:3]

        selected_positions = np.concatenate([pages_by_file[int(fi)] for fi in selected]) if len(selected) else np.asarray([], dtype=np.int64)
        selected_set = set(int(x) for x in selected_positions)
        bm_page_order = rank_positions(page_bm[qi], selected_positions)
        bm_page_rank = {pos: rank for rank, pos in enumerate(bm_page_order, 1) if page_bm[qi, pos] > 0 and languages.get(page_files[pos], "en") == "en"}
        page_rrf = []
        for pos in selected_set:
            score = 1.0 / (k + dense_page_rank[pos])
            if pos in bm_page_rank:
                score += 1.0 / (k + bm_page_rank[pos])
            page_rrf.append((pos, score))
        page_rrf.sort(key=lambda item: (-item[1], item[0]))
        rows.append({
            "selected_files": [file_ids[int(i)] for i in selected],
            "candidate_pages": [page_ids[pos] for pos, _ in page_rrf[:200]],
            "returned_pages": [page_ids[pos] for pos, _ in page_rrf[:20]],
            "returned_scores": [float(score) for _, score in page_rrf[:20]],
        })
    return rows


def run_rrf_page_only(*, page_bm: np.ndarray, file_bm: np.ndarray, dense: np.ndarray, page_ids: list[str], file_ids: list[str], base_rows: list[dict[str, Any]], languages: dict[str, str], k: int = 60) -> list[dict[str, Any]]:
    """Replace only page fusion with RRF; preserve current file parent boost."""
    page_files = [file_id(pid) for pid in page_ids]
    file_pos = {fid: i for i, fid in enumerate(file_ids)}
    page_file_pos = np.asarray([file_pos[fid] for fid in page_files], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for qi, base in enumerate(base_rows):
        selected_files = set(base["selected_files"])
        selected_positions = np.asarray([i for i, fid in enumerate(page_files) if fid in selected_files], dtype=np.int64)
        bm = max_norm(page_bm[qi])
        de = max_norm(dense[qi])
        direct = np.zeros(len(file_ids), dtype=np.float64)
        page_base = np.zeros(len(page_ids), dtype=np.float64)
        for fi, fid in enumerate(file_ids):
            mask = page_file_pos == fi
            if languages.get(fid, "en") == "en":
                page_base[mask] = 0.70 * bm[mask] + 0.30 * de[mask]
                direct[fi] = 0.70 * float(max_norm(file_bm[qi])[fi]) + 0.30 * float(de[mask].max(initial=0.0))
            else:
                page_base[mask] = de[mask]
                direct[fi] = float(de[mask].max(initial=0.0))
        page_max = np.zeros(len(file_ids), dtype=np.float64)
        np.maximum.at(page_max, page_file_pos, page_base)
        file_score = 0.50 * direct + 0.50 * max_norm(page_max)
        parent = max_norm(file_score)[page_file_pos]
        dense_order = rank_positions(dense[qi], selected_positions)
        dense_rank = {pos: rank for rank, pos in enumerate(dense_order, 1)}
        bm_order = rank_positions(page_bm[qi], selected_positions)
        bm_rank = {pos: rank for rank, pos in enumerate(bm_order, 1) if page_bm[qi, pos] > 0 and languages.get(page_files[pos], "en") == "en"}
        rrf_values = {}
        for pos in selected_positions:
            score = 1.0 / (k + dense_rank[int(pos)])
            if int(pos) in bm_rank:
                score += 1.0 / (k + bm_rank[int(pos)])
            rrf_values[int(pos)] = score
        rrf_peak = max(rrf_values.values(), default=0.0)
        ranked = [(pos, 0.85 * (score / rrf_peak if rrf_peak > 0 else 0.0) + 0.15 * parent[pos]) for pos, score in rrf_values.items()]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        rows.append({
            "selected_files": list(base["selected_files"]),
            "candidate_pages": [page_ids[pos] for pos, _ in ranked[:200]],
            "returned_pages": [page_ids[pos] for pos, _ in ranked[:20]],
            "returned_scores": [float(score) for _, score in ranked[:20]],
        })
    return rows


def metrics(rows: list[dict[str, Any]], qids: list[str], qrels: dict[str, set[str]], depth: int) -> dict[str, float]:
    recall, hit, dcg, files, ceiling = [], [], [], [], []
    for row, qid in zip(rows, qids):
        gold = qrels[qid]
        ranked = row["returned_pages"]
        top = set(ranked[:depth])
        recall.append(len(top & gold) / max(len(gold), 1))
        hit.append(float(bool(top & gold)))
        dcg.append(ndcg(ranked, gold, depth))
        files.append(len(set(row["selected_files"]) & {file_id(page) for page in gold}) / max(len({file_id(page) for page in gold}), 1))
        ceiling.append(len(set(row["candidate_pages"]) & gold) / max(len(gold), 1))
    return {f"page_recall@{depth}": float(np.mean(recall)), f"page_hit@{depth}": float(np.mean(hit)), f"ndcg@{depth}": 100.0 * float(np.mean(dcg)), "file_recall@3": float(np.mean(files)), "candidate_pool_recall@200": float(np.mean(ceiling))}


def main() -> None:
    meta = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    qids, page_ids = meta["query_ids"], meta["page_ids"]
    page_bm = np.load(BASE / "page_bm25_raw.npy", mmap_mode="r")
    file_bm = np.load(BASE / "file_bm25_raw.npy", mmap_mode="r")
    dense = np.load(BASE / "query_page_dense.npy", mmap_mode="r")
    if page_bm.shape != dense.shape or page_bm.shape[0] != len(qids) or page_bm.shape[1] != len(page_ids):
        raise RuntimeError(f"Unexpected matrix shapes: {page_bm.shape}, {file_bm.shape}, {dense.shape}")
    qrels = load_qrels()
    languages = load_language()
    max_current = run_score_based(page_bm=page_bm, file_bm=file_bm, dense=dense, page_ids=page_ids, file_ids=meta["file_ids"], languages=languages, mode="max_current")
    runs = {
        "max_current": max_current,
        "minmax": run_score_based(page_bm=page_bm, file_bm=file_bm, dense=dense, page_ids=page_ids, file_ids=meta["file_ids"], languages=languages, mode="minmax"),
        "rrf_page_k60": run_rrf_page_only(page_bm=page_bm, file_bm=file_bm, dense=dense, page_ids=page_ids, file_ids=meta["file_ids"], base_rows=max_current, languages=languages),
        "rrf_full_k60": run_rrf(page_bm=page_bm, file_bm=file_bm, dense=dense, page_ids=page_ids, file_ids=meta["file_ids"], languages=languages),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "protocol": {"queries": 220, "files": 103, "pages": 5091, "file_k": 3, "candidate_depth": 200, "saved_depth": 20, "language_route": "BM25+dense only for detected English files; dense-only otherwise", "qrels_used_for_ranking": False},
        "arms": {},
    }
    groups = {"all_220": lambda q: True, "physics": lambda q: q.startswith("vidore_physics::"), "industrial": lambda q: q.startswith("vidore_industrial::"), "ohr": lambda q: q.startswith("ohrbench::")}
    for name, rows in runs.items():
        out_rows = [{"query_id": qid, **row} for qid, row in zip(qids, rows)]
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in out_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        report["arms"][name] = {}
        for group, predicate in groups.items():
            selected_qids = [qid for qid in qids if predicate(qid)]
            selected_rows = [row for row, qid in zip(rows, qids) if predicate(qid)]
            report["arms"][name][group] = {"queries": len(selected_qids), "at10": metrics(selected_rows, selected_qids, qrels, 10), "at20": metrics(selected_rows, selected_qids, qrels, 20)}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Fusion normalization screen", "", "Cached matrices only; no parsing, embedding or qrel-guided ranking.", "", "| Arm | Group | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 | Candidate recall@200 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm, groups_data in report["arms"].items():
        for group, value in groups_data.items():
            a10, a20 = value["at10"], value["at20"]
            lines.append(f"| {arm} | {group} | {a10['page_recall@10']:.2%} | {a10['ndcg@10']:.2f} | {a20['page_recall@20']:.2%} | {a20['ndcg@20']:.2f} | {a10['file_recall@3']:.2%} | {a10['candidate_pool_recall@200']:.2%} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
