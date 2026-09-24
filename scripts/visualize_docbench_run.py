#!/usr/bin/env python3
"""Build a self-contained HTML review board for one DocBench run.

The generated page keeps retrieval and QA attached to the same question. It is
intended for qualitative debugging: a reviewer can move through questions,
compare the gold answer with the generated answer, inspect the ranked pages and
chunks, and see the timings/statuses that explain a bad result.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def _read_latest_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row.get("qid") or "")
        if qid:
            rows[qid] = row
    return rows


def _version(path: Path) -> int:
    match = re.search(r"_ver(\d+)", path.name)
    return int(match.group(1)) if match else 0


def _latest_file(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda path: (_version(path), path.stat().st_mtime)) if paths else None


def _json_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _unique(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def _dcg(gains: list[float]) -> float:
    return sum(
        (2.0**gain - 1.0) / (1.0 if rank == 1 else math.log2(rank))
        for rank, gain in enumerate(gains, start=1)
    )


def _score(ranked: list[str], grades: dict[str, float], k: int) -> dict[str, float]:
    unique = _unique(ranked[:k])
    found = sum(1 for doc_id in grades if doc_id in unique)
    gains = [grades.get(doc_id, 0.0) for doc_id in unique]
    ideal = _dcg(sorted(grades.values(), reverse=True)[:k])
    return {
        "recall": found / len(grades) if grades else 0.0,
        "ndcg": _dcg(gains) / ideal if ideal else 0.0,
    }


def _row_payload(retrieval: dict[str, Any], qa: dict[str, Any]) -> dict[str, Any]:
    base = qa or retrieval
    hits = retrieval.get("hits") or []
    chunks = retrieval.get("chunks") or []
    gold_qrels = base.get("gold_qrels") or retrieval.get("gold_qrels") or []
    gold_page_qrels = base.get("gold_page_qrels") or retrieval.get("gold_page_qrels") or []
    file_grades = {
        str(item.get("doc_id") or ""): float(item.get("relevance") or 0.0)
        for item in gold_qrels
        if item.get("doc_id")
    }
    page_grades = {
        str(item.get("page_id") or ""): float(item.get("relevance") or 0.0)
        for item in gold_page_qrels
        if item.get("page_id")
    }
    light_pages = [str(item.get("page_id") or "") for item in hits]
    light_files = [str(item.get("source_uri") or "") for item in hits]
    accurate_pages = [str(item.get("doc_id") or "") for item in chunks]
    context_pages = set(str(item) for item in qa.get("context_page_ids") or [])
    context_chunks = set(str(item) for item in qa.get("context_unit_ids") or [])
    page_to_file = {
        str(item.get("page_id") or ""): str(item.get("source_uri") or "")
        for item in hits
        if item.get("page_id")
    }
    light_page_ranks = {
        page_id: index + 1 for index, page_id in enumerate(light_pages) if page_id
    }
    accurate_page_ranks: dict[str, int] = {}
    for index, page_id in enumerate(accurate_pages, start=1):
        if page_id and page_id not in accurate_page_ranks:
            accurate_page_ranks[page_id] = index
    light_10 = _score(light_pages, page_grades, 10)
    light_20 = _score(light_pages, page_grades, 20)
    accurate_10 = _score(accurate_pages, page_grades, 10)
    accurate_20 = _score(accurate_pages, page_grades, 20)
    file_20 = _score(light_files, file_grades, 20)
    timing = retrieval.get("timing") or qa.get("retrieval_timing") or {}
    sys_ans = str(qa.get("sys_ans") or "")
    abstention = "KHONG_DU_THONG_TIN" in sys_ans.upper()
    return {
        "qid": str(base.get("qid") or retrieval.get("qid") or ""),
        "question": str(base.get("question") or retrieval.get("question") or ""),
        "answer": str(base.get("answer") or retrieval.get("answer") or ""),
        "evidence": str(base.get("evidence") or retrieval.get("evidence") or ""),
        "domain": str(base.get("domain") or retrieval.get("domain") or "Unknown"),
        "type_group": str(base.get("type_group") or retrieval.get("type_group") or ""),
        "doc_id": str(base.get("doc_id") or retrieval.get("doc_id") or ""),
        "source": str(base.get("source") or retrieval.get("source") or ""),
        "gold_qrels": gold_qrels,
        "gold_page_qrels": gold_page_qrels,
        "retrieval": {
            "status": str(retrieval.get("status") or "missing"),
            "error": retrieval.get("error"),
            "parse_complete": retrieval.get("parse_complete"),
            "parse_failed_page_ids": retrieval.get("parse_failed_page_ids") or [],
            "selected_pages": retrieval.get("selected_pages") or [],
            "retrieval_config": retrieval.get("retrieval_config") or {},
            "retrieval_config_hash": retrieval.get("retrieval_config_hash"),
            "hits": hits,
            "chunks": chunks,
            "timing": timing,
            "light_page_ranks": light_page_ranks,
            "accurate_page_ranks": accurate_page_ranks,
            "page_to_file": page_to_file,
            "metrics": {
                "light_recall_10": light_10["recall"],
                "light_recall_20": light_20["recall"],
                "light_ndcg_10": light_10["ndcg"],
                "light_ndcg_20": light_20["ndcg"],
                "accurate_recall_10": accurate_10["recall"],
                "accurate_recall_20": accurate_20["recall"],
                "accurate_ndcg_10": accurate_10["ndcg"],
                "accurate_ndcg_20": accurate_20["ndcg"],
                "file_recall_20": file_20["recall"],
            },
        },
        "qa": {
            "status": str(qa.get("status") or "missing"),
            "error": qa.get("error"),
            "score": qa.get("score"),
            "sys_ans": sys_ans,
            "initial_sys_ans": qa.get("initial_sys_ans"),
            "judge_raw": qa.get("judge_raw") or "",
            "generator": qa.get("generator") or "",
            "judge": qa.get("judge") or "",
            "chunks_used": qa.get("chunks_used"),
            "chars_used": qa.get("chars_used"),
            "context_doc_ids": qa.get("context_doc_ids") or [],
            "context_page_ids": sorted(context_pages),
            "context_unit_ids": sorted(context_chunks),
            "retrieved_page_count": qa.get("retrieved_page_count"),
            "infer_time_seconds": qa.get("infer_time_seconds"),
            "unanswerable_retry_count": qa.get("unanswerable_retry_count", 0),
            "qa_config_hash": qa.get("qa_config_hash"),
            "abstention_like": abstention,
        },
    }


def _discover(run_dir: Path, scope: str | None) -> tuple[str, Path, Path | None]:
    retrieval_dir = run_dir / "retrieval"
    candidates = sorted(retrieval_dir.glob("*_baseline_legacy.jsonl"))
    if scope:
        candidates = [path for path in candidates if path.name == f"{scope}_baseline_legacy.jsonl"]
    if not candidates:
        raise FileNotFoundError(f"No retrieval JSONL found under {retrieval_dir}")
    retrieval_path = max(candidates, key=lambda path: path.stat().st_mtime)
    run_scope = retrieval_path.name.removesuffix("_baseline_legacy.jsonl")
    qa_path = run_dir / "qa" / f"{run_scope}_baseline_legacy.jsonl"
    return run_scope, retrieval_path, qa_path if qa_path.is_file() else None


def _build_payload(run_dir: Path, scope: str | None) -> dict[str, Any]:
    run_scope, retrieval_path, qa_path = _discover(run_dir, scope)
    retrieval_rows = _read_latest_jsonl(retrieval_path)
    qa_rows = _read_latest_jsonl(qa_path) if qa_path else {}
    ordered_qids = list(retrieval_rows)
    ordered_qids.extend(qid for qid in qa_rows if qid not in retrieval_rows)
    questions = [
        _row_payload(retrieval_rows.get(qid, {}), qa_rows.get(qid, {}))
        for qid in ordered_qids
    ]
    report_path = _latest_file(
        list((run_dir / "reports").glob(f"{run_scope}_baseline_legacy_ver*.json"))
    )
    timing_path = _latest_file(
        list((run_dir / "reports").glob(f"{run_scope}_timing_summary_ver*.json"))
    )
    manifest_path = _latest_file(list(run_dir.glob(f"manifest_{run_scope}_ver*.json")))
    return {
        "run": {
            "run_dir": str(run_dir),
            "scope": run_scope,
            "retrieval_path": str(retrieval_path),
            "qa_path": str(qa_path) if qa_path else None,
            "report_path": str(report_path) if report_path else None,
            "timing_path": str(timing_path) if timing_path else None,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "report": _json_file(report_path),
            "timing_summary": _json_file(timing_path),
            "manifest": _json_file(manifest_path),
        },
        "questions": questions,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DocBench run review</title>
<style>
:root {
  --ink: #172333; --muted: #667386; --paper: #f7f4ee; --card: #fffdf9;
  --line: #dcd9d0; --navy: #15324b; --teal: #167b79; --coral: #d65b45;
  --gold: #b98525; --green: #277b56; --red: #b6423d; --shadow: 0 14px 36px #18324712;
}
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: var(--paper); font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif; }
button, input, select { font: inherit; }
button { cursor: pointer; }
.shell { min-height: 100vh; background: radial-gradient(circle at 90% 0%, #dcebe4 0, transparent 30%), radial-gradient(circle at 10% 30%, #f3dfd3 0, transparent 26%); }
.topbar { padding: 28px clamp(18px, 4vw, 56px) 22px; color: #f5f4ed; background: linear-gradient(120deg, #122d45, #1c5262 65%, #227873); position: sticky; top: 0; z-index: 10; box-shadow: 0 8px 30px #102c3a38; }
.eyebrow { margin: 0 0 4px; color: #a6dbd1; font-size: 11px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; line-height: 1.15; }
h1 { margin: 0; font-size: clamp(25px, 4vw, 42px); letter-spacing: -.03em; }
.run-meta { margin-top: 8px; color: #c9e0dc; font-size: 12px; word-break: break-all; }
.layout { display: grid; grid-template-columns: minmax(270px, 345px) minmax(0, 1fr); max-width: 1800px; margin: 0 auto; }
.rail { min-height: calc(100vh - 135px); padding: 20px 16px; border-right: 1px solid var(--line); background: #f2eee6d9; }
.rail-tools { position: sticky; top: 154px; z-index: 2; }
.search { width: 100%; padding: 11px 13px; border: 1px solid #c9c9bf; border-radius: 12px; outline: none; background: #fffdf9; }
.search:focus { border-color: var(--teal); box-shadow: 0 0 0 3px #167b7924; }
.filters { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin: 10px 0; }
.filters select { min-width: 0; padding: 8px 7px; border: 1px solid #d1cec5; border-radius: 9px; color: var(--ink); background: #fffdf9; }
.result-count { margin: 8px 2px 12px; color: var(--muted); font-size: 12px; }
.question-list { display: grid; gap: 7px; }
.q-item { width: 100%; padding: 11px 12px; border: 1px solid transparent; border-radius: 13px; color: var(--ink); text-align: left; background: #fffdf9b5; transition: transform .15s, border .15s, box-shadow .15s; }
.q-item:hover { transform: translateY(-1px); border-color: #b6d2cc; box-shadow: 0 5px 14px #17374c12; }
.q-item.active { border-color: var(--teal); box-shadow: inset 4px 0 var(--teal), 0 6px 16px #17374c18; background: #fff; }
.q-line { display: flex; align-items: center; gap: 7px; }
.q-id { flex: 1; overflow: hidden; color: var(--muted); font: 11px ui-monospace, SFMono-Regular, Consolas, monospace; text-overflow: ellipsis; white-space: nowrap; }
.q-text { display: -webkit-box; margin-top: 4px; overflow: hidden; font-weight: 650; line-height: 1.3; -webkit-box-orient: vertical; -webkit-line-clamp: 2; }
.q-sub { display: flex; gap: 7px; margin-top: 7px; color: var(--muted); font-size: 11px; }
.content { min-width: 0; padding: 25px clamp(16px, 3vw, 45px) 60px; }
.summary { display: grid; grid-template-columns: repeat(6, minmax(100px, 1fr)); gap: 10px; margin-bottom: 22px; }
.stat { padding: 13px 14px; border: 1px solid var(--line); border-radius: 14px; background: #fffdf9cf; box-shadow: var(--shadow); }
.stat-label { color: var(--muted); font-size: 10px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
.stat-value { margin-top: 2px; color: var(--navy); font: 700 25px Georgia, serif; }
.stat-note { color: var(--muted); font-size: 11px; }
.empty { padding: 60px 20px; border: 1px dashed #c6c5bb; border-radius: 18px; color: var(--muted); text-align: center; background: #fffdf980; }
.question-head { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 15px; }
.question-head h2 { margin: 0; color: var(--navy); font-size: clamp(24px, 3vw, 35px); }
.qid { margin: 7px 0 0; color: var(--muted); font: 12px ui-monospace, SFMono-Regular, Consolas, monospace; word-break: break-all; }
.nav-buttons { display: flex; gap: 7px; flex-shrink: 0; }
.nav-buttons button, .copy-btn { padding: 8px 11px; border: 1px solid #c8d3d0; border-radius: 9px; color: var(--navy); background: #eff6f3; }
.nav-buttons button:hover, .copy-btn:hover { background: #dcebe4; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 13px 0 19px; }
.badge { display: inline-flex; align-items: center; min-height: 22px; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 800; letter-spacing: .01em; }
.score-1, .ok { color: #185f3d; background: #cdebd9; }
.score-05, .warn { color: #815d15; background: #f5e4b8; }
.score-0, .bad { color: #8f332f; background: #f3c9c4; }
.neutral { color: #536171; background: #e4e7e5; }
.grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 15px; }
.card { min-width: 0; margin-bottom: 15px; padding: 18px; border: 1px solid var(--line); border-radius: 16px; background: var(--card); box-shadow: var(--shadow); }
.card h3 { margin: 0 0 10px; color: var(--navy); font-size: 19px; }
.card-kicker { margin: -5px 0 12px; color: var(--muted); font-size: 12px; }
.answer-box { min-height: 90px; padding: 13px; border-left: 4px solid var(--teal); border-radius: 7px; background: #eaf3ef; white-space: pre-wrap; }
.answer-box.generated { border-color: var(--coral); background: #fff0e9; }
.answer-box.reference { border-color: var(--gold); background: #fbf4df; }
.answer-box.empty-answer { color: var(--muted); font-style: italic; }
.answer-label { margin: 13px 0 5px; color: var(--muted); font-size: 10px; font-weight: 850; letter-spacing: .1em; text-transform: uppercase; }
.judge-line { display: flex; align-items: center; gap: 12px; margin-top: 12px; }
.big-score { display: grid; width: 58px; height: 58px; place-items: center; border-radius: 14px; color: white; font: 700 26px Georgia, serif; background: var(--navy); }
.judge-raw { color: var(--muted); font-size: 12px; }
.metric-grid { display: grid; grid-template-columns: repeat(4, minmax(90px, 1fr)); gap: 8px; }
.metric { padding: 9px; border-radius: 10px; background: #f0f2ed; }
.metric strong { display: block; color: var(--navy); font-size: 17px; }
.metric span { color: var(--muted); font-size: 10px; }
.gold-list { display: flex; flex-wrap: wrap; gap: 6px; }
.gold-pill { padding: 5px 8px; border: 1px solid #dfc985; border-radius: 8px; color: #765313; background: #fff8e6; font: 11px ui-monospace, SFMono-Regular, Consolas, monospace; }
.tabs { display: flex; gap: 5px; margin-bottom: 10px; border-bottom: 1px solid var(--line); }
.tab { padding: 9px 12px; border: 0; border-bottom: 3px solid transparent; color: var(--muted); background: transparent; font-weight: 750; }
.tab.active { border-color: var(--teal); color: var(--navy); }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.rank-card { margin: 9px 0; padding: 13px; border: 1px solid #deddd5; border-radius: 12px; background: #fff; }
.rank-card.is-gold { border-color: #d7af52; box-shadow: inset 4px 0 #d7af52; }
.rank-card.is-context { border-color: #8bc2b1; box-shadow: inset 4px 0 #2c927b; }
.gold-table { overflow-x: auto; }
.gold-table table { min-width: 720px; }
.gold-table tr.hit-row { background: #f0f8f2; }
.gold-table tr.miss-row { background: #fff7f3; }
.gold-page-id { color: var(--navy); font: 11px ui-monospace, SFMono-Regular, Consolas, monospace; word-break: break-all; }
.hit, .miss { display: inline-flex; align-items: center; gap: 5px; padding: 3px 7px; border-radius: 7px; font-size: 11px; font-weight: 850; white-space: nowrap; }
.hit { color: #185f3d; background: #cdebd9; }
.miss { color: #8f332f; background: #f3c9c4; }
.rank-link { border: 0; color: var(--teal); background: transparent; font: 700 12px ui-monospace, SFMono-Regular, Consolas, monospace; text-decoration: underline; text-decoration-style: dotted; }
.rank-link:hover { color: var(--navy); }
.coverage-strip { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 12px; }
.coverage-item { padding: 8px 10px; border-radius: 9px; background: #f0f2ed; }
.coverage-item strong { color: var(--navy); font-size: 17px; }
.coverage-item span { margin-left: 4px; color: var(--muted); font-size: 11px; }
.rank-head { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; }
.rank-number { display: grid; width: 28px; height: 28px; place-items: center; border-radius: 8px; color: #fff; background: var(--navy); font-weight: 800; }
.rank-title { flex: 1; min-width: 180px; color: var(--navy); font-weight: 750; word-break: break-word; }
.rank-score { color: var(--coral); font: 700 13px ui-monospace, SFMono-Regular, Consolas, monospace; }
.rank-meta { display: flex; flex-wrap: wrap; gap: 5px 12px; margin: 7px 0; color: var(--muted); font: 11px ui-monospace, SFMono-Regular, Consolas, monospace; word-break: break-all; }
.rank-text { max-height: 260px; overflow: auto; padding: 11px; border-radius: 8px; color: #263849; background: #f5f6f2; white-space: pre-wrap; }
.rank-text.collapsed { max-height: 92px; }
.text-toggle { padding: 4px 0; border: 0; color: var(--teal); background: transparent; font-size: 11px; font-weight: 750; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 8px 9px; border-bottom: 1px solid #e5e3dc; text-align: left; vertical-align: top; }
th { color: var(--muted); font-size: 10px; letter-spacing: .07em; text-transform: uppercase; }
td code { color: #365267; font: 11px ui-monospace, SFMono-Regular, Consolas, monospace; word-break: break-all; }
details { margin-top: 10px; }
summary { cursor: pointer; color: var(--teal); font-weight: 750; }
.raw { max-height: 430px; overflow: auto; padding: 12px; border-radius: 9px; background: #182a37; color: #dceee9; font: 11px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; }
.notice { padding: 10px 12px; border-radius: 9px; color: #805015; background: #fff0c8; font-size: 12px; }
@media (max-width: 1100px) { .summary { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 780px) { .layout { display: block; } .rail { min-height: auto; border-right: 0; border-bottom: 1px solid var(--line); } .rail-tools { position: static; } .question-list { max-height: 330px; overflow: auto; } .grid-2 { grid-template-columns: 1fr; } .summary { grid-template-columns: repeat(2, 1fr); } .question-head { display: block; } .nav-buttons { margin-top: 12px; } }
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <p class="eyebrow">DocBench / Evidence Review</p>
    <h1>Retrieval + QA field notes</h1>
    <div id="run-meta" class="run-meta"></div>
  </header>
  <div class="layout">
    <aside class="rail">
      <div class="rail-tools">
        <input id="search" class="search" type="search" placeholder="Search qid, question, answer...">
        <div class="filters">
          <select id="score-filter"><option value="all">All scores</option><option value="1">Score 1</option><option value="0.5">Score 0.5</option><option value="0">Score 0</option><option value="missing">No QA score</option></select>
          <select id="retrieval-filter"><option value="all">All retrieval</option><option value="hit">Gold page in top 20</option><option value="miss">No gold page top 20</option><option value="parse-fail">Parse incomplete</option></select>
        </div>
        <select id="sort" class="search" aria-label="Sort questions">
          <option value="original">Original order</option><option value="score-asc">Lowest score first</option><option value="score-desc">Highest score first</option><option value="retrieval-asc">Slowest retrieval first</option>
        </select>
        <div id="result-count" class="result-count"></div>
        <div id="question-list" class="question-list"></div>
      </div>
    </aside>
    <main class="content">
      <section id="summary" class="summary"></section>
      <section id="detail"></section>
    </main>
  </div>
</div>
<script>
const DATA = __PAYLOAD__;
const questions = DATA.questions;
const state = { filtered: [], selected: 0, tab: 'pages' };
const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (value, digits=3) => value == null || Number.isNaN(Number(value)) ? '—' : Number(value).toFixed(digits);
const pct = (value) => value == null || Number.isNaN(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(1)}%`;
const scoreLabel = (score) => score == null ? '—' : Number(score) === 1 ? '1' : Number(score) === .5 ? '.5' : '0';
const scoreClass = (score) => score == null ? 'neutral' : Number(score) === 1 ? 'score-1' : Number(score) === .5 ? 'score-05' : 'score-0';
const safeJson = (value) => esc(JSON.stringify(value, null, 2));
function timing(row) { return row.retrieval.timing || {}; }
function pageHit(row) { return Object.keys(row.retrieval.light_page_ranks || {}).some(id => (row.gold_page_qrels || []).some(g => g.page_id === id)); }
function retrievalSeconds(row) { return Number(timing(row).overall_seconds || timing(row).query_e2e_seconds || 0); }
function qaScore(row) { return row.qa.score == null ? -1 : Number(row.qa.score); }
function firstPageHit(row, pageId) { return (row.retrieval.hits || []).find(item => item.page_id === pageId) || null; }
function firstChunkHit(row, pageId) { return (row.retrieval.chunks || []).find(item => item.doc_id === pageId) || null; }
function jumpTo(id) { const node = document.getElementById(id); if (node) { node.scrollIntoView({behavior: 'smooth', block: 'center'}); node.animate([{backgroundColor:'#d8f1e8'}, {backgroundColor:''}], {duration:900}); } }
function metricCards(row) {
  const m = row.retrieval.metrics || {};
  return [['Light R@10', pct(m.light_recall_10)], ['Light R@20', pct(m.light_recall_20)], ['Light NDCG@10', num(m.light_ndcg_10)], ['Accurate R@10', pct(m.accurate_recall_10)], ['Accurate NDCG@10', num(m.accurate_ndcg_10)], ['File R@20', pct(m.file_recall_20)]];
}
function badge(text, cls='neutral') { return `<span class="badge ${cls}">${esc(text)}</span>`; }
function summary() {
  const qa = questions.filter(q => q.qa.status === 'ok' && q.qa.score != null);
  const scores = qa.map(q => Number(q.qa.score));
  const hits = questions.filter(pageHit).length;
  const parses = questions.filter(q => q.retrieval.parse_complete !== false).length;
  const avg = values => values.length ? values.reduce((a,b) => a+b, 0) / values.length : null;
  const cards = [['Questions', questions.length, 'retrieval rows'], ['QA complete', qa.length, `${questions.length - qa.length} pending/error`], ['Accuracy', pct(avg(scores)), 'mean judge score'], ['Full / partial', `${scores.filter(s => s === 1).length} / ${scores.filter(s => s === .5).length}`, 'score 1 / 0.5'], ['Page hit@20', pct(questions.length ? hits / questions.length : null), `${hits} questions`], ['Avg retrieval', `${num(avg(questions.map(retrievalSeconds)), 1)}s`, `${parses}/${questions.length} parse complete`]];
  $('summary').innerHTML = cards.map(([label, value, note]) => `<div class="stat"><div class="stat-label">${esc(label)}</div><div class="stat-value">${esc(value)}</div><div class="stat-note">${esc(note)}</div></div>`).join('');
  const report = DATA.run.report || {};
  $('run-meta').textContent = `${DATA.run.scope} · ${DATA.run.run_dir} · report ${report.report_version ? `v${report.report_version}` : 'not found'}`;
}
function updateFiltered() {
  const query = $('search').value.trim().toLowerCase();
  const score = $('score-filter').value;
  const retrieval = $('retrieval-filter').value;
  state.filtered = questions.map((q, i) => ({q, i})).filter(({q}) => {
    const haystack = [q.qid, q.question, q.answer, q.qa.sys_ans, q.domain, q.doc_id].join(' ').toLowerCase();
    if (query && !haystack.includes(query)) return false;
    if (score === 'missing' && q.qa.score != null) return false;
    if (score !== 'all' && score !== 'missing' && String(q.qa.score) !== score) return false;
    if (retrieval === 'hit' && !pageHit(q)) return false;
    if (retrieval === 'miss' && pageHit(q)) return false;
    if (retrieval === 'parse-fail' && q.retrieval.parse_complete !== false) return false;
    return true;
  });
  const sort = $('sort').value;
  if (sort === 'score-asc') state.filtered.sort((a,b) => qaScore(a.q) - qaScore(b.q));
  if (sort === 'score-desc') state.filtered.sort((a,b) => qaScore(b.q) - qaScore(a.q));
  if (sort === 'retrieval-asc') state.filtered.sort((a,b) => retrievalSeconds(b.q) - retrievalSeconds(a.q));
  if (!state.filtered.some(item => item.i === state.selected)) state.selected = state.filtered[0]?.i ?? -1;
  renderList(); renderDetail();
}
function renderList() {
  $('result-count').textContent = `${state.filtered.length} of ${questions.length} questions`;
  $('question-list').innerHTML = state.filtered.length ? state.filtered.map(({q, i}) => {
    const rcls = q.retrieval.status === 'ok' ? (pageHit(q) ? 'ok' : 'warn') : 'bad';
    return `<button class="q-item ${i === state.selected ? 'active' : ''}" data-index="${i}"><div class="q-line"><span class="q-id">${esc(q.qid)}</span>${badge(scoreLabel(q.qa.score), scoreClass(q.qa.score))}</div><div class="q-text">${esc(q.question)}</div><div class="q-sub"><span>${esc(q.domain)}</span><span>${badge(pageHit(q) ? 'page hit' : 'page miss', rcls)}</span></div></button>`;
  }).join('') : '<div class="empty">No questions match these filters.</div>';
  document.querySelectorAll('.q-item').forEach(button => button.addEventListener('click', () => { state.selected = Number(button.dataset.index); state.tab = 'pages'; renderList(); renderDetail(); history.replaceState(null, '', `#${encodeURIComponent(questions[state.selected].qid)}`); }));
}
function answerBox(label, text, cls) { return `<div class="answer-label">${esc(label)}</div><div class="answer-box ${cls} ${text ? '' : 'empty-answer'}">${esc(text || 'No answer recorded')}</div>`; }
function rankCell(item, kind, q) {
  if (!item) return '<span class="miss">MISS</span>';
  const rank = item.rank ?? ((kind === 'page' ? q.retrieval.hits : q.retrieval.chunks).indexOf(item) + 1);
  const anchor = kind === 'page' ? `rank-page-${q.retrieval.hits.indexOf(item)}` : `rank-chunk-${q.retrieval.chunks.indexOf(item)}`;
  return `<button class="rank-link" onclick="jumpTo('${anchor}')">#${esc(rank)}</button><span class="rank-score">(${num(item.score, 3)})</span>`;
}
function goldPageComparison(q) {
  const pages = q.gold_page_qrels || [];
  const topPages = Number(q.retrieval.retrieval_config?.top_k_pages || 20);
  const topChunks = Number(q.retrieval.retrieval_config?.top_k_chunks || 10);
  let lightHits = 0, accurateHits = 0, contextHits = 0;
  const rows = pages.map((gold) => {
    const pageId = String(gold.page_id || '');
    const page = firstPageHit(q, pageId);
    const chunk = firstChunkHit(q, pageId);
    const lightRank = page ? Number(page.rank || q.retrieval.hits.indexOf(page) + 1) : null;
    const chunkRank = chunk ? Number(chunk.rank || q.retrieval.chunks.indexOf(chunk) + 1) : null;
    const lightHit = lightRank != null && lightRank <= topPages;
    const accurateHit = chunkRank != null && chunkRank <= topChunks;
    const contextHit = q.qa.context_page_ids.includes(pageId);
    if (lightHit) lightHits++;
    if (accurateHit) accurateHits++;
    if (contextHit) contextHits++;
    return `<tr class="${lightHit ? 'hit-row' : 'miss-row'}"><td><span class="gold-page-id">${esc(pageId)}</span></td><td>${esc(gold.relevance)}</td><td>${lightHit ? '<span class="hit">HIT</span> ' : '<span class="miss">MISS</span> '}${rankCell(page, 'page', q)}</td><td>${accurateHit ? '<span class="hit">HIT</span> ' : '<span class="miss">MISS</span> '}${rankCell(chunk, 'chunk', q)}</td><td>${contextHit ? '<span class="hit">USED</span>' : '<span class="miss">NO</span>'}</td></tr>`;
  }).join('');
  if (!pages.length) return '<div class="notice">No page-level qrels are available for this question.</div>';
  return `<article class="card"><h3>Gold page coverage</h3><p class="card-kicker">Direct comparison against the stored ranking. A HIT means the gold page is present within the configured top-k boundary; click a rank to jump to its evidence card.</p><div class="coverage-strip"><div class="coverage-item"><strong>${lightHits}/${pages.length}</strong><span>Light BM25 top-${topPages}</span></div><div class="coverage-item"><strong>${accurateHits}/${pages.length}</strong><span>Accurate chunks top-${topChunks}</span></div><div class="coverage-item"><strong>${contextHits}/${pages.length}</strong><span>Pages used by QA</span></div></div><div class="gold-table"><table><thead><tr><th>Gold page</th><th>Rel.</th><th>Light BM25</th><th>Accurate chunk</th><th>QA context</th></tr></thead><tbody>${rows}</tbody></table></div></article>`;
}
function renderOverview(q) {
  const qa = q.qa, r = q.retrieval;
  const retry = Number(qa.unanswerable_retry_count || 0);
  const tags = [badge(qa.status === 'ok' ? 'QA saved' : 'QA missing/error', qa.status === 'ok' ? 'ok' : 'bad'), badge(r.status === 'ok' ? 'retrieval saved' : 'retrieval missing/error', r.status === 'ok' ? 'ok' : 'bad'), badge(`${q.domain} · ${q.type_group || 'unknown'}`, 'neutral')];
  if (qa.abstention_like) tags.push(badge(retry ? `abstention · retried ${retry}x` : 'abstention-like answer', 'warn'));
  if (r.parse_complete === false) tags.push(badge(`parse incomplete · ${(r.parse_failed_page_ids || []).length} pages`, 'bad'));
  const score = qa.score == null ? '—' : scoreLabel(qa.score);
  return `<div class="question-head"><div><h2>${esc(q.question)}</h2><p class="qid">${esc(q.qid)} · target document ${esc(q.doc_id)}</p></div><div class="nav-buttons"><button id="prev">← Previous</button><button id="next">Next →</button></div></div><div class="badges">${tags.join('')}</div>
  <div class="grid-2"><article class="card"><h3>Answer under review</h3>${answerBox('Generated answer', qa.sys_ans, 'generated')}${qa.initial_sys_ans ? answerBox('First answer before unanswerable retry', qa.initial_sys_ans, '') : ''}<div class="judge-line"><div class="big-score ${scoreClass(qa.score)}">${esc(score)}</div><div class="judge-raw"><strong>Judge output:</strong> ${esc(qa.judge_raw || 'not recorded')}<br>${esc(qa.generator || 'generator unknown')} → ${esc(qa.judge || 'judge unknown')} · ${num(qa.infer_time_seconds, 2)}s inference</div></div>${qa.error ? `<div class="notice" style="margin-top:12px">${esc(qa.error)}</div>` : ''}</article>
  <article class="card"><h3>Reference answer</h3>${answerBox('Gold answer', q.answer, 'reference')}<div class="answer-label">Original dataset evidence IDs</div><div class="answer-box">${esc(q.evidence || 'No raw evidence IDs recorded; use the Gold page coverage table below.')}</div><div class="card-kicker" style="margin-top:10px">These IDs can be corpus-internal identifiers rather than PDF page numbers. The normalized Gold pages below are the comparison target.</div></article></div>
  ${goldPageComparison(q)}<article class="card"><h3>Retrieval readout</h3><div class="metric-grid">${metricCards(q).map(([label,value]) => `<div class="metric"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`).join('')}</div><div class="answer-label">Gold files</div><div class="gold-list">${(q.gold_qrels || []).map(item => `<span class="gold-pill">${esc(item.doc_id)} · rel ${esc(item.relevance)}</span>`).join('') || '<span class="muted">No file qrels</span>'}</div><div class="answer-label">Gold pages</div><div class="gold-list">${(q.gold_page_qrels || []).map(item => `<span class="gold-pill">${esc(item.page_id)} · rel ${esc(item.relevance)}</span>`).join('') || '<span class="muted">No page qrels</span>'}</div></article>`;
}
function toggleText(id) { const node = document.getElementById(id); node.classList.toggle('collapsed'); node.nextElementSibling.textContent = node.classList.contains('collapsed') ? 'Show more' : 'Show less'; }
function rankCard(item, index, q, kind) {
  const id = kind === 'page' ? item.page_id : item.doc_id;
  const isGold = (q.gold_page_qrels || []).some(g => g.page_id === id);
  const isContext = kind === 'page' ? q.qa.context_page_ids.includes(id) : q.qa.context_unit_ids.includes(item.chunk_id);
  const title = kind === 'page' ? `${item.source_uri || 'unknown source'} · page ${item.page_number ?? '—'}` : `${item.doc_id || 'unknown page'} · chunk ${item.chunk_id || '—'}`;
  const text = item.text || '';
  const textId = `text-${kind}-${index}`;
  return `<article id="rank-${kind}-${index}" class="rank-card ${isGold ? 'is-gold' : ''} ${isContext ? 'is-context' : ''}"><div class="rank-head"><span class="rank-number">${esc(item.rank ?? index + 1)}</span><span class="rank-title">${esc(title)}</span><span class="rank-score">score ${num(item.score, 4)}</span>${isGold ? badge('GOLD PAGE', 'warn') : ''}${isContext ? badge(kind === 'page' ? 'IN QA CONTEXT' : 'USED BY QA', 'ok') : ''}</div><div class="rank-meta"><span>id: ${esc(id)}</span>${item.file_path ? `<span>file: ${esc(item.file_path)}</span>` : ''}${item.needs_ocr ? `<span>OCR: ${esc(item.ocr_reason || 'yes')}</span>` : ''}</div><pre id="${textId}" class="rank-text ${index > 2 ? 'collapsed' : ''}">${esc(text || 'No text')}</pre>${text.length > 700 ? `<button class="text-toggle" onclick="toggleText('${textId}')">${index > 2 ? 'Show more' : 'Show less'}</button>` : ''}</article>`;
}
function renderTab(q) {
  const r = q.retrieval;
  const tabs = [['pages','Light pages'], ['chunks','Accurate chunks'], ['timing','Timing & raw']];
  let body = '';
  if (state.tab === 'pages') body = `<div class="card"><h3>BM25 page ranking</h3><p class="card-kicker">Gold pages are outlined in gold. Pages marked “IN QA CONTEXT” were included in the generator context.</p>${r.hits.length ? r.hits.map((x,i) => rankCard(x,i,q,'page')).join('') : '<div class="empty">No page hits recorded.</div>'}</div>`;
  if (state.tab === 'chunks') body = `<div class="card"><h3>Chunk ranking sent downstream</h3><p class="card-kicker">The QA generator used ${esc(q.qa.chunks_used ?? '—')} chunks and ${esc(q.qa.chars_used ?? '—')} characters.</p>${r.chunks.length ? r.chunks.map((x,i) => rankCard(x,i,q,'chunk')).join('') : '<div class="empty">No chunks recorded.</div>'}</div>`;
  if (state.tab === 'timing') body = `<div class="grid-2"><div class="card"><h3>Timing breakdown</h3><div class="table-wrap"><table><tbody>${Object.entries(r.timing || {}).filter(([k]) => typeof r.timing[k] !== 'object').map(([key,value]) => `<tr><th>${esc(key)}</th><td>${esc(typeof value === 'number' && key.includes('seconds') ? `${num(value, 4)} s` : value)}</td></tr>`).join('')}</tbody></table></div></div><div class="card"><h3>Run metadata</h3><div class="table-wrap"><table><tbody><tr><th>retrieval config hash</th><td><code>${esc(r.retrieval_config_hash || '—')}</code></td></tr><tr><th>QA config hash</th><td><code>${esc(q.qa.qa_config_hash || '—')}</code></td></tr><tr><th>selected pages</th><td>${esc((r.selected_pages || []).length)}</td></tr><tr><th>retrieved pages</th><td>${esc(q.qa.retrieved_page_count ?? r.hits.length)}</td></tr><tr><th>context page ids</th><td>${esc(q.qa.context_page_ids.join(', ') || '—')}</td></tr></tbody></table></div><details><summary>Raw question payload</summary><pre class="raw">${safeJson(q)}</pre></details></div></div>`;
  return `<div class="tabs">${tabs.map(([id,label]) => `<button class="tab ${state.tab === id ? 'active' : ''}" data-tab="${id}">${esc(label)}</button>`).join('')}</div>${body}`;
}
function renderDetail() {
  const q = questions[state.selected];
  if (!q) { $('detail').innerHTML = '<div class="empty">Select a question from the list.</div>'; return; }
  $('detail').innerHTML = renderOverview(q) + renderTab(q);
  $('prev').addEventListener('click', () => move(-1)); $('next').addEventListener('click', () => move(1));
  document.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => { state.tab = button.dataset.tab; renderDetail(); }));
}
function move(delta) {
  const position = state.filtered.findIndex(item => item.i === state.selected);
  const next = state.filtered[(position + delta + state.filtered.length) % state.filtered.length];
  if (next) { state.selected = next.i; renderList(); renderDetail(); history.replaceState(null, '', `#${encodeURIComponent(questions[state.selected].qid)}`); }
}
function selectFromHash() {
  const raw = decodeURIComponent(location.hash.slice(1));
  const index = questions.findIndex(q => q.qid === raw);
  if (index >= 0) state.selected = index;
}
document.addEventListener('keydown', event => { if (event.target.matches('input,select,textarea')) return; if (event.key === 'j') move(1); if (event.key === 'k') move(-1); });
['search','score-filter','retrieval-filter','sort'].forEach(id => $(id).addEventListener(id === 'search' ? 'input' : 'change', updateFiltered));
selectFromHash(); summary(); updateFiltered();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/benchmark/on_demand_basic"))
    parser.add_argument("--scope", default=None, help="Retrieval scope, e.g. lake or file (auto-detected by default).")
    parser.add_argument("--output", type=Path, default=None, help="HTML output path (default: <run-dir>/retrieval_qa_review.html).")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    payload = _build_payload(run_dir, args.scope)
    output = (args.output or run_dir / "retrieval_qa_review.html").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    serialized = serialized.replace("</", "<\\/")
    output.write_text(HTML_TEMPLATE.replace("__PAYLOAD__", serialized), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Questions: {len(payload['questions'])}; QA rows: {sum(q['qa']['status'] != 'missing' for q in payload['questions'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
