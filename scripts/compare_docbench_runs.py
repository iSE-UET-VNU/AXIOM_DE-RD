#!/usr/bin/env python3
"""Build an offline HTML comparison for two DocBench result packages.

The canonical PDF-inspector baseline stores zero-based page IDs and the
on-demand runner stores one-based PDF page IDs. This script normalizes both to
``doc_id#page=<PDF page number>`` before comparing parse and retrieval output.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _page_number(page_id: str) -> int | None:
    match = re.search(r"#page=(\d+)$", str(page_id or ""))
    return int(match.group(1)) if match else None


def _doc_id(page_id: str) -> str:
    return str(page_id or "").rsplit("#page=", 1)[0]


def _canonical(page_id: str, *, page_number: int | None = None) -> str:
    value = str(page_id or "")
    doc_id = _doc_id(value)
    number = page_number if page_number is not None else _page_number(value)
    return f"{doc_id}#page={number}" if number is not None else value


def _baseline_canonical(row: dict[str, Any]) -> str:
    return _canonical(str(row.get("page_id") or ""), page_number=row.get("page_number"))


def _ondemand_canonical(page_id: str, page_number: int | None = None) -> str:
    # On-demand page IDs already use one-based PDF page numbers.
    return _canonical(page_id, page_number=page_number)


def _clean_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _text_similarity(left: str, right: str) -> float | None:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    # The page comparison is qualitative; cap pathological OCR pages so a
    # single huge page cannot make the whole export prohibitively slow.
    left = _normalized_text(left)[:100_000]
    right = _normalized_text(right)[:100_000]
    return difflib.SequenceMatcher(None, left, right, autojunk=True).ratio()


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _parse_summary(record: dict[str, Any]) -> dict[str, Any]:
    rows = record.get("rows") or []
    row = rows[0] if rows and isinstance(rows[0], dict) else {}
    extraction = row.get("extraction") or {}
    metadata = record.get("metadata") or {}
    return {
        "title": extraction.get("title") or "",
        "language": extraction.get("language") or "",
        "document_type": extraction.get("document_type") or "",
        "tables": len(extraction.get("tables") or []),
        "figures": len(extraction.get("figures") or []),
        "formulas": len(extraction.get("formulas") or []),
        "source_object_id": record.get("source_object_id") or "",
        "source_uri": metadata.get("source_uri") or "",
        "row_count": metadata.get("row_count"),
    }


def _extract_ondemand_parse(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    rows = record.get("rows") or []
    row = rows[0] if rows and isinstance(rows[0], dict) else {}
    extraction = row.get("extraction") or {}
    text = _clean_text(extraction.get("main_text") or row.get("text") or "")
    return text, _parse_summary(record)


def _empty_page() -> dict[str, Any]:
    return {
        "baseline": None,
        "ondemand_parse": None,
        "ondemand_light": None,
    }


def _metric_score(ranked: list[str], grades: dict[str, float], k: int) -> tuple[float, float]:
    unique: list[str] = []
    for page_id in ranked[:k]:
        if page_id and page_id not in unique:
            unique.append(page_id)
    recall = sum(page_id in unique for page_id in grades) / len(grades) if grades else 0.0

    def dcg(values: list[float]) -> float:
        return sum(
            (2.0**gain - 1.0) / (1.0 if rank == 1 else math.log2(rank))
            for rank, gain in enumerate(values, start=1)
        )

    gains = [grades.get(page_id, 0.0) for page_id in unique]
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    return recall, dcg(gains) / ideal if ideal else 0.0


def _latest_report(run_dir: Path, scope: str) -> dict[str, Any]:
    paths = sorted((run_dir / "reports").glob(f"{scope}_baseline_legacy_ver*.json"))
    if not paths:
        return {}
    paths.sort(key=lambda path: int(re.search(r"_ver(\d+)", path.name).group(1)))
    return _read_json(paths[-1])


def _load_baseline(root: Path) -> dict[str, Any]:
    pages = _read_jsonl(root / "preparation" / "pages.jsonl")
    files = _read_jsonl(root / "preparation" / "files.jsonl")
    queries = _read_jsonl(root / "inputs" / "queries.jsonl")
    qrels = _read_jsonl(root / "inputs" / "qrels_page_level.jsonl")
    top20 = _read_jsonl(root / "retrieval" / "run_top20.jsonl")
    candidates = _read_gzip_jsonl(root / "retrieval" / "all_selected_file_pages_ranked.jsonl.gz")

    page_store: dict[str, dict[str, Any]] = {}
    raw_to_canonical: dict[str, str] = {}
    for row in pages:
        canonical = _baseline_canonical(row)
        raw_to_canonical[str(row.get("page_id") or "")] = canonical
        entry = page_store.setdefault(canonical, _empty_page())
        entry["baseline"] = {
            "page_id": row.get("page_id"),
            "page_number": row.get("page_number"),
            "page_index": row.get("page_index"),
            "text": _clean_text(row.get("text")),
            "parse_status": row.get("parse_status"),
            "parse_error": row.get("parse_error"),
            "text_source": row.get("text_source"),
            "needs_ocr": row.get("needs_ocr"),
            "ocr_applied": row.get("ocr_applied"),
            "ocr_word_count": row.get("ocr_word_count"),
            "ocr_mean_confidence": row.get("ocr_mean_confidence"),
            "pdf_type": row.get("pdf_type"),
            "source_path": row.get("source_path"),
            "relative_path": row.get("relative_path"),
        }

    qrel_by_qid: dict[str, list[dict[str, Any]]] = {}
    for row in qrels:
        qid = str(row.get("query_id") or "")
        raw_page = str(row.get("page_id") or "")
        canonical = raw_to_canonical.get(raw_page)
        if canonical is None:
            canonical = _canonical(raw_page, page_number=(_page_number(raw_page) or 0) + 1)
        qrel_by_qid.setdefault(qid, []).append(
            {"page_id": canonical, "relevance": float(row.get("relevance") or 0.0)}
        )

    retrieval: dict[str, dict[str, Any]] = {}
    candidate_by_qid = {str(row.get("query_id") or ""): row for row in candidates}
    for row in top20:
        qid = str(row.get("query_id") or "")
        returned = []
        for rank, (raw_page, score) in enumerate(
            zip(row.get("returned_pages") or [], row.get("returned_scores") or []), start=1
        ):
            page_id = raw_to_canonical.get(str(raw_page))
            if page_id is None:
                page_id = _canonical(str(raw_page), page_number=(_page_number(str(raw_page)) or 0) + 1)
            returned.append({"page_id": page_id, "rank": rank, "score": score})
        candidate = candidate_by_qid.get(qid) or {}
        ranked_candidates = []
        for rank, (raw_page, score) in enumerate(
            zip(candidate.get("ranked_candidate_pages") or [], candidate.get("ranked_candidate_scores") or []), start=1
        ):
            page_id = raw_to_canonical.get(str(raw_page))
            if page_id is None:
                page_id = _canonical(str(raw_page), page_number=(_page_number(str(raw_page)) or 0) + 1)
            ranked_candidates.append({"page_id": page_id, "rank": rank, "score": score})
        retrieval[qid] = {
            "selected_files": row.get("selected_files") or [],
            "returned": returned,
            "candidates": ranked_candidates,
        }

    documents: dict[str, dict[str, Any]] = {}
    for row in files:
        doc_id = str(row.get("file_id") or row.get("doc_id") or "")
        documents[doc_id] = {
            "doc_id": doc_id,
            "source": row.get("source") or "",
            "relative_path": row.get("relative_path") or "",
            "page_count": row.get("page_count"),
            "pdf_type": row.get("pdf_type") or "",
            "detected_language": row.get("detected_language") or "",
        }
    for page_id, entry in page_store.items():
        doc_id = _doc_id(page_id)
        doc = documents.setdefault(doc_id, {"doc_id": doc_id})
        doc["baseline_pages"] = int(doc.get("baseline_pages", 0)) + 1
        if entry.get("baseline", {}).get("parse_status") == "ok":
            doc["baseline_ok_pages"] = int(doc.get("baseline_ok_pages", 0)) + 1

    return {
        "manifest": _read_json(root / "manifest.json"),
        "pages": page_store,
        "documents": documents,
        "queries": {str(row.get("query_id") or ""): row for row in queries},
        "qrels": qrel_by_qid,
        "retrieval": retrieval,
    }


def _load_ondemand(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    manifest = _read_json(next(root.glob("manifest_*_ver*.json")))
    retrieval_rows = _read_jsonl(root / "retrieval" / "lake_baseline_legacy.jsonl")
    qa_rows = _read_jsonl(root / "qa" / "lake_baseline_legacy.jsonl")
    parsed_payload = _read_json(root / "cache" / "lake" / "parsed_pages.json")
    parsed_records = parsed_payload.get("records") or {}
    page_store = baseline["pages"]

    for raw_page_id, record in parsed_records.items():
        canonical = _ondemand_canonical(str(raw_page_id))
        entry = page_store.setdefault(canonical, _empty_page())
        text, summary = _extract_ondemand_parse(record)
        entry["ondemand_parse"] = {
            "page_id": raw_page_id,
            "text": text,
            "summary": summary,
        }

    for row in retrieval_rows:
        for hit in row.get("hits") or []:
            canonical = _ondemand_canonical(
                str(hit.get("page_id") or ""), page_number=hit.get("page_number")
            )
            entry = page_store.setdefault(canonical, _empty_page())
            entry["ondemand_light"] = {
                "page_id": hit.get("page_id"),
                "page_number": hit.get("page_number"),
                "page_index": hit.get("page_index"),
                "text": _clean_text(hit.get("text")),
                "needs_ocr": hit.get("needs_ocr"),
                "ocr_reason": hit.get("ocr_reason"),
                "metadata": hit.get("metadata") or {},
            }

    qa_by_qid = {str(row.get("qid") or ""): row for row in qa_rows}
    retrieval_by_qid: dict[str, dict[str, Any]] = {}
    for row in retrieval_rows:
        qid = str(row.get("qid") or "")
        hits = []
        for hit in row.get("hits") or []:
            hits.append(
                {
                    "page_id": _ondemand_canonical(str(hit.get("page_id") or ""), hit.get("page_number")),
                    "rank": hit.get("rank"),
                    "score": hit.get("score"),
                }
            )
        chunks = []
        for chunk in row.get("chunks") or []:
            chunks.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "page_id": _ondemand_canonical(str(chunk.get("doc_id") or "")),
                    "rank": chunk.get("rank"),
                    "score": chunk.get("score"),
                    "text": _clean_text(chunk.get("text")),
                }
            )
        retrieval_by_qid[qid] = {
            "status": row.get("status"),
            "error": row.get("error"),
            "parse_complete": row.get("parse_complete"),
            "parse_failed_page_ids": row.get("parse_failed_page_ids") or [],
            "hits": hits,
            "chunks": chunks,
            "selected_pages": row.get("selected_pages") or [],
            "timing": row.get("timing") or {},
            "retrieval_config": row.get("retrieval_config") or {},
        }

    queries: dict[str, dict[str, Any]] = {}
    for qid, retrieval in retrieval_by_qid.items():
        qa = qa_by_qid.get(qid) or {}
        source = qa or next((row for row in retrieval_rows if str(row.get("qid")) == qid), {})
        gold_pages = []
        for item in source.get("gold_page_qrels") or []:
            gold_pages.append(
                {"page_id": _ondemand_canonical(str(item.get("page_id") or "")), "relevance": item.get("relevance")}
            )
        if not gold_pages:
            gold_pages = baseline["qrels"].get(qid, [])
        queries[qid] = {
            "question": source.get("question") or baseline["queries"].get(qid, {}).get("query") or "",
            "answer": source.get("answer") or (baseline["queries"].get(qid, {}).get("answers") or [""])[0],
            "doc_id": source.get("doc_id") or "",
            "source": source.get("source") or "",
            "gold_pages": gold_pages,
            "baseline_qrels": baseline["qrels"].get(qid, []),
            "retrieval": retrieval,
            "qa": {
                "available": bool(qa),
                "status": qa.get("status") or "missing",
                "score": qa.get("score"),
                "sys_ans": qa.get("sys_ans") or "",
                "judge_raw": qa.get("judge_raw") or "",
                "generator": qa.get("generator") or "",
                "judge": qa.get("judge") or "",
                "context_page_ids": [_ondemand_canonical(str(x)) for x in qa.get("context_page_ids") or []],
                "context_unit_ids": qa.get("context_unit_ids") or [],
                "chunks_used": qa.get("chunks_used"),
                "infer_time_seconds": qa.get("infer_time_seconds"),
                "error": qa.get("error"),
            },
        }
    report = _latest_report(root, "lake")
    timing_paths = sorted((root / "reports").glob("lake_timing_summary_ver*.json"))
    timing = _read_json(timing_paths[-1]) if timing_paths else {}
    return {
        "manifest": manifest,
        "report": report,
        "timing": timing,
        "parsed_count": len(parsed_records),
        "queries": queries,
        "qa_available": bool(qa_rows),
    }


def _build_payload(baseline_root: Path, ondemand_root: Path) -> dict[str, Any]:
    baseline = _load_baseline(baseline_root)
    ondemand = _load_ondemand(ondemand_root, baseline)
    documents = []
    for doc_id, doc in baseline["documents"].items():
        page_ids = sorted(
            [page_id for page_id in baseline["pages"] if _doc_id(page_id) == doc_id],
            key=lambda page_id: _page_number(page_id) or 0,
        )
        parsed = sum(
            1
            for page_id in page_ids
            if baseline["pages"].get(page_id, {}).get("ondemand_parse") is not None
        )
        changed = 0
        similar_values: list[float] = []
        for page_id in page_ids:
            entry = baseline["pages"].get(page_id) or {}
            left = (entry.get("baseline") or {}).get("text") or ""
            right = (entry.get("ondemand_parse") or {}).get("text") or ""
            similarity = None
            text_changed = None
            if entry.get("ondemand_parse") is not None:
                similarity = _text_similarity(left, right)
                text_changed = _normalized_text(left) != _normalized_text(right)
                if similarity is not None:
                    similar_values.append(similarity)
                if text_changed:
                    changed += 1
            entry["parse_comparison"] = {
                "similarity": similarity,
                "text_changed": text_changed,
            }
        documents.append(
            {
                **doc,
                "page_ids": page_ids,
                "ondemand_parsed_pages": parsed,
                "parse_common_pages": parsed,
                "parse_changed_pages": changed,
                "parse_mean_similarity": sum(similar_values) / len(similar_values) if similar_values else None,
            }
        )

    queries = []
    for qid, base_query in baseline["queries"].items():
        query = ondemand["queries"].get(qid) or {}
        b = baseline["retrieval"].get(qid) or {}
        o = query.get("retrieval") or {}
        baseline_pages = [item["page_id"] for item in b.get("returned") or []]
        ondemand_pages = [item["page_id"] for item in o.get("hits") or []]
        gold = {item["page_id"]: float(item.get("relevance") or 0.0) for item in baseline["qrels"].get(qid, [])}
        if not gold:
            gold = {item["page_id"]: float(item.get("relevance") or 0.0) for item in query.get("gold_pages") or []}
        b10 = _metric_score(baseline_pages, gold, 10)
        b20 = _metric_score(baseline_pages, gold, 20)
        o10 = _metric_score(ondemand_pages, gold, 10)
        o20 = _metric_score(ondemand_pages, gold, 20)
        base_set = set(baseline_pages[:20])
        od_set = set(ondemand_pages[:20])
        queries.append(
            {
                "qid": qid,
                "question": base_query.get("query") or query.get("question") or "",
                "answer": query.get("answer") or (base_query.get("answers") or [""])[0],
                "doc_id": query.get("doc_id") or "",
                "source": base_query.get("source") or query.get("source") or "",
                "gold_pages": [{"page_id": page_id, "relevance": grade} for page_id, grade in gold.items()],
                "baseline": {
                    "selected_files": b.get("selected_files") or [],
                    "returned": b.get("returned") or [],
                    "candidates": b.get("candidates") or [],
                    "metrics": {"recall10": b10[0], "ndcg10": b10[1], "recall20": b20[0], "ndcg20": b20[1]},
                },
                "ondemand": {
                    **o,
                    "metrics": {"recall10": o10[0], "ndcg10": o10[1], "recall20": o20[0], "ndcg20": o20[1]},
                    "selected_files": sorted({_doc_id(item["page_id"]) for item in o.get("hits") or []}),
                },
                "qa": query.get("qa") or {"available": False},
                "diff": {
                    "overlap10": len(set(baseline_pages[:10]) & set(ondemand_pages[:10])),
                    "overlap20": len(base_set & od_set),
                    "baseline_only20": sorted(base_set - od_set),
                    "ondemand_only20": sorted(od_set - base_set),
                    "gold_hit_baseline10": bool(set(baseline_pages[:10]) & set(gold)),
                    "gold_hit_baseline20": bool(base_set & set(gold)),
                    "gold_hit_ondemand10": bool(set(ondemand_pages[:10]) & set(gold)),
                    "gold_hit_ondemand20": bool(od_set & set(gold)),
                    "selected_file_overlap": len(
                        set(b.get("selected_files") or [])
                        & set(o.get("selected_files") or [])
                    ),
                },
            }
        )

    pages = {}
    for page_id, entry in baseline["pages"].items():
        if entry.get("baseline") or entry.get("ondemand_parse") or entry.get("ondemand_light"):
            pages[page_id] = entry
    return {
        "baseline": {"root": str(baseline_root), "manifest": baseline["manifest"]},
        "ondemand": {
            "root": str(ondemand_root),
            "manifest": ondemand["manifest"],
            "report": ondemand["report"],
            "timing": ondemand["timing"],
            "parsed_count": ondemand["parsed_count"],
            "qa_available": ondemand["qa_available"],
        },
        "documents": documents,
        "pages": pages,
        "queries": queries,
        "baseline_qa_available": False,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DocBench baseline comparison</title>
<style>
:root{--ink:#172535;--muted:#667485;--paper:#f6f2ea;--card:#fffdf8;--line:#dcd9cf;--navy:#15344e;--teal:#177b78;--coral:#d76049;--gold:#b7862b;--green:#2a8059;--red:#b5423d;--blue:#4a70a8;--shadow:0 12px 30px #1b354012}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--paper);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}.shell{min-height:100vh;background:radial-gradient(circle at 90% 0,#dcebe4 0,transparent 29%),radial-gradient(circle at 5% 34%,#f1ddd0 0,transparent 25%)}
.top{padding:28px clamp(18px,4vw,56px) 20px;color:#f4f3ed;background:linear-gradient(120deg,#122d45,#1d5664 64%,#237d76);box-shadow:0 9px 28px #102d3c3b;position:sticky;top:0;z-index:20}.eyebrow{margin:0 0 5px;color:#a8dcd3;font-size:11px;font-weight:850;letter-spacing:.16em;text-transform:uppercase}h1,h2,h3{font-family:Georgia,'Times New Roman',serif;line-height:1.15}h1{margin:0;font-size:clamp(25px,4vw,42px);letter-spacing:-.035em}.top-meta{margin-top:8px;color:#cee2de;font-size:12px;word-break:break-all}
.nav{display:flex;gap:6px;margin-top:20px}.nav button{padding:9px 13px;border:1px solid #83b9b4;border-radius:10px;color:#dff3ef;background:#ffffff13;font-weight:800}.nav button.active,.nav button:hover{color:var(--navy);background:#ecfaf6}
.body{display:grid;grid-template-columns:minmax(265px,340px) minmax(0,1fr);max-width:1900px;margin:0 auto}.rail{min-height:calc(100vh - 180px);padding:18px 15px;border-right:1px solid var(--line);background:#f1ece3d9}.rail-tools{position:sticky;top:200px}.search,.select{width:100%;padding:10px 12px;border:1px solid #c8c9c0;border-radius:11px;outline:none;background:#fffdf9;color:var(--ink)}.search:focus,.select:focus{border-color:var(--teal);box-shadow:0 0 0 3px #177b7824}.select{margin-top:8px}.count{margin:10px 2px;color:var(--muted);font-size:12px}.list{display:grid;gap:7px}.item{width:100%;padding:11px 12px;border:1px solid transparent;border-radius:13px;text-align:left;color:var(--ink);background:#fffdf9b5}.item:hover{border-color:#b7d3cd;background:#fff}.item.active{border-color:var(--teal);box-shadow:inset 4px 0 var(--teal),0 5px 15px #17374c18;background:#fff}.item-title{display:-webkit-box;overflow:hidden;font-weight:750;line-height:1.3;-webkit-box-orient:vertical;-webkit-line-clamp:2}.item-sub{display:flex;gap:7px;margin-top:6px;color:var(--muted);font-size:11px}.mono{font:11px ui-monospace,SFMono-Regular,Consolas,monospace;word-break:break-all}
.main{min-width:0;padding:25px clamp(16px,3vw,48px) 65px}.cards{display:grid;grid-template-columns:repeat(6,minmax(100px,1fr));gap:10px;margin-bottom:20px}.stat{padding:13px 14px;border:1px solid var(--line);border-radius:14px;background:#fffdf9cc;box-shadow:var(--shadow)}.stat-label{color:var(--muted);font-size:10px;font-weight:850;letter-spacing:.08em;text-transform:uppercase}.stat-value{margin-top:2px;color:var(--navy);font:700 24px Georgia,serif}.stat-note{color:var(--muted);font-size:11px}.card{min-width:0;margin-bottom:15px;padding:18px;border:1px solid var(--line);border-radius:16px;background:var(--card);box-shadow:var(--shadow)}.card h2,.card h3{margin:0 0 10px;color:var(--navy)}.card h2{font-size:28px}.card h3{font-size:19px}.hint{margin:-4px 0 12px;color:var(--muted);font-size:12px}.empty{padding:42px 18px;border:1px dashed #c7c5bb;border-radius:16px;color:var(--muted);text-align:center;background:#fffdf980}.badge{display:inline-flex;align-items:center;min-height:22px;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:850}.ok{color:#185f3d;background:#cdebd9}.warn{color:#815d15;background:#f5e4b8}.bad{color:#8f332f;background:#f3c9c4}.neutral{color:#536171;background:#e4e7e5}.blue{color:#345b92;background:#dbe7f7}
.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.three{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.metric{padding:10px;border-radius:10px;background:#f0f2ed}.metric strong{display:block;color:var(--navy);font-size:18px}.metric span{color:var(--muted);font-size:11px}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(90px,1fr));gap:8px}.section-label{margin:14px 0 6px;color:var(--muted);font-size:10px;font-weight:850;letter-spacing:.1em;text-transform:uppercase}.notice{padding:11px 13px;border-radius:10px;color:#7b5417;background:#fff0c8;font-size:12px}.danger-notice{color:#8d3631;background:#f9d8d2}.success-notice{color:#185f3d;background:#dff2e4}.answer{min-height:70px;padding:12px;border-left:4px solid var(--coral);border-radius:8px;background:#fff0e9;white-space:pre-wrap}.answer.reference{border-color:var(--gold);background:#fbf4df}
.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:8px 9px;border-bottom:1px solid #e4e2db;text-align:left;vertical-align:top}th{color:var(--muted);font-size:10px;letter-spacing:.07em;text-transform:uppercase}tr.clickable{cursor:pointer}tr.clickable:hover{background:#edf7f3}tr.selected-page{box-shadow:inset 4px 0 var(--teal);font-weight:650}.hit-row{background:#f0f8f2}.miss-row{background:#fff7f3}.same-row{background:#f7f8f5}.diff-row{background:#fff4ed}.page-id{font:11px ui-monospace,SFMono-Regular,Consolas,monospace;word-break:break-all}.rank{font:700 12px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--navy)}
.parse-toolbar,.query-toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:12px}.small-button,.link-button{padding:7px 10px;border:1px solid #c8d4d1;border-radius:9px;color:var(--navy);background:#eef6f3;font-weight:750}.small-button:hover,.link-button:hover{background:#dcebe4}.small-button.active{border-color:var(--teal);color:#fff;background:var(--teal)}.coverage{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}.coverage-box{padding:9px 11px;border-radius:10px;background:#f0f2ed}.coverage-box strong{color:var(--navy);font-size:18px}.coverage-box span{margin-left:4px;color:var(--muted);font-size:11px}.diff-bar{height:10px;overflow:hidden;border-radius:999px;background:#e5e4dc}.diff-bar span{display:block;height:100%;background:linear-gradient(90deg,var(--teal),#82c9af)}
.side-title{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}.side-title h2{margin:0}.side-title .mono{color:var(--muted)}.result-col h3{display:flex;justify-content:space-between;align-items:center}.result-list{display:grid;gap:7px}.result{padding:10px;border:1px solid #e0ded6;border-radius:11px;background:#fff}.result.gold{border-color:#d5ae52;box-shadow:inset 4px 0 var(--gold)}.result.same{border-color:#9fc9bb}.result-head{display:flex;align-items:center;gap:7px}.result-rank{display:grid;width:29px;height:29px;place-items:center;border-radius:8px;color:white;background:var(--navy);font-weight:850}.result-id{flex:1;min-width:0;color:var(--navy);font:11px ui-monospace,SFMono-Regular,Consolas,monospace;word-break:break-all}.result-score{color:var(--coral);font:700 11px ui-monospace,SFMono-Regular,Consolas,monospace}.result-meta{margin-top:5px;color:var(--muted);font-size:11px}.result-text{max-height:100px;overflow:auto;margin:8px 0 0;padding:9px;border-radius:7px;background:#f4f5f1;white-space:pre-wrap}.chunk{padding:10px;border:1px solid #deddd5;border-radius:10px;background:#fff}.chunk.used{border-color:#8bc2b1;box-shadow:inset 4px 0 var(--teal)}.chunk-text{max-height:140px;overflow:auto;margin-top:7px;padding:8px;border-radius:7px;background:#f4f5f1;white-space:pre-wrap}
.page-detail{margin-top:12px}.text-panes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.text-pane{min-width:0}.text-pane h4{margin:0 0 6px;color:var(--navy);font:700 14px Georgia,serif}.text-pane pre{max-height:520px;overflow:auto;margin:0;padding:13px;border:1px solid #deddd5;border-radius:10px;background:#f7f7f3;white-space:pre-wrap;word-break:break-word;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}.text-pane.baseline pre{border-top:4px solid var(--blue)}.text-pane.ondemand pre{border-top:4px solid var(--teal)}.text-diff{padding:10px;border-radius:9px;background:#eef2f0;color:var(--muted);font-size:12px}.missing{padding:14px;border:1px dashed #d1c7bc;border-radius:9px;color:var(--muted);background:#faf4ed}.raw{max-height:400px;overflow:auto;padding:12px;border-radius:9px;background:#182b38;color:#dceee9;white-space:pre-wrap;font:11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
@media(max-width:1150px){.cards{grid-template-columns:repeat(3,1fr)}}@media(max-width:800px){.body{display:block}.rail{min-height:auto;border-right:0;border-bottom:1px solid var(--line)}.rail-tools{position:static}.list{max-height:340px;overflow:auto}.two,.text-panes{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}.three{grid-template-columns:1fr}}
</style>
</head>
<body><div class="shell">
<header class="top"><p class="eyebrow">DocBench / two-run comparison</p><h1>Baseline versus on-demand</h1><div id="top-meta" class="top-meta"></div><nav class="nav"><button data-mode="overview">Run overview</button><button data-mode="parse">Parse compare</button><button data-mode="query">Query compare</button></nav></header>
<div class="body"><aside class="rail"><div class="rail-tools"><input id="search" class="search" type="search" placeholder="Search document or query..."><select id="filter" class="select"></select><div id="count" class="count"></div><div id="list" class="list"></div></div></aside><main class="main"><section id="summary" class="cards"></section><section id="content"></section></main></div>
</div>
<script>
const DATA=__PAYLOAD__;
const state={mode:'overview',selectedDoc:0,selectedQuery:0,selectedPage:null};
const $=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(v,d=3)=>v==null||Number.isNaN(Number(v))?'—':Number(v).toFixed(d);
const pct=v=>v==null||Number.isNaN(Number(v))?'—':`${(Number(v)*100).toFixed(1)}%`;
const pageNum=id=>{const m=String(id||'').match(/#page=(\d+)$/);return m?Number(m[1]):null};
const docId=id=>String(id||'').split('#page=')[0];
const pageEntry=id=>DATA.pages[id]||{};
const goldSet=q=>new Set((q.gold_pages||[]).map(x=>x.page_id));
const scoreClass=s=>s==null?'neutral':Number(s)===1?'ok':Number(s)===.5?'warn':'bad';
const badge=(x,c='neutral')=>`<span class="badge ${c}">${esc(x)}</span>`;
function pageHit(result,gold,k){return (result||[]).slice(0,k).some(x=>gold.has(x.page_id));}
function summary(){
  const q=DATA.queries, docs=DATA.documents, report=DATA.ondemand.report||{};
  const bHits=q.filter(x=>x.diff.gold_hit_baseline20).length, oHits=q.filter(x=>x.diff.gold_hit_ondemand20).length;
  const exactDocs=docs.filter(x=>x.parse_common_pages===x.baseline_pages&&x.parse_changed_pages===0).length;
  const cards=[['Documents',docs.length,'parse comparison'],['Pages baseline',DATA.baseline.manifest.pages||'—','canonical package'],['Pages parsed OD',DATA.ondemand.parsed_count,'KDL cache records'],['Gold hit B/O',`${bHits}/${oHits}`,`queries with gold page in top-20`],['QA accuracy',pct(report.accuracy),'on-demand only'],['Parse exact files',exactDocs,`files with no normalized text change`]];
  $('summary').innerHTML=cards.map(([a,b,c])=>`<div class="stat"><div class="stat-label">${esc(a)}</div><div class="stat-value">${esc(b)}</div><div class="stat-note">${esc(c)}</div></div>`).join('');
  $('top-meta').textContent=`${DATA.baseline.root}  ↔  ${DATA.ondemand.root} · ${q.length} shared queries · baseline QA: ${DATA.baseline_qa_available?'available':'not included in package'}`;
}
function setMode(mode){state.mode=mode;state.selectedPage=null;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('active',x.dataset.mode===mode));buildRail();render();}
function buildRail(){
  const input=$('search'), filter=$('filter');
  input.placeholder=state.mode==='parse'?'Search file, source, language...':state.mode==='query'?'Search qid, question, domain...':'Search documents or queries...';
  const opts=state.mode==='parse'?[['all','All parse pages'],['changed','Changed pages'],['missing','Not parsed by on-demand'],['same','Same normalized text']]:state.mode==='query'?[['all','All queries'],['both','Gold hit in both'],['baseline-only','Baseline hit only'],['ondemand-only','On-demand hit only'],['same','Same top-20 set'],['different','Different top-20 set'],['qa-0','QA score 0'],['qa-1','QA score 1']]:[['all','Overview']];
  filter.innerHTML=opts.map(x=>`<option value="${x[0]}">${x[1]}</option>`).join('');
  const refresh=()=>{renderList();render()};input.oninput=refresh;filter.onchange=refresh;renderList();
}
function parseRows(){
  const query=$('search').value.toLowerCase().trim(), filter=$('filter').value;
  return DATA.documents.map((d,i)=>({d,i})).filter(({d})=>{
    const hay=[d.doc_id,d.relative_path,d.source,d.detected_language].join(' ').toLowerCase();
    if(query&&!hay.includes(query))return false;
    if(filter==='changed'&&d.parse_changed_pages===0)return false;
    if(filter==='missing'&&d.ondemand_parsed_pages===d.baseline_pages)return false;
    if(filter==='same'&&(d.parse_changed_pages!==0||d.ondemand_parsed_pages!==d.baseline_pages))return false;
    return true;
  });
}
function queryRows(){
  const query=$('search').value.toLowerCase().trim(),filter=$('filter').value;
  return DATA.queries.map((q,i)=>({q,i})).filter(({q})=>{
    const hay=[q.qid,q.question,q.source,q.doc_id].join(' ').toLowerCase();
    if(query&&!hay.includes(query))return false;
    if(filter==='both'&&!(q.diff.gold_hit_baseline20&&q.diff.gold_hit_ondemand20))return false;
    if(filter==='baseline-only'&&!(q.diff.gold_hit_baseline20&&!q.diff.gold_hit_ondemand20))return false;
    if(filter==='ondemand-only'&&!(!q.diff.gold_hit_baseline20&&q.diff.gold_hit_ondemand20))return false;
    if(filter==='same'&&q.diff.overlap20!==20)return false;
    if(filter==='different'&&q.diff.overlap20===20)return false;
    if(filter==='qa-0'&&Number(q.qa.score)!==0)return false;
    if(filter==='qa-1'&&Number(q.qa.score)!==1)return false;
    return true;
  });
}
function renderList(){
  if(state.mode==='overview'){$('count').textContent='';$('list').innerHTML='<div class="empty">Choose Parse compare or Query compare.</div>';return;}
  const rows=state.mode==='parse'?parseRows():queryRows();
  const selected=state.mode==='parse'?state.selectedDoc:state.selectedQuery;
  $('count').textContent=`${rows.length} of ${state.mode==='parse'?DATA.documents.length:DATA.queries.length}`;
  const current=state.mode==='parse'?state.selectedDoc:state.selectedQuery;
  const selectedRow=rows.find(row=>row.i===current);
  if(!selectedRow&&rows.length){if(state.mode==='parse')state.selectedDoc=rows[0].i;else state.selectedQuery=rows[0].i;}
  $('list').innerHTML=rows.length?rows.map(({d,q,i})=>{
    if(state.mode==='parse')return `<button class="item ${i===selected?'active':''}" data-index="${i}"><div class="item-title">${esc(d.doc_id)}</div><div class="item-sub"><span>${esc(d.source||'')}</span><span>${d.ondemand_parsed_pages}/${d.baseline_pages} pages parsed</span></div><div class="item-sub">${badge(d.parse_changed_pages?`${d.parse_changed_pages} changed`:'text stable',d.parse_changed_pages?'warn':'ok')}</div></button>`;
    return `<button class="item ${i===selected?'active':''}" data-index="${i}"><div class="item-title">${esc(q.question)}</div><div class="item-sub"><span class="mono">${esc(q.qid)}</span>${badge(q.qa.score==null?'QA —':`QA ${q.qa.score}`,scoreClass(q.qa.score))}</div><div class="item-sub">${badge(`B ${q.diff.gold_hit_baseline20?'HIT':'MISS'}`,q.diff.gold_hit_baseline20?'ok':'bad')}${badge(`O ${q.diff.gold_hit_ondemand20?'HIT':'MISS'}`,q.diff.gold_hit_ondemand20?'ok':'bad')}<span>overlap ${q.diff.overlap20}/20</span></div></button>`;
  }).join(''):'<div class="empty">No matches.</div>';
  document.querySelectorAll('.item').forEach(x=>x.onclick=()=>{if(state.mode==='parse')state.selectedDoc=Number(x.dataset.index);else state.selectedQuery=Number(x.dataset.index);renderList();render();});
}
function overview(){
  const b=DATA.baseline.manifest||{},o=DATA.ondemand.manifest||{},r=DATA.ondemand.report||{};
  return `<article class="card"><h2>What this comparison can prove</h2><p class="hint">Page IDs are normalized to one-based PDF page numbers before comparison. Baseline parse is the canonical PDF-inspector + Tesseract preparation package; on-demand parse is the KDL cache generated only for pages selected by retrieval.</p><div class="two"><div><div class="section-label">Baseline</div><div class="answer reference"><b>${esc(b.arm||'canonical baseline')}</b><br>${esc(b.pages||'—')} pages, ${esc(b.queries||'—')} queries<br>Top-20 retrieval, selected-file hierarchy</div></div><div><div class="section-label">On-demand</div><div class="answer"><b>${esc(o.pipeline||'on-demand')}</b><br>${esc(o.indexed_pages||'—')} indexed pages, ${esc(o.questions_selected||'—')} queries<br>${esc((o.chunking_config||{}).chunker||'chunking')} → hybrid retrieval → QA</div></div></div></article><div class="two"><article class="card"><h3>Parse comparison</h3><p class="hint">Choose a file to see page availability, normalized-text similarity, OCR provenance, and side-by-side page text.</p><div class="metric-grid"><div class="metric"><strong>${DATA.documents.length}</strong><span>documents</span></div><div class="metric"><strong>${DATA.ondemand.parsed_count}</strong><span>OD parsed pages</span></div><div class="metric"><strong>${DATA.documents.filter(x=>x.parse_common_pages>0).length}</strong><span>files with overlap</span></div><div class="metric"><strong>${DATA.documents.reduce((a,x)=>a+x.parse_changed_pages,0)}</strong><span>changed pages</span></div></div></article><article class="card"><h3>Retrieval / QA comparison</h3><p class="hint">Choose a query to compare top-20 pages, score/rank movement, gold coverage, and on-demand QA. The baseline package has no QA JSONL, so that side is explicitly unavailable.</p><div class="metric-grid"><div class="metric"><strong>${DATA.queries.length}</strong><span>shared queries</span></div><div class="metric"><strong>${DATA.queries.filter(q=>q.diff.overlap20===20).length}</strong><span>same top-20 set</span></div><div class="metric"><strong>${DATA.queries.filter(q=>q.diff.gold_hit_baseline20).length}</strong><span>baseline gold hits</span></div><div class="metric"><strong>${DATA.queries.filter(q=>q.diff.gold_hit_ondemand20).length}</strong><span>OD gold hits</span></div></div></article></div>`;
}
function parseDoc(){
  const doc=DATA.documents[state.selectedDoc];if(!doc)return '<div class="empty">No document selected.</div>';
  const ids=doc.page_ids||[];const rows=ids.map((id,index)=>{const p=pageEntry(id),b=p.baseline||{},o=p.ondemand_parse,comparison=p.parse_comparison||{};const sim=comparison.similarity;const changed=comparison.text_changed;return {id,index,p,b,o,sim,changed};});
  const changed=rows.filter(x=>x.changed).length,missing=rows.filter(x=>!x.o).length;
  const selected=state.selectedPage&&rows.find(x=>x.id===state.selectedPage)||rows[0];
  return `<div class="side-title"><div><h2>${esc(doc.doc_id)}</h2><div class="mono">${esc(doc.relative_path||'')} · ${esc(doc.page_count||ids.length)} PDF pages</div></div>${badge(`${doc.ondemand_parsed_pages}/${doc.baseline_pages} parsed`,doc.ondemand_parsed_pages?'blue':'bad')}</div><div class="coverage"><div class="coverage-box"><strong>${ids.length}</strong><span>baseline pages</span></div><div class="coverage-box"><strong>${doc.ondemand_parsed_pages}</strong><span>OD parse pages</span></div><div class="coverage-box"><strong>${changed}</strong><span>normalized text changed</span></div><div class="coverage-box"><strong>${missing}</strong><span>OD parse missing</span></div><div class="coverage-box"><strong>${doc.parse_mean_similarity==null?'—':(doc.parse_mean_similarity*100).toFixed(1)+'%'}</strong><span>mean similarity</span></div></div>${selected?parsePage(selected):''}<article class="card"><h3>Page-level parse matrix</h3><p class="hint">Click a page to inspect the two parse outputs. Similarity is whitespace-normalized, case-insensitive text similarity; it is a diagnostic, not a semantic score.</p><div class="table-wrap"><table><thead><tr><th>PDF page</th><th>Baseline</th><th>On-demand</th><th>Similarity</th><th>Text delta</th><th>Source / OCR</th></tr></thead><tbody>${rows.map(x=>`<tr class="clickable ${x.id===state.selectedPage?'selected-page ':''}${!x.o?'miss-row':x.changed?'diff-row':'same-row'}" data-page="${esc(x.id)}"><td class="rank">${esc(pageNum(x.id))}</td><td>${x.b.parse_status==='ok'?badge('OK','ok'):badge(x.b.parse_status||'—','bad')}<br><span class="mono">${(x.b.text||'').length} chars</span></td><td>${x.o?badge('PARSED','ok'):badge('NOT PARSED','neutral')}<br><span class="mono">${x.o?(x.o.text||'').length:'—'} chars</span></td><td>${x.sim==null?'—':badge((x.sim*100).toFixed(1)+'%',x.sim>.97?'ok':x.sim>.8?'warn':'bad')}</td><td>${x.o?badge(x.changed?'DIFFERENT':'SAME',x.changed?'warn':'ok'):'—'}</td><td>${esc(x.b.text_source||'—')} ${x.b.ocr_applied?badge('OCR','blue'):''}</td></tr>`).join('')}</tbody></table></div></article>`;
}
function _norm(s){return String(s||'').toLowerCase().replace(/\s+/g,' ').trim()}
function _similarity(a,b){if(!a&&!b)return 1;if(!a||!b)return 0;let x=_norm(a),y=_norm(b);if(x.length>100000)x=x.slice(0,100000);if(y.length>100000)y=y.slice(0,100000);return _simpleSimilarity(x,y)}
function _simpleSimilarity(a,b){if(a===b)return 1;const short=a.length<b.length?a:b,long=a.length<b.length?b:a;let hits=0;for(let i=0;i<short.length;i+=7)if(long.includes(short.slice(i,i+7)))hits++;return short.length?Math.min(1,hits/Math.ceil(short.length/7)):0}
function parsePage(x){const b=x.b,o=x.o;return `<article id="parse-page-detail" class="card page-detail"><div class="side-title"><h3>Page ${esc(pageNum(x.id))} detail</h3>${badge(o?(x.changed?'text differs':'text stable'):'on-demand parse missing',o?(x.changed?'warn':'ok'):'neutral')}</div><div class="three"><div class="metric"><strong>${esc(b.text_source||'—')}</strong><span>baseline text source</span></div><div class="metric"><strong>${esc(o?.summary?.language||'—')}</strong><span>OD detected language</span></div><div class="metric"><strong>${o?(o.summary.tables||0)+' / '+(o.summary.figures||0)+' / '+(o.summary.formulas||0):'—'}</strong><span>OD tables / figures / formulas</span></div></div><div class="text-diff" style="margin:12px 0">Baseline: ${esc((b.text||'').length)} chars / ${esc((b.text||'').split(/\s+/).filter(Boolean).length)} words · On-demand: ${o?esc((o.text||'').length):'—'} chars / ${o?esc((o.text||'').split(/\s+/).filter(Boolean).length):'—'} words</div><div class="text-panes"><div class="text-pane baseline"><h4>Baseline PDF-inspector + Tesseract</h4>${b.text?`<pre>${esc(b.text)}</pre>`:'<div class="missing">No baseline text.</div>'}</div><div class="text-pane ondemand"><h4>On-demand KDL parse</h4>${o?`<pre>${esc(o.text)}</pre>`:'<div class="missing">This page was not parsed by on-demand because it was not selected by retrieval.</div>'}</div></div></article>`}
function queryDetail(){
  const q=DATA.queries[state.selectedQuery];if(!q)return '<div class="empty">No query selected.</div>';const gold=goldSet(q),b=q.baseline,o=q.ondemand;
  return `<div class="side-title"><div><h2>${esc(q.question)}</h2><div class="mono">${esc(q.qid)} · target ${esc(q.doc_id||'—')}</div></div>${badge(`QA ${q.qa.score==null?'—':q.qa.score}`,scoreClass(q.qa.score))}</div><div class="coverage"><div class="coverage-box"><strong>${q.diff.overlap10}/10</strong><span>overlap top-10</span></div><div class="coverage-box"><strong>${q.diff.overlap20}/20</strong><span>overlap top-20</span></div><div class="coverage-box"><strong>${q.diff.gold_hit_baseline20?'HIT':'MISS'}</strong><span>baseline gold top-20</span></div><div class="coverage-box"><strong>${q.diff.gold_hit_ondemand20?'HIT':'MISS'}</strong><span>OD gold top-20</span></div><div class="coverage-box"><strong>${q.diff.selected_file_overlap}</strong><span>shared selected files</span></div><div class="coverage-box"><strong>${pct(b.metrics.recall20)} / ${pct(o.metrics.recall20)}</strong><span>page recall B / O</span></div></div><article class="card"><h3>Gold pages and rank coverage</h3><p class="hint">The same canonical page ID is used on both sides. Gold rows are the fastest way to tell which run found the evidence.</p><div class="table-wrap"><table><thead><tr><th>Gold page</th><th>Rel.</th><th>Baseline rank</th><th>On-demand rank</th><th>Rank movement</th></tr></thead><tbody>${(q.gold_pages||[]).map(g=>{const bp=b.returned.find(x=>x.page_id===g.page_id),op=o.hits.find(x=>x.page_id===g.page_id);const br=bp?.rank||null,or=op?.rank||null;return `<tr class="${bp||op?'hit-row':'miss-row'}"><td class="page-id">${esc(g.page_id)}</td><td>${esc(g.relevance)}</td><td>${br?badge('#'+br+' · '+num(bp.score,3), 'ok'):'<span class="badge bad">MISS</span>'}</td><td>${or?badge('#'+or+' · '+num(op.score,3),'ok'):'<span class="badge bad">MISS</span>'}</td><td>${br&&or?esc((or-br>0?'+':'')+(or-br)):'—'}</td></tr>`}).join('')}</tbody></table></div></article><article class="card"><h3>Selected file sets</h3><div class="two"><div><div class="section-label">Baseline selected files</div><div class="mono">${(b.selected_files||[]).map(x=>esc(x)).join('<br>')||'—'}</div></div><div><div class="section-label">On-demand files represented in top-20</div><div class="mono">${(o.selected_files||[]).map(x=>esc(x)).join('<br>')||'—'}</div></div></div></article><div class="two">${resultColumn(q,'baseline','Canonical baseline top-20')}${resultColumn(q,'ondemand','On-demand BM25 top-20')}</div>${differenceTable(q)}${qaPanel(q)}</div>`;
}
function resultCard(item,q,side,index){const p=pageEntry(item.page_id),text=side==='baseline'?p.baseline?.text:p.ondemand_light?.text;const gold=goldSet(q).has(item.page_id);return `<article class="result ${gold?'gold':''}"><div class="result-head"><span class="result-rank">${esc(item.rank)}</span><span class="result-id">${esc(item.page_id)}</span><span class="result-score">${num(item.score,4)}</span>${gold?badge('GOLD','warn'):''}</div><div class="result-meta">${side==='baseline'?'canonical preparation page':'on-demand light page'} · ${text?esc(text.length)+' chars':'text unavailable'}</div>${text?`<pre class="result-text">${esc(text.slice(0,1800))}${text.length>1800?'\n…':''}</pre>`:''}</article>`}
function resultColumn(q,side,title){const items=side==='baseline'?q.baseline.returned:q.ondemand.hits;return `<article class="card result-col"><h3>${esc(title)} <span class="mono">${items.length} pages</span></h3><div class="result-list">${items.length?items.map((item,index)=>resultCard(item,q,side,index)).join(''):'<div class="empty">No retrieval rows.</div>'}</div></article>`}
function differenceTable(q){const b=new Map(q.baseline.returned.map(x=>[x.page_id,x])),o=new Map(q.ondemand.hits.map(x=>[x.page_id,x]));const ids=[...new Set([...b.keys(),...o.keys()])];return `<article class="card"><h3>Rank and membership differences</h3><div class="table-wrap"><table><thead><tr><th>Page</th><th>Baseline</th><th>On-demand</th><th>Delta</th><th>Classification</th></tr></thead><tbody>${ids.map(id=>{const x=b.get(id),y=o.get(id),d=x&&y?y.rank-x.rank:null;const c=x&&y?'SHARED':x?'BASELINE ONLY':'ON-DEMAND ONLY';return `<tr class="${c==='SHARED'?'same-row':'diff-row'}"><td class="page-id">${esc(id)}</td><td>${x?`#${x.rank} · ${num(x.score,3)}`:'—'}</td><td>${y?`#${y.rank} · ${num(y.score,3)}`:'—'}</td><td>${d==null?'—':(d>0?'+':'')+d}</td><td>${badge(c,c==='SHARED'?'ok':c==='BASELINE ONLY'?'blue':'warn')}</td></tr>`}).join('')}</tbody></table></div></article>`}
function qaPanel(q){const qa=q.qa;return `<article class="card"><h3>QA comparison</h3>${DATA.baseline_qa_available?'<div class="two">':'<div class="notice">The baseline_pdf_inspector_tesseract package contains no QA JSONL according to its README. The baseline side is therefore unavailable, not an empty answer.</div><div style="margin-top:12px">'}<div><div class="section-label">Baseline QA</div><div class="missing">${DATA.baseline_qa_available?'Loaded':'Not included'}</div></div><div><div class="section-label">On-demand QA</div><div class="answer">${qa.sys_ans?esc(qa.sys_ans):'No answer recorded'}</div><div style="margin-top:8px">${badge(`score ${qa.score==null?'—':qa.score}`,scoreClass(qa.score))} ${esc(qa.generator||'')} → ${esc(qa.judge||'')} · ${esc(qa.judge_raw||'')}</div>${qa.context_page_ids?.length?`<div class="section-label">Context pages</div><div class="mono">${esc(qa.context_page_ids.join(', '))}</div>`:''}</div></div></article>`}
function render(){if(state.mode==='overview'){$('content').innerHTML=overview();return}if(state.mode==='parse'){$('content').innerHTML=parseDoc();$('content').onclick=event=>{const row=event.target.closest('[data-page]');if(!row)return;state.selectedPage=row.dataset.page;render()};const detail=$('parse-page-detail');if(detail)detail.scrollIntoView({block:'start'});return}$('content').innerHTML=queryDetail();}
document.querySelectorAll('[data-mode]').forEach(x=>x.onclick=()=>setMode(x.dataset.mode));
summary();buildRail();render();
</script></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("data/benchmark/baseline_pdf_inspector_tesseract"),
    )
    parser.add_argument(
        "--ondemand-dir",
        type=Path,
        default=Path("data/benchmark/on_demand_basic_1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmark/on_demand_basic_1/baseline_comparison.html"),
    )
    args = parser.parse_args()
    payload = _build_payload(args.baseline_dir.resolve(), args.ondemand_dir.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    serialized = serialized.replace("</", "<\\/")
    output.write_text(HTML_TEMPLATE.replace("__PAYLOAD__", serialized), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Documents: {len(payload['documents'])}; pages: {len(payload['pages'])}; queries: {len(payload['queries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
