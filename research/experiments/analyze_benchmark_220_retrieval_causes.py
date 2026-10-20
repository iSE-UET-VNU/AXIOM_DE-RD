"""Diagnose why the 220-query benchmark misses evidence pages.

This is an analysis-only script.  It reuses the cached light-preparation and
retrieval artifacts; it does not build an index or call a model.

The report intentionally distinguishes:
  * an observable preparation defect (source PDF text is present but the
    baseline page text loses/garbles a query-relevant term),
  * a useful embedded-image OCR signal (OCR adds a query-relevant term that
    was absent from the baseline page text),
  * a semantic rescue (the text embedding run improves over BM25), and
  * a language mismatch (English query, predominantly French gold page).

The labels are diagnostic evidence and can overlap.  They are not claims of
causality from qrels alone.  The report therefore gives both multi-label
counts and a conservative primary diagnosis.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+")

STOPWORDS = {
    # English function/query words.
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "these", "this", "to", "was", "what", "when", "where",
    "which", "who", "why", "with", "would", "into", "about", "also", "used",
    "use", "using", "determine", "describe", "explain", "give", "find", "shown",
    "following", "according", "related", "following", "number", "name", "type",
    # French function words and common answer scaffolding.
    "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "en", "et",
    "est", "la", "le", "les", "par", "pour", "sur", "une", "un", "qui", "que",
    "se", "son", "sont", "plus", "moins", "comme", "entre", "ainsi", "selon",
}

EN_WORDS = {
    "the", "and", "of", "to", "in", "for", "is", "are", "with", "from", "what",
    "which", "how", "does", "can", "that", "this", "three", "determine", "explain",
}
FR_WORDS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "est", "que", "qui",
    "dans", "pour", "avec", "sur", "par", "aux", "ce", "ces", "en", "sont", "plus",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def tokens(text: str | None) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


def meaningful_terms(text: str | None) -> set[str]:
    return {t for t in tokens(text) if len(t) >= 3 and t not in STOPWORDS}


def query_terms(query: dict[str, Any]) -> set[str]:
    terms = meaningful_terms(query.get("query", ""))
    for answer in query.get("answers", []) or []:
        terms.update(meaningful_terms(answer))
    return terms


def fuzzy_match(term: str, candidate: str) -> bool:
    if len(term) < 4 or len(candidate) < 4:
        return False
    if term == candidate:
        return True
    if term[:4] == candidate[:4]:
        return True
    if term in candidate or candidate in term:
        return abs(len(term) - len(candidate)) <= 3
    return SequenceMatcher(None, term, candidate).ratio() >= 0.82


def equivalent_in_text(term: str, text_tokens: set[str], joined: str) -> bool:
    if term in text_tokens or term in joined:
        return True
    return any(fuzzy_match(term, candidate) for candidate in text_tokens)


def lang_stats(text: str | None) -> dict[str, Any]:
    ts = tokens(text)
    en = sum(t in EN_WORDS for t in ts)
    fr = sum(t in FR_WORDS for t in ts)
    if fr >= 3 and fr > en * 1.20 and fr - en >= 2:
        label = "fr"
    elif en >= 3 and en > fr * 1.20 and en - fr >= 2:
        label = "en"
    else:
        label = "mixed_or_unknown"
    return {"language": label, "english_markers": en, "french_markers": fr}


def page_ids_from_run(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("returned_pages"), list):
        return [str(x) for x in row["returned_pages"]]
    chunks = row.get("chunks", []) or []
    return [str(c.get("page_id")) for c in chunks if c.get("page_id")]


def selected_files_from_run(row: dict[str, Any]) -> list[str]:
    selected = row.get("selected_files", []) or []
    if selected and isinstance(selected[0], str):
        return selected
    return [str(x.get("file_id")) for x in selected if x.get("file_id")]


def recall_at(retrieved: list[str], gold: set[str], depth: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:depth]) & gold) / len(gold)


def hit_at(retrieved: list[str], gold: set[str], depth: int) -> int:
    return int(bool(set(retrieved[:depth]) & gold))


def file_recall(selected_files: list[str], gold_files: set[str], depth: int = 3) -> float:
    if not gold_files:
        return 0.0
    return len(set(selected_files[:depth]) & gold_files) / len(gold_files)


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def find_pdf(dataset_root: Path, relative_path: str) -> Path | None:
    candidate = dataset_root / relative_path
    if candidate.is_file():
        return candidate
    name = Path(relative_path).name
    matches = list((dataset_root / "data").glob(name))
    return matches[0] if matches else None


def load_raw_pdf_pages(dataset_root: Path, page_rows: dict[str, dict[str, Any]], needed: set[str]) -> dict[str, dict[str, Any]]:
    """Read only gold pages from the original PDF text layer."""
    try:
        import pymupdf as fitz  # type: ignore
    except ImportError:  # pragma: no cover - old environments
        import fitz  # type: ignore

    by_pdf: dict[Path, list[str]] = defaultdict(list)
    result: dict[str, dict[str, Any]] = {}
    for page_id in sorted(needed):
        row = page_rows.get(page_id)
        if not row:
            result[page_id] = {"status": "missing_page_metadata", "text": ""}
            continue
        path = find_pdf(dataset_root, row.get("relative_path", ""))
        if path is None or path.suffix.lower() != ".pdf":
            result[page_id] = {"status": "non_pdf_or_missing", "text": ""}
            continue
        by_pdf[path].append(page_id)

    for path, page_ids in by_pdf.items():
        try:
            doc = fitz.open(str(path))
        except Exception as exc:  # pragma: no cover
            for page_id in page_ids:
                result[page_id] = {"status": "open_error", "error": str(exc), "text": ""}
            continue
        try:
            for page_id in page_ids:
                row = page_rows[page_id]
                index = int(row.get("page_index", 0))
                if index < 0 or index >= len(doc):
                    result[page_id] = {"status": "page_out_of_range", "text": ""}
                    continue
                try:
                    text = doc[index].get_text("text") or ""
                    result[page_id] = {"status": "ok", "text": text, "pdf": str(path)}
                except Exception as exc:
                    result[page_id] = {"status": "extract_error", "error": str(exc), "text": ""}
        finally:
            doc.close()
    return result


def classify_primary(labels: list[str], base_loss: bool) -> str:
    if not base_loss:
        return "no_recall_loss_at_20"
    has_parser = "parser_extraction" in labels
    has_image = "embedded_image_ocr" in labels
    has_lang = "language_mismatch" in labels
    has_sem = "semantic_rescue" in labels
    if has_parser and has_image:
        return "mixed_parser_and_embedded_image"
    if has_parser:
        return "parser_extraction"
    if has_image and has_sem:
        return "embedded_image_plus_semantic"
    if has_image:
        return "embedded_image_ocr"
    if has_lang and has_sem:
        return "language_mismatch_semantic_rescue"
    if has_sem:
        return "semantic_mismatch"
    if has_lang:
        return "language_mismatch_unresolved"
    return "unresolved_or_ranking"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/BENCHMARK"))
    parser.add_argument(
        "--base-dir", type=Path,
        default=Path("data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20"),
    )
    parser.add_argument(
        "--embedding-dir", type=Path,
        default=Path("data/work/benchmark_hierarchical_local/embedding_full_embedded_ocr_openrouter_te3s_clean_eng_w48"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    base_dir = args.base_dir
    ocr_dir = base_dir / "embedded_image_ocr_eng_w48"
    output_dir = args.output_dir or (args.embedding_dir / "cause_analysis_220")
    output_dir.mkdir(parents=True, exist_ok=True)

    page_rows = {r["page_id"]: r for r in read_jsonl(base_dir / "preparation/pages.jsonl")}
    queries = {r["query_id"]: r for r in read_jsonl(base_dir / "input/queries.jsonl")}
    qrels_rows = read_jsonl(base_dir / "input/qrels_page_level.jsonl")
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in qrels_rows:
        if int(row.get("relevance", 1)) > 0:
            qrels[row["query_id"]].add(row["page_id"])

    base_rows = {r["query_id"]: r for r in read_jsonl(base_dir / "retrieval/hierarchical_kf3_top20.jsonl")}
    full_rows = {
        r["query_id"]: r
        for r in read_jsonl(ocr_dir / "retrieval_bm25_global_vs_hierarchical_kf3/runs/hierarchical_bm25_kf3_top20.jsonl")
    }
    emb_rows = {r["query_id"]: r for r in read_jsonl(args.embedding_dir / "hierarchical_top20.jsonl")}

    ocr_by_page: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(ocr_dir / "ocr_results.jsonl"):
        text = (row.get("text") or "").strip()
        if text:
            ocr_by_page[row["page_id"]].append(text)

    needed_gold = set().union(*qrels.values()) if qrels else set()
    raw_pages = load_raw_pdf_pages(args.dataset_root, page_rows, needed_gold)

    per_query: list[dict[str, Any]] = []
    page_diagnostics: list[dict[str, Any]] = []
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for qid in sorted(queries):
        query = queries[qid]
        gold = qrels.get(qid, set())
        gold_files = {page_rows[p]["doc_id"] for p in gold if p in page_rows}
        relevant = query_terms(query)
        base = base_rows.get(qid, {})
        full = full_rows.get(qid, {})
        emb = emb_rows.get(qid, {})
        base_pages = page_ids_from_run(base)
        full_pages = page_ids_from_run(full)
        emb_pages = page_ids_from_run(emb)
        base_files = selected_files_from_run(base)
        full_files = selected_files_from_run(full)
        emb_files = selected_files_from_run(emb)

        page_stats: list[dict[str, Any]] = []
        parser_pages: list[str] = []
        image_pages: list[str] = []
        lang_stats_list: list[dict[str, Any]] = []
        for page_id in sorted(gold):
            p = page_rows.get(page_id, {})
            base_text = p.get("text", "") or ""
            raw_info = raw_pages.get(page_id, {"status": "not_loaded", "text": ""})
            raw_text = raw_info.get("text", "") or ""
            added_ocr = "\n".join(ocr_by_page.get(page_id, []))
            full_text = (base_text + "\n" + added_ocr).strip()
            raw_ts = set(tokens(raw_text))
            base_ts = set(tokens(base_text))
            ocr_ts = set(tokens(added_ocr))
            base_joined = "".join(base_ts)

            raw_relevant = raw_ts & relevant
            missing_relevant = sorted(
                t for t in raw_relevant
                if t not in base_ts and t not in base_joined
            )
            split_relevant = sorted(
                t for t in raw_relevant
                if t not in base_ts and t in base_joined
            )
            # OCR is useful only when it contributes a relevant signal absent
            # from both the baseline token stream and its joined-token form.
            ocr_exact = sorted(
                t for t in (ocr_ts & relevant)
                if t not in base_ts and t not in base_joined
            )
            ocr_fuzzy: list[dict[str, str]] = []
            for ocr_token in sorted(ocr_ts):
                if len(ocr_token) < 4:
                    continue
                if equivalent_in_text(ocr_token, base_ts, base_joined):
                    continue
                for term in sorted(relevant):
                    if fuzzy_match(term, ocr_token) and term not in base_ts and term not in base_joined:
                        ocr_fuzzy.append({"term": term, "ocr_token": ocr_token})
            ocr_fuzzy = ocr_fuzzy[:50]

            raw_to_base_coverage = (
                len(raw_ts & base_ts) / len(raw_ts) if raw_ts else None
            )
            parser_issue = bool(missing_relevant or split_relevant)
            # This is a broad extraction warning, retained separately from the
            # query-relevant parser defect above.
            extraction_warning = bool(raw_ts and (not base_ts or (raw_to_base_coverage is not None and raw_to_base_coverage < 0.50)))
            image_issue = bool(ocr_exact or ocr_fuzzy)
            ls = lang_stats(full_text)
            lang_stats_list.append(ls)
            if parser_issue:
                parser_pages.append(page_id)
            if image_issue:
                image_pages.append(page_id)
            page_diag = {
                "page_id": page_id,
                "doc_id": p.get("doc_id"),
                "page_index": p.get("page_index"),
                "source": p.get("source"),
                "raw_status": raw_info.get("status"),
                "raw_text_chars": len(raw_text),
                "baseline_text_chars": len(base_text),
                "embedded_ocr_chars": len(added_ocr),
                "raw_token_count": len(raw_ts),
                "baseline_token_count": len(base_ts),
                "embedded_ocr_token_count": len(ocr_ts),
                "raw_to_baseline_token_coverage": raw_to_base_coverage,
                "baseline_placeholder_count": base_text.count("[Image:"),
                "parser_missing_relevant_terms": missing_relevant,
                "parser_split_or_garbled_relevant_terms": split_relevant,
                "embedded_ocr_new_relevant_terms": ocr_exact,
                "embedded_ocr_fuzzy_relevant_terms": ocr_fuzzy,
                "parser_issue": parser_issue,
                "broad_extraction_warning": extraction_warning,
                "embedded_image_ocr_signal": image_issue,
                "language": ls,
            }
            page_stats.append(page_diag)
            page_diagnostics.append(page_diag | {"query_id": qid})

        majority_fr = sum(x["language"] == "fr" for x in lang_stats_list) > (len(lang_stats_list) / 2 if lang_stats_list else 0)
        language_mismatch = majority_fr
        base_r20 = recall_at(base_pages, gold, 20)
        full_r20 = recall_at(full_pages, gold, 20)
        emb_r20 = recall_at(emb_pages, gold, 20)
        base_r10 = recall_at(base_pages, gold, 10)
        full_r10 = recall_at(full_pages, gold, 10)
        emb_r10 = recall_at(emb_pages, gold, 10)
        base_fr3 = file_recall(base_files, gold_files)
        full_fr3 = file_recall(full_files, gold_files)
        emb_fr3 = file_recall(emb_files, gold_files)
        semantic_rescue = (emb_r20 > full_r20 + 1e-12) or (emb_fr3 > full_fr3 + 1e-12)
        semantic_page_rescue = emb_r20 > full_r20 + 1e-12
        semantic_file_rescue = emb_fr3 > full_fr3 + 1e-12
        base_loss = base_r20 < 1.0 - 1e-12

        labels: list[str] = []
        if parser_pages:
            labels.append("parser_extraction")
        if image_pages:
            labels.append("embedded_image_ocr")
        if semantic_rescue:
            labels.append("semantic_rescue")
        if language_mismatch:
            labels.append("language_mismatch")
        primary = classify_primary(labels, base_loss)
        row = {
            "query_id": qid,
            "source": query.get("source"),
            "query": query.get("query"),
            "answers": query.get("answers", []),
            "gold_page_count": len(gold),
            "gold_file_count": len(gold_files),
            "gold_page_ids": sorted(gold),
            "base_bm25": {
                "page_recall@10": base_r10,
                "page_recall@20": base_r20,
                "page_hit@10": hit_at(base_pages, gold, 10),
                "file_recall@3": base_fr3,
                "selected_files": base_files,
            },
            "full_embedded_ocr_bm25": {
                "page_recall@10": full_r10,
                "page_recall@20": full_r20,
                "page_hit@10": hit_at(full_pages, gold, 10),
                "file_recall@3": full_fr3,
                "selected_files": full_files,
            },
            "full_embedded_ocr_embedding": {
                "page_recall@10": emb_r10,
                "page_recall@20": emb_r20,
                "page_hit@10": hit_at(emb_pages, gold, 10),
                "file_recall@3": emb_fr3,
                "selected_files": emb_files,
            },
            "deltas_embedding_vs_full_ocr_bm25": {
                "page_recall@10": emb_r10 - full_r10,
                "page_recall@20": emb_r20 - full_r20,
                "file_recall@3": emb_fr3 - full_fr3,
            },
            "diagnosis": {
                "base_recall_loss_at_20": base_loss,
                "parser_extraction_signal": bool(parser_pages),
                "embedded_image_ocr_signal": bool(image_pages),
                "semantic_rescue": semantic_rescue,
                "semantic_page_rescue": semantic_page_rescue,
                "semantic_file_rescue": semantic_file_rescue,
                "language_mismatch": language_mismatch,
                "embedding_improves_language_case": language_mismatch and semantic_rescue,
                "labels": labels,
                "primary": primary,
                "parser_signal_pages": parser_pages,
                "embedded_ocr_signal_pages": image_pages,
                "gold_page_language_counts": dict(Counter(x["language"] for x in lang_stats_list)),
            },
            "gold_page_diagnostics": page_stats,
        }
        per_query.append(row)
        source = str(query.get("source", "unknown"))
        source_counts[source][primary] += 1

    # Counts are intentionally reported twice: all 220 queries and the subset
    # where the base top-20 did not contain every gold page.
    all_rows = per_query
    loss_rows = [r for r in per_query if r["diagnosis"]["base_recall_loss_at_20"]]

    def count(pred, rows=all_rows) -> int:
        return sum(bool(pred(r)) for r in rows)

    language_rows = [r for r in all_rows if r["diagnosis"]["language_mismatch"]]
    language_deltas = [r["deltas_embedding_vs_full_ocr_bm25"]["page_recall@20"] for r in language_rows]
    parser_missing_queries = [r for r in all_rows if any(d["parser_missing_relevant_terms"] for d in r["gold_page_diagnostics"])]
    parser_split_queries = [r for r in all_rows if any(d["parser_split_or_garbled_relevant_terms"] for d in r["gold_page_diagnostics"])]
    parser_missing_pages = [d for d in page_diagnostics if d["parser_missing_relevant_terms"]]
    parser_split_pages = [d for d in page_diagnostics if d["parser_split_or_garbled_relevant_terms"]]
    ocr_exact_queries = [r for r in all_rows if any(d["embedded_ocr_new_relevant_terms"] for d in r["gold_page_diagnostics"])]
    ocr_fuzzy_queries = [r for r in all_rows if any(d["embedded_ocr_fuzzy_relevant_terms"] for d in r["gold_page_diagnostics"])]
    signal_names = [
        "parser_extraction_signal",
        "embedded_image_ocr_signal",
        "semantic_rescue",
        "language_mismatch",
        "embedding_improves_language_case",
    ]
    source_signals: dict[str, dict[str, int]] = {}
    for source in sorted({str(r.get("source", "unknown")) for r in all_rows}):
        source_rows = [r for r in all_rows if str(r.get("source", "unknown")) == source]
        source_signals[source] = {
            "queries": len(source_rows),
            "base_recall_loss_at_20": sum(r["diagnosis"]["base_recall_loss_at_20"] for r in source_rows),
            **{
                name: sum(r["diagnosis"][name] for r in source_rows)
                for name in signal_names
            },
        }
    label_combinations = Counter(
        "+".join(sorted(r["diagnosis"]["labels"])) or "none"
        for r in all_rows
    )
    summary = {
        "protocol": {
            "queries": len(per_query),
            "corpus_pages": len(page_rows),
            "base_run": str(base_dir / "retrieval/hierarchical_kf3_top20.jsonl"),
            "full_ocr_bm25_run": str(ocr_dir / "retrieval_bm25_global_vs_hierarchical_kf3/runs/hierarchical_bm25_kf3_top20.jsonl"),
            "full_ocr_embedding_run": str(args.embedding_dir / "hierarchical_top20.jsonl"),
            "embedding_model": "openai/text-embedding-3-small via OpenRouter",
            "embedding_language": "English query + full OCR page text",
            "base_loss_definition": "base BM25 page recall@20 < 100%",
            "semantic_rescue_definition": "full-OCR embedding improves page recall@20 or file recall@3 over full-OCR BM25",
            "language_definition": "majority of gold pages classified French by deterministic function-word heuristic",
        },
        "metrics_overall": {
            "base_page_recall@10": safe_mean([r["base_bm25"]["page_recall@10"] for r in all_rows]),
            "base_page_recall@20": safe_mean([r["base_bm25"]["page_recall@20"] for r in all_rows]),
            "full_ocr_bm25_page_recall@10": safe_mean([r["full_embedded_ocr_bm25"]["page_recall@10"] for r in all_rows]),
            "full_ocr_bm25_page_recall@20": safe_mean([r["full_embedded_ocr_bm25"]["page_recall@20"] for r in all_rows]),
            "full_ocr_embedding_page_recall@10": safe_mean([r["full_embedded_ocr_embedding"]["page_recall@10"] for r in all_rows]),
            "full_ocr_embedding_page_recall@20": safe_mean([r["full_embedded_ocr_embedding"]["page_recall@20"] for r in all_rows]),
        },
        "counts_all_220": {
            "base_recall_loss_at_20": len(loss_rows),
            "parser_extraction_signal": count(lambda r: r["diagnosis"]["parser_extraction_signal"]),
            "embedded_image_ocr_signal": count(lambda r: r["diagnosis"]["embedded_image_ocr_signal"]),
            "semantic_rescue": count(lambda r: r["diagnosis"]["semantic_rescue"]),
            "semantic_page_rescue": count(lambda r: r["diagnosis"]["semantic_page_rescue"]),
            "semantic_file_rescue": count(lambda r: r["diagnosis"]["semantic_file_rescue"]),
            "language_mismatch": len(language_rows),
            "embedding_improves_language_case": count(lambda r: r["diagnosis"]["embedding_improves_language_case"]),
        },
        "counts_recall_loss_subset": {
            "queries": len(loss_rows),
            "parser_extraction_signal": count(lambda r: r["diagnosis"]["parser_extraction_signal"], loss_rows),
            "embedded_image_ocr_signal": count(lambda r: r["diagnosis"]["embedded_image_ocr_signal"], loss_rows),
            "semantic_rescue": count(lambda r: r["diagnosis"]["semantic_rescue"], loss_rows),
            "semantic_page_rescue": count(lambda r: r["diagnosis"]["semantic_page_rescue"], loss_rows),
            "semantic_file_rescue": count(lambda r: r["diagnosis"]["semantic_file_rescue"], loss_rows),
            "language_mismatch": count(lambda r: r["diagnosis"]["language_mismatch"], loss_rows),
            "embedding_improves_language_case": count(lambda r: r["diagnosis"]["embedding_improves_language_case"], loss_rows),
        },
        "primary_diagnosis_all": dict(Counter(r["diagnosis"]["primary"] for r in all_rows)),
        "primary_diagnosis_recall_loss_subset": dict(Counter(r["diagnosis"]["primary"] for r in loss_rows)),
        "language_case_embedding_delta": {
            "queries": len(language_rows),
            "improved_queries": count(lambda r: r["diagnosis"]["embedding_improves_language_case"], language_rows),
            "page_recall_improved_queries": count(lambda r: r["diagnosis"]["semantic_page_rescue"], language_rows),
            "file_recall_improved_queries": count(lambda r: r["diagnosis"]["semantic_file_rescue"], language_rows),
            "mean_page_recall20_delta": safe_mean(language_deltas),
            "median_page_recall20_delta": float(median(language_deltas)) if language_deltas else 0.0,
            "mean_file_recall3_delta": safe_mean([r["deltas_embedding_vs_full_ocr_bm25"]["file_recall@3"] for r in language_rows]),
        },
        "preparation_detail": {
            "parser_missing_relevant_term_queries": len(parser_missing_queries),
            "parser_split_or_garbled_relevant_term_queries": len(parser_split_queries),
            "parser_missing_relevant_term_pages": len(parser_missing_pages),
            "parser_split_or_garbled_relevant_term_pages": len(parser_split_pages),
            "embedded_ocr_exact_new_term_queries": len(ocr_exact_queries),
            "embedded_ocr_fuzzy_new_term_queries": len(ocr_fuzzy_queries),
        },
        "by_source_primary": {source: dict(counter) for source, counter in source_counts.items()},
        "by_source_signal": source_signals,
        "label_combinations": dict(label_combinations),
        "notes": [
            "The four cause labels are multi-label diagnostic signals and may overlap.",
            "Parser extraction is measured against the original PDF text layer on gold pages; a split/garbled token is reported separately from a truly absent token.",
            "Embedded-image OCR is counted only when OCR adds a query/answer-relevant term absent from the baseline page text.",
            "Semantic rescue is an observational comparison of cached runs, not a proof that embeddings caused the improvement.",
            "Language classification is a lightweight French/English marker heuristic; it is not a language-identification model.",
        ],
    }

    (output_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "per_query.jsonl", per_query)
    write_jsonl(output_dir / "gold_page_diagnostics.jsonl", page_diagnostics)

    lines = [
        "# Benchmark 220-query retrieval cause analysis",
        "",
        "This report compares base BM25, full embedded-image OCR BM25, and full embedded-image OCR + OpenRouter text embeddings.",
        "The cause counts are diagnostic multi-label counts; one query can appear in several rows.",
        "",
        "## Overall cached-run means",
        "",
        "| Arm | Page recall@10 | Page recall@20 |",
        "|---|---:|---:|",
        f"| Base BM25 | {pct(summary['metrics_overall']['base_page_recall@10'])} | {pct(summary['metrics_overall']['base_page_recall@20'])} |",
        f"| Full embedded-image OCR + BM25 | {pct(summary['metrics_overall']['full_ocr_bm25_page_recall@10'])} | {pct(summary['metrics_overall']['full_ocr_bm25_page_recall@20'])} |",
        f"| Full embedded-image OCR + embeddings | {pct(summary['metrics_overall']['full_ocr_embedding_page_recall@10'])} | {pct(summary['metrics_overall']['full_ocr_embedding_page_recall@20'])} |",
        "",
        f"Base misses at least one gold page in top-20 for **{len(loss_rows)}/{len(all_rows)}** queries.",
        "",
        "## Cause signals",
        "",
        "| Signal | All 220 | Recall-loss subset | Definition |",
        "|---|---:|---:|---|",
        f"| Parser extraction / garbling | {summary['counts_all_220']['parser_extraction_signal']} | {summary['counts_recall_loss_subset']['parser_extraction_signal']} | Raw PDF has a relevant term absent or split in baseline extracted text |",
        f"| Embedded-image OCR signal | {summary['counts_all_220']['embedded_image_ocr_signal']} | {summary['counts_recall_loss_subset']['embedded_image_ocr_signal']} | OCR adds a relevant term absent from baseline text |",
        f"| Semantic rescue | {summary['counts_all_220']['semantic_rescue']} | {summary['counts_recall_loss_subset']['semantic_rescue']} | Embedding improves page recall@20 or file recall@3 over full-OCR BM25 |",
        f"| Semantic page rescue only | {summary['counts_all_220']['semantic_page_rescue']} | {summary['counts_recall_loss_subset']['semantic_page_rescue']} | Embedding improves page recall@20 |",
        f"| Language mismatch | {summary['counts_all_220']['language_mismatch']} | {summary['counts_recall_loss_subset']['language_mismatch']} | Majority of gold pages classified French while queries are English |",
        f"| Embedding improves language-mismatch case | {summary['counts_all_220']['embedding_improves_language_case']} | {summary['counts_recall_loss_subset']['embedding_improves_language_case']} | Both language mismatch and embedding rescue |",
        "",
        "## Conservative primary diagnosis",
        "",
        "| Primary diagnosis | All 220 | Recall-loss subset |",
        "|---|---:|---:|",
    ]
    keys = sorted(set(summary["primary_diagnosis_all"]) | set(summary["primary_diagnosis_recall_loss_subset"]))
    for key in keys:
        lines.append(f"| {key} | {summary['primary_diagnosis_all'].get(key, 0)} | {summary['primary_diagnosis_recall_loss_subset'].get(key, 0)} |")
    lines += [
        "",
        "## Signals by dataset source",
        "",
        "| Source | Queries | Base loss@20 | Parser | Embedded OCR | Semantic rescue | Language mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source, vals in summary["by_source_signal"].items():
        lines.append(
            f"| {source} | {vals['queries']} | {vals['base_recall_loss_at_20']} | "
            f"{vals['parser_extraction_signal']} | {vals['embedded_image_ocr_signal']} | "
            f"{vals['semantic_rescue']} | {vals['language_mismatch']} |"
        )
    lines += [
        "",
        "## Language-mismatch embedding comparison",
        "",
        f"- Language-mismatch queries: **{len(language_rows)}**.",
        f"- Improved with embeddings: **{summary['language_case_embedding_delta']['improved_queries']}**.",
        f"- Page recall@20 improved: **{summary['language_case_embedding_delta']['page_recall_improved_queries']}**; file recall@3 improved: **{summary['language_case_embedding_delta']['file_recall_improved_queries']}**.",
        f"- Mean page recall@20 delta: **{summary['language_case_embedding_delta']['mean_page_recall20_delta']:+.4f}** ({summary['language_case_embedding_delta']['mean_page_recall20_delta']*100:+.2f} points).",
        f"- Median page recall@20 delta: **{summary['language_case_embedding_delta']['median_page_recall20_delta']:+.4f}**.",
        f"- Mean file recall@3 delta: **{summary['language_case_embedding_delta']['mean_file_recall3_delta']:+.4f}** ({summary['language_case_embedding_delta']['mean_file_recall3_delta']*100:+.2f} points).",
        "",
        "## Preparation detail",
        "",
        f"- Gold pages with a relevant raw-PDF term truly absent from baseline text: **{summary['preparation_detail']['parser_missing_relevant_term_pages']} pages / {summary['preparation_detail']['parser_missing_relevant_term_queries']} queries**.",
        f"- Gold pages with a relevant term split/garbled by extraction: **{summary['preparation_detail']['parser_split_or_garbled_relevant_term_pages']} pages / {summary['preparation_detail']['parser_split_or_garbled_relevant_term_queries']} queries**.",
        f"- Queries with an exact new relevant embedded-image OCR term: **{summary['preparation_detail']['embedded_ocr_exact_new_term_queries']}**; fuzzy/spelling-variant term: **{summary['preparation_detail']['embedded_ocr_fuzzy_new_term_queries']}**.",
        "",
        "## Interpretation limits",
        "",
        "- These are evidence-based diagnostics, not a causal intervention per query. A query can have a parser defect and still retrieve its gold page through another term.",
        "- The embedding comparison uses English queries and full OCR page text. It tests semantic retrieval, but it does not isolate language translation from embedding quality.",
        "- The language labels are a simple marker heuristic and should be audited with samples before being used as a routing feature.",
        "",
        "Artifacts: `per_query.jsonl` contains every query, its three retrieval outcomes, labels, and gold-page diagnostics; `gold_page_diagnostics.jsonl` contains page-level preparation evidence.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
