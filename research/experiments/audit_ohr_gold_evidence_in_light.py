"""Audit whether OHR gold evidence survives light preparation and retrieval.

This is intentionally an *evidence-preservation* audit, not a new ranking run:
the benchmark-provided ``evidence_context`` is used only after retrieval to
describe the gold page.  It is never supplied to a scorer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import unicodedata
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/work/ohr_replacement_20260920/inputs"
LIGHT_PAGES = ROOT / "data/work/ohr_replacement_20260920/embedded_ocr_filtered/size_filter_only/pages.jsonl"
HYBRID = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/hybrid_bm25_embeddings.jsonl"
COHERE = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/cohere/cohere_top20.jsonl"
KDL = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803/kdl/kdl_pages.jsonl"
OUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/analysis/ohr_gold_evidence_in_light"

TOKEN = re.compile(r"[\w]+(?:[.,:/-][\w]+)*", re.UNICODE)
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "to", "with",
    "what", "which", "who", "where", "when", "how", "was", "were", "this", "that", "these", "those", "does", "do", "did",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    return "".join(char for char in value if not unicodedata.combining(char))


def tokens(value: str) -> list[str]:
    return [token for token in TOKEN.findall(norm(value)) if len(token) > 1 and token not in STOP]


def token_coverage(reference: str, observed: str) -> tuple[float, list[str], list[str]]:
    reference_terms = list(dict.fromkeys(tokens(reference)))
    observed_terms = set(tokens(observed))
    found = [term for term in reference_terms if term in observed_terms]
    missing = [term for term in reference_terms if term not in observed_terms]
    return (len(found) / len(reference_terms) if reference_terms else 1.0, found, missing)


def answer_exact(answer: str, text: str) -> bool:
    # Spaces/punctuation often change in OCR; preserve alphanumerics for a fair
    # exact-content check while still requiring the complete answer sequence.
    compact_answer = re.sub(r"\W", "", norm(answer))
    compact_text = re.sub(r"\W", "", norm(text))
    return bool(compact_answer) and compact_answer in compact_text


def rank(pages: list[str], gold: str) -> int | None:
    try:
        return pages.index(gold) + 1
    except ValueError:
        return None


def excerpt(text: str, anchors: list[str], limit: int = 360) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    lowered = norm(clean)
    positions = [lowered.find(anchor) for anchor in anchors if anchor and lowered.find(anchor) >= 0]
    if not positions:
        return clean[:limit]
    start = max(0, min(positions) - 90)
    return clean[start:start + limit]


def content_state(answer_exactly_present: bool, answer_cov: float, evidence_cov: float, light_chars: int) -> str:
    if answer_exactly_present:
        return "answer_exact_in_light"
    if answer_cov >= 0.60 or evidence_cov >= 0.60:
        return "partial_evidence_in_light"
    if light_chars == 0:
        return "light_text_empty"
    return "evidence_weak_or_absent"


def main() -> None:
    queries = {row["query_id"]: row for row in read_jsonl(INPUT / "queries.jsonl")}
    gold_by_query: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(INPUT / "qrels.jsonl"):
        gold_by_query[row["query_id"]].append(f"{row['doc_id']}#page={int(row['page_index'])}")
    ohr_qids = sorted(gold_by_query)
    if len(ohr_qids) != 65 or any(not qid.startswith("ohrbench::") for qid in ohr_qids):
        raise RuntimeError(f"Expected 65 OHR qids, got {len(ohr_qids)}")

    light = {row["page_id"]: row for row in read_jsonl(LIGHT_PAGES)}
    kdl = {row["page_id"]: row for row in read_jsonl(KDL) if row["page_id"].startswith("ohrbench::")}
    hybrid = {row["query_id"]: row for row in read_jsonl(HYBRID) if row["query_id"].startswith("ohrbench::")}
    cohere = {row["query_id"]: row for row in read_jsonl(COHERE) if row["query_id"].startswith("ohrbench::")}
    if set(ohr_qids) != set(hybrid) or set(ohr_qids) != set(cohere):
        raise RuntimeError("OHR run/query alignment mismatch")

    audit: list[dict[str, Any]] = []
    for qid in ohr_qids:
        query = queries[qid]
        metadata = query.get("metadata") or {}
        hybrid_row, cohere_row = hybrid[qid], cohere[qid]
        selected_files = set(hybrid_row.get("selected_files") or [])
        for page_id in sorted(gold_by_query[qid]):
            page = light[page_id]
            light_text = str(page.get("text") or "")
            kdl_text = str(kdl.get(page_id, {}).get("text") or "")
            answer = " | ".join(str(value) for value in query.get("answers") or [])
            evidence_context = str(metadata.get("evidence_context") or "")
            ans_cov, ans_found, ans_missing = token_coverage(answer, light_text)
            ev_cov, ev_found, ev_missing = token_coverage(evidence_context, light_text)
            exact = any(answer_exact(str(value), light_text) for value in query.get("answers") or [])
            hybrid_rank_200 = rank(list(hybrid_row.get("candidate_pages") or []), page_id)
            hybrid_rank_20 = rank(list(hybrid_row.get("returned_pages") or []), page_id)
            cohere_rank_20 = rank(list(cohere_row.get("returned_pages") or cohere_row.get("reranked_pages") or []), page_id)
            state = content_state(exact, ans_cov, ev_cov, len(light_text))
            if page["doc_id"] not in selected_files:
                retrieval_state = "gold_file_not_in_light_top3"
            elif hybrid_rank_200 is None:
                retrieval_state = "gold_page_not_in_candidate_pool"
            elif cohere_rank_20 is None:
                retrieval_state = "candidate_present_but_not_in_cohere_top20"
            else:
                retrieval_state = "cohere_top20_hit"
            audit.append({
                "query_id": qid,
                "query": query["query"],
                "answer": answer,
                "evidence_context": evidence_context,
                "visual_class": metadata.get("visual_class"),
                "answer_form": metadata.get("answer_form"),
                "gold_page_id": page_id,
                "gold_doc_id": page["doc_id"],
                "gold_page_index": page["page_index"],
                "light_text_source": page.get("text_source"),
                "light_needs_ocr": bool(page.get("needs_ocr")),
                "light_ocr_word_count": int(page.get("ocr_word_count") or 0),
                "light_ocr_mean_confidence": float(page.get("ocr_mean_confidence") or 0),
                "light_chars": len(light_text),
                "kdl_chars": len(kdl_text),
                "answer_exact_in_light": exact,
                "answer_token_coverage_in_light": ans_cov,
                "evidence_token_coverage_in_light": ev_cov,
                "answer_terms_found": ans_found,
                "answer_terms_missing": ans_missing,
                "evidence_terms_found": ev_found,
                "evidence_terms_missing": ev_missing,
                "content_state": state,
                "light_evidence_excerpt": excerpt(light_text, ans_found or ev_found),
                "kdl_evidence_excerpt": excerpt(kdl_text, tokens(answer) or tokens(evidence_context)),
                "gold_file_in_light_top3": page["doc_id"] in selected_files,
                "hybrid_candidate_rank_up_to_200": hybrid_rank_200,
                "hybrid_final_rank_up_to_20": hybrid_rank_20,
                "cohere_final_rank_up_to_20": cohere_rank_20,
                "retrieval_state": retrieval_state,
            })

    if len(audit) != 65:
        raise RuntimeError(f"Expected one gold page per OHR query, got {len(audit)}")
    summary = {
        "protocol": {
            "queries": 65,
            "gold_pages": len(audit),
            "light_text": "PDF-inspector + selective full-page Tesseract + filtered embedded-image OCR",
            "retrieval": "official language-routed hybrid and Cohere reranker; Kf=3, up-to-200 page candidate pool, final top-20",
            "important_caveat": "evidence_context and answers are used only for post-hoc audit of the gold page, never for retrieval.",
        },
        "content_state_counts": dict(Counter(row["content_state"] for row in audit)),
        "retrieval_state_counts": dict(Counter(row["retrieval_state"] for row in audit)),
        "cross_tab_content_by_retrieval": {
            state: dict(Counter(row["retrieval_state"] for row in audit if row["content_state"] == state))
            for state in sorted({row["content_state"] for row in audit})
        },
        "exact_answer_in_light": sum(row["answer_exact_in_light"] for row in audit),
        "mean_answer_token_coverage": sum(row["answer_token_coverage_in_light"] for row in audit) / len(audit),
        "mean_evidence_token_coverage": sum(row["evidence_token_coverage_in_light"] for row in audit) / len(audit),
        "by_visual_class": {
            visual: {
                "pages": len(rows),
                "exact_answer_in_light": sum(row["answer_exact_in_light"] for row in rows),
                "mean_evidence_token_coverage": sum(row["evidence_token_coverage_in_light"] for row in rows) / len(rows),
                "gold_file_in_top3": sum(row["gold_file_in_light_top3"] for row in rows),
                "cohere_top20": sum(row["cohere_final_rank_up_to_20"] is not None for row in rows),
            }
            for visual, rows in sorted(((visual, [r for r in audit if r["visual_class"] == visual]) for visual in {r["visual_class"] for r in audit}), key=lambda x: str(x[0]))
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_query_gold_evidence.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit), encoding="utf-8"
    )
    (OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# OHR gold-evidence audit: light preparation and retrieval",
        "",
        "This is a post-hoc audit of the annotated gold page for every OHR query. The answer and `evidence_context` are **not** used by retrieval.",
        "",
        "## Summary",
        "",
        f"- Gold pages audited: {len(audit)} (one per OHR query).",
        f"- Exact annotated answer retained in light text: {summary['exact_answer_in_light']}/{len(audit)}.",
        f"- Mean answer-token coverage in light text: {100 * summary['mean_answer_token_coverage']:.1f}%.",
        f"- Mean annotated-evidence token coverage in light text: {100 * summary['mean_evidence_token_coverage']:.1f}%.",
        f"- Gold file absent from light top-3: {summary['retrieval_state_counts'].get('gold_file_not_in_light_top3', 0)}/{len(audit)}.",
        f"- Gold page in Cohere top-20: {summary['retrieval_state_counts'].get('cohere_top20_hit', 0)}/{len(audit)}.",
        "",
        "## Every gold page",
        "",
        "| QID | Evidence class | Content retained by light | Retrieval state | Answer / annotated evidence |",
        "|---|---|---|---|---|",
    ]
    for row in audit:
        evidence = (row["answer"] or row["evidence_context"]).replace("|", "\\|").replace("\n", " ")[:180]
        lines.append(f"| `{row['query_id'].split('::', 1)[1][:8]}` | {row['visual_class']} | {row['content_state']} ({100 * row['evidence_token_coverage_in_light']:.0f}%) | {row['retrieval_state']} | {evidence} |")
    lines += ["", "Full per-query evidence, OCR fields, rank positions and excerpts: `per_query_gold_evidence.jsonl`."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
