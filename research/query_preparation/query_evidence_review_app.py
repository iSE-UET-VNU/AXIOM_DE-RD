"""Local UI for analysing queries: gold evidence reader + retriever top-N inspector.

Run from the repository root with:
    streamlit run research/query_preparation/query_evidence_review_app.py

Tabs
* Gold evidence: read the gold pages of each gold file (page navigator, neighbouring
  pages, PDF excerpt containing only the gold pages, optional full PDF).
* Retriever top-N: the pages returned by the data-discovery light retriever
  (BM25 + text-embedding-3-small over PP-OCRv5 text, saved run
  ``light_run_top100``), labelled gold page / gold file / other file, with the
  query terms that matched or are missing on each page.

Gold qrels are analysis-only; never feed them to query preparation or retrieval.
"""

from __future__ import annotations

import html
import json
import re
import sys
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_scores import (  # noqa: E402  (BM25 / dense / file-prior breakdown of the saved run)
    TOKEN,
    MissingEmbedding,
    RetrieverScorer,
    query_terms,
    token_stems,
)

QUERY_PATH = ROOT / "data" / "raw" / "0.1. BENCHMARK" / "queries.jsonl"
DOCUMENT_PATH = ROOT / "data" / "raw" / "0.1. BENCHMARK" / "documents.jsonl"
OLD_QRELS_PATH = (
    ROOT
    / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20"
    / "bundle_debug_light-retrieval/input/qrels_page_level.jsonl"
)
OHR_QRELS_PATH = ROOT / "data/work/ohr_replacement_20260920/inputs/qrels.jsonl"
PAGE_TEXT_PATH = (
    ROOT
    / "data/work/ohr_replacement_20260920/retrieval/pymupdf_all_improvements_220"
    / "preparation/pages_pymupdf_plus_embedded_ocr.jsonl"
)
HANDOFF = (
    ROOT
    / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803"
)
GATE_PATH = HANDOFF / "light_prep/gate_ppocrv5_ungated.json"
RETRIEVER_PAGES_PATH = HANDOFF / "light_prep/pages_ppocrv5.jsonl"

PLACEHOLDER = re.compile(r"\[Image:[^\]]*\]")
STATUS_GOLD, STATUS_FILE, STATUS_NOISE = "✅ Gold page", "🟡 Gold file · trang khác", "🔴 File khác"
STATUS_TINT = {
    STATUS_GOLD: "rgba(46, 160, 67, 0.28)",
    STATUS_FILE: "rgba(210, 153, 34, 0.25)",
    STATUS_NOISE: "rgba(248, 81, 73, 0.12)",
}


# ----------------------------------------------------------------------------- data

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@st.cache_data(show_spinner=False)
def load_data() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, str]]:
    queries = read_jsonl(QUERY_PATH)
    documents = {row["doc_id"]: row for row in read_jsonl(DOCUMENT_PATH)}

    gold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(OLD_QRELS_PATH):
        if row.get("query_id", "").startswith(("vidore_physics::", "vidore_industrial::")):
            gold[row["query_id"]].append(
                {
                    "page_id": row["page_id"],
                    "doc_id": row["doc_id"],
                    "page_index": int(row["page_index"]),
                    "relevance": row.get("relevance", 1),
                }
            )
    for row in read_jsonl(OHR_QRELS_PATH):
        query_id = row.get("query_id", "")
        doc_id = row.get("doc_id", "")
        page_index = int(row.get("page_index", 0))
        gold[query_id].append(
            {
                "page_id": f"{doc_id}#page={page_index}",
                "doc_id": doc_id,
                "page_index": page_index,
                "relevance": row.get("relevance", 1),
            }
        )

    page_text = {
        row["page_id"]: str(row.get("text") or "")
        for row in read_jsonl(PAGE_TEXT_PATH)
        if row.get("page_id")
    }
    return queries, documents, dict(gold), page_text


@st.cache_data(show_spinner="Đang nạp kết quả retriever…")
def load_retriever() -> tuple[dict[str, list[str]], dict[str, str], str]:
    """Saved light-retriever top-100 per query and the PP-OCRv5 text it indexed."""
    run: dict[str, list[str]] = {}
    description = "không tìm thấy gate_ppocrv5_ungated.json"
    if GATE_PATH.exists():
        gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
        run = {qid: list(pages) for qid, pages in gate["light_run_top100"].items()}
        description = (
            f"{gate.get('variant', 'light retriever')} · {gate.get('dense_model', '')} · "
            f"{gate.get('n_corpus_pages', '?')} trang"
        )
    text = {
        row["page_id"]: str(row.get("text") or "")
        for row in read_jsonl(RETRIEVER_PAGES_PATH)
        if row.get("page_id")
    }
    return run, text, description


def repository_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    # Benchmark paths are relative to a corpus data/ folder, not always ROOT.
    candidates = (
        ROOT / path,
        ROOT / "data" / "raw" / "0.1. BENCHMARK" / "data" / path.name,
        ROOT / "data" / "raw" / "0.1. BENCHMARK" / path.name,
        ROOT / "data" / "raw" / "BENCHMARK" / "data" / path.name,
        ROOT / "data" / "raw" / "BENCHMARK" / path.name,
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def open_in_browser(pdf_path: Path, page_number: int) -> bool:
    """Open the raw PDF in a new browser tab at a 1-based page (``#page=N`` works in Chrome/Edge/Firefox).

    The file is opened by the machine that runs Streamlit, which is the local machine here.
    """
    url = f"{pdf_path.resolve().as_uri()}#page={page_number}"
    try:
        return bool(webbrowser.open_new_tab(url))
    except Exception:
        return False


def split_page_id(page_id: str) -> tuple[str, int]:
    doc_id, _, index = page_id.partition("#page=")
    return doc_id, int(index)


# --------------------------------------------------------------------------- pdf io

@st.cache_data(show_spinner=False, max_entries=400)
def render_page(pdf_path: str, page_index: int, zoom: float) -> bytes | None:
    try:
        with fitz.open(pdf_path) as pdf:
            if page_index < 0 or page_index >= pdf.page_count:
                return None
            pixmap = pdf.load_page(page_index).get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return pixmap.tobytes("png")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def pdf_page_count(pdf_path: str) -> int:
    try:
        with fitz.open(pdf_path) as pdf:
            return pdf.page_count
    except Exception:
        return 0


@st.cache_data(show_spinner=False, max_entries=64)
def gold_excerpt_pdf(pdf_path: str, page_indices: tuple[int, ...]) -> bytes | None:
    """A small PDF with only the requested pages (selectable text, native zoom)."""
    try:
        with fitz.open(pdf_path) as source, fitz.open() as excerpt:
            for index in page_indices:
                if 0 <= index < source.page_count:
                    excerpt.insert_pdf(source, from_page=index, to_page=index)
            return excerpt.tobytes()
    except Exception:
        return None


# ----------------------------------------------------------------------- text helpers

@st.cache_resource(show_spinner="Đang dựng chỉ mục BM25 và nạp vector embedding…")
def load_scorer() -> RetrieverScorer | None:
    try:
        return RetrieverScorer()
    except Exception as error:  # missing cache files: UI keeps working without score breakdown
        st.session_state["scorer_error"] = str(error)
        return None


def clean_text(text: str) -> str:
    return PLACEHOLDER.sub("▣", text)


def match_terms(text: str, terms: dict[str, str]) -> tuple[list[str], list[str]]:
    """Query words found / not found on the page after the BM25 analyzer (images ignored)."""
    found: set[str] = set()
    for match in TOKEN.finditer(PLACEHOLDER.sub(" ", text)):
        found.update(token_stems(match.group(0)))
    matched: list[str] = []
    missing: list[str] = []
    for stem, word in terms.items():
        target = matched if stem in found else missing
        if word not in target and word not in matched:
            target.append(word)
    missing = [word for word in missing if word not in matched]
    return matched, missing


def highlight(text: str, terms: dict[str, str]) -> str:
    out: list[str] = []
    last = 0
    for match in TOKEN.finditer(text):
        if any(stem in terms for stem in token_stems(match.group(0))):
            out.append(html.escape(text[last:match.start()]))
            out.append(
                "<mark style='background:#ffe066;color:#000;padding:0 2px;border-radius:3px'>"
                f"{html.escape(match.group(0))}</mark>"
            )
            last = match.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def text_block(text: str, terms: dict[str, str], height: int = 480) -> None:
    body = highlight(clean_text(text), terms) if text.strip() else "<i>(không có text)</i>"
    st.html(
        "<div style='white-space:pre-wrap;font-size:13px;line-height:1.45;overflow:auto;"
        f"max-height:{height}px;padding:.7rem .85rem;border:1px solid rgba(128,128,128,.35);"
        f"border-radius:8px;background:rgba(128,128,128,.07)'>{body}</div>"
    )


# ----------------------------------------------------------------------------- app

st.set_page_config(page_title="Query evidence review", page_icon="🔎", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1500px; padding-top: 1.5rem;}
    [data-testid="stMetric"] {background: rgba(128,128,128,.10); padding:.75rem; border-radius:8px;}
    .query-card {padding:1rem 1.1rem; border:1px solid rgba(128,128,128,.35); border-radius:10px;
                 background: rgba(128,128,128,.07);}
    </style>
    """,
    unsafe_allow_html=True,
)

queries, documents, gold_by_query, page_text = load_data()
retriever_run, retriever_text, retriever_info = load_retriever()
st.title("Query evidence review")
st.caption("Phân tích từng query: đọc gold evidence và xem nhiễu trong top-N của retriever.")

if not queries:
    st.error(f"Không đọc được danh sách query: {QUERY_PATH}")
    st.stop()


def doc_label(doc_id: str) -> str:
    doc = documents.get(doc_id, {})
    meta = doc.get("metadata") or {}
    return str(meta.get("source_doc_id") or Path(doc.get("path", doc_id)).name)


def doc_pdf(doc_id: str) -> Path | None:
    path = documents.get(doc_id, {}).get("path")
    if not path:
        return None
    resolved = repository_path(str(path))
    return resolved if resolved.exists() else None


def step_query(delta: int, labels: list[str]) -> None:
    current = st.session_state.get("query_label")
    st.session_state["query_label"] = (
        labels[(labels.index(current) + delta) % len(labels)] if current in labels else labels[0]
    )


source_names = sorted({str(row.get("source", "unknown")) for row in queries})
default_sources = [name for name in source_names if name.startswith("vidore")] or source_names
with st.sidebar:
    st.header("Tìm query")
    chosen_sources = st.multiselect("Dataset", source_names, default=default_sources)
    search = st.text_input("Tìm theo query hoặc ID", placeholder="Ví dụ: entropy, GL-414…")

filtered = [
    row
    for row in queries
    if row.get("source") in chosen_sources
    and (
        not search.strip()
        or search.casefold() in str(row.get("query", "")).casefold()
        or search.casefold() in str(row.get("query_id", "")).casefold()
    )
]
def query_order(row: dict[str, Any]) -> tuple[str, int, str]:
    suffix = str(row.get("query_id", "")).rsplit("::", 1)[-1]
    return str(row.get("source", "")), int(suffix) if suffix.isdigit() else 0, suffix


filtered.sort(key=query_order)

if not filtered:
    st.info("Không có query khớp bộ lọc.")
    st.stop()

options = {
    f"{row.get('source', '')} · {row.get('query_id', '')} · {row.get('query', '')[:95]}": row
    for row in filtered
}
labels = list(options)
if st.session_state.get("query_label") not in options:
    st.session_state.pop("query_label", None)

with st.sidebar:
    nav_prev, nav_next = st.columns(2)
    nav_prev.button("◀ Query trước", on_click=step_query, args=(-1, labels), use_container_width=True)
    nav_next.button("Query sau ▶", on_click=step_query, args=(1, labels), use_container_width=True)
    st.caption(f"{len(labels)} query khớp bộ lọc")
    st.divider()
    st.header("Hiển thị")
    zoom = st.slider("Độ phóng ảnh trang", 0.8, 2.5, 1.4, 0.1)
    top_n = st.slider("Retriever: số trang top-N", 5, 100, 20, 5)

selected_label = st.selectbox("Query", labels, key="query_label")
query = options[selected_label]
query_id = query["query_id"]
query_text = str(query.get("query", ""))
terms = query_terms(query_text)
scorer = load_scorer()
scores = None
if scorer is not None:
    try:
        scores = scorer.score(query_text)
    except MissingEmbedding:
        st.sidebar.warning("Query này chưa có embedding trong cache nên không bóc tách được điểm.")
elif st.session_state.get("scorer_error"):
    st.sidebar.warning(f"Không nạp được bộ tính điểm: {st.session_state['scorer_error'][:160]}")
gold_pages = gold_by_query.get(query_id, [])
gold_ids = {evidence["page_id"] for evidence in gold_pages}
gold_docs = {evidence["doc_id"] for evidence in gold_pages}

file_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
for evidence in gold_pages:
    file_groups[evidence["doc_id"]].append(evidence)

full_run = retriever_run.get(query_id, [])
rank_of = {page_id: position + 1 for position, page_id in enumerate(full_run)}

metric_cols = st.columns(4)
metric_cols[0].metric("Dataset", query.get("source", "—"))
metric_cols[1].metric("Gold files", len(file_groups))
metric_cols[2].metric("Gold pages", len(gold_pages))
best = min((rank_of[p] for p in gold_ids if p in rank_of), default=None)
metric_cols[3].metric("Gold page tốt nhất trong retriever", f"hạng {best}" if best else "ngoài top-100")

st.markdown("#### Query")
st.markdown(
    f"<div class='query-card'>{html.escape(query_text)}</div>",
    unsafe_allow_html=True,
)
answers = query.get("answers") or []
if answers:
    st.markdown("**Reference answer**")
    st.write(answers[0])
metadata = query.get("metadata") or {}
if metadata.get("evidence_context"):
    with st.expander("Benchmark evidence context"):
        st.write(metadata["evidence_context"])


def page_status(page_id: str) -> str:
    if page_id in gold_ids:
        return STATUS_GOLD
    return STATUS_FILE if split_page_id(page_id)[0] in gold_docs else STATUS_NOISE


def rank_label(rank: int, raw: float | None = None) -> str:
    total = len(scorer.page_ids) if scorer else 0
    return "không khớp" if raw is not None and raw <= 0 else f"#{rank}/{total}"


def score_panel(page_id: str) -> None:
    """Why this page is ranked where it is: BM25, dense and file-prior parts of the final score."""
    if scores is None or scorer is None or page_id not in scorer.page_index:
        return
    i = scorer.page_index[page_id]
    part_bm25, part_dense, part_file = scores.contributions(i)
    final = float(scores.final[i]) or 1e-12
    cols = st.columns(4)
    cols[0].metric(f"Điểm cuối · {rank_label(int(scores.final_rank[i]))}", f"{scores.final[i]:.3f}")
    cols[1].metric(f"BM25 · {rank_label(int(scores.bm25_rank[i]), scores.bm25_raw[i])}", f"{scores.bm25_raw[i]:.1f}",
                   help=f"BM25 thô. Chuẩn hoá theo max toàn corpus = {scores.bm25_norm[i]:.3f}.")
    cols[2].metric(f"Dense · {rank_label(int(scores.dense_rank[i]))}", f"{scores.dense[i]:.3f}",
                   help=f"Cosine lớn nhất giữa query và các chunk của trang. Chuẩn hoá theo max = {scores.dense_norm[i]:.3f}.")
    cols[3].metric("File prior", f"{scores.file_prior[i]:.3f}",
                   help="Trang dense tốt nhất của file này, chuẩn hoá theo file tốt nhất toàn corpus.")
    st.caption(
        f"Điểm cuối = 0.85 × (0.3 × BM25 + 0.7 × dense) + 0.15 × file prior → đóng góp: "
        f"BM25 {part_bm25:.3f} ({100 * part_bm25 / final:.0f}%) · Dense {part_dense:.3f} "
        f"({100 * part_dense / final:.0f}%) · File {part_file:.3f} ({100 * part_file / final:.0f}%)"
    )
    with st.expander("BM25 theo từng từ của query"):
        term_rows = scorer.bm25_terms(query_text, i)
        if term_rows:
            term_frame = pd.DataFrame(term_rows).rename(
                columns={"word": "Từ", "idf": "IDF", "df": "Số trang chứa từ", "contribution": "Đóng góp BM25"}
            )
            st.dataframe(term_frame.sort_values("Đóng góp BM25", ascending=False), hide_index=True, width="stretch")
            st.caption("IDF thấp / số trang chứa từ lớn = từ chung chung, ít phân biệt. Đóng góp 0 = từ không có trên trang.")
    chunk, similarity = scorer.best_chunk(scores, i)
    if chunk:
        with st.expander(f"Đoạn gần query nhất theo embedding (cosine {similarity:.3f})"):
            text_block(chunk, terms, height=260)


def score_ranks(page_id: str) -> str:
    """' [BM25 #x · dense #y]' for a page, empty when scores are unavailable."""
    if scores is None or scorer is None or page_id not in scorer.page_index:
        return ""
    i = scorer.page_index[page_id]
    bm25 = f"#{scores.bm25_rank[i]}" if scores.bm25_raw[i] > 0 else "không khớp"
    return f" [BM25 {bm25} · dense #{scores.dense_rank[i]}]"


def show_page(page_id: str, badge: str) -> None:
    """Page image (left) + retriever text with highlighted query terms (right)."""
    doc_id, page_index = split_page_id(page_id)
    pdf_path = doc_pdf(doc_id)
    st.markdown(badge)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        if pdf_path:
            image = render_page(str(pdf_path), page_index, zoom)
            if image:
                st.image(image, caption=f"{doc_label(doc_id)} · trang {page_index + 1}", width="stretch")
            else:
                st.info("Không render được trang PDF này.")
        else:
            st.info("Không tìm thấy PDF cục bộ; chỉ có text đã trích xuất.")
    with right:
        main_text = retriever_text.get(page_id, "")
        matched, missing = match_terms(main_text, terms)
        if terms:
            st.markdown(
                f"**Từ query khớp ({len(matched)}/{len(matched) + len(missing)}):** "
                + (":green[" + ", ".join(matched) + "]" if matched else "_không có_")
                + ("  \n**Thiếu:** :red[" + ", ".join(missing) + "]" if missing else "")
            )
        score_panel(page_id)
        alt_text = page_text.get(page_id, "")
        if alt_text.strip() and alt_text.strip() != main_text.strip():
            tab_main, tab_alt = st.tabs(["Text retriever (PP-OCRv5)", "Text PyMuPDF + OCR"])
            with tab_main:
                text_block(main_text, terms)
            with tab_alt:
                text_block(alt_text, terms)
        else:
            text_block(main_text or alt_text, terms)
        st.caption("Vàng = từ khớp query sau analyzer BM25 (stem EN/FR). ▣ = ảnh nhúng, không được lập chỉ mục.")


def sync_pills(cur_key: str, pill_key: str, gold_numbers: list[int]) -> None:
    current = st.session_state[cur_key]
    st.session_state[pill_key] = current if current in gold_numbers else None


def shift_page(cur_key: str, pill_key: str, gold_numbers: list[int], delta: int, high: int) -> None:
    st.session_state[cur_key] = min(high, max(1, st.session_state[cur_key] + delta))
    sync_pills(cur_key, pill_key, gold_numbers)


def pick_gold(cur_key: str, pill_key: str) -> None:
    value = st.session_state.get(pill_key)
    if value is not None:
        st.session_state[cur_key] = value


tab_gold, tab_retriever = st.tabs(["📄 Gold evidence", f"🔍 Retriever top-{top_n}"])

# ------------------------------------------------------------------ gold evidence tab
with tab_gold:
    if not gold_pages:
        st.warning("Không tìm thấy qrel cho query này trong các artifact hiện có.")
    else:
        ranks_text = " · ".join(
            f"p.{split_page_id(p)[1] + 1} → "
            + (f"hạng {rank_of[p]}" if p in rank_of else ">100")
            + score_ranks(p)
            for p in sorted(gold_ids, key=lambda p: (split_page_id(p)[0], split_page_id(p)[1]))
        )
        st.caption(f"Vị trí của gold pages trong retriever: {ranks_text}")

        for doc_id, pages in sorted(file_groups.items()):
            doc = documents.get(doc_id, {})
            doc_meta = doc.get("metadata") or {}
            gold_indices = sorted({item["page_index"] for item in pages})
            gold_numbers = [index + 1 for index in gold_indices]
            title = f"{doc_label(doc_id)} · {len(gold_indices)} gold page{'s' if len(gold_indices) != 1 else ''}"
            with st.expander(title, expanded=len(file_groups) == 1):
                pdf_path = doc_pdf(doc_id)
                st.caption(
                    f"Gold file ID: `{doc_id}` · File: `{pdf_path if pdf_path else doc.get('path', 'unknown')}`"
                )
                page_count = pdf_page_count(str(pdf_path)) if pdf_path else int(doc_meta.get("page_count") or 0)
                if page_count:
                    st.caption(f"Số trang trong file: {page_count}")
                high = page_count or 100_000

                cur_key = f"cur::{query_id}::{doc_id}"
                pill_key = f"pill::{query_id}::{doc_id}"
                if cur_key not in st.session_state:
                    st.session_state[cur_key] = gold_numbers[0]
                    st.session_state[pill_key] = gold_numbers[0]

                if pdf_path:
                    action_cols = st.columns([1.3, 1.3, 1.3, 1.1])
                    show_excerpt = action_cols[0].toggle(
                        "Mở PDF chỉ gồm gold pages", key=f"excerpt::{query_id}::{doc_id}"
                    )
                    show_full = action_cols[1].toggle(
                        "Mở toàn bộ tài liệu gốc", key=f"full::{query_id}::{doc_id}"
                    )
                    if action_cols[3].button(
                        f"🌐 Mở trong trình duyệt (trang {gold_numbers[0]})",
                        key=f"browser::{query_id}::{doc_id}",
                        help=f"Mở {pdf_path.name} ở tab mới, tại trang gold đầu tiên (trang {gold_numbers[0]}).",
                        width="stretch",
                    ):
                        if open_in_browser(pdf_path, gold_numbers[0]):
                            st.toast(f"Đã mở {pdf_path.name} tại trang {gold_numbers[0]}", icon="🌐")
                        else:
                            st.toast("Không mở được trình duyệt. Hãy mở đường dẫn file bên dưới.", icon="⚠️")
                    excerpt = gold_excerpt_pdf(str(pdf_path), tuple(gold_indices))
                    if excerpt:
                        action_cols[2].download_button(
                            "Tải PDF gold pages",
                            data=excerpt,
                            file_name=f"gold_{query_id.replace('::', '_')}_{pdf_path.stem}.pdf",
                            mime="application/pdf",
                            key=f"dl-excerpt::{query_id}::{doc_id}",
                        )
                    if show_excerpt and excerpt:
                        st.pdf(excerpt, height=820, key=f"excerpt-pdf::{query_id}::{doc_id}")
                    if show_full:
                        st.pdf(pdf_path.read_bytes(), height=820, key=f"full-pdf::{query_id}::{doc_id}")

                st.markdown("**Đọc từng trang**")
                nav = st.columns([1, 1, 1.2, 4])
                nav[0].button("◀ Trang trước", key=f"prev::{cur_key}", on_click=shift_page,
                              args=(cur_key, pill_key, gold_numbers, -1, high), use_container_width=True)
                nav[1].button("Trang sau ▶", key=f"next::{cur_key}", on_click=shift_page,
                              args=(cur_key, pill_key, gold_numbers, 1, high), use_container_width=True)
                nav[2].number_input("Tới trang", min_value=1, max_value=high, step=1, key=cur_key,
                                    on_change=sync_pills, args=(cur_key, pill_key, gold_numbers),
                                    label_visibility="collapsed")
                with nav[3]:
                    st.pills("Gold pages", gold_numbers, selection_mode="single", key=pill_key,
                             format_func=lambda number: f"★ p.{number}", on_change=pick_gold,
                             args=(cur_key, pill_key), label_visibility="collapsed")

                current = st.session_state[cur_key]
                page_id = f"{doc_id}#page={current - 1}"
                is_gold = page_id in gold_ids
                rank = rank_of.get(page_id)
                badge = (
                    ("**★ Gold page**" if is_gold else "**Trang lân cận — không phải gold**")
                    + f" · trang {current}"
                    + (f" · retriever hạng #{rank}" if rank else " · ngoài top-100 của retriever")
                )
                show_page(page_id, badge)

# -------------------------------------------------------------------- retriever tab
with tab_retriever:
    if not full_run:
        st.warning("Chưa có kết quả retriever đã lưu cho query này.")
    else:
        st.caption(f"Retriever đã lưu: {retriever_info}. Top-100 rút gọn xuống top-{top_n}.")
        if str(query.get("source", "")).endswith("physics"):
            st.caption(
                "Physics: query tiếng Anh, tài liệu tiếng Pháp. Retriever bù bằng embedding cho cặp khác "
                "ngôn ngữ, nên số từ khớp BM25 ở đây chỉ mang tính tham khảo."
            )
        ranked = full_run[:top_n]
        statuses = [page_status(page_id) for page_id in ranked]
        n_gold = sum(status == STATUS_GOLD for status in statuses)
        n_file = sum(status == STATUS_FILE for status in statuses)
        n_noise = sum(status == STATUS_NOISE for status in statuses)
        distinct_files = {split_page_id(page_id)[0] for page_id in ranked}

        summary = st.columns(5)
        summary[0].metric(f"Gold trong top-{top_n}", f"{n_gold}/{len(gold_ids)}")
        summary[1].metric("Trang cùng gold file", n_file)
        summary[2].metric("Trang file khác (noise)", n_noise)
        summary[3].metric("Số file trong top", len(distinct_files))
        summary[4].metric("Gold files có mặt", f"{len(distinct_files & gold_docs)}/{len(gold_docs)}")

        missed = [p for p in sorted(gold_ids) if p not in set(ranked)]
        if missed:
            st.warning(
                "Gold pages chưa vào top-"
                f"{top_n}: "
                + " · ".join(
                    f"{doc_label(split_page_id(p)[0])} p.{split_page_id(p)[1] + 1} "
                    + (f"(hạng {rank_of[p]})" if p in rank_of else "(>100)")
                    + score_ranks(p)
                    for p in missed
                )
            )

        rows = []
        chart_rows = []
        for position, (page_id, status) in enumerate(zip(ranked, statuses), start=1):
            doc_id, page_index = split_page_id(page_id)
            matched, missing = match_terms(retriever_text.get(page_id, ""), terms)
            row = {
                "Hạng": position,
                "Loại": status,
                "File": doc_label(doc_id),
                "Trang": page_index + 1,
            }
            if scores is not None and scorer is not None and page_id in scorer.page_index:
                i = scorer.page_index[page_id]
                part_bm25, part_dense, part_file = scores.contributions(i)
                total = float(scores.final[i]) or 1e-12
                row.update(
                    {
                        "Điểm": float(scores.final[i]),
                        "BM25": float(scores.bm25_raw[i]),
                        "Hạng BM25": int(scores.bm25_rank[i]) if scores.bm25_raw[i] > 0 else None,
                        "Dense": float(scores.dense[i]),
                        "Hạng dense": int(scores.dense_rank[i]),
                        "Nguồn điểm": f"BM25 {100 * part_bm25 / total:.0f}% · Dense {100 * part_dense / total:.0f}% · File {100 * part_file / total:.0f}%",
                    }
                )
                chart_rows.append(
                    {"Hạng": f"{position:03d}", "BM25": part_bm25, "Dense": part_dense, "File prior": part_file}
                )
            row.update(
                {
                    "Khớp": f"{len(matched)}/{len(matched) + len(missing)}",
                    "Từ khớp": ", ".join(matched),
                    "Từ thiếu": ", ".join(missing),
                    "Ký tự": len(PLACEHOLDER.sub("", retriever_text.get(page_id, ""))),
                }
            )
            rows.append(row)
        frame = pd.DataFrame(rows)
        if chart_rows:
            st.markdown("##### Điểm cuối được tạo từ đâu (theo hạng)")
            st.bar_chart(pd.DataFrame(chart_rows).set_index("Hạng"), stack=True, height=230)
            st.caption(
                "Mỗi cột là một trang trong top-N, chia theo phần đóng góp vào điểm cuối. Trang nhiễu thường có "
                "phần Dense lớn nhưng BM25 nhỏ, hoặc nhờ File prior từ file có trang dense rất cao."
            )
        formats = {c: "{:.3f}" for c in ("Điểm", "Dense") if c in frame} | ({"BM25": "{:.1f}"} if "BM25" in frame else {})
        st.dataframe(
            frame.style.apply(lambda r: [f"background-color: {STATUS_TINT[r['Loại']]}"] * len(r), axis=1).format(
                formats, na_rep="—"
            ),
            hide_index=True,
            width="stretch",
            height=min(36 * (len(rows) + 1) + 3, 760),
        )

        with st.expander("Phân bố theo file"):
            by_file = (
                frame.assign(_doc=[split_page_id(p)[0] for p in ranked])
                .groupby("_doc")
                .agg(**{"Số trang": ("Hạng", "count"), "Các hạng": ("Hạng", lambda r: ", ".join(map(str, r)))})
                .reset_index()
            )
            by_file["File"] = by_file["_doc"].map(doc_label)
            by_file["Gold file"] = by_file["_doc"].map(lambda d: "✅" if d in gold_docs else "")
            st.dataframe(
                by_file.sort_values("Số trang", ascending=False)[["File", "Gold file", "Số trang", "Các hạng"]],
                hide_index=True,
                width="stretch",
            )

        st.markdown("##### Soi từng trang")
        rk_key = f"rk::{query_id}"
        if rk_key not in st.session_state or st.session_state[rk_key] > len(ranked):
            st.session_state[rk_key] = 1

        def move_rank(delta: int) -> None:
            st.session_state[rk_key] = (st.session_state[rk_key] - 1 + delta) % len(ranked) + 1

        def jump_rank(want: str) -> None:
            """Next rank (wrapping) that is a gold page, or any non-gold page for 'noise'."""
            start = st.session_state[rk_key]
            for step in range(1, len(ranked) + 1):
                candidate = (start - 1 + step) % len(ranked)
                if (statuses[candidate] == STATUS_GOLD) == (want == "gold"):
                    st.session_state[rk_key] = candidate + 1
                    return

        controls = st.columns([1, 1, 1.2, 1.4, 1.4, 3])
        controls[0].button("◀ Hạng trước", key=f"rk-prev::{query_id}", on_click=move_rank, args=(-1,),
                           use_container_width=True)
        controls[1].button("Hạng sau ▶", key=f"rk-next::{query_id}", on_click=move_rank, args=(1,),
                           use_container_width=True)
        controls[2].number_input("Hạng", min_value=1, max_value=len(ranked), step=1, key=rk_key,
                                 label_visibility="collapsed")
        controls[3].button("⏭ Noise kế tiếp", key=f"rk-noise::{query_id}", on_click=jump_rank, args=("noise",),
                           use_container_width=True)
        controls[4].button("⏭ Gold kế tiếp", key=f"rk-gold::{query_id}", on_click=jump_rank, args=("gold",),
                           use_container_width=True)

        current_rank = st.session_state[rk_key]
        page_id = ranked[current_rank - 1]
        doc_id, page_index = split_page_id(page_id)
        show_page(
            page_id,
            f"**Hạng #{current_rank}** · {statuses[current_rank - 1]} · {doc_label(doc_id)} · trang {page_index + 1}",
        )

        if st.toggle(f"Thư viện ảnh top-{top_n} (đối chiếu nhanh)", key="gallery"):
            columns_per_row = 4
            for start in range(0, len(ranked), columns_per_row):
                cols = st.columns(columns_per_row)
                for offset, col in enumerate(cols):
                    if start + offset >= len(ranked):
                        break
                    rank_number = start + offset + 1
                    gallery_id = ranked[start + offset]
                    gallery_doc, gallery_index = split_page_id(gallery_id)
                    gallery_pdf = doc_pdf(gallery_doc)
                    with col:
                        image = render_page(str(gallery_pdf), gallery_index, 0.55) if gallery_pdf else None
                        caption = f"#{rank_number} {statuses[start + offset].split(' ')[0]} {doc_label(gallery_doc)[:28]} p.{gallery_index + 1}"
                        if image:
                            st.image(image, caption=caption, width="stretch")
                        else:
                            st.caption(caption + " (không có ảnh)")

st.divider()
st.caption(
    "Lưu ý: gold files/pages là benchmark qrels để phục vụ phân tích evidence. "
    "Không dùng thông tin này làm input cho query preparation hoặc retrieval."
)
