"""Interactive demo UI for the AXIOM document-processing pipeline."""

from __future__ import annotations

from datetime import datetime
import html
from importlib.util import find_spec
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import uuid

import streamlit as st

from src.demo.data import (
    PROJECT_ROOT,
    RunOverview,
    build_run_zip,
    discover_runs,
    extract_document_view,
    find_source_file,
    format_bytes,
    list_documents,
    load_document,
    load_stage_metadata,
)
from src.pipeline import run_pipeline
from src.reading_order import (
    citation_ids,
    component_path_parts,
    source_blocks_from_extraction,
)
from src.utils.config import load_config
from src.utils.env import load_dotenv_file


st.set_page_config(
    page_title="AXIOM · Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUPPORTED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg"]


def main() -> None:
    load_dotenv_file(PROJECT_ROOT)
    _inject_styles()
    next_page = st.session_state.pop("_next_page", None)
    if next_page:
        st.session_state["page"] = next_page
    page = _sidebar()
    runs = discover_runs()
    if page == "Chạy pipeline":
        _render_pipeline_runner()
    else:
        _render_run_explorer(runs)


def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-mark">AX</div>
            <div class="brand-title">AXIOM</div>
            <div class="brand-subtitle">DOCUMENT INTELLIGENCE</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sidebar-rule'></div>", unsafe_allow_html=True)
        page = st.radio(
            "Điều hướng",
            ["Chạy pipeline", "Khám phá kết quả"],
            label_visibility="collapsed",
            key="page",
        )
        st.markdown("<div class='sidebar-spacer'></div>", unsafe_allow_html=True)
        st.caption("AXIOM_DE-RD · Demo console")
    return page


def _render_pipeline_runner() -> None:
    _hero(
        "Chạy pipeline",
        "Upload PDF hoặc ảnh để parse và tạo output artifact.",
        "NEW RUN",
    )
    parser_key_ready = bool(os.getenv("DATALAB_API_KEY"))
    parser_sdk_ready = find_spec("datalab_sdk") is not None
    if not parser_sdk_ready:
        st.error("Thiếu `datalab-python-sdk`. Chạy `pip install -e .` rồi khởi động lại Streamlit.")
    elif not parser_key_ready:
        st.warning("Thêm `DATALAB_API_KEY` vào file `.env` để chạy parser.")

    uploads = st.file_uploader(
        "Chọn PDF, PNG hoặc JPEG",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        help="Các file được chọn sẽ được xử lý chung trong một pipeline run.",
    )
    if uploads:
        for item in uploads:
            cols = st.columns([5, 1])
            cols[0].write(item.name)
            cols[1].caption(format_bytes(item.size))

    st.caption("Datalab Lift parser · Local hash embedding")
    run_clicked = st.button(
        "Run pipeline",
        type="primary",
        width="stretch",
        disabled=not uploads or not parser_key_ready or not parser_sdk_ready,
    )
    if run_clicked and uploads:
        _execute_upload_run(uploads)


def _execute_upload_run(uploads: list) -> None:
    upload_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    upload_root = PROJECT_ROOT / "data" / "raw" / "demo_uploads" / upload_id
    upload_root.mkdir(parents=True, exist_ok=False)
    used_names: set[str] = set()
    for uploaded in uploads:
        file_name = _unique_file_name(_safe_upload_name(uploaded.name), used_names)
        (upload_root / file_name).write_bytes(uploaded.getvalue())

    progress = st.progress(8, text="Đã lưu tài liệu upload")
    config_path: Path | None = None
    try:
        config_path = _demo_config()
        progress.progress(18, text="Đang parse và xử lý tài liệu…")
        with st.spinner("Pipeline đang chạy. Với PDF lớn, bước parser có thể mất vài phút."):
            state = run_pipeline(config_path, local_raw=upload_root)
        progress.progress(100, text="Pipeline hoàn tất")
        st.session_state["selected_run"] = state.run_id
        st.session_state["run_notice"] = (
            f"Run `{state.run_id}` hoàn tất với {len(state.errors)} lỗi/quarantine."
            if state.errors
            else f"Run `{state.run_id}` hoàn tất với {len(state.data_objects)} tài liệu."
        )
        st.session_state["_next_page"] = "Khám phá kết quả"
        st.rerun()
    except Exception as exc:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            logging.getLogger(__name__).warning(
                "Streamlit pipeline stopped by provider HTTP %s: %s",
                status_code,
                exc,
            )
        else:
            logging.getLogger(__name__).exception("Streamlit pipeline run failed")
        progress.empty()
        st.error(f"Pipeline chưa thể hoàn tất: {_pipeline_error_message(exc)}")
    finally:
        if config_path is not None:
            config_path.unlink(missing_ok=True)


def _demo_config() -> Path:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    embeddings = config.setdefault("indexing", {}).setdefault("embeddings", {})
    embeddings.update(
        {
            "enabled": True,
            "provider": "local_hash",
            "model": "local-hash-embedding-v1",
            "dimension": 128,
            "fail_on_error": True,
        }
    )

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="axiom-streamlit-",
        encoding="utf-8",
        delete=False,
    )
    with handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    return Path(handle.name)


def _pipeline_error_message(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    messages = {
        401: "Datalab API key không hợp lệ hoặc đã hết hiệu lực.",
        402: (
            "Datalab API trả về 402 Payment Required. Tài khoản/API key hiện không "
            "còn credit; hãy nạp credit trên Datalab rồi chạy lại pipeline."
        ),
        403: "Datalab API key không có quyền thực hiện yêu cầu này.",
        429: "Datalab API đang giới hạn tần suất yêu cầu; hãy đợi rồi chạy lại.",
    }
    return messages.get(status_code, str(error))


def _render_run_explorer(runs: list[RunOverview]) -> None:
    _hero(
        "Khám phá kết quả",
        "Chọn một pipeline run và xem nội dung đã trích xuất theo từng tài liệu.",
        "RESULTS",
    )
    notice = st.session_state.pop("run_notice", None)
    if notice:
        st.success(notice)
    if not runs:
        st.info("Chưa có artifact nào để hiển thị.")
        return

    run_ids = [run.run_id for run in runs]
    selected_from_state = st.session_state.get("selected_run")
    default_index = run_ids.index(selected_from_state) if selected_from_state in run_ids else 0
    selected_run_id = st.selectbox(
        "Pipeline run",
        run_ids,
        index=default_index,
        format_func=lambda value: _run_option(next(item for item in runs if item.run_id == value)),
    )
    selected_run = next(item for item in runs if item.run_id == selected_run_id)
    st.session_state["selected_run"] = selected_run_id

    run_info, download = st.columns([3, 1])
    with run_info:
        st.caption(
            f"{selected_run.document_count} tài liệu · {selected_run.status.replace('_', ' ')}"
        )
    with download:
        st.download_button(
            "Download run artifacts (.zip)",
            data=build_run_zip(selected_run_id),
            file_name=f"axiom-{selected_run_id}.zip",
            mime="application/zip",
            width="stretch",
        )
    documents = list_documents(selected_run_id)
    if not documents:
        st.warning("Run này chưa có document artifact.")
        return

    search = st.text_input("Tìm tài liệu", placeholder="Tên file, title hoặc document ID…")
    if search:
        needle = search.casefold()
        documents = [
            item
            for item in documents
            if needle in " ".join((item.file_name, item.title or "", item.document_id)).casefold()
        ]
    if not documents:
        st.info("Không tìm thấy tài liệu phù hợp.")
        return

    selected_document_id = st.selectbox(
        "Document",
        [item.document_id for item in documents],
        format_func=lambda value: _document_option(next(item for item in documents if item.document_id == value)),
    )
    document = next(item for item in documents if item.document_id == selected_document_id)
    payload = load_document(selected_run_id, selected_document_id)
    view = extract_document_view(payload)
    _render_document(selected_run, document, payload, view)


def _render_document(run: RunOverview, document, payload: dict, view: dict) -> None:
    identity = view["identity"]
    ingested_payload = load_document(
        run.run_id,
        document.document_id,
        stage="ingested",
    )
    parsed_payload = ingested_payload.get("parsed") if ingested_payload else None
    parsing_view = extract_document_view(ingested_payload) if ingested_payload else view
    output_content = view["content"]
    content = output_content if output_content else parsing_view["content"]
    retrieval = view["retrieval"]
    items = retrieval.get("items", []) if isinstance(retrieval.get("items", []), list) else []
    main_text = str(content.get("main_text") or "")
    tables = content.get("tables", []) if isinstance(content.get("tables"), list) else []
    figures = content.get("figures", []) if isinstance(content.get("figures"), list) else []
    source_file = find_source_file(run.run_id, document)

    st.markdown("---")
    title_col, badge_col = st.columns([5, 1])
    title_col.markdown(f"## {identity.get('title') or document.file_name}")
    title_col.caption(f"{document.file_name}  ·  `{document.document_id}`")
    badge_col.markdown(_status_badge(document.status), unsafe_allow_html=True)

    overview_tab, content_tab, chunking_tab, lineage_tab, json_tab = st.tabs(
        ["Overview", "Content", "Chunking & Embedding", "Lineage", "Raw JSON"]
    )
    with overview_tab:
        info = {
            "Document type": identity.get("document_type") or "—",
            "Language": identity.get("language") or "—",
            "Content type": document.content_type or "—",
            "File size": format_bytes(document.size_bytes),
            "Contract": view["contract_version"],
            "Source": document.source_uri or "—",
        }
        info_columns = st.columns(2)
        for index, (label, value) in enumerate(info.items()):
            with info_columns[index % 2]:
                st.caption(label.upper())
                st.write(value)
        metric_cols = st.columns(4)
        _mini_metric(metric_cols[0], "Characters", f"{len(main_text):,}")
        _mini_metric(metric_cols[1], "Tables", len(tables))
        _mini_metric(metric_cols[2], "Figures", len(figures))
        _mini_metric(metric_cols[3], "Chunks", len(items))

    with content_tab:
        source_column, parsed_column = st.columns([1, 1.1], gap="large")
        with source_column:
            st.markdown("#### Tài liệu gốc")
            _render_source_preview(source_file, document)
        with parsed_column:
            st.markdown("#### Nội dung parsing")
            has_final_reading_order = bool(
                output_content.get("blocks") and output_content.get("reading_order")
            )
            if has_final_reading_order:
                st.caption(f"data/output/{run.run_id}/documents/{document.document_id}.json")
                with st.container(height=650, border=True):
                    _render_output_content(output_content)
            elif parsed_payload is not None:
                st.caption(f"data/ingested/{run.run_id}/documents/{document.document_id}.json")
                with st.container(height=650, border=True):
                    _render_parsed_content(parsed_payload)
            else:
                st.info("Artifact không có reading order hoặc payload parsing khả dụng.")

    with chunking_tab:
        if not items:
            st.info("Không có chunk hoặc embedding trong artifact này.")
        else:
            type_counts: dict[str, int] = {}
            for item in items:
                item_type = str(item.get("type") or item.get("index_type") or "unknown")
                type_counts[item_type] = type_counts.get(item_type, 0) + 1
            cols = st.columns(max(1, min(len(type_counts), 4)))
            for column, (item_type, count) in zip(cols, sorted(type_counts.items())):
                _mini_metric(column, item_type.replace("_", " ").title(), count)
            for index, item in enumerate(items, 1):
                item_type = item.get("type") or item.get("index_type") or "item"
                with st.expander(f"{index:02d} · {str(item_type).upper()}"):
                    content_value = item.get("content")
                    if isinstance(content_value, dict) and content_value.get("text"):
                        st.write(content_value["text"])
                    st.json(item, expanded=False)

    with lineage_tab:
        lineage = view.get("lineage", {})
        if lineage:
            st.json(lineage, expanded=True)
        else:
            stage_data = {
                stage: load_stage_metadata(run.run_id, stage)
                for stage in run.available_stages
            }
            st.json({"run_id": run.run_id, "stages": list(stage_data)}, expanded=True)

    with json_tab:
        st.download_button(
            "Download document JSON",
            data=json.dumps(payload, ensure_ascii=False, indent=2),
            file_name=f"{document.document_id}.json",
            mime="application/json",
        )
        st.json(payload, expanded=False)


def _render_source_preview(source_file: Path | None, document) -> None:
    """Render the persisted raw input alongside its parsed representation."""
    if source_file is None:
        st.markdown(
            "<div class='document-placeholder'><span>DOC</span><b>Không tìm thấy file gốc</b>"
            "<small>Artifact có thể đến từ một run cũ hoặc source ngoài hệ thống</small></div>",
            unsafe_allow_html=True,
        )
        return

    suffix = source_file.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        st.image(str(source_file), caption=source_file.name, width="stretch")
    elif suffix == ".pdf":
        st.pdf(source_file.read_bytes(), height=650)
    else:
        st.markdown(
            "<div class='document-placeholder'><span>DOC</span><b>Không hỗ trợ preview</b>"
            "<small>Hãy tải file gốc để xem nội dung</small></div>",
            unsafe_allow_html=True,
        )

    st.download_button(
        "Tải file gốc",
        data=source_file.read_bytes(),
        file_name=source_file.name,
        mime=document.content_type or "application/octet-stream",
        key=f"source-{document.document_id}",
    )


def _render_parsed_content(parsed_payload: object) -> None:
    """Render semantic extraction fields while preserving parser reading order."""
    if not isinstance(parsed_payload, dict):
        st.info("Payload parsing không có content khả dụng.")
        return

    rows = parsed_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        text = parsed_payload.get("text")
        if text:
            _parsed_text_block(str(text))
        else:
            st.info("Payload parsing không có content khả dụng.")
        return

    rendered = False
    for row_index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        extraction = row.get("extraction")
        if not isinstance(extraction, dict):
            text = row.get("text")
            if text:
                _parsed_text_block(str(text))
                rendered = True
            continue

        if len(rows) > 1:
            st.caption(f"PARSED ROW {row_index}")
        title = extraction.get("title")
        if title:
            st.markdown(f"### {html.escape(str(title))}", unsafe_allow_html=True)

        metadata = [
            str(value)
            for value in (extraction.get("document_type"), extraction.get("language"))
            if value
        ]
        if metadata:
            st.caption(" · ".join(metadata))

        citations = _citation_ids(extraction.get("main_text_citations"))
        main_text = extraction.get("main_text") or row.get("text")
        reading_order_is_mapped = False
        if main_text:
            reading_order_is_mapped = _render_reading_order_blocks(
                str(main_text),
                citations,
                extraction,
            )
            if not reading_order_is_mapped:
                _parsed_text_block(str(main_text))
            rendered = True

        _render_parsed_collection("Tables", extraction.get("tables"))
        _render_parsed_collection("Figures", extraction.get("figures"))
        _render_parsed_collection("Formulas", extraction.get("formulas"))

        if citations and not reading_order_is_mapped:
            with st.expander(f"Source citations chưa ánh xạ · {len(citations)} blocks"):
                st.caption(
                    "Artifact này không chứa đủ dữ liệu block-level để gắn chính xác "
                    "từng component ID vào từng đoạn văn."
                )
                ordered_blocks = "\n".join(
                    f"{index:02d}  {citation}"
                    for index, citation in enumerate(citations, 1)
                )
                st.code(ordered_blocks, language=None)

    if not rendered:
        st.info("Payload parsing không có main text khả dụng.")


def _render_output_content(content: dict) -> None:
    """Render the final output contract without reconstructing component order."""
    blocks = content.get("blocks")
    reading_order = content.get("reading_order")
    metadata = content.get("reading_order_meta")
    if not isinstance(blocks, list) or not isinstance(reading_order, list):
        _parsed_text_block(str(content.get("main_text") or ""))
        return

    block_by_id = {
        block.get("component_id"): block
        for block in blocks
        if isinstance(block, dict) and isinstance(block.get("component_id"), str)
    }
    ordered_blocks = [
        block_by_id[component_id]
        for component_id in reading_order
        if isinstance(component_id, str) and component_id in block_by_id
    ]
    meta = metadata if isinstance(metadata, dict) else {}
    source = str(meta.get("source") or "unknown")
    complete = bool(meta.get("complete"))
    status = "COMPLETE" if complete else "RECONSTRUCTED"
    st.caption(
        f"READING ORDER · {len(ordered_blocks)} SOURCE BLOCKS · {source.upper()} · {status}"
    )
    _render_source_blocks(ordered_blocks)

    _render_parsed_collection("Tables", content.get("tables"))
    _render_parsed_collection("Figures", content.get("figures"))
    _render_parsed_collection("Formulas", content.get("formulas"))


def _parsed_text_block(text: str) -> None:
    escaped = html.escape(text)
    st.markdown(
        f"<article class='parsed-article'>{escaped}</article>",
        unsafe_allow_html=True,
    )


def _citation_ids(value: object) -> list[str]:
    """Return non-empty parser component IDs without inventing replacements."""
    return citation_ids(value)


def _reading_order_blocks(text: str, citations: list[str]) -> list[tuple[str, str]]:
    """Map paragraphs to citations only when the artifact provides a 1:1 order."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", text.strip())
        if paragraph.strip()
    ]
    if not paragraphs or len(paragraphs) != len(citations):
        return []
    return list(zip(citations, paragraphs))


def _component_path_parts(component_id: str) -> tuple[int, int, str] | None:
    return component_path_parts(component_id)


def _all_reading_order_blocks(
    text: str,
    citations: list[str],
    extraction: dict,
) -> list[tuple[str, str]]:
    """Merge all visible source components into the parser's document order."""
    enriched_extraction = {
        **extraction,
        "main_text": text,
        "main_text_citations": citations,
    }
    return [
        (block["component_id"], str(block.get("text") or ""))
        for block in source_blocks_from_extraction(enriched_extraction)
    ]


def _render_reading_order_blocks(
    text: str,
    citations: list[str],
    extraction: dict,
) -> bool:
    blocks = _all_reading_order_blocks(text, citations, extraction)
    if not blocks:
        return False

    st.caption(f"READING ORDER · {len(blocks)} SOURCE BLOCKS")
    _render_source_blocks(
        [
            {
                "component_id": component_id,
                "text": block_text,
                "type": (
                    _component_path_parts(component_id)[2]
                    if _component_path_parts(component_id)
                    else "Block"
                ),
            }
            for component_id, block_text in blocks
        ]
    )
    return True


def _render_source_blocks(blocks: list[dict]) -> None:
    for index, block in enumerate(blocks, 1):
        component_id = str(block.get("component_id") or "")
        block_text = str(block.get("text") or block.get("semantic_text") or "")
        escaped_id = html.escape(component_id, quote=True)
        escaped_text = html.escape(block_text or "[Không có biểu diễn text cho block này]")
        path_parts = _component_path_parts(component_id)
        block_type = str(block.get("type") or (path_parts[2] if path_parts else "Block"))
        st.markdown(
            (
                f"<section id='{escaped_id}' class='parsed-block' "
                f"data-component-id='{escaped_id}'>"
                "<header class='parsed-block-meta'>"
                f"<span>{index:02d}</span>"
                f"<b>{html.escape(block_type.upper())}</b>"
                f"<code>{escaped_id}</code>"
                "</header>"
                f"<div class='parsed-block-text'>{escaped_text}</div>"
                "</section>"
            ),
            unsafe_allow_html=True,
        )


def _render_parsed_collection(label: str, value: object) -> None:
    if value is None:
        return
    items = value if isinstance(value, list) else [value]
    if not items:
        return
    with st.expander(f"{label} · {len(items)}"):
        for index, item in enumerate(items, 1):
            if len(items) > 1:
                st.caption(f"{label[:-1]} {index}")
            if isinstance(item, dict):
                caption = item.get("caption")
                content = item.get("content") or item.get("description") or item.get("text")
                if caption:
                    st.markdown(f"**{caption}**")
                if content:
                    st.markdown(str(content))
                elif not caption:
                    st.write(item)
            else:
                st.code(str(item), language=None)


def _mini_metric(container, label: str, value) -> None:
    container.metric(label, value)


def _hero(title: str, subtitle: str, eyebrow: str) -> None:
    st.markdown(
        f"<section class='hero'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></section>",
        unsafe_allow_html=True,
    )


def _status_badge(status: str) -> str:
    normalized = status.casefold()
    css_class = "warning" if "error" in normalized or "quarant" in normalized else "success"
    label = status.replace("_", " ").upper()
    return f"<span class='status-badge {css_class}'>{label}</span>"


def _date_label(run: RunOverview | None) -> str:
    if not run or not run.created_at:
        return "No runs yet"
    return run.created_at[:10]


def _run_option(run: RunOverview) -> str:
    return f"{run.run_id}  ·  {run.document_count} docs  ·  {_date_label(run)}"


def _document_option(document) -> str:
    title = f" — {document.title}" if document.title and document.title != document.file_name else ""
    return f"{document.file_name}{title}  [{document.status}]"


def _unique_file_name(name: str, used: set[str]) -> str:
    safe_name = name or "document"
    candidate = safe_name
    counter = 2
    while candidate.casefold() in used:
        path = Path(safe_name)
        candidate = f"{path.stem}-{counter}{path.suffix}"
        counter += 1
    used.add(candidate.casefold())
    return candidate


def _safe_upload_name(name: str) -> str:
    """Strip browser-supplied path components from an uploaded file name."""
    safe_name = Path(str(name).replace("\\", "/")).name.strip()
    if safe_name in {"", ".", ".."}:
        return "document"
    return safe_name


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700&display=swap');
        :root { --ink:#17211c; --muted:#66706a; --paper:#f5f4ee; --acid:#d8ff3e; --green:#184d39; }
        html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
        .stApp { background:var(--paper); color:var(--ink); }
        [data-testid="stSidebar"] { background:#17211c; }
        [data-testid="stSidebar"] * { color:#eef1e9; }
        [data-testid="stSidebar"] [role="radiogroup"] label { padding:.48rem .7rem; border-radius:8px; }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#28342e; }
        [data-testid="stSidebar"] hr { border-color:#344139; }
        .block-container { max-width:1440px; padding-top:2.2rem; padding-bottom:4rem; }
        .brand-mark { width:42px; height:42px; display:grid; place-items:center; background:var(--acid); color:#17211c!important; font-family:'DM Mono'; font-weight:500; border-radius:3px; margin-bottom:.8rem; }
        .brand-title { font-size:1.25rem; font-weight:700; letter-spacing:.14em; }
        .brand-subtitle { font-family:'DM Mono'; font-size:.66rem; color:#98a49d!important; letter-spacing:.14em; }
        .sidebar-rule { height:1px; background:#344139; margin:1.5rem 0; }
        .sidebar-spacer { height:12rem; }
        .hero { padding:.6rem 0 1.25rem; border-bottom:1px solid #d8d8d0; margin-bottom:1.6rem; }
        .hero .eyebrow { color:#466452; font-family:'DM Mono'; font-size:.72rem; letter-spacing:.12em; margin-bottom:.7rem; }
        .hero h1 { max-width:900px; font-size:clamp(2rem,3vw,3rem); line-height:1.08; letter-spacing:-.04em; margin:0; color:var(--ink); }
        .hero p { max-width:760px; color:var(--muted); font-size:1.03rem; margin:.95rem 0 0; line-height:1.65; }
        .status-badge { display:inline-block; font-family:'DM Mono'; font-size:.64rem; letter-spacing:.05em; padding:.35rem .55rem; border-radius:2px; background:#e4eadf; color:#24513d; }
        .status-badge.warning { background:#fff0d2; color:#8a5616; }
        .document-placeholder { min-height:330px; background:#e8e8e0; display:flex; flex-direction:column; align-items:center; justify-content:center; border:1px solid #d3d4cb; }
        .document-placeholder span { width:72px; height:88px; display:grid; place-items:center; background:#faf9f4; border:1px solid #c9cbc1; font-family:'DM Mono'; color:#788078; }
        .document-placeholder b { margin-top:1rem; }
        .document-placeholder small { color:#747c77; margin-top:.3rem; }
        .parsed-article { white-space:pre-wrap; line-height:1.72; color:#243029; font-size:.94rem; }
        .parsed-block { margin:0 0 .8rem; border:1px solid #d9ddd4; background:#faf9f4; border-radius:4px; overflow:hidden; }
        .parsed-block-meta { display:flex; align-items:center; gap:.65rem; padding:.42rem .65rem; border-bottom:1px solid #e2e4dc; background:#f0f1e9; }
        .parsed-block-meta span { min-width:1.7rem; color:#587062; font-family:'DM Mono'; font-size:.68rem; }
        .parsed-block-meta b { color:#476152; font-family:'DM Mono'; font-size:.62rem; font-weight:500; letter-spacing:.05em; }
        .parsed-block-meta code { overflow-wrap:anywhere; color:#315442; background:transparent; font-size:.72rem; }
        .parsed-block-text { padding:.72rem .8rem .82rem; white-space:pre-wrap; line-height:1.68; color:#243029; font-size:.94rem; }
        .stButton>button[kind="primary"], .stDownloadButton>button { border-radius:3px; }
        .stButton>button[kind="primary"] { background:#173f31; border-color:#173f31; }
        .stButton>button[kind="primary"]:hover { background:#235844; border-color:#235844; }
        [data-testid="stMetric"] { background:#faf9f4; border:1px solid #dedfd6; padding:.8rem; }
        @media(max-width:800px) { .hero h1 { font-size:2.2rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
