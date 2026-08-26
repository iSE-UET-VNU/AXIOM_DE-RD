"""Interactive demo UI for the AXIOM document-processing pipeline."""

from __future__ import annotations

import base64
from datetime import datetime
import html
from html.parser import HTMLParser
from importlib.util import find_spec
import json
import logging
import math
import mimetypes
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
    persist_dataeng_response,
)
from src.dispatcher import dispatch_dataeng_inputs
from src.ingestion.parsing.lift.client import SUPPORTED_EXTENSIONS as LIFT_SUPPORTED_EXTENSIONS
from src.reading_order import (
    citation_ids,
    component_path_parts,
    source_blocks_from_extraction,
)
from src.table_agent.client import DEFAULT_WORKBOOK_EXTENSIONS
from src.utils.config import load_config
from src.utils.env import load_dotenv_file


st.set_page_config(
    page_title="AXIOM · Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PIPELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def _normalized_extensions(value: object, *, fallback: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Read config extensions in the same normalized form used by the router."""
    if not isinstance(value, list):
        return fallback
    extensions: list[str] = []
    for item in value:
        extension = str(item).strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        if extension not in extensions:
            extensions.append(extension)
    return tuple(extensions) or fallback


_PIPELINE_CONFIG = load_config(PIPELINE_CONFIG_PATH)
_LOCAL_INPUT_CONFIG = _PIPELINE_CONFIG.get("local_input", {})
_TABLE_AGENT_CONFIG = _PIPELINE_CONFIG.get("table_agent", {})
CONFIGURED_INPUT_EXTENSIONS = _normalized_extensions(
    _LOCAL_INPUT_CONFIG.get("include_extensions")
)
CONFIGURED_WORKBOOK_EXTENSIONS = _normalized_extensions(
    _TABLE_AGENT_CONFIG.get("supported_extensions"),
    fallback=DEFAULT_WORKBOOK_EXTENSIONS,
)
CONFIGURED_UPLOAD_EXTENSIONS = _normalized_extensions(
    [*CONFIGURED_INPUT_EXTENSIONS, *CONFIGURED_WORKBOOK_EXTENSIONS]
)
SUPPORTED_UPLOAD_TYPES = [extension.removeprefix(".") for extension in CONFIGURED_UPLOAD_EXTENSIONS]
DISPLAY_TITLE_OVERRIDES = {
    "5.4. Data size analysis (RQ4)": "Analysis",
}
RUN_PAGE = "Run pipeline"
RESULTS_PAGE = "Explore results"
LEGACY_PAGE_NAMES = {
    "Chạy pipeline": RUN_PAGE,
    "Khám phá kết quả": RESULTS_PAGE,
    "Pipeline overview": RUN_PAGE,
}


class _OutputHTMLSanitizer(HTMLParser):
    """Keep parser content markup while dropping executable or styling markup."""

    ALLOWED_TAGS = {
        "a",
        "b",
        "blockquote",
        "br",
        "caption",
        "code",
        "div",
        "em",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "i",
        "li",
        "math",
        "mi",
        "mn",
        "mo",
        "mrow",
        "msub",
        "msup",
        "ol",
        "p",
        "pre",
        "span",
        "strong",
        "sub",
        "sup",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
    VOID_TAGS = {"br"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.casefold()
        if normalized not in self.ALLOWED_TAGS:
            return
        safe_attrs = ""
        if normalized in {"td", "th"}:
            spans = []
            for name, value in attrs:
                if name.casefold() in {"colspan", "rowspan"} and value and value.isdigit():
                    spans.append(f" {name.casefold()}='{html.escape(value, quote=True)}'")
            safe_attrs = "".join(spans)
        self.parts.append(f"<{normalized}{safe_attrs}>")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in self.ALLOWED_TAGS and normalized not in self.VOID_TAGS:
            self.parts.append(f"</{normalized}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        self.parts.append(html.escape(data))


def main() -> None:
    load_dotenv_file(PROJECT_ROOT)
    _inject_styles()
    next_page = st.session_state.pop("_next_page", None)
    if next_page:
        st.session_state["page"] = LEGACY_PAGE_NAMES.get(next_page, next_page)
    current_page = st.session_state.get("page")
    if current_page in LEGACY_PAGE_NAMES:
        st.session_state["page"] = LEGACY_PAGE_NAMES[current_page]
    page = _sidebar()
    runs = discover_runs()
    if page == RUN_PAGE:
        _render_pipeline_runner()
    else:
        _render_run_explorer(runs)


def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-title">AXIOM</div>
            <div class="brand-subtitle">DOCUMENT INTELLIGENCE</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sidebar-rule'></div>", unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            [RUN_PAGE, RESULTS_PAGE],
            label_visibility="collapsed",
            key="page",
        )
        st.markdown("<div class='sidebar-spacer'></div>", unsafe_allow_html=True)
        st.caption("AXIOM_DE-RD · Demo console")
    return page


def _render_pipeline_runner() -> None:
    _hero(
        "Run pipeline",
        "Upload documents or Excel workbooks and route them to the right processor.",
        "NEW RUN",
    )
    parser_key_ready = bool(os.getenv("DATALAB_API_KEY"))
    parser_sdk_ready = find_spec("datalab_sdk") is not None

    uploads = st.file_uploader(
        "Choose files enabled in configs/pipeline.yaml",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        help=(
            "The upload types come from local_input.include_extensions and "
            "table_agent.supported_extensions. Excel workbooks are sent to "
            "TableAgent; other configured files continue through the normal "
            "AXIOM pipeline."
        ),
    )
    if uploads:
        for item in uploads:
            cols = st.columns([5, 1])
            cols[0].write(item.name)
            cols[1].caption(format_bytes(item.size))

    needs_lift_parser = any(
        Path(item.name).suffix.lower() in LIFT_SUPPORTED_EXTENSIONS
        for item in uploads or []
    )
    if needs_lift_parser and not parser_sdk_ready:
        st.error("Missing `datalab-python-sdk`. Run `pip install -e .`, then restart Streamlit.")
    elif needs_lift_parser and not parser_key_ready:
        st.warning("Add `DATALAB_API_KEY` to `.env` to process PDF or image files.")

    st.caption(
        "Configured types: "
        + ", ".join(SUPPORTED_UPLOAD_TYPES)
        + " · Excel → TableAgent · PDF/images → Datalab Lift"
    )
    run_clicked = st.button(
        "Run pipeline",
        type="primary",
        width="stretch",
        disabled=(
            not uploads
            or (
                needs_lift_parser
                and (not parser_key_ready or not parser_sdk_ready)
            )
        ),
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
        upload_path = upload_root / file_name
        upload_path.write_bytes(uploaded.getvalue())

    progress = st.progress(8, text="Uploaded files saved")
    config_path: Path | None = None
    try:
        config_path = _demo_config()
        progress.progress(18, text="Routing files to the appropriate processor…")
        with st.spinner("Processing files. Large documents may take several minutes."):
            dispatch_result = dispatch_dataeng_inputs(
                config_path=config_path,
                local_raw=upload_root,
            )
        progress.progress(100, text="Pipeline completed")
        state = dispatch_result.pipeline_state
        if dispatch_result.table_document_count:
            run_id = persist_dataeng_response(dispatch_result.response)
            st.session_state["selected_run"] = run_id
            st.session_state["run_notice"] = (
                f"Run `{run_id}` completed with "
                f"{dispatch_result.table_document_count} workbook(s) "
                "processed by TableAgent."
            )
            st.session_state["_next_page"] = RESULTS_PAGE
            st.rerun()

        if state is None:
            raise RuntimeError("The document pipeline did not return a run state.")
        st.session_state["selected_run"] = state.run_id
        st.session_state["run_notice"] = (
            f"Run `{state.run_id}` completed with {len(state.errors)} errors/quarantined documents."
            if state.errors
            else f"Run `{state.run_id}` completed with {len(state.data_objects)} documents."
        )
        st.session_state["_next_page"] = RESULTS_PAGE
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
        st.error(f"Pipeline could not complete: {_pipeline_error_message(exc)}")
    finally:
        if config_path is not None:
            config_path.unlink(missing_ok=True)


def _demo_config() -> Path:
    config = load_config(PROJECT_ROOT / "configs" / "pipeline.yaml")
    chunking_embedding = config.setdefault("chunking_embedding", {})
    chunking_embedding["embedder"] = "local_hash"
    chunking_embedding["embedder_params"] = {
        "model": "local-hash-embedding-v1",
        "dimension": 128,
    }

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
        401: "The Datalab API key is invalid or expired.",
        402: (
            "Datalab returned 402 Payment Required. The account or API key has no "
            "remaining credits; add Datalab credits and run the pipeline again."
        ),
        403: "The Datalab API key is not authorized to perform this request.",
        429: "Datalab is rate-limiting requests; wait briefly and try again.",
    }
    return messages.get(status_code, str(error))


def _render_run_explorer(runs: list[RunOverview]) -> None:
    _hero(
        "Explore results",
        "Select a pipeline run and inspect the extracted content for each document.",
        "RESULTS",
    )
    notice = st.session_state.pop("run_notice", None)
    if notice:
        st.success(notice)
    if not runs:
        st.info("No artifacts are available yet.")
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
            f"{selected_run.document_count} documents · {selected_run.status.replace('_', ' ')}"
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
        st.warning("This run has no document artifacts.")
        return

    search = st.text_input("Search documents", placeholder="File name, title, or document ID…")
    if search:
        needle = search.casefold()
        documents = [
            item
            for item in documents
            if needle
            in " ".join(
                (
                    item.file_name,
                    item.title or "",
                    _display_title(item.title),
                    item.document_id,
                )
            ).casefold()
        ]
    if not documents:
        st.info("No matching documents found.")
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
    title_col.markdown(f"## {_display_title(identity.get('title')) or document.file_name}")
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
        has_final_reading_order = bool(
            output_content.get("blocks") and output_content.get("reading_order")
        )
        display_mode = "Rendered"
        if has_final_reading_order:
            display_mode = st.segmented_control(
                "Content display",
                ["Rendered", "Raw"],
                default="Rendered",
                key=f"content-display-{run.run_id}-{document.document_id}",
                help="Rendered displays block HTML. Raw shows every content field stored in output JSON.",
            ) or "Rendered"
        interactive_rendered = False
        if has_final_reading_order:
            interactive_rendered = _render_interactive_content(
                source_file,
                document,
                output_content,
                raw=display_mode == "Raw",
            )

        if not interactive_rendered:
            source_column, parsed_column = st.columns([1, 1.1], gap="large")
            with source_column:
                st.markdown("#### Source document")
                _render_source_preview(source_file, document)
            with parsed_column:
                st.markdown("#### Parsed content")
                if has_final_reading_order:
                    st.caption(
                        f"data/output/{run.run_id}/documents/{document.document_id}.json"
                    )
                    with st.container(height=650, border=True):
                        _render_output_content(
                            output_content,
                            raw=display_mode == "Raw",
                        )
                elif parsed_payload is not None:
                    st.caption(
                        f"data/ingested/{run.run_id}/documents/{document.document_id}.json"
                    )
                    with st.container(height=650, border=True):
                        _render_parsed_content(parsed_payload)
                else:
                    st.info("This artifact has no reading order or usable parsing payload.")

    with chunking_tab:
        if not items:
            st.info("This artifact has no chunks or embeddings.")
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
            "<div class='document-placeholder'><span>DOC</span><b>Source file not found</b>"
            "<small>The artifact may come from an older run or an external source</small></div>",
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
            "<div class='document-placeholder'><span>DOC</span><b>Preview not supported</b>"
            "<small>Download the source file to view its content</small></div>",
            unsafe_allow_html=True,
        )

    st.download_button(
        "Download source file",
        data=source_file.read_bytes(),
        file_name=source_file.name,
        mime=document.content_type or "application/octet-stream",
        key=f"source-{document.document_id}",
    )


def _render_interactive_content(
    source_file: Path | None,
    document,
    content: dict,
    *,
    raw: bool = False,
) -> bool:
    """Render synchronized source boxes and reading-order blocks for image inputs."""
    component_html = _build_content_inspector_html(source_file, content, raw=raw)
    if component_html is None:
        return False

    st.caption(
        "Click a box on the source image to highlight its parsed block. "
        "You can also select a parsed block to locate it on the image."
    )
    st.iframe(component_html, height=780, tab_index=0)
    st.download_button(
        "Download source file",
        data=source_file.read_bytes(),
        file_name=source_file.name,
        mime=document.content_type or "application/octet-stream",
        key=f"interactive-source-{document.document_id}",
    )
    return True


def _build_content_inspector_html(
    source_file: Path | None,
    content: dict,
    *,
    raw: bool = False,
) -> str | None:
    """Build a self-contained image/block inspector, or return None for fallback."""
    if source_file is None or not source_file.is_file():
        return None
    if source_file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        return None

    blocks = content.get("blocks")
    reading_order = content.get("reading_order")
    if not isinstance(blocks, list) or not isinstance(reading_order, list):
        return None

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
    if not ordered_blocks:
        return None

    try:
        from PIL import Image

        with Image.open(source_file) as image:
            image_width, image_height = image.size
        image_bytes = source_file.read_bytes()
    except (OSError, ValueError):
        return None
    if image_width <= 0 or image_height <= 0:
        return None

    canvas_width, canvas_height = _bbox_canvas_size(
        ordered_blocks,
        image_width,
        image_height,
    )
    overlay_parts: list[str] = []
    card_parts: list[str] = []
    boxed_ids: set[str] = set()
    for index, block in enumerate(ordered_blocks, 1):
        component_id = str(block.get("component_id") or "")
        escaped_id = html.escape(component_id, quote=True)
        path_parts = _component_path_parts(component_id)
        block_type = str(block.get("type") or (path_parts[2] if path_parts else "Block"))
        escaped_type = html.escape(block_type.upper())
        card_content = _output_block_html(block, raw=raw)
        bbox = _valid_bbox(block.get("bbox"), canvas_width, canvas_height)
        page = block.get("page")
        page_label = f"PAGE {page + 1}" if isinstance(page, int) else "PAGE —"

        if bbox is not None and page in (None, 0):
            left, top, right, bottom = bbox
            style = (
                f"left:{left / canvas_width * 100:.5f}%;"
                f"top:{top / canvas_height * 100:.5f}%;"
                f"width:{(right - left) / canvas_width * 100:.5f}%;"
                f"height:{(bottom - top) / canvas_height * 100:.5f}%;"
            )
            overlay_parts.append(
                f"<button class='source-box' style='{style}' data-id='{escaped_id}' "
                f"aria-label='Select block {index}: {escaped_type}' "
                f"title='{index:02d} · {escaped_type}'><span>{index:02d}</span></button>"
            )
            boxed_ids.add(component_id)

        box_state = "BOXED" if component_id in boxed_ids else "NO BOX"
        bbox_label = (
            " · [" + ", ".join(f"{coordinate:g}" for coordinate in bbox) + "]"
            if bbox is not None
            else ""
        )
        card_parts.append(
            f"<article class='parsed-card' data-id='{escaped_id}' "
            "role='button' tabindex='0'>"
            "<span class='card-meta'>"
            f"<i>{index:02d}</i><b>{escaped_type}</b>"
            f"<small>{html.escape(page_label)} · {box_state}{bbox_label}</small>"
            "</span>"
            f"<code>{escaped_id}</code>"
            f"{card_content}"
            "</article>"
        )

    if not overlay_parts:
        return None

    mime_type = mimetypes.guess_type(source_file.name)[0] or "image/png"
    image_data = base64.b64encode(image_bytes).decode("ascii")
    image_name = html.escape(source_file.name)
    overlay_html = "".join(overlay_parts)
    cards_html = "".join(card_parts)

    return f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --black:#11110f; --black-soft:#1d1d19; --gold:#c7a34a;
    --gold-dark:#9a792d; --paper:#f2efe7; --surface:#fbf9f3;
    --line:#d9d0b9; --ink:#1b1b17; --muted:#706b5d;
  }}
  * {{ box-sizing:border-box; }}
  html, body {{ margin:0; color:var(--ink); background:var(--paper); font-family:Inter,Manrope,Arial,sans-serif; }}
  .inspector {{ height:758px; display:grid; grid-template-columns:minmax(0,1fr) minmax(360px,1.08fr); border:1px solid #b9aa83; border-radius:5px; overflow:hidden; background:var(--surface); }}
  .pane {{ min-width:0; display:flex; flex-direction:column; overflow:hidden; }}
  .source-pane {{ background:#25241f; border-right:1px solid #3c392f; }}
  .pane-header {{ flex:none; height:58px; display:flex; align-items:center; justify-content:space-between; gap:12px; padding:0 16px; border-bottom:1px solid #d0c39f; background:var(--surface); }}
  .source-pane .pane-header {{ color:#f8f2df; background:var(--black-soft); border-color:#3c392f; }}
  .pane-header div {{ min-width:0; }}
  .pane-header strong {{ display:block; font-size:14px; letter-spacing:.01em; }}
  .pane-header small {{ display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-top:3px; color:var(--muted); font-size:10px; letter-spacing:.07em; }}
  .source-pane .pane-header small {{ color:#aaa38e; }}
  .count {{ flex:none; padding:5px 7px; color:var(--black); background:var(--gold); border-radius:2px; font:700 10px ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .source-scroll, .parsed-scroll {{ flex:1; min-height:0; overflow:auto; }}
  .source-scroll {{ padding:18px; }}
  .image-stage {{ position:relative; width:100%; line-height:0; box-shadow:0 7px 24px rgba(0,0,0,.28); }}
  .image-stage img {{ display:block; width:100%; height:auto; background:#fff; }}
  .source-box {{ position:absolute; z-index:2; min-width:10px; min-height:10px; margin:0; padding:0; cursor:pointer; border:1.5px solid rgba(154,121,45,.92); border-radius:1px; background:rgba(199,163,74,.10); box-shadow:inset 0 0 0 1px rgba(255,255,255,.18); transition:background .12s,border-color .12s,box-shadow .12s; }}
  .source-box span {{ position:absolute; top:-1px; left:-1px; min-width:20px; padding:2px 3px; color:#fff8e5; background:rgba(17,17,15,.82); font:700 8px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .source-box:hover {{ z-index:4; border-color:#f0cf73; background:rgba(199,163,74,.27); }}
  .source-box.active {{ z-index:5; border:2.5px solid #e8bd53; background:rgba(232,189,83,.30); box-shadow:0 0 0 2px rgba(17,17,15,.78),0 0 14px rgba(232,189,83,.82); }}
  .source-box.active span {{ color:var(--black); background:#e8bd53; }}
  .parsed-scroll {{ padding:12px; background:#eee9dd; }}
  .parsed-card {{ display:block; width:100%; margin:0 0 10px; padding:0; overflow:hidden; text-align:left; color:var(--ink); cursor:pointer; border:1px solid #d3c7a8; border-radius:4px; background:var(--surface); box-shadow:0 2px 0 rgba(17,17,15,.04); transition:border-color .13s,box-shadow .13s,transform .13s; }}
  .parsed-card:hover {{ border-color:#9a8248; transform:translateY(-1px); box-shadow:3px 3px 0 rgba(154,121,45,.18); }}
  .parsed-card.active {{ border:2px solid var(--gold-dark); box-shadow:4px 4px 0 rgba(154,121,45,.28); }}
  .card-meta {{ display:flex; align-items:center; gap:8px; min-height:34px; padding:6px 9px; color:#d6cfbd; background:var(--black-soft); }}
  .parsed-card.active .card-meta {{ color:#fff4d8; background:var(--black); }}
  .card-meta i {{ min-width:25px; padding:3px; color:var(--black); background:var(--gold); text-align:center; font:normal 700 10px ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .card-meta b {{ color:var(--gold); font:600 10px ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.05em; }}
  .card-meta small {{ margin-left:auto; color:#aaa38e; font:9px ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .parsed-card code {{ display:block; padding:7px 10px; overflow-wrap:anywhere; color:#756d59; border-bottom:1px solid #e3dccb; background:#f5f1e8; font:10px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .card-output {{ padding:10px 12px 12px; overflow-x:auto; white-space:pre-wrap; font-size:13px; line-height:1.6; }}
  .card-output p:first-child, .card-output div:first-child {{ margin-top:0; }}
  .card-output p:last-child, .card-output div:last-child {{ margin-bottom:0; }}
  .card-output table {{ width:100%; min-width:max-content; border-collapse:collapse; background:#fffdf7; font-size:11px; line-height:1.4; white-space:normal; }}
  .card-output th, .card-output td {{ padding:6px 8px; border:1px solid #cfc19d; text-align:left; vertical-align:top; }}
  .card-output th {{ color:#f6e8bd; background:#2a2923; font-weight:700; }}
  .card-output tr:nth-child(even) td {{ background:#f2ecdc; }}
  .card-output caption {{ margin-bottom:7px; color:#514a3a; font-weight:700; text-align:left; }}
  .card-raw {{ margin:0; padding:10px 12px 12px; overflow:auto; color:#302e27; background:#fffdf7; white-space:pre-wrap; overflow-wrap:anywhere; font:11px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace; }}
  @media (max-width:720px) {{
    .inspector {{ height:758px; grid-template-columns:1fr; grid-template-rows:50% 50%; }}
    .source-pane {{ border-right:0; border-bottom:1px solid #3c392f; }}
  }}
</style>
</head>
<body>
  <main class="inspector">
    <section class="pane source-pane">
      <header class="pane-header"><div><strong>Source document</strong><small>{image_name}</small></div><span class="count">{len(overlay_parts)} BOXES</span></header>
      <div class="source-scroll">
        <div class="image-stage">
          <img src="data:{mime_type};base64,{image_data}" alt="Raw source document">
          {overlay_html}
        </div>
      </div>
    </section>
    <section class="pane parsed-pane">
      <header class="pane-header"><div><strong>Parsed content</strong><small>JSON READING ORDER · {'JSON OUTPUT' if raw else 'RENDERED'}</small></div><span class="count">{len(ordered_blocks)} BLOCKS</span></header>
      <div class="parsed-scroll">{cards_html}</div>
    </section>
  </main>
<script>
  const boxes = Array.from(document.querySelectorAll('.source-box'));
  const cards = Array.from(document.querySelectorAll('.parsed-card'));

  function activate(componentId, scrollTarget) {{
    boxes.forEach((box) => box.classList.toggle('active', box.dataset.id === componentId));
    cards.forEach((card) => card.classList.toggle('active', card.dataset.id === componentId));
    if (scrollTarget === 'card') {{
      const card = cards.find((item) => item.dataset.id === componentId);
      if (card) card.scrollIntoView({{behavior:'smooth', block:'center'}});
    }}
    if (scrollTarget === 'box') {{
      const box = boxes.find((item) => item.dataset.id === componentId);
      if (box) box.scrollIntoView({{behavior:'smooth', block:'center', inline:'center'}});
    }}
  }}

  boxes.forEach((box) => box.addEventListener('click', () => activate(box.dataset.id, 'card')));
  cards.forEach((card) => card.addEventListener('click', () => activate(card.dataset.id, 'box')));
  cards.forEach((card) => card.addEventListener('keydown', (event) => {{
    if (event.key === 'Enter' || event.key === ' ') {{
      event.preventDefault();
      activate(card.dataset.id, 'box');
    }}
  }}));
  if (boxes.length) activate(boxes[0].dataset.id, null);
</script>
</body>
</html>
"""


def _valid_bbox(
    value: object,
    canvas_width: float,
    canvas_height: float,
) -> tuple[float, float, float, float] | None:
    """Validate and clamp one parser bbox to its parser coordinate canvas."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    left, top, right, bottom = (float(item) for item in value)
    left = max(0.0, min(left, canvas_width))
    top = max(0.0, min(top, canvas_height))
    right = max(0.0, min(right, canvas_width))
    bottom = max(0.0, min(bottom, canvas_height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _output_block_html(block: dict, *, raw: bool = False) -> str:
    """Render one output block without merging in any other output collection."""
    if raw:
        value = _output_block_source_value(block)
        if value is None:
            value = "[No content representation is available for this block]"
        return f"<pre class='card-raw'>{html.escape(value)}</pre>"

    parser_html = block.get("html")
    if isinstance(parser_html, str) and parser_html.strip():
        sanitized = _sanitize_output_html(parser_html)
        visible_text = re.sub(r"<[^>]+>", "", sanitized).strip()
        if visible_text:
            return f"<div class='card-output'>{sanitized}</div>"

    for field in ("text", "semantic_text"):
        value = block.get(field)
        if isinstance(value, str) and value:
            return f"<div class='card-output'>{html.escape(value)}</div>"
    return (
        "<div class='card-output'>"
        "[No content representation is available for this block]"
        "</div>"
    )


def _output_block_source_value(block: dict) -> str | None:
    """Return all content representations stored in the output JSON block."""
    content_fields = {
        field: block[field]
        for field in ("text", "semantic_text", "html")
        if field in block
    }
    if not content_fields:
        return None
    return json.dumps(content_fields, ensure_ascii=False, indent=2)


def _sanitize_output_html(value: str) -> str:
    sanitizer = _OutputHTMLSanitizer()
    sanitizer.feed(value)
    sanitizer.close()
    return "".join(sanitizer.parts)


def _bbox_canvas_size(
    blocks: list[dict],
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    """Resolve the parser canvas while preserving the source image aspect ratio."""
    for block in blocks:
        page_bbox = block.get("page_bbox")
        if not _is_numeric_bbox(page_bbox):
            continue
        left, top, right, bottom = (float(item) for item in page_bbox)
        if right > left and bottom > top:
            return right - left, bottom - top

    # Legacy artifacts did not retain the Page polygon. Datalab image canvases
    # are bucketed in 128 px steps, so find the smallest bucket that contains
    # every box on both axes. Deriving height from the source aspect ratio is
    # important: using the last detected box independently on each axis warps
    # the overlay and makes the error grow toward the bottom/right of the page.
    right_edges: list[float] = []
    bottom_edges: list[float] = []
    for block in blocks:
        value = block.get("bbox")
        if not _is_numeric_bbox(value):
            continue
        left, top, right, bottom = (float(item) for item in value)
        if right > left and bottom > top:
            right_edges.append(right)
            bottom_edges.append(bottom)
    if not right_edges:
        return float(image_width), float(image_height)

    required_width = max(
        max(right_edges),
        max(bottom_edges) * image_width / image_height,
        1536.0,
    )
    canvas_width = math.ceil(required_width / 128.0) * 128.0
    canvas_height = canvas_width * image_height / image_width
    return canvas_width, canvas_height


def _is_numeric_bbox(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(
            not isinstance(item, bool) and isinstance(item, (int, float))
            for item in value
        )
    )


def _render_parsed_content(parsed_payload: object) -> None:
    """Render semantic extraction fields while preserving parser reading order."""
    if not isinstance(parsed_payload, dict):
        st.info("The parsing payload has no usable content.")
        return

    rows = parsed_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        text = parsed_payload.get("text")
        if text:
            _parsed_text_block(str(text))
        else:
            st.info("The parsing payload has no usable content.")
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
            with st.expander(f"Unmapped source citations · {len(citations)} blocks"):
                st.caption(
                    "This artifact does not contain enough block-level data to map "
                    "each component ID to an exact paragraph."
                )
                ordered_blocks = "\n".join(
                    f"{index:02d}  {citation}"
                    for index, citation in enumerate(citations, 1)
                )
                st.code(ordered_blocks, language=None)

    if not rendered:
        st.info("The parsing payload has no usable main text.")


def _render_output_content(content: dict, *, raw: bool = False) -> None:
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
    _render_source_blocks(ordered_blocks, raw=raw)

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


def _render_source_blocks(blocks: list[dict], *, raw: bool = False) -> None:
    for index, block in enumerate(blocks, 1):
        component_id = str(block.get("component_id") or "")
        escaped_id = html.escape(component_id, quote=True)
        block_content = _output_block_html(block, raw=raw)
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
                f"<div class='parsed-block-text'>{block_content}</div>"
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
    display_title = _display_title(document.title)
    title = f" — {display_title}" if display_title and display_title != document.file_name else ""
    return f"{document.file_name}{title}  [{document.status}]"


def _display_title(value: object) -> str:
    if value is None:
        return ""
    title = str(value)
    return DISPLAY_TITLE_OVERRIDES.get(title, title)


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
        :root {
            --black:#11110f;
            --black-soft:#1d1d19;
            --yellow:#c7a34a;
            --yellow-deep:#a17d2b;
            --yellow-soft:#eee2bc;
            --ink:#1b1b17;
            --muted:#706b5d;
            --paper:#f2efe7;
            --surface:#fbf9f3;
            --line:#d9d0b9;
        }
        html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
        .stApp {
            color:var(--ink);
            background:
                radial-gradient(circle at 88% 4%, rgba(199,163,74,.09), transparent 24rem),
                var(--paper);
        }
        [data-testid="stSidebar"] {
            background:var(--black);
            border-right:1px solid #2f2e26;
            box-shadow:8px 0 28px rgba(17,17,15,.12);
        }
        [data-testid="stSidebar"]::before {
            content:"";
            display:block;
            height:3px;
            background:var(--yellow);
        }
        [data-testid="stSidebar"] * { color:#f7f2e5; }
        [data-testid="stSidebar"] [role="radiogroup"] { gap:.45rem; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding:.7rem .8rem;
            border:1px solid #303029;
            border-radius:4px;
            transition:all .16s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background:#292821;
            border-color:#5a5542;
            transform:translateX(2px);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background:var(--yellow);
            border-color:var(--yellow);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {
            color:var(--black)!important;
            font-weight:700;
        }
        [data-testid="stSidebar"] hr { border-color:#38372f; }
        .block-container { max-width:1440px; padding-top:1.8rem; padding-bottom:4rem; }
        .brand-title { padding-top:.35rem; font-size:1.3rem; font-weight:700; letter-spacing:.16em; }
        .brand-subtitle { font-family:'DM Mono'; font-size:.64rem; color:#aaa38e!important; letter-spacing:.14em; }
        .sidebar-rule { height:1px; background:#37362e; margin:1.5rem 0; }
        .sidebar-spacer { height:12rem; }
        .hero {
            position:relative;
            overflow:hidden;
            padding:1.55rem 1.75rem 1.65rem;
            margin-bottom:1.75rem;
            border:1px solid var(--black);
            border-left:6px solid var(--yellow);
            border-radius:5px;
            background:
                linear-gradient(115deg, transparent 72%, rgba(199,163,74,.08) 72%),
                var(--black);
            box-shadow:5px 5px 0 rgba(199,163,74,.38);
        }
        .hero .eyebrow { color:var(--yellow); font-family:'DM Mono'; font-size:.7rem; letter-spacing:.15em; margin-bottom:.65rem; }
        .hero h1 { position:relative; z-index:1; max-width:900px; font-size:clamp(2rem,3vw,3rem); line-height:1.08; letter-spacing:-.04em; margin:0; color:#fffdf4; }
        .hero p { position:relative; z-index:1; max-width:760px; color:#beb8a7; font-size:1rem; margin:.75rem 0 0; line-height:1.65; }
        .status-badge { display:inline-block; font-family:'DM Mono'; font-size:.64rem; font-weight:500; letter-spacing:.05em; padding:.4rem .58rem; border:1px solid var(--black); border-radius:2px; background:var(--yellow); color:var(--black); }
        .status-badge.warning { background:#ead8a4; color:#574719; border-color:#92742e; }
        .document-placeholder { min-height:330px; background:#f7f2e4; display:flex; flex-direction:column; align-items:center; justify-content:center; border:1px dashed #9b8244; border-radius:4px; }
        .document-placeholder span { width:72px; height:88px; display:grid; place-items:center; background:var(--black); border:2px solid var(--yellow); font-family:'DM Mono'; color:#d8bd78; box-shadow:4px 4px 0 #d8c993; }
        .document-placeholder b { margin-top:1rem; }
        .document-placeholder small { color:#786f5b; margin-top:.3rem; }
        .parsed-article { white-space:pre-wrap; line-height:1.72; color:var(--ink); font-size:.94rem; }
        .parsed-block { margin:0 0 .85rem; border:1px solid #d8ceaf; background:var(--surface); border-radius:4px; overflow:hidden; box-shadow:0 2px 0 rgba(17,17,15,.04); }
        .parsed-block:hover { border-color:#a58a4c; box-shadow:3px 3px 0 rgba(199,163,74,.18); }
        .parsed-block-meta { display:flex; align-items:center; gap:.65rem; padding:.48rem .65rem; border-bottom:1px solid #3b392f; background:var(--black-soft); }
        .parsed-block-meta span { min-width:1.75rem; color:var(--black); background:var(--yellow); padding:.16rem .25rem; text-align:center; font-family:'DM Mono'; font-size:.68rem; font-weight:500; }
        .parsed-block-meta b { color:var(--yellow); font-family:'DM Mono'; font-size:.62rem; font-weight:500; letter-spacing:.06em; }
        .parsed-block-meta code { overflow-wrap:anywhere; color:#c7c0ad; background:transparent; font-size:.7rem; }
        .parsed-block-text { padding:.8rem .9rem .9rem; white-space:pre-wrap; line-height:1.7; color:#28271f; font-size:.94rem; }
        .parsed-block-text .card-output { padding:0; overflow-x:auto; }
        .parsed-block-text .card-raw { margin:0; overflow:auto; color:#302e27; background:#fffdf7; white-space:pre-wrap; overflow-wrap:anywhere; font:11px/1.55 'DM Mono',monospace; }
        .parsed-block-text table { width:100%; min-width:max-content; border-collapse:collapse; background:#fffdf7; font-size:.82rem; white-space:normal; }
        .parsed-block-text th, .parsed-block-text td { padding:.4rem .55rem; border:1px solid #cfc19d; text-align:left; vertical-align:top; }
        .parsed-block-text th { color:#f6e8bd; background:#2a2923; }

        /* Streamlit controls */
        [data-testid="stFileUploaderDropzone"] { background:#f8f3e6; border:1.5px dashed #9d8448; border-radius:5px; }
        [data-testid="stFileUploaderDropzone"] button { background:var(--black); color:var(--yellow); border-color:var(--black); }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stTextInput"] input { background:var(--surface); border-color:#c9bd9e; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid #cfc3a4; }
        [data-testid="stTabs"] button[role="tab"] { color:#665f4f; padding:.7rem .95rem; border-radius:4px 4px 0 0; }
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:var(--black); background:var(--yellow); font-weight:700; }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--black); }
        [data-testid="stExpander"] { background:var(--surface); border-color:#d6caab; border-radius:4px; }
        [data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,253,246,.74); border-color:#d4c7a6!important; }
        .stButton>button, .stDownloadButton>button { border-radius:3px; font-weight:700; transition:all .14s ease; }
        .stButton>button[kind="primary"] { background:var(--yellow); color:var(--black); border:1px solid var(--black); box-shadow:4px 4px 0 var(--black); }
        .stButton>button[kind="primary"]:hover { background:#d4b45f; color:var(--black); border-color:var(--black); transform:translate(-1px,-1px); box-shadow:5px 5px 0 var(--black); }
        .stDownloadButton>button { background:var(--black); color:var(--yellow); border:1px solid var(--black); }
        .stDownloadButton>button:hover { background:#2c2b23; color:#d9bc72; border-color:#2c2b23; }
        [data-testid="stMetric"] { background:var(--surface); border:1px solid #d5caad; border-top:4px solid var(--yellow); padding:.85rem; box-shadow:2px 2px 0 rgba(17,17,15,.07); }
        [data-testid="stAlert"] { border-radius:4px; }
        code { color:#6b5200; }
        ::selection { color:var(--black); background:var(--yellow); }
        ::-webkit-scrollbar { width:10px; height:10px; }
        ::-webkit-scrollbar-track { background:#ebe4d3; }
        ::-webkit-scrollbar-thumb { background:#89816c; border:2px solid #ebe4d3; border-radius:8px; }
        ::-webkit-scrollbar-thumb:hover { background:var(--yellow-deep); }
        @media(max-width:800px) {
            .hero { padding:1.2rem; box-shadow:3px 3px 0 rgba(199,163,74,.4); }
            .hero h1 { font-size:2.2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
