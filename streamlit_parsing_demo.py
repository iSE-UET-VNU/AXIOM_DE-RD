"""Standalone parsing showcase for AXIOM-DE.

This intentionally stays separate from ``streamlit_app.py``. It is a compact
portfolio demo for PDF and image parsing only: upload one document, call the
Datalab Lift parser, and inspect parser bounding boxes page by page.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
import html
from html.parser import HTMLParser
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import tempfile
from typing import Any

import pymupdf
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

from src.ingestion.parsing.lift.client import LiftAPIConfig, LiftAPIParserClient
from src.models import DataObject
from src.utils.env import load_dotenv_file


PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_TYPES = ["pdf", "png", "jpg", "jpeg", "webp"]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 25


st.set_page_config(
    page_title="AXIOM · Parsing Showcase",
    page_icon="◈",
    layout="wide",
)


def main() -> None:
    load_dotenv_file(PROJECT_ROOT)
    _styles()
    _sidebar()
    _hero(
        "Inspect document parsing",
        "Upload a PDF or image, run document understanding, and review detected elements against the original page.",
        "PARSING SHOWCASE",
    )

    if not _requirements_ready():
        return

    st.markdown("<div class='section-label'>SOURCE DOCUMENT</div>", unsafe_allow_html=True)
    upload = st.file_uploader(
        "Choose a PDF or image",
        type=SUPPORTED_TYPES,
        help="PDF, PNG, JPG, JPEG, and WEBP only. Keep uploads below 20 MB.",
    )
    if upload is None:
        _empty_state()
        return

    source_bytes = upload.getvalue()
    suffix = Path(upload.name).suffix.lower()
    if len(source_bytes) > MAX_UPLOAD_BYTES:
        st.error("This demo accepts files up to 20 MB.")
        return
    if suffix == ".pdf":
        try:
            page_count = len(pymupdf.open(stream=source_bytes, filetype="pdf"))
        except (pymupdf.FileDataError, RuntimeError) as exc:
            st.error(f"The PDF could not be opened: {exc}")
            return
        if page_count > MAX_PDF_PAGES:
            st.error(f"This demo accepts PDFs with at most {MAX_PDF_PAGES} pages.")
            return
    else:
        page_count = 1

    file_hash = hashlib.sha256(source_bytes).hexdigest()
    st.caption(
        f"`{upload.name}` · {len(source_bytes) / 1024:.1f} KB · {page_count} page(s)"
    )
    if st.button("Run parsing", type="primary", use_container_width=True):
        _run_parser(upload.name, source_bytes, suffix, file_hash)

    result = st.session_state.get("parsing_showcase_result")
    if not isinstance(result, dict) or result.get("file_hash") != file_hash:
        st.info("Upload a file and select **Run parsing** to inspect its layout.")
        return

    _render_result(result)


def _requirements_ready() -> bool:
    if not st.session_state.get("datalab_key_checked"):
        import os

        st.session_state["datalab_key_checked"] = bool(os.getenv("DATALAB_API_KEY"))
    if not st.session_state["datalab_key_checked"]:
        st.error("`DATALAB_API_KEY` is not configured. Add it to `.env` before running the demo.")
        return False
    return True


def _empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
          <div class="empty-icon">01</div>
          <div><b>Ready for a source document</b><br>
          <span>PDF and image parsing only. Upload a file to inspect text, tables, figures, and layout bounding boxes.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_parser(name: str, source_bytes: bytes, suffix: str, file_hash: str) -> None:
    safe_name = Path(name).name or f"upload{suffix}"
    object_id = f"demo-{file_hash[:16]}"
    content_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    config = LiftAPIConfig(
        mode="fast",
        schema_path="src/ingestion/parsing/lift/schemas/document_components.json",
        output_dir=str(PROJECT_ROOT / ".tmp" / "parsing-showcase-assets"),
        fallback_to_local=False,
        extract_images=False,
        save_raw_outputs=False,
        project_root=str(PROJECT_ROOT),
    )
    source = DataObject(
        object_id=object_id,
        uri=f"demo://{safe_name}",
        content_type=content_type,
        metadata={"file_name": safe_name, "format": suffix.removeprefix(".")},
    )

    try:
        with tempfile.TemporaryDirectory(prefix="axiom-parsing-demo-") as temp_dir:
            source_path = Path(temp_dir) / safe_name
            source_path.write_bytes(source_bytes)
            with st.spinner("Parsing document…"):
                parsed = LiftAPIParserClient(config).parse_file(source_path, source)
    except Exception as exc:
        st.error(f"Parsing failed: {exc}")
        return

    row = parsed.rows[0] if parsed.rows else {}
    blocks = row.get("source_blocks", []) if isinstance(row, dict) else []
    extraction = row.get("extraction", {}) if isinstance(row, dict) else {}
    if isinstance(blocks, list) and isinstance(extraction, dict):
        _attach_structured_html(blocks, extraction)
    st.session_state["parsing_showcase_result"] = {
        "file_hash": file_hash,
        "file_name": safe_name,
        "suffix": suffix,
        "source_bytes": source_bytes,
        "blocks": blocks if isinstance(blocks, list) else [],
        "text": parsed.text or "",
        "metadata": parsed.metadata,
    }
    st.rerun()


class _OutputHTMLSanitizer(HTMLParser):
    """Keep table markup from the parser while dropping unsafe markup."""

    ALLOWED = {"table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "br", "p", "div", "span", "strong", "em"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag not in self.ALLOWED:
            return
        safe_attrs = ""
        if tag in {"td", "th"}:
            for name, value in attrs:
                if name.casefold() in {"colspan", "rowspan"} and value and value.isdigit():
                    safe_attrs += f" {name.casefold()}='{html.escape(value, quote=True)}'"
        self.parts.append(f"<{tag}{safe_attrs}>")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self.ALLOWED and tag != "br":
            self.parts.append(f"</{tag}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        self.parts.append(html.escape(data))


def _sanitize_html(value: str) -> str:
    sanitizer = _OutputHTMLSanitizer()
    sanitizer.feed(value)
    sanitizer.close()
    return "".join(sanitizer.parts)


def _attach_structured_html(blocks: list[dict[str, Any]], extraction: dict[str, Any]) -> None:
    """Attach extracted table HTML to its cited layout block."""
    by_id = {str(block.get("component_id")): block for block in blocks if isinstance(block, dict)}
    tables = extraction.get("tables", [])
    if not isinstance(tables, list):
        return
    for table in tables:
        if not isinstance(table, dict):
            continue
        content = table.get("content")
        if not isinstance(content, str) or "<table" not in content.casefold():
            continue
        for field in ("content_citations", "description_citations", "caption_citations"):
            citations = table.get(field)
            if not isinstance(citations, list):
                continue
            for component_id in citations:
                block = by_id.get(str(component_id))
                if block is not None:
                    block["html"] = content


def _render_result(result: dict[str, Any]) -> None:
    blocks = [item for item in result.get("blocks", []) if isinstance(item, dict)]
    page_numbers = sorted({int(block.get("page", 0)) for block in blocks if isinstance(block.get("page"), int)})
    source_bytes = result["source_bytes"]
    suffix = str(result["suffix"])
    inferred_pages = _source_page_count(source_bytes, suffix)
    page_numbers = page_numbers or list(range(inferred_pages))
    page_index = st.selectbox(
        "Page",
        page_numbers,
        format_func=lambda value: f"Page {value + 1}",
    )
    page_blocks = [block for block in blocks if int(block.get("page", 0)) == page_index]
    boxed_count = sum(_bbox(block) is not None for block in page_blocks)

    metrics = st.columns(4)
    metrics[0].metric("Pages", inferred_pages)
    metrics[1].metric("Layout blocks", len(blocks))
    metrics[2].metric("Boxes on page", boxed_count)
    metrics[3].metric("Parser latency", f"{result['metadata'].get('latency_seconds', '—')} s")

    image_bytes, width, height = _render_page(source_bytes, suffix, page_index)
    components.html(
        _overlay_html(
            image_bytes=image_bytes,
            image_name=result["file_name"],
            image_width=width,
            image_height=height,
            blocks=page_blocks,
        ),
        height=760,
        scrolling=False,
    )

    content_tab, json_tab = st.tabs(["Extracted content", "Raw parsing result"])
    with content_tab:
        if not page_blocks:
            st.info("The parser returned no content blocks for this page.")
        for index, block in enumerate(page_blocks, 1):
            block_type = str(block.get("type") or "block").upper()
            label = f"{index:02d} · {block_type}"
            with st.expander(label):
                st.caption(str(block.get("component_id") or ""))
                st.write(str(block.get("text") or "[No text representation]"))
                if _bbox(block) is not None:
                    st.code(f"bbox = {_bbox(block)}", language=None)
    with json_tab:
        st.download_button(
            "Download parsing JSON",
            data=json.dumps(result["raw_result"], ensure_ascii=False, indent=2, default=str),
            file_name=f"{Path(result['file_name']).stem}_parsing.json",
            mime="application/json",
        )
        st.json(result["raw_result"], expanded=False)


def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<div class='brand-title'>AXIOM</div><div class='brand-subtitle'>DOCUMENT INTELLIGENCE</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sidebar-rule'></div>", unsafe_allow_html=True)
        st.markdown("<div class='nav-current'>PARSING SHOWCASE</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-spacer'></div>", unsafe_allow_html=True)
        st.caption("AXIOM-DE | Portfolio demo")


def _hero(title: str, subtitle: str, eyebrow: str) -> None:
    st.markdown(
        f"<section class='hero'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></section>",
        unsafe_allow_html=True,
    )


def _render_result(result: dict[str, Any]) -> None:
    """Render the focused AXIOM-style public page inspector."""
    blocks = [item for item in result.get("blocks", []) if isinstance(item, dict)]
    source_bytes = result["source_bytes"]
    suffix = str(result["suffix"])
    inferred_pages = _source_page_count(source_bytes, suffix)
    page_numbers = sorted(
        {int(block.get("page", 0)) for block in blocks if isinstance(block.get("page"), int)}
    ) or list(range(inferred_pages))

    controls, note = st.columns([1, 3])
    with controls:
        page_index = st.selectbox(
            "Viewing page", page_numbers, format_func=lambda value: f"Page {value + 1}"
        )
    with note:
        st.markdown(
            "<div class='inspector-note'>Click a gold region or its parsed-content card to move between the source and its extracted result.</div>",
            unsafe_allow_html=True,
        )

    page_blocks = [block for block in blocks if int(block.get("page", 0)) == page_index]
    boxed_count = sum(_bbox(block) is not None for block in page_blocks)
    metrics = st.columns(4)
    metrics[0].metric("Pages", inferred_pages)
    metrics[1].metric("Layout blocks", len(blocks))
    metrics[2].metric("Boxes on page", boxed_count)
    metrics[3].metric("Parser latency", f"{result['metadata'].get('latency_seconds', 'n/a')} s")

    image_bytes, width, height = _render_page(source_bytes, suffix, page_index)
    components.html(
        _overlay_html(
            image_bytes=image_bytes,
            image_name=result["file_name"],
            image_width=width,
            image_height=height,
            blocks=page_blocks,
        ),
        height=760,
        scrolling=False,
    )


def _source_page_count(source_bytes: bytes, suffix: str) -> int:
    if suffix == ".pdf":
        with pymupdf.open(stream=source_bytes, filetype="pdf") as document:
            return len(document)
    return 1


def _render_page(source_bytes: bytes, suffix: str, page_index: int) -> tuple[bytes, int, int]:
    if suffix == ".pdf":
        with pymupdf.open(stream=source_bytes, filetype="pdf") as document:
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
            return pixmap.tobytes("png"), pixmap.width, pixmap.height
    with Image.open(BytesIO(source_bytes)) as image:
        converted = image.convert("RGB")
        output = BytesIO()
        converted.save(output, format="PNG")
        return output.getvalue(), converted.width, converted.height


def _bbox(block: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = block.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return None
    left, top, right, bottom = (float(item) for item in value)
    return (left, top, right, bottom) if right > left and bottom > top else None


def _canvas_size(blocks: list[dict[str, Any]], image_width: int, image_height: int) -> tuple[float, float]:
    for block in blocks:
        value = block.get("page_bbox")
        if isinstance(value, (list, tuple)) and len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            left, top, right, bottom = (float(item) for item in value)
            if right > left and bottom > top:
                return right - left, bottom - top
    boxes = [box for block in blocks if (box := _bbox(block)) is not None]
    if not boxes:
        return float(image_width), float(image_height)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    if max_x <= 0 or max_y <= 0:
        return float(image_width), float(image_height)
    return max_x, max_y


def _overlay_html(
    *,
    image_bytes: bytes,
    image_name: str,
    image_width: int,
    image_height: int,
    blocks: list[dict[str, Any]],
) -> str:
    canvas_width, canvas_height = _canvas_size(blocks, image_width, image_height)
    boxes: list[str] = []
    cards: list[str] = []
    for index, block in enumerate(blocks, 1):
        bbox = _bbox(block)
        block_type = html.escape(str(block.get("type") or "block").upper())
        parser_html = block.get("html")
        if isinstance(parser_html, str) and parser_html.strip():
            content = f"<div class='card-output'>{_sanitize_html(parser_html)}</div>"
        else:
            content = f"<p>{html.escape(str(block.get('text') or '[No text representation]'))}</p>"
        block_id = html.escape(str(block.get("component_id") or index), quote=True)
        if bbox is not None:
            left, top, right, bottom = bbox
            style = (
                f"left:{left / canvas_width * 100:.5f}%;top:{top / canvas_height * 100:.5f}%;"
                f"width:{(right - left) / canvas_width * 100:.5f}%;height:{(bottom - top) / canvas_height * 100:.5f}%;"
            )
            boxes.append(f"<button class='box' data-id='{block_id}' style='{style}'>{index}</button>")
        cards.append(
            f"<article class='card' data-id='{block_id}'><b>{index:02d} · {block_type}</b>"
            f"{content}</article>"
        )
    image_data = base64.b64encode(image_bytes).decode("ascii")
    return f"""
    <style>
      * {{ box-sizing:border-box; }} body {{ margin:0; font-family:Arial,sans-serif; color:#1b1b17; }}
      .shell {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(320px,.85fr); height:750px; border:1px solid #bda96e; border-radius:5px; overflow:hidden; }}
      .source,.parsed {{ min-width:0; display:flex; flex-direction:column; overflow:hidden; }}
      .source {{ background:#1d1c18; border-right:1px solid #3c392f; }} .source-scroll,.parsed-scroll {{ flex:1; min-height:0; overflow:auto; }} .source-scroll {{ padding:16px; }} .stage {{ position:relative; line-height:0; }}
      img {{ width:100%; height:auto; display:block; background:#fff; }}
      .pane-header {{ flex:none; height:56px; display:flex; align-items:center; justify-content:space-between; padding:0 16px; border-bottom:1px solid #d0c39f; background:#fbf9f3; }} .source .pane-header {{ color:#f8f2df; background:#1d1d19; border-color:#3c392f; }}
      .pane-header strong {{ display:block; font-size:14px; }} .pane-header small {{ display:block; margin-top:3px; color:#706b5d; font-size:10px; letter-spacing:.07em; }} .source .pane-header small {{ color:#aaa38e; }} .count {{ padding:5px 7px; color:#11110f; background:#c7a34a; border-radius:2px; font:700 10px monospace; }}
      .box {{ position:absolute; border:1.5px solid rgba(199,163,74,.95); background:rgba(199,163,74,.12); color:#fff8e5; cursor:pointer; font-size:10px; line-height:16px; min-width:18px; min-height:18px; padding:0; }}
      .box.active {{ border:2.5px solid #e8bd53; background:rgba(232,189,83,.30); box-shadow:0 0 0 2px rgba(17,17,15,.78),0 0 14px rgba(232,189,83,.82); z-index:2; }}
      .parsed-scroll {{ padding:12px; background:#eee9dd; }} .card {{ cursor:pointer; border:1px solid #d5c8a7; padding:0; margin:0 0 10px; background:#fffdf8; border-radius:4px; overflow:hidden; }}
      .card.active {{ border:2px solid #a17d2b; background:#fff7df; }} .card b {{ display:block; padding:8px 10px; color:#c7a34a; background:#1d1d19; font:600 11px monospace; }} .card p {{ margin:0; padding:10px; white-space:pre-wrap; font-size:13px; line-height:1.45; }}
      .card-output {{ padding:10px; overflow-x:auto; }} .card-output table {{ width:100%; border-collapse:collapse; background:#fffdf7; font-size:11px; line-height:1.4; }} .card-output th,.card-output td {{ padding:6px 8px; border:1px solid #cfc19d; text-align:left; vertical-align:top; }} .card-output th {{ color:#f6e8bd; background:#2a2923; }} .card-output tr:nth-child(even) td {{ background:#f2ecdc; }}
      @media (max-width:700px) {{ .shell {{ grid-template-columns:1fr; grid-template-rows:52% 48%; }} }}
    </style>
    <main class='shell'>
      <section class='source'><header class='pane-header'><div><strong>Source document</strong><small>{html.escape(image_name)}</small></div><span class='count'>{len(boxes)} BOXES</span></header><div class='source-scroll'><div class='stage'><img src='data:image/png;base64,{image_data}' alt='{html.escape(image_name)}'>{''.join(boxes)}</div></div></section>
      <section class='parsed'><header class='pane-header'><div><strong>Parsed content</strong><small>READING ORDER</small></div><span class='count'>{len(blocks)} BLOCKS</span></header><div class='parsed-scroll'>{''.join(cards) or '<p>No blocks found.</p>'}</div></section>
    </main>
    <script>
      const boxes=[...document.querySelectorAll('.box')], cards=[...document.querySelectorAll('.card')];
      function select(id, scroll) {{
        boxes.forEach(x=>x.classList.toggle('active',x.dataset.id===id)); cards.forEach(x=>x.classList.toggle('active',x.dataset.id===id));
        const target=(scroll==='card'?cards:boxes).find(x=>x.dataset.id===id); if(target) target.scrollIntoView({{block:'center',behavior:'smooth'}});
      }}
      boxes.forEach(x=>x.onclick=()=>select(x.dataset.id,'card')); cards.forEach(x=>x.onclick=()=>select(x.dataset.id,'box'));
    </script>
    """


def _styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700&display=swap');
          :root { --black:#11110f; --black-soft:#1d1d19; --gold:#c7a34a; --gold-deep:#a17d2b; --paper:#f2efe7; --surface:#fbf9f3; --line:#d9d0b9; --ink:#1b1b17; --muted:#706b5d; }
          html, body, [class*='css'] { font-family:'Manrope',sans-serif; }
          .stApp { color:var(--ink); background:radial-gradient(circle at 88% 4%,rgba(199,163,74,.09),transparent 24rem),var(--paper); }
          [data-testid='stSidebar'] { background:var(--black); border-right:1px solid #2f2e26; box-shadow:8px 0 28px rgba(17,17,15,.12); }
          [data-testid='stSidebar']::before { content:''; display:block; height:3px; background:var(--gold); }
          [data-testid='stSidebar'] * { color:#f7f2e5; }
          .block-container { max-width:1440px; padding-top:1.8rem; padding-bottom:4rem; }
          .brand-title { padding-top:.35rem; font-size:1.3rem; font-weight:700; letter-spacing:.16em; }
          .brand-subtitle { font-family:'DM Mono',monospace; font-size:.64rem; color:#aaa38e!important; letter-spacing:.14em; }
          .sidebar-rule { height:1px; background:#37362e; margin:1.5rem 0; }
          .nav-current { padding:.7rem .8rem; border:1px solid var(--gold); border-radius:4px; background:var(--gold); color:var(--black)!important; font:700 .7rem 'DM Mono',monospace; letter-spacing:.06em; }
          .sidebar-spacer { height:12rem; }
          .hero { position:relative; overflow:hidden; padding:1.55rem 1.75rem 1.65rem; margin-bottom:1.75rem; border:1px solid var(--black); border-left:6px solid var(--gold); border-radius:5px; background:linear-gradient(115deg,transparent 72%,rgba(199,163,74,.08) 72%),var(--black); box-shadow:5px 5px 0 rgba(199,163,74,.38); }
          .hero .eyebrow { color:var(--gold); font: .7rem 'DM Mono',monospace; letter-spacing:.15em; margin-bottom:.65rem; }
          .hero h1 { max-width:900px; font-size:clamp(2rem,3vw,3rem); line-height:1.08; letter-spacing:-.04em; margin:0; color:#fffdf4; }
          .hero p { max-width:760px; color:#beb8a7; font-size:1rem; margin:.75rem 0 0; line-height:1.65; }
          .section-label { margin:0 0 .55rem; color:#6e5720; font:500 .7rem 'DM Mono',monospace; letter-spacing:.12em; }
          .document-summary { min-height:64px; padding:.75rem .9rem; border:1px solid #d5caad; background:var(--surface); box-shadow:2px 2px 0 rgba(17,17,15,.07); }
          .document-summary b { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.93rem; }
          .document-summary span { display:block; margin-top:.35rem; color:var(--muted); font:.66rem 'DM Mono',monospace; letter-spacing:.05em; }
          .empty-state { display:flex; align-items:center; gap:1rem; margin-top:2.2rem; padding:1.5rem; border:1px dashed #9b8244; background:#f8f3e6; color:#4d4534; }
          .empty-icon { display:grid; flex:none; width:3rem; height:3rem; place-items:center; color:var(--gold); border:1px solid var(--gold); background:var(--black); font:.75rem 'DM Mono',monospace; }
          .empty-state b { font-size:1rem; } .empty-state span { color:var(--muted); font-size:.9rem; }
          .inspector-note { height:100%; display:flex; align-items:flex-end; padding:0 0 .35rem; color:var(--muted); font-size:.82rem; line-height:1.4; }
          [data-testid='stFileUploaderDropzone'] { background:#f8f3e6; border:1.5px dashed #9d8448; border-radius:5px; }
          [data-testid='stFileUploaderDropzone'] button { background:var(--black); color:var(--gold); border-color:var(--black); }
          [data-testid='stTabs'] [data-baseweb='tab-list'] { gap:.35rem; border-bottom:1px solid #cfc3a4; }
          [data-testid='stTabs'] button[role='tab'] { color:#665f4f; padding:.7rem .95rem; border-radius:4px 4px 0 0; }
          [data-testid='stTabs'] button[role='tab'][aria-selected='true'] { color:var(--black); background:var(--gold); font-weight:700; }
          [data-testid='stTabs'] [data-baseweb='tab-highlight'] { background:var(--black); }
          [data-testid='stSelectbox'] [data-baseweb='select'] > div { background:var(--surface); border-color:#c9bd9e; }
          .stButton>button, .stDownloadButton>button { border-radius:3px; font-weight:700; transition:all .14s ease; }
          .stButton>button[kind='primary'] { background:var(--gold); color:var(--black); border:1px solid var(--black); box-shadow:4px 4px 0 var(--black); }
          .stButton>button[kind='primary']:hover { background:#d4b45f; color:var(--black); transform:translate(-1px,-1px); box-shadow:5px 5px 0 var(--black); }
          .stDownloadButton>button { background:var(--black); color:var(--gold); border:1px solid var(--black); }
          [data-testid='stMetric'] { background:var(--surface); border:1px solid #d5caad; border-top:4px solid var(--gold); padding:.85rem; box-shadow:2px 2px 0 rgba(17,17,15,.07); }
          ::selection { color:var(--black); background:var(--gold); }
          @media(max-width:800px) { .hero { padding:1.2rem; box-shadow:3px 3px 0 rgba(199,163,74,.4); } .hero h1 { font-size:2.2rem; } .sidebar-spacer { height:3rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
