"""Build a visual debug bundle for selected Industrial zero-recall queries.

The bundle is intentionally self-contained: rendered page PNGs are stored next
to the HTML, while the HTML keeps the query, answer, extraction text, scores,
ranks and preparation metadata together.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - compatibility with older environments
    import fitz  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = REPO_ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20"
OUT_ROOT = BUNDLE_ROOT / "industrial_zero_recall_visualization"
DATASET_ROOT = REPO_ROOT / "data/raw/BENCHMARK"
PREP_PATH = BUNDLE_ROOT / "preparation/pages.jsonl"
ZERO_PATH = BUNDLE_ROOT / "queries/zero_recall_at20.jsonl"

CASE_CONFIG: dict[str, dict[str, Any]] = {
    "vidore_industrial::111": {
        "label": "q111 — environmental effects",
        "anchors": ["temperature", "pollution", "sand and dust", "fumes", "corrosion"],
    },
    "vidore_industrial::112": {
        "label": "q112 — corrosion categories",
        "anchors": ["uniform surface corrosion", "galvanic corrosion", "pitting corrosion"],
    },
    "vidore_industrial::239": {
        "label": "q239 — 25mm gun safety pin",
        "anchors": ["25mm gun", "safety pin", "safety pin stowage bracket"],
    },
    "vidore_industrial::102": {
        "label": "q102 — igniter installation visuals",
        "anchors": ["ignitor assembly installation steps", "ignitor dimensions", "ignitor"],
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact(value: str, limit: int = 900) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value if len(value) <= limit else value[:limit].rstrip() + " …"


def highlight(value: str, anchors: list[str]) -> str:
    escaped = html.escape(value or "")
    for anchor in sorted(anchors, key=len, reverse=True):
        pattern = re.escape(html.escape(anchor))
        escaped = re.sub(
            pattern,
            lambda match: f'<mark class="anchor">{match.group(0)}</mark>',
            escaped,
            flags=re.IGNORECASE,
        )
    return escaped


def safe_name(page_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", page_id)


TESSERACT = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def render_page(page: dict[str, Any], docs: dict[Path, Any], image_dir: Path) -> tuple[str, int, int]:
    relative = Path(page["relative_path"])
    pdf_path = DATASET_ROOT / relative
    output = image_dir / f"{safe_name(page['page_id'])}.png"
    if not output.exists():
        document = docs.get(pdf_path)
        if document is None:
            document = fitz.open(pdf_path)
            docs[pdf_path] = document
        pdf_page = document[page["page_index"]]
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
        pixmap.save(str(output))
    with fitz.open(str(output)) as rendered:
        width = int(rendered[0].rect.width)
        height = int(rendered[0].rect.height)
    return output.relative_to(OUT_ROOT).as_posix(), width, height


def ocr_anchor_boxes(image_path: Path, anchors: list[str]) -> tuple[list[dict[str, Any]], str]:
    """Locate answer/query anchor words on short or placeholder-heavy pages."""
    if not Path(TESSERACT).exists():
        return [], ""
    wanted = set(re.findall(r"[a-z0-9]+", " ".join(anchors).lower()))
    try:
        result = subprocess.run(
            [TESSERACT, str(image_path), "stdout", "--psm", "3", "tsv"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
        )
    except (OSError, subprocess.TimeoutExpired):
        return [], ""
    boxes: list[dict[str, Any]] = []
    words: list[str] = []
    for row in result.stdout.splitlines()[1:]:
        fields = row.split("\t")
        if len(fields) < 12:
            continue
        word = fields[11].strip()
        if not word:
            continue
        words.append(word)
        tokens = re.findall(r"[a-z0-9]+", word.lower())
        if not tokens or not any(token in wanted for token in tokens):
            continue
        try:
            boxes.append({
                "text": word,
                "left": int(fields[6]),
                "top": int(fields[7]),
                "width": int(fields[8]),
                "height": int(fields[9]),
            })
        except ValueError:
            continue
    return boxes, " ".join(words)


def prepare_page(
    page_id: str,
    prep: dict[str, dict[str, Any]],
    docs: dict[Path, Any],
    image_dir: Path,
    anchors: list[str],
    *,
    rank: int | None = None,
    score: float | None = None,
    is_gold: bool = False,
) -> dict[str, Any]:
    page = dict(prep[page_id])
    image_rel, image_width, image_height = render_page(page, docs, image_dir)
    page["image"] = image_rel
    page["image_width"] = image_width
    page["image_height"] = image_height
    page["rank"] = rank
    page["score"] = score
    page["is_gold"] = is_gold
    page["anchor_hits"] = [anchor for anchor in anchors if re.search(re.escape(anchor), page["text"], re.IGNORECASE)]
    page["ocr_boxes"] = []
    page["ocr_text"] = ""
    if len(page["text"]) < 350:
        page["ocr_boxes"], page["ocr_text"] = ocr_anchor_boxes(image_dir / Path(image_rel).name, anchors)
    page["text_html"] = highlight(page["text"], anchors)
    page["preview_html"] = highlight(compact(page["text"]), anchors)
    return page


def score_label(score: float | None) -> str:
    return "—" if score is None else f"{score:.6f}"


def page_card(page: dict[str, Any], *, full: bool) -> str:
    rank = "gold evidence" if page["is_gold"] else f"rank {page['rank']}"
    score = "" if page["is_gold"] else f" · score {score_label(page['score'])}"
    status = " · anchor found" if page["anchor_hits"] else ""
    text = page["text_html"] if full else page["preview_html"]
    image_class = "gold-image" if full else "noise-image"
    overlay = "".join(
        f'<span class="ocr-box" style="left:{box["left"] / page["image_width"] * 100:.3f}%;top:{box["top"] / page["image_height"] * 100:.3f}%;width:{box["width"] / page["image_width"] * 100:.3f}%;height:{box["height"] / page["image_height"] * 100:.3f}%" aria-label="OCR anchor {html.escape(box["text"])}"></span>'
        for box in page["ocr_boxes"]
    )
    image = f'<div class="image-wrap"><img class="{image_class}" src="{html.escape(page["image"])}" alt="Rendered {html.escape(page["page_id"])}">{overlay}</div>'
    ocr_note = ""
    if page["ocr_boxes"]:
        words = ", ".join(box["text"] for box in page["ocr_boxes"])
        ocr_note = f'<div class="ocr-note">Red boxes: OCR located anchor words on the rendered page ({html.escape(words)}).</div>'
    return f"""
    <article class=\"page-card {'gold-card' if full else 'noise-card'}\">
      <div class=\"page-heading\">
        <div><strong>{html.escape(rank)}</strong>{html.escape(score)}{html.escape(status)}</div>
        <code>{html.escape(page['page_id'])}</code>
      </div>
      <div class=\"page-meta\">source={html.escape(page['text_source'])} · parse={html.escape(page['parse_status'])} · OCR={str(bool(page['ocr_applied'])).lower()} · extracted chars={len(page['text'])} · images/placeholders={page['text'].count('[Image:')}</div>
      <div class=\"page-body {'gold-body' if full else ''}\">
        {image}
        {ocr_note}
        <pre class=\"extracted\">{text}</pre>
      </div>
    </article>
    """


def case_section(case: dict[str, Any]) -> str:
    gold_cards = "\n".join(page_card(page, full=True) for page in case["gold_pages"])
    noise_cards = "\n".join(page_card(page, full=False) for page in case["top20"])
    selected = ", ".join(case["selected_files"])
    root_cause = {
        "file_retrieval": "Gold file is outside Kf=3.",
        "page_ranking": "Gold file is selected, but gold pages are below the top-20 page ranking.",
    }.get(case["failure_stage"], case["failure_stage"])
    return f"""
    <section class=\"case\" id=\"{html.escape(case['query_id'].replace(':', '-'))}\">
      <div class=\"case-title\">
        <div><span class=\"case-number\">{case['case_number']}</span><h2>{html.escape(case['label'])}</h2></div>
        <span class=\"stage\">{html.escape(case['failure_stage'])}</span>
      </div>
      <div class=\"query-answer\">
        <h3>Query</h3><p class=\"query\">{html.escape(case['query'])}</p>
        <h3>Gold answer</h3><p>{html.escape(' | '.join(case['answers']))}</p>
      </div>
      <div class=\"facts\">
        <div><b>Gold pages</b><span>{len(case['gold_pages'])}</span></div>
        <div><b>File selected?</b><span>{'yes' if case['candidate_recall'] else 'no'}</span></div>
        <div><b>Page recall@20</b><span>{case['page_recall_at20'] * 100:.2f}%</span></div>
        <div><b>Candidate pages</b><span>{case['candidate_page_count']}</span></div>
      </div>
      <p class=\"diagnosis\"><b>Failure interpretation:</b> {html.escape(root_cause)}<br><b>Selected files:</b> <code>{html.escape(selected)}</code></p>
      <h3>Gold evidence pages — rendered page + extracted text</h3>
      <p class=\"legend\">Yellow highlights show answer/query anchors found in the extracted text. If the image contains the answer but the text panel does not, that is a preparation loss.</p>
      <div class=\"gold-grid\">{gold_cards}</div>
      <details class=\"noise\"><summary>Top-20 retrieved pages and noise (click to expand)</summary><div class=\"noise-grid\">{noise_cards}</div></details>
    </section>
    """


def build() -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    image_dir = OUT_ROOT / "pages"
    image_dir.mkdir(parents=True, exist_ok=True)
    prep = {page["page_id"]: page for page in load_jsonl(PREP_PATH)}
    zero_rows = {row["query_id"]: row for row in load_jsonl(ZERO_PATH)}
    docs: dict[Path, Any] = {}
    cases: list[dict[str, Any]] = []

    for case_number, (query_id, config) in enumerate(CASE_CONFIG.items(), start=1):
        row = zero_rows[query_id]
        gold_ids = list(row["gold_pages"])
        top20 = row["components"]["top20"]
        gold = [
            prepare_page(page_id, prep, docs, image_dir, config["anchors"], is_gold=True)
            for page_id in gold_ids
        ]
        top = [
            prepare_page(
                item["page_id"],
                prep,
                docs,
                image_dir,
                config["anchors"],
                rank=int(item["rank"]),
                score=float(item["score"]),
                is_gold=False,
            )
            for item in top20
        ]
        case = {
            "case_number": case_number,
            "query_id": query_id,
            "label": config["label"],
            "anchors": config["anchors"],
            "query": row["query"],
            "answers": row["answers"],
            "failure_stage": row["failure_stage"],
            "candidate_recall": float(row["candidate_recall"]),
            "page_recall_at20": float(row["page_recall@20"]),
            "candidate_page_count": int(row["candidate_page_count"]),
            "selected_files": row["selected_file_ids"],
            "gold_pages": gold,
            "top20": top,
        }
        cases.append(case)

    (OUT_ROOT / "report.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    sections = "\n".join(case_section(case) for case in cases)
    nav = "".join(
        f'<button type="button" data-case="{case["query_id"].replace(":", "-")}">'
        f'{html.escape(case["label"])}</button>'
        for case in cases
    )
    html_doc = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Industrial zero-recall visual debug</title>
<style>
:root {{ color-scheme: light; --ink:#1d2733; --muted:#5f6b78; --line:#d8dee6; --soft:#f4f7fa; --gold:#fff2a8; --red:#b42318; --blue:#175cd3; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font:14px/1.45 Segoe UI,Arial,sans-serif; color:var(--ink); background:#f7f9fb; }}
main {{ max-width:1500px; margin:0 auto; padding:24px; }} h1 {{ margin:0 0 6px; font-size:28px; }} h2 {{ margin:0; font-size:20px; }} h3 {{ margin:18px 0 8px; font-size:16px; }} p {{ margin:8px 0; }} code {{ background:#eef2f6; padding:2px 4px; border-radius:3px; word-break:break-word; }}
.intro {{ color:var(--muted); margin-bottom:18px; }} .nav {{ display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 22px; }} .nav button {{ border:1px solid var(--line); background:#fff; color:var(--ink); padding:9px 12px; border-radius:6px; cursor:pointer; }} .nav button.active {{ background:#e8f0ff; border-color:#8db2f4; color:#114a9e; }}
.case {{ display:none; }} .case.active {{ display:block; }} .case-title {{ display:flex; justify-content:space-between; align-items:center; gap:12px; border-bottom:2px solid var(--ink); padding-bottom:10px; }} .case-title>div {{ display:flex; align-items:center; gap:10px; }} .case-number {{ display:inline-flex; width:30px; height:30px; align-items:center; justify-content:center; border-radius:50%; background:#e8f0ff; color:#114a9e; font-weight:600; }} .stage {{ color:var(--red); font-weight:600; text-transform:uppercase; font-size:12px; letter-spacing:.04em; }}
.query-answer {{ display:grid; grid-template-columns:140px 1fr; gap:2px 16px; padding:14px 0; }} .query-answer h3 {{ margin:0; color:var(--muted); font-size:13px; }} .query-answer p {{ margin:0 0 8px; }} .query {{ font-size:17px; font-weight:500; }}
.facts {{ display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); gap:10px; margin:8px 0 12px; }} .facts div {{ background:#fff; border:1px solid var(--line); border-radius:7px; padding:10px 12px; }} .facts b {{ display:block; color:var(--muted); font-size:12px; font-weight:400; }} .facts span {{ display:block; margin-top:3px; font-size:18px; font-weight:600; }} .diagnosis {{ background:#fffaf0; border-left:4px solid #d99b00; padding:10px 12px; }} .legend {{ color:var(--muted); }}
.gold-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:14px; }} .noise-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:12px; margin-top:14px; }} .page-card {{ background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; }} .gold-card {{ border-top:4px solid #d99b00; }} .noise-card {{ border-top:3px solid #b6c0cc; }} .page-heading {{ display:flex; justify-content:space-between; gap:8px; padding:9px 11px 5px; font-size:13px; }} .page-heading code {{ text-align:right; max-width:65%; }} .page-meta {{ color:var(--muted); font-size:11px; padding:0 11px 9px; }} .page-body {{ display:flex; gap:12px; align-items:flex-start; padding:0 11px 12px; }} .gold-body {{ display:block; }} .image-wrap {{ position:relative; display:block; line-height:0; background:#f0f2f5; }} .image-wrap img {{ display:block; }} .gold-image {{ width:100%; max-height:680px; object-fit:contain; object-position:top; }} .noise-image {{ width:220px; max-height:300px; object-fit:contain; object-position:top; flex:0 0 220px; }} .ocr-box {{ position:absolute; border:2px solid #d92d20; background:rgba(255, 0, 0, .10); pointer-events:none; }} .ocr-note {{ color:#b42318; font-size:11px; padding:6px 0; line-height:1.3; }} .extracted {{ white-space:pre-wrap; overflow-wrap:anywhere; background:var(--soft); border:1px solid #e7ebf0; padding:9px; margin:0; max-height:420px; overflow:auto; font:12px/1.45 Consolas,monospace; }} .gold-body .extracted {{ max-height:340px; margin-top:10px; }} mark.anchor {{ background:var(--gold); color:inherit; padding:0 1px; }} details.noise {{ margin-top:22px; border-top:1px solid var(--line); padding-top:14px; }} summary {{ cursor:pointer; font-weight:600; color:var(--blue); }}
@media (max-width:720px) {{ main {{ padding:14px; }} h1 {{ font-size:23px; }} .query-answer {{ grid-template-columns:1fr; gap:4px; }} .facts {{ grid-template-columns:repeat(2,minmax(120px,1fr)); }} .gold-grid {{ grid-template-columns:1fr; }} .page-body {{ display:block; }} .noise-image {{ width:100%; max-height:420px; }} .page-heading {{ display:block; }} .page-heading code {{ display:block; max-width:100%; text-align:left; margin-top:5px; }} }}
</style></head><body><main>
<h1>Industrial recall@20 = 0 — visual evidence debugger</h1>
<p class=\"intro\">Default hierarchical BM25 · Kf=3 · top-20 saved. Gold pages are shown in full; retrieved top-20 pages are collapsed below each case. Yellow marks indicate query/answer anchors found in extracted text.</p>
<nav class=\"nav\" aria-label=\"Cases\">{nav}</nav>
{sections}
</main><script>
const buttons=[...document.querySelectorAll('[data-case]')]; const cases=[...document.querySelectorAll('.case')];
function show(id){{ buttons.forEach(b=>b.classList.toggle('active',b.dataset.case===id)); cases.forEach(c=>c.classList.toggle('active',c.id===id)); }}
buttons.forEach(b=>b.addEventListener('click',()=>show(b.dataset.case))); show(cases[0].id);
</script></body></html>"""
    output = OUT_ROOT / "index.html"
    output.write_text(html_doc, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build())
