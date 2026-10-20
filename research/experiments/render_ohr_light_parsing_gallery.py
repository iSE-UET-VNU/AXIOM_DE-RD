"""Render an inline visual gallery for OHR gold evidence not exactly retained by light text."""
from __future__ import annotations

import base64
from html import escape
import io
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "data/work/ohr_replacement_20260920/retrieval/analysis/ohr_gold_evidence_in_light/per_query_gold_evidence.jsonl"
OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/analysis/ohr_gold_evidence_in_light/ohr-light-parsing-gallery.html"


def thumbnail_data_url(pdf: Path, page_number: int) -> str:
    with tempfile.TemporaryDirectory(prefix="ohr-gallery-") as raw:
        prefix = Path(raw) / "page"
        subprocess.run(
            ["pdftoppm", "-f", str(page_number), "-l", str(page_number), "-jpeg", "-r", "75", "-jpegopt", "quality=45,optimize=y", str(pdf), str(prefix)],
            check=True, capture_output=True,
        )
        source = next(Path(raw).glob("page-*.jpg"))
        with Image.open(source) as image:
            image.thumbnail((340, 440))
            stream = io.BytesIO()
            image.convert("RGB").save(stream, format="JPEG", quality=42, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def compact(value: str, length: int = 440) -> str:
    value = value.replace("\\n", " ").replace("\\r", " ")
    value = " ".join(value.split())
    return value if len(value) <= length else value[:length - 1] + "…"


def main() -> None:
    rows = [json.loads(line) for line in AUDIT.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if not row["answer_exact_in_light"]]
    rows.sort(key=lambda row: (row["content_state"], row["query_id"]))
    cards = []
    for index, row in enumerate(rows, start=1):
        pdf = Path(str(row["gold_page_id"]).replace("#page=" + str(row["gold_page_index"]), ""))
        # OHR canonical path is deterministic from the page/doc id; never use qrels to select it.
        filename = f"ohrbench__{row['gold_doc_id'].split('::', 1)[1]}.pdf"
        source_pdf = ROOT / "data/raw/0.1. BENCHMARK/data" / filename
        image = thumbnail_data_url(source_pdf, int(row["gold_page_index"]) + 1)
        state = row["content_state"].replace("_", " ")
        retrieve = row["retrieval_state"].replace("_", " ")
        cards.append(f'''<article class="evidence-card" data-state="{escape(row['content_state'])}">
  <div class="card-heading"><strong>{index:02d}. {escape(str(row['visual_class']))}</strong><span class="viz-badge">{escape(state)}</span></div>
  <div class="evidence-layout">
    <figure><img src="{image}" alt="Original gold evidence page for query {index}"><figcaption>Gold page {int(row['gold_page_index']) + 1}</figcaption></figure>
    <div class="evidence-copy">
      <p><strong>Query</strong><br>{escape(row['query'])}</p>
      <p><strong>Answer</strong><br>{escape(row['answer'])}</p>
      <p><strong>Annotated evidence</strong><br>{escape(compact(row['evidence_context'], 260))}</p>
      <p><strong>Light preparation</strong> <span class="text-muted">({escape(str(row['light_text_source']))}; {row['light_chars']} chars; OCR {row['light_ocr_word_count']} words)</span><br><code>{escape(compact(row['light_evidence_excerpt'])) or '∅'}</code></p>
      <p><strong>Legacy KDL</strong> <span class="text-muted">({row['kdl_chars']} chars)</span><br><code>{escape(compact(row['kdl_evidence_excerpt'])) or '∅'}</code></p>
      <p class="text-small"><strong>Retrieval:</strong> {escape(retrieve)}; hybrid candidate rank: {row['hybrid_candidate_rank_up_to_200'] or '—'}; Cohere top-20 rank: {row['cohere_final_rank_up_to_20'] or '—'}. Evidence-token coverage in light: {100 * row['evidence_token_coverage_in_light']:.0f}%.</p>
    </div>
  </div>
</article>''')
    fragment = f'''<div id="ohr-light-gallery">
  <h2>OHR: gold evidence not exactly retained by light preparation</h2>
  <div class="viz-controls" aria-label="Evidence filter">
    <label class="form-check"><input class="form-check-input" type="checkbox" id="show-partial" checked> <span class="form-check-label">Include partial evidence</span></label>
    <span class="text-muted" id="shown-count" aria-live="polite"></span>
  </div>
  <div id="gallery-list">{''.join(cards)}</div>
</div>
<style>
#ohr-light-gallery .evidence-card {{ margin: 16px 0; padding: 12px 0; border-bottom: 1px solid var(--border); }}
#ohr-light-gallery .card-heading {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }}
#ohr-light-gallery .evidence-layout {{ display:grid; grid-template-columns:minmax(180px, 34%) 1fr; gap:16px; align-items:start; }}
#ohr-light-gallery figure {{ margin:0; }}
#ohr-light-gallery img {{ width:100%; max-height:440px; object-fit:contain; background:var(--muted); }}
#ohr-light-gallery figcaption {{ color:var(--muted-foreground); }}
#ohr-light-gallery p {{ margin:0 0 10px; }}
#ohr-light-gallery code {{ white-space:normal; overflow-wrap:anywhere; }}
@media (max-width:600px) {{ #ohr-light-gallery .evidence-layout {{ grid-template-columns:1fr; }} }}
</style>
<script>
(() => {{
  const root = document.getElementById('ohr-light-gallery');
  const checkbox = root.querySelector('#show-partial');
  const cards = [...root.querySelectorAll('.evidence-card')];
  const count = root.querySelector('#shown-count');
  function update() {{
    let visible = 0;
    cards.forEach(card => {{
      const show = checkbox.checked || card.dataset.state !== 'partial_evidence_in_light';
      card.hidden = !show;
      if (show) visible += 1;
    }});
    count.textContent = `${{visible}} gold pages shown`;
  }}
  checkbox.addEventListener('change', update);
  update();
}})();
</script>'''
    if len(fragment.encode("utf-8")) > 1_000_000:
        raise RuntimeError(f"Gallery is {len(fragment.encode('utf-8'))} bytes; must stay under 1 MB")
    OUTPUT.write_text(fragment, encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "cards": len(rows), "bytes": len(fragment.encode('utf-8'))}, indent=2))


if __name__ == "__main__":
    main()
