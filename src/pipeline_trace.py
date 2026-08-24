"""Helpers for visualizing pipeline stages and chunking inputs/outputs."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("ingested", "cleaned", "enriched", "embedded", "output")

CALL_SEQUENCE = [
    ("run_pipeline", "Orchestrates the full batch and writes stage artifacts."),
    ("ingestion.run", "Parses raw files into ParsedData and quarantine records."),
    ("parse_raw_file", "Chooses Lift API or local parsing for one source file."),
    ("cleaning.run", "Passes parsed rows through the cleaning contract."),
    ("enrichment.run", "Passes cleaned rows through the enrichment contract."),
    ("chunking_embedding.run", "Chunks enriched extractions and embeds them into retrieval records."),
    ("integration.run", "Applies the integration pass-through stage."),
    ("artifacts.write_*", "Writes the ingested, cleaned, enriched, embedded, and output artifacts."),
]


@dataclass(frozen=True)
class TraceSummary:
    run_id: str
    project_root: str
    stages: list[dict[str, Any]]
    call_sequence: list[dict[str, str]]
    chunking_embedding: dict[str, Any] | None = None


def build_trace_summary(
    run_id: str,
    *,
    project_root: str | Path = PROJECT_ROOT,
    chunk_dir: str | Path | None = None,
) -> TraceSummary:
    root = Path(project_root).resolve()
    stage_entries = [
        _stage_entry(root, stage, run_id)
        for stage in STAGES
    ]
    chunking = _chunking_entry(root, chunk_dir) if chunk_dir else None
    return TraceSummary(
        run_id=run_id,
        project_root=str(root),
        stages=stage_entries,
        call_sequence=[
            {"function": name, "purpose": purpose}
            for name, purpose in CALL_SEQUENCE
        ],
        chunking_embedding=chunking,
    )


def export_enriched_data(
    run_id: str,
    *,
    project_root: str | Path = PROJECT_ROOT,
    output_dir: str | Path | None = None,
) -> Path:
    """Write ``enriched_data.json`` from a completed pipeline run.

    The chunking stage consumes enrichment output, so this reads the enriched
    stage rather than the ingested one. Flattening the per-document artifacts
    into a single list is the only thing this adds: ``run_cli`` can also read
    ``data/enriched/<run_id>/`` directly.
    """
    root = Path(project_root).resolve()
    enriched_dir = root / "data" / "enriched" / run_id / "documents"
    if not enriched_dir.is_dir():
        raise FileNotFoundError(f"Enriched documents directory does not exist: {enriched_dir}")

    enriched_records: list[dict[str, Any]] = []
    skipped = 0
    for path in sorted(enriched_dir.glob("*.json")):
        payload = _load_json(path)
        record = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(record, dict):
            enriched_records.append(record)
        else:
            skipped += 1

    if output_dir is None:
        output_root = root / "data" / "work" / f"chunking_embedding_{run_id}"
    else:
        output_root = Path(output_dir)
        if not output_root.is_absolute():
            output_root = root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    enriched_path = output_root / "enriched_data.json"
    enriched_path.write_text(
        json.dumps(enriched_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "run_id": run_id,
        "source_dir": str(enriched_dir),
        "enriched_count": len(enriched_records),
        "skipped_count": skipped,
        "enriched_data_path": str(enriched_path),
    }
    (output_root / "enriched_data.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_root


def render_trace_html(summary: TraceSummary) -> str:
    stage_cards = "".join(_render_stage_card(stage) for stage in summary.stages)
    call_rows = "".join(
        f"<tr><td>{escape(item['function'])}</td><td>{escape(item['purpose'])}</td></tr>"
        for item in summary.call_sequence
    )
    chunking_section = ""
    if summary.chunking_embedding:
        chunking_section = _render_chunking_card(summary.chunking_embedding)

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pipeline Trace {escape(summary.run_id)}</title>
  <style>
    :root {{ --bg:#f5f7fb; --surface:#fff; --ink:#17212b; --mut:#5f6b78; --line:#d9e1ea; --accent:#0b7285; --accent2:#eef7f8; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f1520; --surface:#16202c; --ink:#e6eef7; --mut:#98a6b5; --line:#253241; --accent:#45c1d1; --accent2:#10353b; }} }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height:1.5; }}
    .wrap {{ max-width:1180px; margin:0 auto; padding:32px 24px 96px; }}
    h1 {{ margin:0 0 6px; font-size:26px; }}
    .sub {{ margin:0 0 26px; color:var(--mut); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12.5px; word-break: break-all; }}
    h2 {{ margin:28px 0 12px; font-size:15px; text-transform:uppercase; letter-spacing:.06em; color:var(--mut); }}
    .card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin:12px 0; }}
    .card h3 {{ margin:0 0 10px; font-size:15px; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:8px 16px; color:var(--mut); font-size:13px; margin-bottom:10px; }}
    .pill {{ background:var(--accent2); color:var(--accent); border-radius:999px; padding:2px 8px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:11px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ border-top:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
    th {{ color:var(--mut); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }}
    pre {{ margin:10px 0 0; background:rgba(127,127,127,.07); border:1px solid var(--line); border-radius:10px; padding:12px; overflow:auto; }}
    details summary {{ cursor:pointer; color:var(--accent); font-weight:600; }}
    .samples {{ display:grid; grid-template-columns:1fr; gap:10px; }}
    .sample {{ border:1px solid var(--line); border-radius:10px; padding:10px 12px; background:rgba(127,127,127,.04); }}
    .sample h4 {{ margin:0 0 6px; font-size:13px; }}
    .kv {{ display:grid; grid-template-columns:180px 1fr; gap:6px 12px; font-size:12.5px; }}
    .kv div:nth-child(odd) {{ color:var(--mut); }}
    .mut {{ color:var(--mut); }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Pipeline Trace</h1>
    <p class="sub">{escape(summary.run_id)} · {escape(summary.project_root)}</p>

    <h2>Call Sequence</h2>
    <div class="card">
      <table>
        <thead><tr><th>Function</th><th>Role</th></tr></thead>
        <tbody>{call_rows}</tbody>
      </table>
    </div>

    <h2>Stages</h2>
    {stage_cards}

    {chunking_section}
  </div>
</body>
</html>"""


def write_trace_report(
    run_id: str,
    *,
    project_root: str | Path = PROJECT_ROOT,
    chunk_dir: str | Path | None = None,
    out: str | Path | None = None,
) -> tuple[Path, Path]:
    summary = build_trace_summary(run_id, project_root=project_root, chunk_dir=chunk_dir)
    root = Path(project_root).resolve()
    if out is None:
        out_path = root / "data" / "output" / "traces" / run_id
    else:
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = root / out_path
    if out_path.suffix.lower() == ".html":
        html_path = out_path
        json_path = out_path.with_suffix(".json")
        html_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        html_path = out_path / "report.html"
        json_path = out_path / "report.json"

    html_path.write_text(render_trace_html(summary), encoding="utf-8")
    json_path.write_text(
        json.dumps(_summary_to_dict(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return html_path, json_path


def _render_stage_card(stage: dict[str, Any]) -> str:
    status = "missing" if stage.get("missing") else "present"
    meta = stage.get("metadata") or {}
    samples = stage.get("samples") or []
    progress = stage.get("progress")
    document_count = stage.get("document_count")
    sample_html = "".join(_render_sample(sample) for sample in samples) if samples else '<div class="mut">No samples available.</div>'
    return f"""
    <div class="card">
      <h3>{escape(stage['name'])}</h3>
      <div class="meta">
        <span class="pill">{escape(status)}</span>
        <span>{escape(str(stage.get('metadata_path') or 'n/a'))}</span>
        <span>{escape(f"documents: {document_count}")}</span>
      </div>
      {f'<div class="meta"><span>progress: {escape(json.dumps(progress, ensure_ascii=False))}</span></div>' if progress else ''}
      <details open>
        <summary>metadata</summary>
        <pre>{escape(json.dumps(meta, indent=2, ensure_ascii=False))}</pre>
      </details>
      <h4>samples</h4>
      <div class="samples">{sample_html}</div>
    </div>"""


def _render_sample(sample: dict[str, Any]) -> str:
    rows = "".join(
        f"<div>{escape(str(key))}</div><div>{escape(_stringify(value))}</div>"
        for key, value in sample.items()
    )
    return f'<div class="sample"><div class="kv">{rows}</div></div>'


def _render_chunking_card(chunking: dict[str, Any]) -> str:
    samples = "".join(_render_sample(sample) for sample in chunking.get("samples", []))
    return f"""
    <h2>Chunking Embedding</h2>
    <div class="card">
      <div class="meta">
        <span class="pill">{escape(str(chunking.get("docs", {}).get("indexed", 0)))} indexed</span>
        <span>{escape(str(chunking.get("chunk_dir") or 'n/a'))}</span>
      </div>
      <details open>
        <summary>embedding report</summary>
        <pre>{escape(json.dumps(chunking.get("report", {}), indent=2, ensure_ascii=False))}</pre>
      </details>
      <h4>sample chunks</h4>
      <div class="samples">{samples or '<div class="mut">No chunk samples available.</div>'}</div>
    </div>"""


def _stage_entry(root: Path, stage: str, run_id: str) -> dict[str, Any]:
    stage_dir = root / "data" / stage / run_id
    metadata_path = stage_dir / "metadata.json"
    if not metadata_path.is_file():
        return {
            "name": stage,
            "metadata_path": str(metadata_path),
            "missing": True,
            "document_count": 0,
            "metadata": {},
            "samples": [],
        }

    metadata = _load_json(metadata_path)
    documents_dir = stage_dir / "documents"
    samples = _stage_samples(stage, documents_dir, metadata)
    return {
        "name": stage,
        "metadata_path": str(metadata_path),
        "missing": False,
        "document_count": metadata.get("document_count"),
        "progress": metadata.get("progress"),
        "metadata": metadata,
        "samples": samples,
    }


def _stage_samples(stage: str, documents_dir: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if stage == "ingested":
        return [_summarize_ingested_document(documents_dir / f"{item['document_id']}.json") for item in _first_docs(metadata)]
    if stage == "cleaned":
        return [_summarize_cleaned_document(documents_dir / f"{item['document_id']}.json") for item in _first_docs(metadata)]
    if stage == "enriched":
        return [_summarize_enriched_document(documents_dir / f"{item['document_id']}.json") for item in _first_docs(metadata)]
    if stage == "embedded":
        return [_summarize_embedded_document(documents_dir / f"{item['document_id']}.json") for item in _first_docs(metadata)]
    if stage == "output":
        return [_summarize_output_document(documents_dir / f"{item['document_id']}.json") for item in _first_docs(metadata)]
    return []


def _first_docs(metadata: dict[str, Any], limit: int = 2) -> list[dict[str, Any]]:
    documents = metadata.get("documents")
    if not isinstance(documents, list):
        return []
    out: list[dict[str, Any]] = []
    for item in documents[:limit]:
        if isinstance(item, dict):
            out.append(item)
    return out


def _summarize_ingested_document(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    parsed = payload.get("parsed") if isinstance(payload.get("parsed"), dict) else {}
    failure = payload.get("failure") if isinstance(payload.get("failure"), dict) else {}
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    rows = parsed.get("rows") if isinstance(parsed.get("rows"), list) else []
    metadata = parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
    return {
        "document_id": payload.get("document_id"),
        "status": payload.get("status"),
        "source_uri": source.get("uri") or payload.get("source_uri"),
        "content_type": source.get("content_type") or payload.get("content_type"),
        "file_name": source.get("metadata", {}).get("file_name") if isinstance(source.get("metadata"), dict) else None,
        "rows": len(rows),
        "text_chars": len(parsed.get("text") or ""),
        "parser": metadata.get("parser"),
        "page_count": metadata.get("page_count"),
        "failure_codes": [reason.get("code") for reason in failure.get("reasons", []) if isinstance(reason, dict)] if failure else [],
    }


def _summarize_cleaned_document(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    rows = data.get("rows") if isinstance(data.get("rows"), list) else []
    issues = data.get("issues") if isinstance(data.get("issues"), list) else []
    return {
        "document_id": payload.get("document_id"),
        "schema_id": payload.get("schema_id"),
        "rows": len(rows),
        "issues": len(issues),
        "source_format": data.get("metadata", {}).get("source_format") if isinstance(data.get("metadata"), dict) else None,
    }


def _summarize_enriched_document(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    rows = data.get("rows") if isinstance(data.get("rows"), list) else []
    annotations = data.get("annotations") if isinstance(data.get("annotations"), dict) else {}
    return {
        "document_id": payload.get("document_id"),
        "schema_id": payload.get("schema_id"),
        "rows": len(rows),
        "annotation_keys": sorted(annotations)[:8],
    }


def _summarize_embedded_document(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    retrieval = payload.get("retrieval") if isinstance(payload.get("retrieval"), dict) else {}
    items = retrieval.get("items") if isinstance(retrieval.get("items"), list) else []
    item_types = sorted({item.get("type") for item in items if isinstance(item, dict)})
    return {
        "document_id": payload.get("document_id"),
        "item_count": len(items),
        "item_types": item_types,
    }


def _summarize_output_document(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
    retrieval = payload.get("retrieval") if isinstance(payload.get("retrieval"), dict) else {}
    items = retrieval.get("items") if isinstance(retrieval.get("items"), list) else []
    blocks = content.get("blocks") if isinstance(content.get("blocks"), list) else []
    reading_order = content.get("reading_order") if isinstance(content.get("reading_order"), list) else []
    return {
        "document_id": document.get("document_id"),
        "title": document.get("title"),
        "document_type": document.get("document_type"),
        "language": document.get("language"),
        "blocks": len(blocks),
        "reading_order": len(reading_order),
        "retrieval_items": len(items),
    }


def _chunking_entry(root: Path, chunk_dir: str | Path) -> dict[str, Any]:
    chunk_root = Path(chunk_dir)
    if not chunk_root.is_absolute():
        chunk_root = root / chunk_root
    report_path = chunk_root / "embedding_report.json"
    chunks_path = chunk_root / "chunk_records.json"
    vectors_path = chunk_root / "vector_records.json"
    if not report_path.is_file():
        return {
            "chunk_dir": str(chunk_root),
            "report": {},
            "docs": {},
            "samples": [],
        }
    report = _load_json(report_path)
    chunks = _load_json(chunks_path) if chunks_path.is_file() else []
    vectors = _load_json(vectors_path) if vectors_path.is_file() else []
    samples = []
    for chunk in chunks[:3] if isinstance(chunks, list) else []:
        if isinstance(chunk, dict):
            samples.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_type": chunk.get("chunk_type"),
                    "field_path": chunk.get("field_path"),
                    "chars": f"{chunk.get('char_start')}..{chunk.get('char_end')}",
                    "content": _truncate(chunk.get("content")),
                    "config_hash": chunk.get("metadata", {}).get("config_hash") if isinstance(chunk.get("metadata"), dict) else None,
                }
            )
    return {
        "chunk_dir": str(chunk_root),
        "report": report,
        "docs": report.get("docs", {}) if isinstance(report, dict) else {},
        "chunk_count": len(chunks) if isinstance(chunks, list) else 0,
        "vector_count": len(vectors) if isinstance(vectors, list) else 0,
        "samples": samples,
    }


def _summary_to_dict(summary: TraceSummary) -> dict[str, Any]:
    return {
        "run_id": summary.run_id,
        "project_root": summary.project_root,
        "stages": summary.stages,
        "call_sequence": summary.call_sequence,
        "chunking_embedding": summary.chunking_embedding,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _truncate(value: Any, limit: int = 220) -> str:
    text = _stringify(value).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"
