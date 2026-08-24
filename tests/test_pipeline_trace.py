from __future__ import annotations

import json
from pathlib import Path

from src.pipeline_trace import build_trace_summary, export_enriched_data, write_trace_report


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stage_metadata(run_id: str, document_id: str) -> dict:
    return {
        "contract_version": "stage-metadata-v1",
        "run_id": run_id,
        "stage": "test",
        "document_count": 1,
        "documents": [{"document_id": document_id}],
        "schema": {"document_file": {}, "records": {}},
        "progress": {"status": "completed", "processed_document_count": 1},
    }


def _build_run_tree(root: Path, run_id: str, document_id: str) -> Path:
    data_root = root / "data"
    ingested_dir = data_root / "ingested" / run_id
    cleaned_dir = data_root / "cleaned" / run_id
    enriched_dir = data_root / "enriched" / run_id
    embedded_dir = data_root / "embedded" / run_id
    output_dir = data_root / "output" / run_id

    _write_json(
        ingested_dir / "metadata.json",
        _stage_metadata(run_id, document_id) | {"stage": "ingested"},
    )
    _write_json(
        ingested_dir / "documents" / f"{document_id}.json",
        {
            "contract_version": "ingested-document-v2",
            "stage": "ingested",
            "status": "succeeded",
            "document_id": document_id,
            "source": {
                "uri": "data/subset/paper1.png",
                "content_type": "image/png",
                "metadata": {"file_name": "paper1.png"},
            },
            "parsed": {
                "object_id": document_id,
                "source_uri": "data/subset/paper1.png",
                "source_format": "png",
                "rows": [{"extraction": {"main_text": "hello"}}],
                "text": "hello",
                "metadata": {"parser": "lift-api", "page_count": 1},
            },
            "schema_id": "schema-1",
            "failure": None,
        },
    )

    _write_json(
        cleaned_dir / "metadata.json",
        _stage_metadata(run_id, document_id) | {"stage": "cleaned"},
    )
    _write_json(
        cleaned_dir / "documents" / f"{document_id}.json",
        {
            "contract_version": "cleaned-document-v1",
            "stage": "cleaned",
            "document_id": document_id,
            "data": {"rows": [{"x": 1}], "issues": [], "metadata": {"source_format": "png"}},
            "schema_id": "cleaned-1",
        },
    )

    _write_json(
        enriched_dir / "metadata.json",
        _stage_metadata(run_id, document_id) | {"stage": "enriched"},
    )
    _write_json(
        enriched_dir / "documents" / f"{document_id}.json",
        {
            "contract_version": "enriched-document-v1",
            "stage": "enriched",
            "document_id": document_id,
            "data": {
                "source_object_id": document_id,
                "rows": [{"extraction": {"main_text": "hello"}}],
                "annotations": {"foo": "bar"},
                "profile": {},
                "metadata": {"parser": "lift-api", "source_uri": "data/subset/paper1.png"},
            },
            "schema_id": "enriched-1",
        },
    )

    _write_json(
        embedded_dir / "metadata.json",
        _stage_metadata(run_id, document_id) | {"stage": "embedded"},
    )
    _write_json(
        embedded_dir / "documents" / f"{document_id}.json",
        {
            "contract_version": "embedded-document-v3",
            "stage": "embedded",
            "document_id": document_id,
            "retrieval": {
                "items": [
                    {"item_id": "a", "type": "text", "position": 0, "content": "hello", "embeddings": []},
                ]
            },
        },
    )

    _write_json(
        output_dir / "metadata.json",
        _stage_metadata(run_id, document_id) | {"stage": "output"},
    )
    _write_json(
        output_dir / "documents" / f"{document_id}.json",
        {
            "contract_version": "output-document-v4",
            "document": {
                "document_id": document_id,
                "source_uri": "data/subset/paper1.png",
                "file_name": "paper1.png",
                "content_type": "image/png",
                "title": "Test Doc",
                "document_type": "paper",
                "language": "en",
            },
            "content": {
                "main_text": "hello",
                "tables": [],
                "figures": [],
                "formulas": [],
                "blocks": [],
                "reading_order": [],
                "reading_order_meta": {"source": "parser_json", "complete": True, "block_count": 1},
            },
            "retrieval": {
                "items": [
                    {"item_id": "a", "type": "text", "position": 0, "content": "hello", "embeddings": []},
                ]
            },
            "lineage": {"run_id": run_id, "status": "succeeded", "schema_ids": {}, "completed_stages": []},
        },
    )

    chunk_dir = data_root / "output" / f"chunking_embedding_{run_id}"
    _write_json(
        chunk_dir / "chunk_records.json",
        [
            {
                "chunk_id": "chunk-1",
                "chunk_type": "text_chunk",
                "doc_id": document_id,
                "field_path": "main_text",
                "content": "hello",
                "char_start": 0,
                "char_end": 5,
                "level": 0,
                "parent_id": None,
                "children": [],
                "metadata": {"config_hash": "abc", "contract_version": "chunk-embed-v1"},
            }
        ],
    )
    _write_json(
        chunk_dir / "vector_records.json",
        [{"id": "chunk-1", "vector": [0.1, 0.2], "model": "local-hash-embedding-v1", "dim": 2}],
    )
    _write_json(
        chunk_dir / "embedding_report.json",
        {
            "docs": {"total": 1, "indexed": 1, "skipped_existing": 0, "skipped_failed": 0},
            "chunk_counts": {"text_chunk": 1},
            "report": "ok",
        },
    )
    return chunk_dir


def test_trace_report_exports_and_renders(tmp_path: Path) -> None:
    run_id = "run-1"
    document_id = "doc-1"
    chunk_dir = _build_run_tree(tmp_path, run_id, document_id)

    enriched_out = export_enriched_data(run_id, project_root=tmp_path, output_dir=tmp_path / "data" / "work" / "chunking_embedding_run-1")
    enriched_data = json.loads((enriched_out / "enriched_data.json").read_text())
    assert len(enriched_data) == 1
    assert enriched_data[0]["source_object_id"] == document_id
    assert enriched_data[0]["metadata"]["parser"] == "lift-api"

    summary = build_trace_summary(run_id, project_root=tmp_path, chunk_dir=chunk_dir)
    assert summary.run_id == run_id
    assert len(summary.stages) == 5
    assert summary.chunking_embedding is not None
    assert summary.chunking_embedding["docs"]["indexed"] == 1

    html_path, json_path = write_trace_report(
        run_id,
        project_root=tmp_path,
        chunk_dir=chunk_dir,
        out=tmp_path / "trace_out",
    )
    html = html_path.read_text(encoding="utf-8")
    trace_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert "Pipeline Trace" in html
    assert "Chunking Embedding" in html
    assert trace_json["run_id"] == run_id
    assert trace_json["chunking_embedding"]["docs"]["indexed"] == 1
