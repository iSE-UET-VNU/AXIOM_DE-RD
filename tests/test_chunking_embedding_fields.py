from __future__ import annotations

import json
from pathlib import Path

from src.chunking_embedding import FieldContext, route_document

FIXTURE = Path(__file__).parent / "fixtures" / "indexing_enriched_data.json"


def _extraction() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))[0]["rows"][0]["extraction"]


def _ctx(**overrides) -> FieldContext:
    defaults = dict(
        doc_id="aaaa000011112222",
        chunker_name="paragraph",
        chunker_params={},
        max_rows_per_chunk=20,
        language="en",
    )
    defaults.update(overrides)
    return FieldContext(**defaults)


def _route(extraction: dict, ctx: FieldContext | None = None):
    return route_document(extraction, ctx or _ctx())


def test_routes_every_field_type() -> None:
    chunks = _route(_extraction())
    by_type = {}
    for chunk in chunks:
        by_type.setdefault(chunk.chunk_type, []).append(chunk)
    assert len(by_type["text_chunk"]) == 4  # 3 paragraphs + 1 formula
    assert len(by_type["table_chunk"]) == 1
    assert len(by_type["figure_chunk"]) == 1  # the empty second figure is skipped


def test_main_text_offsets_are_within_field() -> None:
    extraction = _extraction()
    main_text = extraction["main_text"]
    for chunk in _route(extraction):
        if chunk.field_path == "main_text":
            assert chunk.text == main_text[chunk.start:chunk.end]
            assert chunk.metadata["page"] == 0
            assert chunk.metadata["chunker"] == "paragraph"


def test_small_table_is_single_chunk() -> None:
    extraction = _extraction()
    chunks = [c for c in _route(extraction) if c.chunk_type == "table_chunk"]
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.field_path == "tables[0]"
    assert chunk.text == extraction["tables"][0]["content"]
    assert chunk.metadata["caption"] == "Canada-U.S. merchandise trade"
    assert chunk.metadata["n_rows"] == 5
    assert chunk.embedding_text().startswith("Canada-U.S. merchandise trade\n")


def test_large_table_row_groups_repeat_header() -> None:
    extraction = _extraction()
    content = extraction["tables"][0]["content"]
    chunks = [
        c
        for c in _route(extraction, _ctx(max_rows_per_chunk=2))
        if c.chunk_type == "table_chunk"
    ]
    assert len(chunks) == 3  # 5 rows -> groups of 2, 2, 1
    header = "| Year | Imports | Exports |\n|---|---|---|"
    for i, chunk in enumerate(chunks):
        assert chunk.text.startswith(header)
        assert chunk.metadata["row_group"] == i
        assert chunk.metadata["header_repeated"] is True
        assert chunk.text == f"{header}\n{content[chunk.start:chunk.end]}"
    assert [c.metadata["n_rows"] for c in chunks] == [2, 2, 1]
    assert "| 2016 |" in chunks[0].text and "| 2020 |" in chunks[2].text


def test_figures_compose_caption_and_description() -> None:
    chunks = [c for c in _route(_extraction()) if c.chunk_type == "figure_chunk"]
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.field_path == "figures[0]"
    assert "Monthly imports and exports" in chunk.text
    assert "Line graph showing trade" in chunk.text
    assert chunk.metadata["composed_from"] == ["caption", "description"]
    assert (chunk.start, chunk.end) == (0, len(chunk.text))


def test_formulas_are_atomic_chunks() -> None:
    chunks = [c for c in _route(_extraction()) if c.field_path.startswith("formulas")]
    assert len(chunks) == 1
    assert chunks[0].text == "surplus = exports - imports"
    assert chunks[0].metadata["component_type"] == "formula"


def test_empty_and_missing_fields_are_skipped_silently() -> None:
    extraction = {
        "main_text": "",
        "tables": [],
        "figures": None,
        "formulas": None,
        "language": "en",
    }
    assert _route(extraction) == []
    assert _route({}) == []


def test_chunk_ids_are_stable_across_runs_and_unique() -> None:
    extraction = _extraction()
    ids_a = [c.chunk_id for c in _route(extraction)]
    ids_b = [c.chunk_id for c in _route(extraction)]
    assert ids_a == ids_b
    assert len(ids_a) == len(set(ids_a))
    other_doc = [c.chunk_id for c in _route(extraction, _ctx(doc_id="ffff000011112222"))]
    assert set(ids_a).isdisjoint(other_doc)
