"""MMDocIR adapter, against fixtures written in the published schema.

The 11.2 GB dataset is not a test dependency, so these fixtures reproduce the
real field names from the dataset card: ``doc_name`` / ``questions[].Q|A|type|
page_id|layout_mapping`` in the annotations, ``passage_id|ocr_text|vlm_text`` in
the pages table, ``layout_id|page_id|type|bbox`` in the layouts table.

The behaviours pinned here are the ones that were wrong on the first pass and
would have produced quietly meaningless numbers.
"""

from __future__ import annotations

from pathlib import Path
import json

import pytest

pytest.importorskip("pyarrow")
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.evaluation.benchmarks import load  # noqa: E402
from src.evaluation.benchmarks.base import Benchmark  # noqa: E402

ANNOTATION = {
    "doc_name": "finance_report",
    "domain": "finance",
    "page_indices": [0, 2],
    # The dataset card spells this "layout_indinces"; kept verbatim.
    "layout_indinces": [0, 2],
    "questions": [
        {
            "Q": "What was Q3 revenue?",
            "A": "12.4M",
            "type": "table",
            "page_id": [1],
            "layout_mapping": [
                {"page": 1, "page_size": [612, 792], "bbox": [100.0, 200.0, 300.0, 260.0]}
            ],
        },
        {
            "Q": "Describe the chart trend.",
            "A": "upward",
            "type": "chart",
            "page_id": [2],
            "layout_mapping": [
                {"page": 2, "page_size": [612, 792], "bbox": [50.0, 50.0, 250.0, 150.0]}
            ],
        },
    ],
}


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("mmdocir")
    (path / "MMDocIR_annotations.jsonl").write_text(json.dumps(ANNOTATION), encoding="utf-8")
    pq.write_table(
        pa.table(
            {
                "doc_name": ["finance_report"] * 3,
                "domain": ["finance"] * 3,
                "passage_id": ["0", "1", "2"],
                "ocr_text": ["cover", "Q3 revenue 12.4M", "chart noise"],
                "vlm_text": ["cover", "Q3 revenue was 12.4M", "the chart trends upward"],
            }
        ),
        path / "MMDocIR_pages.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "doc_name": ["finance_report"] * 2,
                "domain": ["finance"] * 2,
                "type": ["table", "chart"],
                "layout_id": [10, 11],
                "page_id": [1, 2],
                "text": ["", ""],
                "ocr_text": ["Q3 revenue 12.4M", "chart noise"],
                "vlm_text": ["Q3 revenue was 12.4M", "the chart trends upward"],
                "bbox": [[100.0, 200.0, 300.0, 260.0], [50.0, 50.0, 250.0, 150.0]],
                "page_size": [[612.0, 792.0], [612.0, 792.0]],
            }
        ),
        path / "MMDocIR_layouts.parquet",
    )
    return path


def test_conforms(root: Path):
    assert isinstance(load("mmdocir", root=root), Benchmark)


def test_synthesized_qids_are_stable(root: Path):
    """The source has no question id; ours is positional and must not drift."""
    qids = [q.qid for q in load("mmdocir", root=root).questions()]
    assert qids == ["finance_report::0", "finance_report::1"]


def test_question_modality_reaches_the_metric_layer(root: Path):
    """The per-modality breakdown is the reason for running this dataset."""
    by_qid = {q.qid: q.modalities for q in load("mmdocir", root=root).questions()}
    assert by_qid["finance_report::0"] == ("table",)
    assert by_qid["finance_report::1"] == ("chart",)


def test_bbox_join_recovers_real_layout_ids_and_types(root: Path):
    """``layout_mapping`` carries a bbox, not a layout id -- it is a geometric join."""
    regions = load("mmdocir", root=root).gold_regions("finance_report::0")
    assert [r.region_id for r in regions] == ["finance_report#page=1#layout=10"]
    assert regions[0].modality == "table"


def test_region_doc_id_is_the_containing_page(root: Path):
    """Page-level retrieval can only return a page.

    Scoring region recall against a layout id reported 0 for a page that was
    correctly retrieved -- the bug this pins.
    """
    regions = load("mmdocir", root=root).gold_regions("finance_report::0")
    assert regions[0].doc_id == "finance_report#page=1"


def test_layout_units_carry_their_page(root: Path):
    """Without the page component, layout-level page recall reads 0."""
    ids = [d.doc_id for d in load("mmdocir", root=root, level="layout").corpus()]
    assert ids == ["finance_report#page=1#layout=10", "finance_report#page=2#layout=11"]


def test_regions_work_without_the_layouts_table(root: Path, tmp_path: Path):
    """Annotations alone are 770 kB; the layouts table is 2.5 GB.

    Falling back to a geometric region id keeps the modality breakdown working
    for anyone who has not downloaded the large file.
    """
    lean = tmp_path / "lean"
    lean.mkdir()
    (lean / "MMDocIR_annotations.jsonl").write_text(json.dumps(ANNOTATION), encoding="utf-8")
    regions = load("mmdocir", root=lean).gold_regions("finance_report::0")
    assert regions[0].region_id.endswith("#bbox=100.0,200.0,300.0,260.0")
    assert regions[0].modality == "table"


def test_scope_is_the_questions_own_document(root: Path):
    """MMDocIR retrieves inside one document, not across the corpus."""
    benchmark = load("mmdocir", root=root)
    assert benchmark.scope_for("finance_report::0") == [
        "finance_report#page=0",
        "finance_report#page=1",
        "finance_report#page=2",
    ]


def test_text_source_selects_the_published_comparison(root: Path):
    vlm = [d.text for d in load("mmdocir", root=root, text_source="vlm_text").corpus()]
    ocr = [d.text for d in load("mmdocir", root=root, text_source="ocr_text").corpus()]
    assert vlm != ocr
    assert "the chart trends upward" in vlm


def test_invalid_options_are_rejected(root: Path):
    with pytest.raises(ValueError, match="text_source"):
        load("mmdocir", root=root, text_source="raw_text")
    with pytest.raises(ValueError, match="level"):
        load("mmdocir", root=root, level="document")


def test_missing_dataset_names_the_download(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="MMDocIR_annotations.jsonl"):
        load("mmdocir", root=tmp_path / "absent")
