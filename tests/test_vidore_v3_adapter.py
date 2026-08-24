"""ViDoRe V3 adapter, against fixtures written in the published schema."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.evaluation.benchmarks import load  # noqa: E402
from src.evaluation.benchmarks.base import Benchmark  # noqa: E402

BBOX = pa.list_(
    pa.struct([("annotator", pa.int64()), ("x1", pa.int64()),
               ("x2", pa.int64()), ("y1", pa.int64()), ("y2", pa.int64())])
)


def _write_subset(root: Path, subset: str, *, marker: str) -> None:
    """One subset in the real schema."""
    path = root / subset
    path.mkdir(parents=True)

    pq.write_table(
        pa.table({
            "corpus_id": [10, 20, 30],
            # Present and unread: projecting it is a ~12 GB download.
            "image": [b"\x89PNG-fake", b"\x89PNG-fake", b"\x89PNG-fake"],
            "doc_id": [f"{marker}_report", f"{marker}_report", f"{marker}_deck"],
            "markdown": [f"{marker} cover page", f"{marker} revenue was 12.4M",
                         f"{marker} chart trends upward"],
            "page_number_in_doc": [0, 1, 0],
        }),
        path / "corpus.parquet",
    )

    # 0/1 English, 2/3 their French translations: each variant has its own id.
    pq.write_table(
        pa.table({
            "query_id": [0, 1, 2, 3],
            "query": [f"{marker} revenue?", f"{marker} trend?",
                      f"{marker} chiffre d'affaires?", f"{marker} tendance?"],
            "language": ["english", "english", "french", "french"],
            "query_types": [["extractive"], ["open-ended"],
                            ["extractive"], ["open-ended"]],
            "query_format": ["question"] * 4,
            "content_type": [["Table"], ["Chart"], ["Table"], ["Chart"]],
            "raw_answers": [["12.4M"], ["upward"], ["12,4M"], ["hausse"]],
            "source_type": ["summary"] * 4,
            "answer": ["12.4M", "upward", "12,4M", "hausse"],
        }),
        path / "queries.parquet",
    )

    # Graded 2 vs 1 on the same query, so binary and graded metrics diverge.
    pq.write_table(
        pa.table({
            "query_id": [0, 0, 1, 2, 2, 3],
            "corpus_id": [20, 10, 30, 20, 10, 30],
            "score": [2, 1, 2, 2, 1, 2],
            "content_type": [["Table"], ["Text"], ["Chart"],
                             ["Table"], ["Text"], ["Chart"]],
            "bounding_boxes": pa.array(
                [[{"annotator": 0, "x1": 10, "x2": 90, "y1": 10, "y2": 50}]] * 6,
                type=BBOX,
            ),
        }),
        path / "qrels.parquet",
    )

    pq.write_table(
        pa.table({
            "file_name": [f"{marker}_report.pdf", f"{marker}_deck.pdf"],
            "doc_id": [f"{marker}_report", f"{marker}_deck"],
            "doc_type": ["report", "slides"],
            "doc_language": ["english", "english"],
            "page_number": [2, 1],
        }),
        path / "documents_metadata.parquet",
    )


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("vidore_v3")
    _write_subset(path, "hr", marker="hr")
    _write_subset(path, "energy", marker="en")
    return path


def bench(root: Path, subset: str = "hr", language: str = "english"):
    return load("vidore_v3", root=root, subset=subset, language=language)


# -- seam: Benchmark protocol vs adapter ---------------------------------------


def test_conforms(root: Path):
    assert isinstance(bench(root), Benchmark)


# -- seam: language filter vs qrel drop ----------------------------------------


def test_language_is_required_with_no_default(root: Path):
    """A multilingual average must be impossible to produce by accident."""
    with pytest.raises(ValueError, match="language"):
        load("vidore_v3", root=root, subset="hr")
    with pytest.raises(ValueError, match="language"):
        load("vidore_v3", root=root, subset="hr", language="all")


def test_language_filter_drops_the_other_languages_qrels(root: Path):
    """Filter queries, then drop qrels whose query_id did not survive."""
    english = bench(root, language="english")
    assert [q.qid for q in english.questions()] == ["hr::0", "hr::1"]
    assert sum(len(v) for v in english.qrels().values()) == 3

    french = bench(root, language="french")
    assert [q.qid for q in french.questions()] == ["hr::2", "hr::3"]
    assert sum(len(v) for v in french.qrels().values()) == 3


# -- seam: qrels corpus_id vs corpus corpus_id ---------------------------------


def test_gold_units_resolve_through_the_corpus_id_join(root: Path):
    """``corpus_id`` is a key, not a row offset."""
    gold = bench(root).gold_docs("hr::0")
    assert set(gold.flat()) == {"hr::hr_report#page=1", "hr::hr_report#page=0"}


def test_join_rate_is_counted_not_assumed(root: Path):
    """If the join degrades, every number degrades with it and nothing errors."""
    benchmark = bench(root)
    benchmark.qrels()
    assert benchmark.join_stats == {"attempted": 3, "matched": 3, "unreachable": 0}


def test_unreachable_qrels_are_reported_not_dropped(root: Path, tmp_path: Path):
    """A qrel pointing outside the corpus must be counted, not silently skipped."""
    broken = tmp_path / "broken"
    _write_subset(broken, "hr", marker="hr")
    table = pq.read_table(broken / "hr" / "qrels.parquet")
    pq.write_table(
        table.set_column(
            table.schema.get_field_index("corpus_id"),
            "corpus_id",
            pa.array([999, 10, 30, 20, 10, 30], type=pa.int64()),
        ),
        broken / "hr" / "qrels.parquet",
    )
    benchmark = load("vidore_v3", root=broken, subset="hr", language="english")
    benchmark.qrels()
    assert benchmark.join_stats["unreachable"] == 1


# -- seam: per-subset id namespacing -------------------------------------------


def test_ids_from_two_subsets_never_collide(root: Path):
    """``query_id`` and ``corpus_id`` restart near zero in every subset."""
    hr, energy = bench(root, "hr"), bench(root, "energy")

    assert {q.qid for q in hr.questions()}.isdisjoint(q.qid for q in energy.questions())
    assert hr.qrels().keys().isdisjoint(energy.qrels().keys())

    hr_units = {d.doc_id for d in hr.corpus()}
    assert hr_units.isdisjoint(d.doc_id for d in energy.corpus())
    assert all(u.startswith("hr::") for u in hr_units)


# -- graded gold, shaped for pytrec_eval ---------------------------------------


def test_qrels_carry_the_graded_score_untranslated(root: Path):
    """NDCG@10 over graded relevance is the published metric."""
    qrels = bench(root).qrels()
    assert qrels["hr::0"] == {"hr::hr_report#page=1": 2, "hr::hr_report#page=0": 1}
    assert sorted(set(s for v in qrels.values() for s in v.values())) == [1, 2]


def test_qrels_feed_pytrec_eval_without_translation(root: Path):
    """The metric seam: our gold shape must be the one the official code uses."""
    pytrec_eval = pytest.importorskip("pytrec_eval")

    qrels = bench(root).qrels()
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})

    good = {"hr::0": {"hr::hr_report#page=1": 2.0, "hr::hr_report#page=0": 1.0}}
    bad = {"hr::0": {"hr::hr_report#page=1": 1.0, "hr::hr_report#page=0": 2.0}}
    hi = evaluator.evaluate(good)["hr::0"]["ndcg_cut_10"]
    lo = evaluator.evaluate(bad)["hr::0"]["ndcg_cut_10"]

    assert hi == pytest.approx(1.0)
    assert lo < hi


def test_corpus_units_match_the_qrel_key_space(root: Path):
    """A run keyed differently from qrels scores zero without erroring."""
    benchmark = bench(root)
    units = {d.doc_id for d in benchmark.corpus()}
    for graded in benchmark.qrels().values():
        assert set(graded).issubset(units)


# -- corpus projection ---------------------------------------------------------


def test_image_column_is_never_read(root: Path):
    """Materializing ``image`` is a ~12 GB download for a text-only run."""
    from src.evaluation.benchmarks.vidore_v3 import CORPUS_COLUMNS

    assert "image" not in CORPUS_COLUMNS
    assert [d.text for d in bench(root).corpus()][1] == "hr revenue was 12.4M"


def test_modality_and_language_reach_the_metric_layer(root: Path):
    questions = {q.qid: q for q in bench(root).questions()}
    assert questions["hr::0"].modalities == ("Table",)
    assert questions["hr::0"].taxonomy == "vidore_v3"
    assert questions["hr::0"].answer_type == "judge"
    assert bench(root).language == "english"


def test_regions_carry_bboxes_and_page(root: Path):
    regions = bench(root).gold_regions("hr::0")
    assert regions is not None
    assert regions[0].bbox == (10.0, 10.0, 90.0, 50.0)
    assert regions[0].doc_id == "hr::hr_report#page=1"


def test_unknown_subset_names_the_public_eight(root: Path):
    with pytest.raises(ValueError, match="finance_de"):
        load("vidore_v3", root=root, subset="finance_de", language="english")
