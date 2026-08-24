"""The Benchmark adapter, and the fidelity gate on IseChallenge.

The refactor's job was to generalize the harness without changing what it
measures. These tests pin the parts where "generalize" could quietly become
"change": the grading route, the recall denominator, and which questions are in
the set.
"""

from __future__ import annotations

import pytest

from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.base import (
    Benchmark,
    GoldSpec,
    Question,
    normalize_answer_type,
)
from src.evaluation.evalset import read_jsonl

QUESTIONS = "data/benchmark/questions.jsonl"


@pytest.fixture(scope="module")
def ise():
    return load("ise", questions_path=QUESTIONS)


def test_adapter_conforms(ise):
    assert isinstance(ise, Benchmark)


def test_question_set_matches_the_raw_evalset(ise):
    raw = {q.qid for q in read_jsonl(QUESTIONS) if q.resolvable}
    assert {q.qid for q in ise.questions()} == raw


def test_grading_route_survives_the_answer_type_rename(ise):
    """``Exact Match`` became ``em``; the grader must route identically.

    Getting this wrong sends every exact-match question to the LLM judge, which
    changes the numbers the refactor was supposed to preserve.
    """
    raw = {q.qid: q.answer_type for q in read_jsonl(QUESTIONS) if q.resolvable}
    for question in ise.questions():
        was_exact = raw[question.qid].lower().startswith("exact")
        assert (question.answer_type == "em") is was_exact


def test_lake_reports_no_page_or_region_labels(ise):
    """None, not an empty list.

    Empty would average into the report as "retrieved no regions"; None means
    "this source has no region annotation", which is why MMDocIR is worth running.
    """
    assert ise.gold_pages("1") is None
    assert ise.gold_regions("1") is None


# --------------------------------------------------------------------------
# any-of semantics -- not exercised by the synthetic fixture, so pinned here
# --------------------------------------------------------------------------


def test_any_of_group_counts_once_not_once_per_member():
    """A directory reference is one piece of evidence.

    Flattening it would make q24's 45-file group demand all 45 documents and
    score a correct retrieval as 1/45.
    """
    spec = GoldSpec(docs=(), any_of=(tuple(f"f{i}.mp3" for i in range(45)),))
    assert spec.required == 1
    assert spec.recall(["f7.mp3"]) == 1.0
    assert spec.recall(["nope.mp3"]) == 0.0


def test_named_and_any_of_evidence_combine():
    spec = GoldSpec(docs=("a.pdf", "b.pdf"), any_of=(("x.png", "y.png"),))
    assert spec.required == 3
    assert spec.recall(["a.pdf"]) == pytest.approx(1 / 3)
    assert spec.recall(["a.pdf", "b.pdf", "y.png"]) == 1.0


def test_flat_dedupes_but_keeps_every_candidate():
    spec = GoldSpec(docs=("a.pdf",), any_of=(("a.pdf", "b.pdf"),))
    assert spec.flat() == ["a.pdf", "b.pdf"]


def test_real_any_of_question_is_satisfied_by_one_member(ise):
    """q24's evidence is ``IELTS-Listening/*`` -- 45 files, any one sufficient."""
    spec = ise.gold_docs("24")
    assert spec.required == 1 and len(spec.any_of[0]) == 45
    assert spec.recall([spec.any_of[0][0]]) == 1.0


# --------------------------------------------------------------------------
# answer-type normalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Exact Match", "em"),
        ("exact_match", "em"),
        ("LLM as a Judge", "judge"),
        ("multiple_choice", "mc"),
        ("", "judge"),
        ("something new", "judge"),
    ],
)
def test_answer_type_normalization(raw: str, expected: str):
    assert normalize_answer_type(raw) == expected


def test_unknown_answer_type_is_rejected_at_construction():
    with pytest.raises(ValueError, match="answer_type"):
        Question(qid="1", query="q", answer="a", answer_type="Exact Match")
