"""A benchmark grades its own answers, or the taxonomy silently changes.

ViDoRe is three-way and the harness default is binary. Scoring ViDoRe through
the binary judge renders fine and drops Partially Correct into Incorrect, which
moves the only significant end-to-end effect we have measured.
"""

from __future__ import annotations

import pytest

from src.evaluation.answer_style import DefaultStyle, Judged, style_for
from src.evaluation.benchmarks.vidore_v3_judge import ViDoreStyle
from src.evaluation.generate import Generation


def gen(answer="réponse", error=None):
    return Generation(qid="q1", answer=answer, abstained=False, chunks_used=1,
                      chars_used=10, context_doc_ids=["d#page=0"], error=error)


class Bench:
    def answer_style(self):
        return ViDoreStyle()


class Plain:
    pass


def test_a_benchmark_without_a_style_gets_the_binary_default():
    assert isinstance(style_for(Plain()), DefaultStyle)


def test_a_benchmark_with_a_style_gets_its_own():
    assert isinstance(style_for(Bench()), ViDoreStyle)


def test_the_two_styles_are_not_the_same_taxonomy():
    assert DefaultStyle().name != ViDoreStyle().name


def test_vidore_prompt_uses_the_papers_wording_not_the_harness_one():
    """The harness prompt offers an abstain option; Figure 25 has none."""
    body = ViDoreStyle().render_prompt("quelle vitesse?", ["page un"])
    assert "expert at answering query based on documents" in body
    assert "KHONG_DU_THONG_TIN" not in body


def test_default_prompt_still_offers_abstain():
    body = DefaultStyle().render_prompt("how fast?", ["page one"])
    assert "KHONG_DU_THONG_TIN" in body


def test_vidore_prompt_numbers_the_documents():
    body = ViDoreStyle().render_prompt("q", ["alpha", "beta"])
    assert "[1] alpha" in body and "[2] beta" in body


@pytest.mark.parametrize("label,correct,credited", [
    ("Correct", True, True),
    ("Partially Correct", False, True),
    ("Incorrect", False, False),
])
def test_partial_credit_is_kept_apart_from_strict_credit(monkeypatch, label, correct, credited):
    """The failure this exists for: collapsing three labels into one bool."""
    import src.evaluation.benchmarks.vidore_v3_judge as mod

    monkeypatch.setattr(mod, "judge_answer",
                        lambda *a, **k: mod.ViDoreVerdict("q1", label))
    verdict = ViDoreStyle().judge("q", "or", gen(), "", model="j", generator_model="g")
    assert (verdict.label, verdict.correct, verdict.credited) == (label, correct, credited)


def test_a_generation_error_is_never_credited():
    verdict = ViDoreStyle().judge("q", "or", gen(error="boom"), "",
                                  model="j", generator_model="g")
    assert not verdict.correct and not verdict.credited and verdict.error == "boom"


def test_binary_style_reports_equal_strict_and_credited(monkeypatch):
    """A binary benchmark has no partial band; the two readings must not diverge."""
    import src.evaluation.judge as mod

    monkeypatch.setattr(mod, "judge",
                        lambda *a, **k: mod.Verdict("q1", True, False, "llm_judge"))
    verdict = DefaultStyle().judge("q", "gold", gen(), "", model="j", generator_model="g")
    assert verdict.correct == verdict.credited is True


def test_judged_carries_the_raw_label_not_only_a_bool():
    assert Judged("Partially Correct", False, True, "llm_judge").label == "Partially Correct"
