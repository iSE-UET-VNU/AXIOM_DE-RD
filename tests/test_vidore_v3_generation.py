"""ViDoRe V3 generation leg: pinned prompts, three-way verdict, two aggregations."""

from __future__ import annotations

import pytest

from src.evaluation.benchmarks.vidore_v3_judge import (  # noqa: E402
    ANSWER_PROMPT,
    JUDGE_PROMPT,
    ViDoreVerdict,
    parse_judgment,
    score,
)


# -- seam: three-way label vs the two aggregations -----------------------------


def test_partial_makes_the_two_aggregations_differ():
    """The whole point of storing three labels instead of two."""
    verdicts = [
        ViDoreVerdict("hr::0", "Correct"),
        ViDoreVerdict("hr::1", "Partially Correct"),
        ViDoreVerdict("hr::2", "Incorrect"),
    ]
    result = score(verdicts)
    assert result["correct_only"] == pytest.approx(1 / 3)
    assert result["correct_plus_partial"] == pytest.approx(2 / 3)
    assert result["correct_only"] != result["correct_plus_partial"]
    assert result["n"] == 3


def test_the_headline_metric_excludes_partial():
    """Correct-only is the headline, matching the paper's stated metric."""
    verdicts = [ViDoreVerdict("q", "Partially Correct")]
    assert score(verdicts)["correct_only"] == 0.0
    assert score(verdicts)["correct_plus_partial"] == 1.0


def test_raw_label_survives_to_the_record():
    """Collapsing at parse time is the defect; the label must reach storage."""
    assert ViDoreVerdict("q", "Partially Correct").judgment == "Partially Correct"


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"explanation": "x", "judgment": "Correct"}', "Correct"),
        ('{"explanation": "x", "judgment": "Partially Correct"}', "Partially Correct"),
        ('{"explanation": "x", "judgment": "Incorrect"}', "Incorrect"),
        ('```json\n{"explanation": "x", "judgment": "Correct"}\n```', "Correct"),
    ],
)
def test_parses_the_published_json_shape(reply: str, expected: str):
    assert parse_judgment(reply) == expected


def test_unparseable_judgment_raises_rather_than_defaulting():
    """A silent default would land every malformed reply in one bucket."""
    with pytest.raises(ValueError, match="judgment"):
        parse_judgment("looks correct to me")
    with pytest.raises(ValueError, match="judgment"):
        parse_judgment('{"explanation": "x", "judgment": "Mostly Right"}')


# -- abstention must be impossible, not merely unlikely ------------------------


def test_abstention_is_rejected_not_bucketed():
    """The paper's prompt has no abstain option."""
    with pytest.raises(ValueError, match="abstain"):
        score([ViDoreVerdict("q", "Correct"), ViDoreVerdict("q2", "Abstained")])


# -- seam: pinned prompts vs the paper -----------------------------------------


def test_judge_prompt_is_byte_identical_to_figure_24():
    """A prompt edit six months from now would move the number silently."""
    assert JUDGE_PROMPT == (
        'You are an expert judge evaluating the accuracy of a test answer against a '
        'gold-standard true answer. Your goal is to determine if the test answer '
        'captures the essential "core information."\n'
        "\n"
        "### Evaluation Criteria:\n"
        "- Correct: The test answer contains all core information of the true answer. "
        "Minor omissions of non-essential details or the addition of minor, "
        'non-contradictory information should still be marked as "Correct."\n'
        "- Partially Correct: The test answer captures some of the core information, "
        "but suffers from significant omissions or includes substantial extra "
        "information that was not requested or present in the true answer.\n"
        "- Incorrect: The test answer is fundamentally wrong, contradicts the true "
        "answer, or misses the core information entirely.\n"
        "\n"
        "### Input Data:\n"
        "Query: {query}\n"
        "True Answer: {true_answer}\n"
        "Test Answer: {test_answer}\n"
        "\n"
        "### Output Format:\n"
        "Provide a very brief explanation for your judgment. You must output your "
        'final response in a JSON format with two fields: "explanation" and '
        '"judgment" (which must be "Correct", "Partially Correct", or "Incorrect").'
    )


def test_answer_prompt_is_byte_identical_to_figure_25():
    assert ANSWER_PROMPT == (
        "You are an expert at answering query based on documents.\n"
        "Here is a list of relevant documents:\n"
        "{documents}\n"
        "\n"
        "Based on the above documents, answer the following query:\n"
        "{query}\n"
        "\n"
        "Keep the response short when appropriate. Output the answer only."
    )


def test_our_own_judge_prompt_is_not_reused():
    """One word CORRECT/INCORRECT and a three-way JSON rubric are not variants."""
    from src.evaluation import judge as house_judge

    assert JUDGE_PROMPT != house_judge.PROMPT
    assert "Partially Correct" not in house_judge.PROMPT
    assert "CORRECT or INCORRECT" not in JUDGE_PROMPT


# -- judge replies that are not valid JSON -------------------------------------

# Verbatim shape of the reply that killed a run: a single backslash before "(",
# which JSON rejects as an escape. Doubling it here would make the fixture valid
# JSON and the test would pass against the unfixed parser.
LATEX_REPLY = (
    '```json\n{\n  "explanation": "It uses \\((a, b)\\) instead of c and z0, '
    'and \\frac{1}{2} differs.",\n  "judgment": "Correct"\n}\n```'
)


def test_a_latex_explanation_does_not_lose_the_judgment():
    """Physics judges emit \\( and \\frac, which JSON rejects as escapes."""
    from src.evaluation.benchmarks.vidore_v3_judge import parse_judgment

    assert parse_judgment(LATEX_REPLY) == "Correct"


def test_every_label_survives_an_unparsable_explanation():
    from src.evaluation.benchmarks.vidore_v3_judge import parse_judgment

    for label in ("Correct", "Partially Correct", "Incorrect"):
        reply = '{"explanation": "bad \\escape", "judgment": "%s"}' % label
        assert parse_judgment(reply) == label


def test_a_reply_with_no_judgment_field_still_raises():
    """Recovery must not become a default; that shifts every failure one way."""
    from src.evaluation.benchmarks.vidore_v3_judge import parse_judgment
    import pytest as _pytest

    with _pytest.raises(ValueError, match="could not read a judgment"):
        parse_judgment('{"explanation": "bad \\escape"}')


def test_an_unknown_label_is_still_refused_after_recovery():
    from src.evaluation.benchmarks.vidore_v3_judge import parse_judgment
    import pytest as _pytest

    with _pytest.raises(ValueError, match="is not one of"):
        parse_judgment('{"explanation": "bad \\escape", "judgment": "Mostly Right"}')
