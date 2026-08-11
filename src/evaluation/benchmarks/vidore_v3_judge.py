"""ViDoRe V3 generation and judging, pinned to the paper's own prompts.

Upstream is retrieval-only, so this is built from Figures 24 and 25. See
``src/evaluation/vidore_notes.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import json
import re

from ..llm import complete

JUDGE_MODEL = "gpt-5.2"
JUDGE_REASONING_EFFORT = "medium"
TEMPERATURE = 0.0

# Anything else, including an abstention, is an error rather than a fourth bucket.
LABELS = ("Correct", "Partially Correct", "Incorrect")

# Figure 24. Do not edit; test_judge_prompt_is_byte_identical_to_figure_24 pins it.
JUDGE_PROMPT = (
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

# Figure 25. Note what is absent: an abstain option.
ANSWER_PROMPT = (
    "You are an expert at answering query based on documents.\n"
    "Here is a list of relevant documents:\n"
    "{documents}\n"
    "\n"
    "Based on the above documents, answer the following query:\n"
    "{query}\n"
    "\n"
    "Keep the response short when appropriate. Output the answer only."
)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_JUDGMENT = re.compile(r'"judgment"\s*:\s*"([^"]+)"')


@dataclass(frozen=True)
class ViDoreVerdict:
    """One judged answer; ``judgment`` is the raw three-way label, not a bool."""

    qid: str
    judgment: str
    explanation: str = ""
    error: str | None = None


def parse_judgment(reply: str) -> str:
    """Raises rather than defaulting; a default shifts the score one way."""
    text = _FENCE.sub("", (reply or "").strip())
    try:
        payload = json.loads(text)
        label = str(payload["judgment"]).strip()
    except (ValueError, KeyError, TypeError) as error:
        # LaTeX in the explanation -- "\(", "\frac" -- is not valid JSON escaping
        # and physics answers produce it constantly. The judgment is a plain
        # quoted literal, so read it directly; still raise when it is absent,
        # because a default would shift every unparsed reply the same way.
        match = _JUDGMENT.search(text)
        if not match:
            raise ValueError(
                f"could not read a judgment from the judge reply {reply!r}: {error}"
            ) from error
        label = match.group(1).strip()
    if label not in LABELS:
        raise ValueError(f"judgment {label!r} is not one of {list(LABELS)}")
    return label


def judge_answer(
    qid: str,
    query: str,
    true_answer: str,
    test_answer: str,
    *,
    model: str,
    generator_model: str,
) -> ViDoreVerdict:
    if model == generator_model:
        raise ValueError(
            f"judge model must differ from generator model (both {model!r}); "
            "self-preference bias would favour the generator's own phrasing"
        )
    prompt = JUDGE_PROMPT.format(
        query=query, true_answer=true_answer, test_answer=test_answer
    )
    try:
        reply = complete(model, prompt, temperature=TEMPERATURE, max_output_tokens=256)
    except Exception as error:  # noqa: BLE001 - recorded, never scored as correct
        return ViDoreVerdict(qid, "Incorrect", error=str(error))
    return ViDoreVerdict(qid, parse_judgment(reply))


def render_documents(texts: Sequence[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(texts))


class ViDoreStyle:
    """The paper's Figure 24/25 prompts and its three-way label.

    ``credited`` is Correct+Partially Correct: the only significant end-to-end
    effect we have measured sits in that band, so collapsing it to a bool at
    grading time would discard the result rather than report it.
    """

    name = "vidore_three_way"

    def render_prompt(self, query: str, texts: Sequence[str]) -> str:
        return ANSWER_PROMPT.format(documents=render_documents(texts), query=query)

    def judge(self, question, gold, generation, answer_type, *, model, generator_model):
        from ..answer_style import Judged

        if generation.error:
            return Judged("Incorrect", False, False, "error", error=generation.error)
        verdict = judge_answer(generation.qid, question, gold, generation.answer,
                               model=model, generator_model=generator_model)
        return Judged(verdict.judgment, verdict.judgment == "Correct",
                      verdict.judgment in ("Correct", "Partially Correct"),
                      "llm_judge", error=verdict.error)


def score(verdicts: Iterable[ViDoreVerdict]) -> dict[str, float]:
    """Both aggregations from one judge pass; report both, always."""
    items = list(verdicts)
    if not items:
        return {"correct_only": 0.0, "correct_plus_partial": 0.0, "n": 0}

    for verdict in items:
        if verdict.judgment not in LABELS:
            raise ValueError(
                f"{verdict.qid}: judgment {verdict.judgment!r} is not one of "
                f"{list(LABELS)}. The paper's prompt has no abstain option, and a "
                "fourth bucket would make our denominator differ from theirs."
            )

    total = len(items)
    correct = sum(1 for v in items if v.judgment == "Correct")
    partial = sum(1 for v in items if v.judgment == "Partially Correct")
    return {
        "correct_only": correct / total,
        "correct_plus_partial": (correct + partial) / total,
        "n": total,
    }
