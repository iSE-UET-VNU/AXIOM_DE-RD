"""How a benchmark asks for an answer and how it grades one.

Generation and judging are benchmark properties, not harness properties. ViDoRe
scores three ways against the paper's own prompts; the iSE lake scores a binary
correct with an abstain option and an exact-match path. Running one through the
other's grader silently changes the denominator, so an adapter that has its own
supplies it here and ``run_answer`` uses whatever the benchmark hands it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Judged:
    """One graded answer, in whatever taxonomy the benchmark uses.

    ``correct`` is the strict reading and ``credited`` the lenient one. A binary
    benchmark sets them equal; a graded one lets partial credit differ, and both
    are reported so quoting one cannot pick the conclusion.
    """

    label: str
    correct: bool
    credited: bool
    grader: str
    abstained: bool = False
    error: str | None = None


@runtime_checkable
class AnswerStyle(Protocol):
    name: str

    def render_prompt(self, query: str, texts: Sequence[str]) -> str: ...

    def judge(self, question: str, gold: str, generation: Any, answer_type: str, *,
              model: str, generator_model: str) -> Judged: ...


class DefaultStyle:
    """The harness prompt and the binary judge, unchanged."""

    name = "binary"

    def render_prompt(self, query: str, texts: Sequence[str]) -> str:
        from .generate import ABSTAIN, PROMPT, render_texts

        return PROMPT.format(abstain=ABSTAIN, context=render_texts(texts), question=query)

    def judge(self, question: str, gold: str, generation: Any, answer_type: str, *,
              model: str, generator_model: str) -> Judged:
        from .judge import judge as judge_binary

        verdict = judge_binary(question, gold, generation, answer_type,
                               model=model, generator_model=generator_model)
        label = "Correct" if verdict.correct else "Incorrect"
        return Judged(label, verdict.correct, verdict.correct, verdict.grader,
                      verdict.abstained, verdict.error)


def style_for(benchmark: object) -> AnswerStyle:
    """The benchmark's own style, or the binary default."""
    supply = getattr(benchmark, "answer_style", None)
    return supply() if callable(supply) else DefaultStyle()
