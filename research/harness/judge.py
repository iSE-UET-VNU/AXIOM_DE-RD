"""Grading generated answers against the sheet's ground truth.

The question sheet declares how each question should be graded -- "Exact Match"
or "LLM as a Judge" -- so the grader follows the sheet rather than choosing.
Exact-match questions carry an explicit format instruction in the question text
("trả về duy nhất dưới dạng chữ số"), which is what makes string comparison fair
for them.

The judge model must differ from the generator model. LLM judges prefer their
own outputs, and with the generator frozen across every arm that bias would land
on whichever arm happened to produce text in the generator's own idiom.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from .generate import ABSTAIN, Generation
from .llm import complete

JUDGE_MODEL = "llm-judge"
TEMPERATURE = 0.0

PROMPT = """You are grading one answer against a reference answer.

Question: {question}
Reference answer: {gold}
Candidate answer: {candidate}

The candidate is CORRECT if it conveys the same fact as the reference, even if
worded differently, in a different language, or with extra formatting. It is
INCORRECT if it states a different fact, is missing the fact, or refuses.

Reply with exactly one word: CORRECT or INCORRECT."""

_PUNCT = re.compile(r"[\s​]+")
_EDGE = re.compile(r"^[\s\"'`(\[.,;:]+|[\s\"'`)\].,;:]+$")


@dataclass
class Verdict:
    qid: str
    correct: bool
    abstained: bool
    grader: str
    error: str | None = None


def normalize(text: str) -> str:
    """Fold formatting, keep meaning.

    NFC only -- diacritics are NOT stripped. "Đại học" and "Dai hoc" are
    different strings in Vietnamese, and folding them would silently forgive an
    OCR arm for losing the diacritics this benchmark is meant to detect.
    """
    folded = unicodedata.normalize("NFC", text or "")
    folded = _PUNCT.sub(" ", folded).strip()
    folded = _EDGE.sub("", folded)
    return folded.casefold()


def exact_match(candidate: str, gold: str) -> bool:
    left, right = normalize(candidate), normalize(gold)
    if not right:
        return False
    return left == right or right in left.split() or left == right.rstrip(".0")


def judge(
    question: str,
    gold: str,
    generation: Generation,
    answer_type: str,
    *,
    model: str = JUDGE_MODEL,
    generator_model: str,
) -> Verdict:
    if model == generator_model:
        raise ValueError(
            f"judge model must differ from generator model (both {model!r}); "
            "self-preference bias would favour the generator's own phrasing"
        )
    if generation.error:
        return Verdict(generation.qid, False, False, "error", error=generation.error)
    if generation.abstained or not generation.answer.strip():
        return Verdict(generation.qid, False, True, "abstain")

    # Accepts both the sheet's own wording ("Exact Match") and the adapter's
    # normalized form ("em"). The adapter is the future, but the raw spelling
    # must keep working or the refactor silently reroutes every exact-match
    # question to the LLM judge and changes the numbers it was meant to preserve.
    kind = answer_type.strip().lower()
    if kind == "em" or kind.startswith("exact"):
        return Verdict(
            generation.qid, exact_match(generation.answer, gold), False, "exact_match"
        )

    prompt = PROMPT.format(question=question, gold=gold, candidate=generation.answer)
    try:
        reply = complete(model, prompt, temperature=TEMPERATURE, max_output_tokens=16)
    except Exception as error:  # noqa: BLE001
        return Verdict(generation.qid, False, False, "error", error=str(error))
    verdict = "INCORRECT" not in reply.upper() and "CORRECT" in reply.upper()
    return Verdict(generation.qid, verdict, False, "llm_judge")
