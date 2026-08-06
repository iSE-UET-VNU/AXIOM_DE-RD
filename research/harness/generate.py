"""Answer generation from retrieved context, frozen so arms stay comparable.

The generator is a fixed instrument, not a variable. Every knob that could move
an answer -- model, prompt, temperature, context budget -- is pinned here, so a
difference between two arms is attributable to parsing, chunking or retrieval
rather than to decoding.

The context budget is counted in CHARACTERS, not chunks. Passing ``top_k=5`` to
every arm would hand a 3400-char chunker three times the evidence of a 512-word
one, and chunk size would become the treatment. Filling a fixed character budget
removes that confound; ``chunks_used`` then reports what each arm bought with it,
which is itself a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .llm import complete

GENERATOR_MODEL = "llm-default"
TEMPERATURE = 0.0
MAX_CONTEXT_CHARS = 12000
MAX_OUTPUT_TOKENS = 512

# A refusal must be detectable, because "the evidence was not retrieved" and
# "the evidence was retrieved and misread" are different failures and the
# parsing comparison turns on telling them apart.
ABSTAIN = "KHONG_DU_THONG_TIN"

PROMPT = """Answer the question using only the numbered context passages below.

Rules:
- Use only the passages. Do not use outside knowledge.
- If the passages do not contain the answer, reply with exactly: {abstain}
- If the question specifies an answer format, follow it exactly.
- Answer in the language of the question.
- Give the answer only, with no explanation and no citation.

Context:
{context}

Question: {question}

Answer:"""


@dataclass(frozen=True)
class ContextChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float = 0.0


@dataclass
class Generation:
    qid: str
    answer: str
    abstained: bool
    chunks_used: int
    chars_used: int
    context_doc_ids: list[str] = field(default_factory=list)
    error: str | None = None


def pack_context(
    chunks: Sequence[ContextChunk], max_chars: int = MAX_CONTEXT_CHARS
) -> list[ContextChunk]:
    """Fill the budget in rank order, whole chunks only.

    A chunk that does not fit is skipped rather than truncated, and lower-ranked
    chunks are still considered: truncating would feed the generator a passage
    cut mid-sentence, which penalises large-chunk arms for a packing decision
    rather than for their retrieval quality.
    """
    packed: list[ContextChunk] = []
    used = 0
    for chunk in chunks:
        size = len(chunk.text)
        if used + size > max_chars:
            continue
        packed.append(chunk)
        used += size
    return packed


def render_context(chunks: Sequence[ContextChunk]) -> str:
    return "\n\n".join(f"[{i + 1}] {chunk.text}" for i, chunk in enumerate(chunks))


def generate(
    qid: str,
    question: str,
    chunks: Sequence[ContextChunk],
    *,
    model: str = GENERATOR_MODEL,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> Generation:
    packed = pack_context(chunks, max_chars)
    doc_ids: list[str] = []
    for chunk in packed:
        if chunk.doc_id not in doc_ids:
            doc_ids.append(chunk.doc_id)
    chars = sum(len(chunk.text) for chunk in packed)

    if not packed:
        return Generation(qid, ABSTAIN, True, 0, 0, doc_ids)

    prompt = PROMPT.format(
        abstain=ABSTAIN, context=render_context(packed), question=question
    )
    try:
        answer = complete(
            model, prompt, temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS
        ).strip()
    except Exception as error:  # noqa: BLE001 - recorded, never scored as wrong
        return Generation(qid, "", False, len(packed), chars, doc_ids, error=str(error))

    return Generation(
        qid=qid,
        answer=answer,
        abstained=ABSTAIN in answer,
        chunks_used=len(packed),
        chars_used=chars,
        context_doc_ids=doc_ids,
    )
