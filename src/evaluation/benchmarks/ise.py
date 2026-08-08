"""The iSE Summer Challenge lake as a Benchmark.

First adapter, and the fidelity gate: it must reproduce the numbers the harness
produced before the refactor, question for question, or the refactor has changed
the measurement rather than generalizing it.

The lake has file-level labels only -- ``gold_pages`` and ``gold_regions`` return
None. That is the honest answer and the reason MMDocIR is worth running: it
carries the layout annotation this source cannot provide.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import repo_root
from typing import Iterator
import json

from ..evalset import EvalQuestion, read_jsonl
from ..lake import Lake, modality_of
from .base import Benchmark, GoldSpec, Question, Region, SourceDoc, normalize_answer_type

PROJECT_ROOT = repo_root(__file__)
DEFAULT_QUESTIONS = PROJECT_ROOT / "data" / "benchmark" / "questions.jsonl"
DEFAULT_LAKE = PROJECT_ROOT / "[iSE Summer Challenge 2026] Data Lake"
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "benchmark" / "corpus.jsonl"


class IseChallenge(Benchmark):
    name = "ise"

    def __init__(
        self,
        questions_path: Path | str = DEFAULT_QUESTIONS,
        lake_path: Path | str = DEFAULT_LAKE,
        subset: str = "resolvable",
        corpus_path: Path | str = DEFAULT_CORPUS,
    ) -> None:
        self.questions_path = Path(questions_path)
        self.lake_path = Path(lake_path)
        self.corpus_path = Path(corpus_path)
        self.subset = subset
        self._records: list[EvalQuestion] = [
            q for q in read_jsonl(self.questions_path) if self._keep(q)
        ]
        self._by_qid = {q.qid: q for q in self._records}

    def _keep(self, question: EvalQuestion) -> bool:
        if self.subset == "all":
            return True
        if self.subset == "text_only":
            return question.is_text_only
        return question.resolvable

    def corpus(self) -> Iterator[SourceDoc]:
        """Extracted documents, not raw lake files.

        ``corpus.jsonl`` is what retrieval actually runs over -- the lake holds
        523 files the extractors cannot read, and yielding those as units would
        put empty documents in the index and count them as retrievable misses.
        Units here are whole documents; the caller chunks them, because chunking
        is an experimental variable rather than a property of the source.
        """
        for line in self.corpus_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            doc_id = str(payload.get("doc_id") or "")
            blocks = payload.get("blocks") or []
            text = payload.get("text") or "\n\n".join(
                b.get("text", "") for b in blocks if isinstance(b, dict)
            )
            yield SourceDoc(
                doc_id=doc_id,
                text=text,
                modality=modality_of(doc_id),
                meta={"blocks": blocks, "title": payload.get("title", "")},
            )

    def questions(self) -> Iterator[Question]:
        for record in self._records:
            yield Question(
                qid=record.qid,
                query=record.question,
                answer=record.answer,
                answer_type=normalize_answer_type(record.answer_type),
                level=record.level,
                modalities=tuple(record.modalities),
            )

    def gold_docs(self, qid: str) -> GoldSpec:
        record = self._by_qid[str(qid)]
        return GoldSpec(
            docs=tuple(record.gold_doc_ids),
            any_of=tuple(tuple(group) for group in record.gold_any_of),
        )

    def gold_pages(self, qid: str) -> list[str] | None:
        return None

    def gold_regions(self, qid: str) -> list[Region] | None:
        return None
