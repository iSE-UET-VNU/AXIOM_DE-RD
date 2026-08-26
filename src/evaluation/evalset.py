"""Question sheet -> typed evaluation records with resolved gold documents.

The sheet is the only ground truth we have, so nothing here silently repairs it.
A question whose evidence cannot be resolved keeps its unresolved references and
is excluded from scored subsets by ``resolvable``, never dropped on load.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable
import csv
import json

from .lake import Lake, modality_of


@dataclass
class EvalQuestion:
    qid: str
    question: str
    answer: str
    answer_type: str
    level: str
    evidence_refs: list[str] = field(default_factory=list)
    gold_doc_ids: list[str] = field(default_factory=list)
    gold_any_of: list[list[str]] = field(default_factory=list)
    unresolved_refs: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    resolution: dict[str, str] = field(default_factory=dict)

    @property
    def resolvable(self) -> bool:
        return bool(self.gold_doc_ids or self.gold_any_of) and not self.unresolved_refs

    @property
    def is_text_only(self) -> bool:
        return self.resolvable and set(self.modalities) == {"text"}

    @property
    def all_gold_doc_ids(self) -> list[str]:
        """Every document that could serve as evidence, flattened.

        Use for coverage and reachability. Do NOT use as the denominator of
        recall: a directory reference expands to a group where finding any one
        member satisfies the question, and counting all of them as required
        would score a correct retrieval as a 1/45 miss.
        """
        flat = list(self.gold_doc_ids)
        for group in self.gold_any_of:
            flat.extend(doc for doc in group if doc not in flat)
        return flat

    def recall(self, retrieved: Iterable[str]) -> float:
        """Set recall over evidence, all-of for named files, any-of per group."""
        found = set(retrieved)
        required = len(self.gold_doc_ids) + len(self.gold_any_of)
        if not required:
            return 0.0
        got = len(found & set(self.gold_doc_ids))
        got += sum(1 for group in self.gold_any_of if found & set(group))
        return got / required

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolvable"] = self.resolvable
        payload["is_text_only"] = self.is_text_only
        return payload


def parse_evidence(raw: str) -> list[str]:
    """Split the Evidences cell, tolerating malformed JSON in the sheet."""
    text = (raw or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except ValueError:
        value = [part for part in text.strip("[]").split(",") if part.strip()]
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def load_questions(csv_path: Path | str, lake: Lake) -> list[EvalQuestion]:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    questions: list[EvalQuestion] = []
    for row in rows:
        refs = parse_evidence(row.get("Evidences", ""))
        gold: list[str] = []
        any_of: list[list[str]] = []
        unresolved: list[str] = []
        modalities: set[str] = set()
        resolution: dict[str, str] = {}
        for ref in refs:
            paths, strategy = lake.resolve_all(ref)
            if not paths:
                unresolved.append(ref.strip())
                continue
            resolution[ref.strip()] = strategy
            doc_ids = [lake.doc_id(path) for path in paths]
            # A reference that expanded to a folder or glob names a group where
            # any member is sufficient. Merging it into ``gold`` would silently
            # turn one piece of evidence into 45 required documents.
            if len(doc_ids) > 1:
                any_of.append(doc_ids)
            elif doc_ids[0] not in gold:
                gold.append(doc_ids[0])
            modalities.update(modality_of(path) for path in paths)
        questions.append(
            EvalQuestion(
                qid=str(row.get("STT", "")).strip(),
                question=(row.get("Question") or "").strip(),
                answer=(row.get("Groundtruth") or "").strip(),
                answer_type=(row.get("Answer type") or "").strip(),
                level=(row.get("Level") or "").strip(),
                evidence_refs=[ref.strip() for ref in refs],
                gold_doc_ids=gold,
                gold_any_of=any_of,
                unresolved_refs=unresolved,
                modalities=sorted(modalities),
                resolution=resolution,
            )
        )
    return questions


def write_jsonl(path: Path | str, questions: Iterable[EvalQuestion]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path | str) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        payload = json.loads(line)
        payload.pop("resolvable", None)
        payload.pop("is_text_only", None)
        questions.append(EvalQuestion(**payload))
    return questions
