"""The dataset interface every benchmark implements.

The harness was welded to the iSE evalset: ``run_answer.py`` read one file path
and one schema. External benchmarks carry labels our lake does not have -- pages,
layout regions, bounding boxes, per-region modality tags -- so the interface is
shaped by the richest source rather than the poorest, and adapters report what
they have.

Three gold granularities, because they answer different questions:

* ``gold_docs``   -- which file. All we have for the iSE lake; the Stage-1 metric.
* ``gold_pages``  -- which page. MMDocIR's page-level retrieval task.
* ``gold_regions``-- which paragraph/table/figure, with a modality tag. The
  annotation that makes chunker evaluation possible at all.

The modality tag travels with the region rather than sitting in a side table,
because the per-modality breakdown is the entire reason for running these sets:
an aggregate score hides the table-and-figure failure the parsing work is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

# The harness grades by the source's own declared type. Normalized to three
# kinds so a judge/EM decision never depends on a per-dataset spelling.
ANSWER_TYPES = ("em", "judge", "mc")


@dataclass(frozen=True)
class SourceDoc:
    """One retrievable source: a file to parse, or an already-rendered page."""

    doc_id: str
    path: str | None = None
    text: str | None = None
    page: str | None = None
    modality: str = "text"
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Region:
    """A layout-level gold span: paragraph, table, figure, equation, chart."""

    doc_id: str
    region_id: str
    modality: str
    page: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    text: str = ""


@dataclass(frozen=True)
class GoldSpec:
    """Evidence for one question, with any-of groups preserved.

    ``any_of`` exists because a reference naming a directory or a glob denotes a
    group where finding one member is sufficient. Flattening it into ``docs``
    would turn a single piece of evidence into 45 required documents and score a
    correct retrieval as a 1/45 miss.
    """

    docs: Sequence[str] = ()
    any_of: Sequence[Sequence[str]] = ()

    @property
    def required(self) -> int:
        return len(self.docs) + len(self.any_of)

    def recall(self, retrieved: Iterable[str]) -> float:
        found = set(retrieved)
        if not self.required:
            return 0.0
        hits = len(found & set(self.docs))
        hits += sum(1 for group in self.any_of if found & set(group))
        return hits / self.required

    def flat(self) -> list[str]:
        out = list(self.docs)
        for group in self.any_of:
            out.extend(d for d in group if d not in out)
        return out


@dataclass(frozen=True)
class Question:
    qid: str
    query: str
    answer: str
    answer_type: str
    level: str = ""
    modalities: Sequence[str] = ()
    choices: Sequence[str] = ()
    # Which vocabulary ``modalities`` is drawn from. Sources are not always
    # internally consistent -- MMDocIR ships two disjoint labelling schemes on
    # different documents -- and averaging across incompatible vocabularies
    # produces a breakdown that looks fine and means nothing.
    taxonomy: str = ""

    def __post_init__(self) -> None:
        if self.answer_type not in ANSWER_TYPES:
            raise ValueError(
                f"answer_type {self.answer_type!r} must be one of {ANSWER_TYPES}; "
                "adapters normalize their source's spelling."
            )


@runtime_checkable
class Benchmark(Protocol):
    """A dataset the harness can evaluate end to end."""

    name: str

    def corpus(self) -> Iterator[SourceDoc]: ...

    def questions(self) -> Iterator[Question]: ...

    def gold_docs(self, qid: str) -> GoldSpec: ...

    def gold_pages(self, qid: str) -> list[str] | None:
        """Page ids, or None when the source has no page labels."""
        ...

    def gold_regions(self, qid: str) -> list[Region] | None:
        """Layout regions, or None when the source has no region labels."""
        ...


def iter_parquet(path: Any, columns: Sequence[str], batch_size: int = 2048) -> Iterator[dict]:
    """Stream parquet file(s) row by row, never holding the whole table.

    ``read_table(...).to_pylist()`` materializes twice -- once as Arrow buffers,
    once as Python objects -- which killed the process on MMDocIR's 170,338
    layout rows with three text columns. Row-group batches keep peak memory at
    one batch instead of the whole file.

    Shared here rather than in one adapter because every large benchmark needs
    it: M3DocVQA is 41,005 pages, ViDoRe V3 ~26,000.
    """
    import pyarrow.parquet as pq

    from pathlib import Path

    if isinstance(path, (list, tuple)):
        paths = [Path(item) for item in path]
    else:
        source = Path(path)
        paths = sorted(source.glob("*.parquet")) if source.is_dir() else [source]
    if not paths:
        raise FileNotFoundError(f"No parquet files found at {path}")

    for parquet_path in paths:
        reader = pq.ParquetFile(parquet_path)
        for batch in reader.iter_batches(batch_size=batch_size, columns=list(columns)):
            for row in batch.to_pylist():
                yield row


class MixedTaxonomy(ValueError):
    """Refuses to average modality labels drawn from different vocabularies."""


def check_single_taxonomy(taxonomies: Iterable[str], context: str = "") -> str:
    """Raise unless every question's modality labels share one vocabulary.

    A hard rule rather than a convention, because the failure is invisible:
    MMDocIR ships ``text-only`` on some documents and ``Pure-text
    (Plain-text)`` on others, and a per-modality table averaging across both
    renders perfectly while comparing two different things. This is the same
    class of failure as the analyzer bug -- a plausible number instead of an
    error -- so it errors.

    Callers that genuinely want both must group by taxonomy first and report
    the groups separately.
    """
    present = {t for t in taxonomies if t}
    if len(present) > 1:
        raise MixedTaxonomy(
            f"Cannot aggregate across taxonomies {sorted(present)}"
            f"{f' in {context}' if context else ''}. Group by taxonomy and "
            "report separately; averaging them compares different vocabularies."
        )
    return next(iter(present), "")


def normalize_answer_type(raw: str) -> str:
    """Map a source's own spelling onto the three kinds the grader knows.

    Defaults to ``judge`` rather than ``em``: grading a free-text answer by
    string equality scores correct answers as wrong, which is a silent,
    one-directional bias. An over-lenient judge is visible in the transcript; an
    over-strict exact match is not.
    """
    value = (raw or "").strip().lower()
    if value.startswith("exact") or value in {"em", "exact_match"}:
        return "em"
    if value in {"mc", "multiple_choice", "multiple-choice", "choice"}:
        return "mc"
    return "judge"
