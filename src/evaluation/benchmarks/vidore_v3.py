"""ViDoRe V3 as a Benchmark (ACL 2026, aclanthology.org/2026.acl-long.755).

Retrieval, generation and localization labels over one corpus, with a published
per-dataset baseline for each. See ``src/evaluation/vidore_notes.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import repo_root
from typing import Any, Iterator, Mapping

from .base import (
    Benchmark, GoldSpec, Question, Region, SourceDoc,
    iter_parquet, normalize_answer_type,
)

PROJECT_ROOT = repo_root(__file__)
DEFAULT_ROOT = PROJECT_ROOT / "data" / "benchmark" / "vidore_v3"

# nuclear and telecom are private hold-outs and deliberately unreachable.
PUBLIC_SUBSETS = (
    "hr", "energy", "computer_science", "physics",
    "finance_en", "finance_fr", "industrial", "pharmaceuticals",
)

# Source documents are English and French; the other four are query translations.
LANGUAGES = ("english", "french", "spanish", "italian", "german", "portuguese")

# ``image`` excluded on purpose: ~12 GB of page renders no text arm reads.
CORPUS_COLUMNS = ("corpus_id", "doc_id", "markdown", "page_number_in_doc")
QUERY_COLUMNS = ("query_id", "query", "language", "query_types", "query_format",
                 "content_type", "source_type", "answer")
QREL_COLUMNS = ("query_id", "corpus_id", "score", "content_type", "bounding_boxes")

TAXONOMY = "vidore_v3"

# A leaked annotation-form option on 96 released qrels; not a modality.
NOT_A_MODALITY = "N/A (If relevance score=0)"


def unit_id(subset: str, doc_id: Any, page: Any) -> str:
    """Retrievable unit, keyed on the globally unique ``doc_id``."""
    return f"{subset}::{doc_id}#page={page}"


def question_id(subset: str, query_id: Any) -> str:
    """Namespaced qid; ``query_id`` restarts near zero in every subset."""
    return f"{subset}::{query_id}"


def modalities_of(raw: Any) -> tuple[str, ...]:
    labels = tuple(str(x) for x in (raw or []) if str(x) != NOT_A_MODALITY)
    return labels or ("unknown",)


@dataclass(frozen=True)
class _Query:
    qid: str
    query: str
    answer: str
    language: str
    modalities: tuple[str, ...]
    query_types: tuple[str, ...]
    query_format: str


class ViDoreV3(Benchmark):
    name = "vidore_v3"

    def __init__(
        self,
        root: Path | str = DEFAULT_ROOT,
        subset: str = "",
        language: str = "",
    ) -> None:
        if subset not in PUBLIC_SUBSETS:
            raise ValueError(
                f"Unknown subset {subset!r}. The 8 public ViDoRe V3 subsets are "
                f"{list(PUBLIC_SUBSETS)}; nuclear and telecom are private hold-outs."
            )
        if language not in LANGUAGES:
            raise ValueError(
                f"language is required and must be one of {list(LANGUAGES)}, got "
                f"{language!r}. No default and no 'all': a multilingual average "
                "measures cross-lingual mismatch, not retrieval quality."
            )
        self.root = Path(root)
        self.subset = subset
        self.language = language

        # Counted, not assumed: if the join degrades every number degrades silently.
        self.join_stats: dict[str, int] = {"attempted": 0, "matched": 0, "unreachable": 0}

        self._pages = self._read_corpus()
        self._queries = self._read_queries()
        self._by_qid = {q.qid: q for q in self._queries}
        self._qrels: dict[str, dict[str, int]] | None = None
        self._regions: dict[str, list[Region]] = {}

    # -- source ------------------------------------------------------------

    def _path(self, name: str) -> Path:
        # Evaluation exports use ``<root>/<subset>/<name>.parquet``. The raw
        # Hugging Face download uses
        # ``<root>/vidore_v3_<subset>/<name>/test-*.parquet``. Read either layout
        # directly so the multi-gigabyte image-bearing corpus is never copied just
        # to consolidate its shards.
        roots = (self.root / self.subset, self.root / f"vidore_v3_{self.subset}")
        for subset_root in roots:
            flat = subset_root / f"{name}.parquet"
            if flat.is_file():
                return flat
            sharded = subset_root / name
            if sharded.is_dir() and any(sharded.glob("*.parquet")):
                return sharded
        raise FileNotFoundError(
            f"ViDoRe V3 {self.subset}/{name} not found below {self.root}. Expected "
            f"either <root>/{self.subset}/{name}.parquet or "
            f"<root>/vidore_v3_{self.subset}/{name}/*.parquet (see "
            "docs/vidore_v3_setup.md)."
        )

    def _read_corpus(self) -> dict[int, dict[str, Any]]:
        return {
            row["corpus_id"]: row
            for row in iter_parquet(self._path("corpus"), CORPUS_COLUMNS)
        }

    def _read_queries(self) -> list[_Query]:
        # Mirrors dataset_loader.py:66-111 at commit a70f23af; see vidore_notes.md.
        out: list[_Query] = []
        for row in iter_parquet(self._path("queries"), QUERY_COLUMNS):
            if row["language"] != self.language:
                continue
            out.append(
                _Query(
                    qid=question_id(self.subset, str(row["query_id"])),
                    query=str(row["query"] or ""),
                    answer=str(row["answer"] or ""),
                    language=str(row["language"]),
                    modalities=modalities_of(row.get("content_type")),
                    query_types=tuple(str(x) for x in (row.get("query_types") or [])),
                    query_format=str(row.get("query_format") or ""),
                )
            )
        return out

    # -- protocol ----------------------------------------------------------

    def corpus(self) -> Iterator[SourceDoc]:
        """Pages, text-only: ``markdown`` is the release's own page text."""
        for row in self._pages.values():
            yield SourceDoc(
                doc_id=unit_id(self.subset, row["doc_id"], row["page_number_in_doc"]),
                text=row.get("markdown") or "",
                page=str(row["page_number_in_doc"]),
                modality="page",
                meta={
                    "doc_id": row["doc_id"],
                    "corpus_id": row["corpus_id"],
                    "subset": self.subset,
                },
            )

    def questions(self) -> Iterator[Question]:
        for record in self._queries:
            yield Question(
                qid=record.qid,
                query=record.query,
                answer=record.answer,
                # Every answer is free text; the paper grades with an LLM judge.
                answer_type=normalize_answer_type(""),
                modalities=record.modalities,
                taxonomy=TAXONOMY,
            )

    def qrels(self) -> dict[str, dict[str, int]]:
        """Graded judgements shaped for ``pytrec_eval``, gains untranslated."""
        if self._qrels is not None:
            return self._qrels

        surviving = {q.qid for q in self._queries}
        graded: dict[str, dict[str, int]] = {}
        regions: dict[str, list[Region]] = {}

        for row in iter_parquet(self._path("qrels"), QREL_COLUMNS):
            qid = question_id(self.subset, str(row["query_id"]))
            # Without this drop the gold set inflates by the number of languages.
            if qid not in surviving:
                continue

            self.join_stats["attempted"] += 1
            page = self._pages.get(row["corpus_id"])
            if page is None:
                self.join_stats["unreachable"] += 1
                continue
            self.join_stats["matched"] += 1

            unit = unit_id(self.subset, page["doc_id"], page["page_number_in_doc"])
            graded.setdefault(qid, {})[unit] = int(row["score"])
            regions.setdefault(qid, []).extend(
                _regions_from(row, unit, str(page["page_number_in_doc"]))
            )

        self._qrels = graded
        self._regions = regions
        return graded

    @property
    def unreachable_n(self) -> int:
        self.qrels()
        return self.join_stats["unreachable"]

    def gold_docs(self, qid: str) -> GoldSpec:
        """Binary view; the graded distinction lives in ``qrels`` for NDCG."""
        return GoldSpec(docs=tuple(self.qrels().get(str(qid), {})))

    def gold_pages(self, qid: str) -> list[str] | None:
        units = self.qrels().get(str(qid))
        if not units:
            return None
        return [unit.rsplit("#page=", 1)[-1] for unit in units]

    def gold_regions(self, qid: str) -> list[Region] | None:
        """Boxes in rendered-page pixel space; scoring them is deferred."""
        self.qrels()
        return self._regions.get(str(qid)) or None

    def corpus_identity(self) -> str:
        """Names this arm in the ``index_id``; both axes select disjoint data."""
        return f"{self.subset}-{self.language}"

    def scope_for(self, qid: str) -> list[str]:
        """The whole subset: retrieval is open-corpus, unlike MMDocIR."""
        return [d.doc_id for d in self.corpus()]

    def answer_style(self):
        """Graded, not binary: the harness default would drop partial credit."""
        from .vidore_v3_judge import ViDoreStyle

        return ViDoreStyle()


def _regions_from(row: Mapping[str, Any], unit: str, page: str) -> list[Region]:
    modality = modalities_of(row.get("content_type"))[0]
    out: list[Region] = []
    for box in row.get("bounding_boxes") or []:
        x1, y1 = float(box["x1"]), float(box["y1"])
        x2, y2 = float(box["x2"]), float(box["y2"])
        # 84 released boxes are degenerate; zero area would score as a miss.
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(
            Region(
                doc_id=unit,
                region_id=f"{unit}#bbox={x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}"
                          f"#ann={box.get('annotator')}",
                modality=modality,
                page=page,
                bbox=(x1, y1, x2, y2),
            )
        )
    return out
