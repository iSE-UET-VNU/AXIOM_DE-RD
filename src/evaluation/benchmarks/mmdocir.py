"""MMDocIR as a Benchmark (EMNLP 2025, arXiv 2501.08828).

Why this dataset: it carries **layout-level gold labels with bounding boxes and
modality tags**, which our lake does not have. That is the annotation that makes
chunker evaluation possible at all, and it arrives with a published leaderboard
to check the pipeline against.

Three things about the source shape drive this adapter:

* **Retrieval is within a document, not across a corpus.** MMDocIR's page task is
  "find the relevant pages inside this 65-page document". So each question's
  search space is its own document's pages, and ``scope`` carries that. Running
  it as an open-corpus task would measure something the leaderboard does not.
* **Questions have no id.** They are nested under documents, so a stable
  ``qid`` is synthesized as ``{doc_name}::{index}`` — positional, but the file is
  a frozen release.
* **``layout_mapping`` gives a bbox, not a layout id.** Linking a question to its
  gold layouts is a geometric join against the layouts table, not a key lookup.
  Regions therefore work from the annotations alone (770 kB) and are *enriched*
  with real layout ids and types only when the 2.5 GB parquet is present.

Only the 1,658 expert-annotated questions are exposed. The ~173,843
bootstrapped-label questions are training-scale supervision, not an evaluation
set, and are deliberately unreachable through this adapter.

(The paper's abstract says 1,685; its body, Table 1 and Table 3 all say 1,658,
with 2,107 page-level and 2,638 layout-level labels. Our parse matches the body
exactly on all three counts. The abstract is the outlier -- nothing is missing.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import repo_root
from typing import Any, Iterator, Sequence
import json

from .base import (
    Benchmark, GoldSpec, Question, Region, SourceDoc,
    iter_parquet, normalize_answer_type,
)

PROJECT_ROOT = repo_root(__file__)
DEFAULT_ROOT = PROJECT_ROOT / "data" / "benchmark" / "mmdocir"

ANNOTATIONS = "MMDocIR_annotations.jsonl"
PAGES = "MMDocIR_pages.parquet"
LAYOUTS = "MMDocIR_layouts.parquet"

# Which text column stands in for a page. The published finding is that VLM-text
# beats OCR-text; running both is how we replicate it rather than assume it.
TEXT_SOURCES = ("vlm_text", "ocr_text")


def canonical_doc(name: Any) -> str:
    """One document key across all three files.

    The annotations store ``doc_name`` WITH a ``.pdf`` suffix; both parquet
    tables store it without. Measured: 0/313 documents join on the raw value,
    313/313 with the suffix stripped.

    Nothing errors on the mismatch -- the layout join finds no rows and falls
    back to geometric ids, and page units get a doc_id the gold labels never
    contain, so every recall number comes out zero and reads as a broken
    retriever. Canonicalizing on the way in is the only place this can be fixed
    once.
    """
    text = str(name or "")
    return text[:-4] if text.lower().endswith(".pdf") else text


def page_doc_id(doc_name: str, page_id: Any) -> str:
    """Page-level retrieval unit.

    The page identity lives in the doc_id rather than in our chunk_id, which
    stays offset-based. Encoding pages into chunk identity would change a
    pipeline contract for the benefit of one benchmark.
    """
    return f"{doc_name}#page={page_id}"


def layout_doc_id(doc_name: str, layout_id: Any, page_id: Any = None) -> str:
    """Layout-level unit, carrying its page.

    The page component is not decoration: page recall is computed from the
    retrieved ids, so a layout id without a page reports 0 pages found for a run
    that retrieved exactly the right layouts.
    """
    if page_id is None:
        return f"{doc_name}#layout={layout_id}"
    return f"{doc_name}#page={page_id}#layout={layout_id}"


# The ``type`` field carries TWO disjoint labelling schemes because MMDocIR is
# assembled from two sources with different annotation procedures: 794 questions
# reviewed and validated from MMLongBench-Doc, and 864 annotated from scratch
# for DocBench (794 + 864 = 1,658). They never co-occur within a document.
# Averaging across them would compare incompatible vocabularies in a table that
# renders fine and means nothing, so the source travels with the labels.
EVIDENCE_LABELS = {
    "Pure-text (Plain-text)": "text",
    "Generalized-text (Layout)": "layout_text",
    "Figure": "figure",
    "Table": "table",
    "Chart": "chart",
}


def parse_type(raw: Any) -> tuple[tuple[str, ...], str]:
    """Return ``(modalities, taxonomy)`` from one ``type`` value.

    The list form is a *Python repr*, not JSON -- single-quoted, so json.loads
    rejects it. ``ast.literal_eval`` is the correct reader and is safe on
    literals.
    """
    text = str(raw or "").strip()
    if text.startswith("["):
        import ast

        try:
            labels = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return ("unknown",), "mmlongbench"
        normalized = tuple(
            EVIDENCE_LABELS.get(str(label), str(label).lower()) for label in labels
        )
        return (normalized or ("none",)), "mmlongbench"
    return ((text or "unknown"),), "docbench"


@dataclass(frozen=True)
class _QA:
    qid: str
    doc_name: str
    query: str
    answer: str
    modalities: tuple[str, ...]
    taxonomy: str
    page_ids: tuple[Any, ...]
    layout_mapping: tuple[dict[str, Any], ...]


class MMDocIR(Benchmark):
    name = "mmdocir"

    def __init__(
        self,
        root: Path | str = DEFAULT_ROOT,
        text_source: str = "vlm_text",
        level: str = "page",
    ) -> None:
        if text_source not in TEXT_SOURCES:
            raise ValueError(f"text_source must be one of {TEXT_SOURCES}")
        if level not in ("page", "layout"):
            raise ValueError("level must be 'page' or 'layout'")
        self.root = Path(root)
        self.text_source = text_source
        self.level = level
        self._records = list(self._read_annotations())
        self._by_qid = {qa.qid: qa for qa in self._records}
        # The bbox join is the entire gold linkage at layout level. If it
        # degrades, every layout-level number degrades with it and nothing
        # errors, so the rate is counted and reportable rather than assumed.
        self.join_stats: dict[str, int] = {"attempted": 0, "matched": 0, "no_table": 0}

    # -- source ------------------------------------------------------------

    def _path(self, name: str) -> Path:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download the MMDocIR evaluation dataset into "
                f"{self.root} (see docs/mmdocir_setup.md)."
            )
        return path

    def _read_annotations(self) -> Iterator[_QA]:
        for line in self._path(ANNOTATIONS).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            doc_name = canonical_doc(payload["doc_name"])
            for index, item in enumerate(payload.get("questions") or []):
                modalities, taxonomy = parse_type(item.get("type"))
                yield _QA(
                    qid=f"{doc_name}::{index}",
                    doc_name=doc_name,
                    query=str(item.get("Q") or ""),
                    answer=str(item.get("A") or ""),
                    # Carried through to the metric layer -- the per-modality
                    # split is the reason for running this dataset.
                    modalities=modalities,
                    taxonomy=taxonomy,
                    page_ids=tuple(item.get("page_id") or []),
                    layout_mapping=tuple(item.get("layout_mapping") or []),
                )

    # -- protocol ----------------------------------------------------------

    def corpus(self) -> Iterator[SourceDoc]:
        if self.level == "page":
            for row in iter_parquet(
                self._path(PAGES), ["doc_name", "passage_id", self.text_source]
            ):
                yield SourceDoc(
                    doc_id=page_doc_id(canonical_doc(row["doc_name"]), row["passage_id"]),
                    text=row.get(self.text_source) or "",
                    page=str(row["passage_id"]),
                    modality="page",
                    meta={"doc_name": canonical_doc(row["doc_name"])},
                )
            return

        for row in iter_parquet(
            self._path(LAYOUTS),
            ["doc_name", "layout_id", "page_id", "type", "text", *TEXT_SOURCES],
        ):
            # Layout text lives in different columns by type: `text` for text and
            # equations, ocr/vlm for figures and tables. Reading only one column
            # silently empties half the corpus.
            body = row.get("text") or row.get(self.text_source) or row.get("ocr_text") or ""
            yield SourceDoc(
                doc_id=layout_doc_id(canonical_doc(row["doc_name"]), row["layout_id"], row["page_id"]),
                text=body,
                page=str(row["page_id"]),
                modality=str(row.get("type") or "unknown"),
                meta={"doc_name": canonical_doc(row["doc_name"])},
            )

    def questions(self) -> Iterator[Question]:
        for qa in self._records:
            yield Question(
                qid=qa.qid,
                query=qa.query,
                answer=qa.answer,
                # No answer-type column: every answer is free text, so grading is
                # by judge. normalize_answer_type already defaults there.
                answer_type=normalize_answer_type(""),
                modalities=qa.modalities,
                taxonomy=qa.taxonomy,
            )

    def gold_docs(self, qid: str) -> GoldSpec:
        """Gold at the configured retrieval unit.

        Page level is exact. Layout level needs the geometric join, so it falls
        back to the gold pages when the layouts table is absent -- a coarser but
        honest label rather than a silently empty one.
        """
        qa = self._by_qid[str(qid)]
        if self.level == "page":
            return GoldSpec(docs=tuple(page_doc_id(qa.doc_name, p) for p in qa.page_ids))
        regions = self.gold_regions(qid) or []
        ids = tuple(r.region_id for r in regions if r.region_id.startswith(qa.doc_name))
        if not ids:
            return GoldSpec(docs=tuple(page_doc_id(qa.doc_name, p) for p in qa.page_ids))
        return GoldSpec(docs=ids)

    def gold_pages(self, qid: str) -> list[str] | None:
        qa = self._by_qid[str(qid)]
        return [str(p) for p in qa.page_ids] or None

    def gold_regions(self, qid: str) -> list[Region] | None:
        qa = self._by_qid[str(qid)]
        if not qa.layout_mapping:
            return None
        table = self._layout_index(qa.doc_name)
        regions: list[Region] = []
        for mapping in qa.layout_mapping:
            page = mapping.get("page")
            bbox = tuple(mapping.get("bbox") or ())
            if table:
                self.join_stats["attempted"] += 1
                matched = _match_bbox(table, page, bbox)
                self.join_stats["matched"] += matched is not None
            else:
                self.join_stats["no_table"] += 1
                matched = None
            regions.append(
                Region(
                    # ``doc_id`` is the CONTAINING unit, ``region_id`` is the
                    # region itself. Retrieval at page level can only ever return
                    # the page, so scoring region recall against a layout id
                    # would report 0 for a page that was correctly retrieved.
                    doc_id=page_doc_id(qa.doc_name, page),
                    region_id=(
                        layout_doc_id(qa.doc_name, matched["layout_id"], page)
                        if matched is not None
                        # No layouts table: keep the geometric label, so the
                        # modality breakdown still works from annotations alone.
                        else f"{qa.doc_name}#page={page}#bbox={_fmt(bbox)}"
                    ),
                    modality=str((matched or {}).get("type") or qa.modalities[0]),
                    page=str(page),
                    bbox=bbox if len(bbox) == 4 else None,
                )
            )
        return regions

    # -- helpers -----------------------------------------------------------

    def graded_region_recall(self, qid: str, retrieved: Sequence[str]) -> float | None:
        """Layout recall as graded box overlap, matching the paper's protocol.

        MMDocIR scores layout retrieval by the *overlap* between retrieved and
        gold boxes, because layout detectors produce differing boxes for the
        same content and a binary matched/not-matched verdict throws that away.
        Our binary ``region_recall`` (IoU >= 0.5) is therefore NOT comparable to
        their published layout numbers; this is, and both are reported.

        Returns None at page level, where retrieved units are pages and have no
        box to overlap -- a 0.0 there would read as total failure rather than
        "not applicable".
        """
        qa = self._by_qid[str(qid)]
        if not qa.layout_mapping:
            return None
        boxes = self._retrieved_boxes(retrieved)
        if not boxes:
            return None
        scores: list[float] = []
        for mapping in qa.layout_mapping:
            bbox = mapping.get("bbox") or []
            if len(bbox) != 4:
                continue
            gold = tuple(float(v) for v in bbox)
            page = str(mapping.get("page"))
            scores.append(
                max(
                    (_iou(gold, box) for (bp, box) in boxes if bp == page),
                    default=0.0,
                )
            )
        return round(sum(scores) / len(scores), 4) if scores else None

    def _retrieved_boxes(self, retrieved: Sequence[str]) -> list[tuple[str, tuple[float, ...]]]:
        index = self._layout_box_index()
        out: list[tuple[str, tuple[float, ...]]] = []
        for unit in retrieved:
            if "#layout=" not in unit:
                continue
            box = index.get(unit)
            if box:
                out.append(box)
        return out

    def _layout_box_index(self) -> dict[str, tuple[str, tuple[float, ...]]]:
        if not hasattr(self, "_box_cache"):
            cache: dict[str, tuple[str, tuple[float, ...]]] = {}
            path = self.root / LAYOUTS
            if path.exists():
                for row in iter_parquet(
                    path, ["doc_name", "layout_id", "page_id", "bbox"]
                ):
                    box = row.get("bbox") or []
                    if len(box) == 4:
                        unit = layout_doc_id(
                            canonical_doc(row["doc_name"]), row["layout_id"], row["page_id"]
                        )
                        cache[unit] = (str(row["page_id"]), tuple(float(v) for v in box))
            self._box_cache = cache
        return self._box_cache

    def scope_for(self, qid: str) -> list[str]:
        """Search space for one question: its own document's units.

        MMDocIR retrieves *within* a document. Passing this as ``scope`` is what
        makes our numbers comparable to the leaderboard instead of measuring a
        harder open-corpus task nobody else ran.
        """
        qa = self._by_qid[str(qid)]
        if not hasattr(self, "_scope_cache"):
            # Grouped once. Scanning every unit per question is
            # O(questions x units) -- 1,658 x ~20,000 on the real dataset.
            grouped: dict[str, list[str]] = {}
            for doc in self._doc_units():
                grouped.setdefault(str(doc.meta.get("doc_name")), []).append(doc.doc_id)
            self._scope_cache = grouped
        return self._scope_cache.get(qa.doc_name, [])

    def _doc_units(self) -> list[SourceDoc]:
        if not hasattr(self, "_units_cache"):
            self._units_cache = list(self.corpus())
        return self._units_cache

    def _layout_index(self, doc_name: str) -> list[dict[str, Any]] | None:
        if not hasattr(self, "_layouts_cache"):
            self._layouts_cache: dict[str, list[dict[str, Any]]] | None = None
            path = self.root / LAYOUTS
            if path.exists():
                grouped: dict[str, list[dict[str, Any]]] = {}
                for row in iter_parquet(
                    path, ["doc_name", "layout_id", "page_id", "type", "bbox"]
                ):
                    grouped.setdefault(canonical_doc(row["doc_name"]), []).append(row)
                self._layouts_cache = grouped
        if self._layouts_cache is None:
            return None
        return self._layouts_cache.get(doc_name)


def _fmt(bbox: tuple[Any, ...]) -> str:
    return ",".join(f"{float(v):.1f}" for v in bbox) if bbox else "none"


MIN_IOU = 0.5


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    overlap = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    area_b = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def _match_bbox(
    rows: list[dict[str, Any]],
    page: Any,
    bbox: tuple[Any, ...],
    min_iou: float = MIN_IOU,
) -> dict[str, Any] | None:
    """Join an annotation bbox to a layout row on the same page, by overlap.

    Both sides store absolute pixel coordinates (verified: 2,638 annotation
    boxes, all absolute, none normalized), but they come from different
    pipelines, so exact equality matches nothing and a fixed pixel tolerance is
    wrong at both extremes -- 2px is too tight for a re-detected box and too
    loose for the 18px-tall boxes in this set.

    IoU is scale-invariant, which is what makes one threshold work across page
    sizes ranging from 612x792 to 880x1583. Best overlap on the page wins, and
    anything below the floor is reported as a miss rather than silently taking
    the nearest box -- a wrong layout id is worse than an honest gap.
    """
    if len(bbox) != 4:
        return None
    target = tuple(float(v) for v in bbox)  # type: ignore[assignment]
    best, best_iou = None, 0.0
    for row in rows:
        if str(row.get("page_id")) != str(page):
            continue
        candidate = row.get("bbox") or []
        if len(candidate) != 4:
            continue
        score = _iou(target, tuple(float(v) for v in candidate))  # type: ignore[arg-type]
        if score > best_iou:
            best, best_iou = row, score
    return best if best_iou >= min_iou else None
