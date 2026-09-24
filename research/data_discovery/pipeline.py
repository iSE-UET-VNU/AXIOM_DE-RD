"""A small baseline for query-driven ingestion.

The discovery corpus is deliberately separate from the production retrieval
corpus.  It contains one cheap text record per PDF page, while the selected
pages are later passed through the normal ingestion stages.

Page numbers exposed by this module are one-based (the user-facing PDF
convention); ``page_index`` remains zero-based for PDF APIs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence
import json
import re
import zipfile

from src.retrieval.sparse import BM25Index, positions_to_ids

if TYPE_CHECKING:
    from src import ingestion as ingestion_runner


@dataclass(frozen=True)
class PageEvidence:
    """Cheap, page-level evidence produced by the discovery parser."""

    page_id: str
    file_path: str
    source_uri: str
    page_index: int
    page_number: int
    text: str
    needs_ocr: bool = False
    ocr_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Keep the two preparation sources separate so retrieval results remain
    # auditable after the canonical text is rebuilt.
    inspector_text: str | None = None
    ocr_text: str = ""
    text_source: str = "pdf_inspector"
    ocr_applied: bool = False
    ocr_word_count: int = 0
    ocr_mean_confidence: float | None = None
    ocr_seconds: float = 0.0
    ocr_error: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.page_id,
            "doc_id": self.source_uri,
            "text": self.text,
        }


@dataclass(frozen=True)
class DiscoveryHit:
    evidence: PageEvidence
    score: float
    rank: int

    @property
    def page_id(self) -> str:
        return self.evidence.page_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "page_id": self.evidence.page_id,
            "file_path": self.evidence.file_path,
            "source_uri": self.evidence.source_uri,
            "page_index": self.evidence.page_index,
            "page_number": self.evidence.page_number,
            "text": self.evidence.text,
            "text_source": self.evidence.text_source,
            "ocr_applied": self.evidence.ocr_applied,
            "ocr_word_count": self.evidence.ocr_word_count,
            "ocr_mean_confidence": self.evidence.ocr_mean_confidence,
            "needs_ocr": self.evidence.needs_ocr,
            "ocr_reason": self.evidence.ocr_reason,
            "metadata": self.evidence.metadata,
        }


class PdfInspectorPageParser:
    """Extract one cheap full-page text region per PDF page.

    ``pdf-inspector`` is optional in the base install.  The constructor keeps
    its API injectable so discovery can be unit-tested without the extra and
    so callers can share one imported API object across many files.
    """

    def __init__(
        self,
        api: Any | None = None,
        *,
        page_dimensions: Callable[[Path], Sequence[tuple[float, float]]] | None = None,
        page_id_factory: Callable[[str, int], str] | None = None,
    ) -> None:
        from src.ingestion.parsing.pdf_inspector import (
            PdfInspectorClassifier,
            PdfInspectorRegionExtractor,
        )

        self.classifier = PdfInspectorClassifier(api=api)
        self.extractor = PdfInspectorRegionExtractor(api=api)
        self._page_dimensions = page_dimensions or _pdf_page_dimensions
        self._page_id_factory = page_id_factory or page_id

    def parse(
        self,
        path: str | Path,
        *,
        source_uri: str | None = None,
    ) -> list[PageEvidence]:
        file_path = Path(path).resolve()
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Page discovery currently supports PDF only: {path}")

        classification = self.classifier.classify(file_path)
        dimensions = list(self._page_dimensions(file_path))
        page_count = min(int(classification.page_count), len(dimensions))
        if page_count <= 0:
            return []

        page_regions = [
            (
                page_index,
                [[0.0, 0.0, float(width), float(height)]],
            )
            for page_index, (width, height) in enumerate(dimensions[:page_count])
        ]
        extracted = self.extractor.extract_pages(file_path, page_regions)
        uri = source_uri or str(file_path)
        out: list[PageEvidence] = []
        for page_index, regions in enumerate(extracted):
            region = regions[0] if regions else None
            text = region.text.strip() if region else ""
            out.append(
                PageEvidence(
                    page_id=getattr(self, "_page_id_factory", page_id)(
                        uri, page_index
                    ),
                    file_path=str(file_path),
                    source_uri=uri,
                    page_index=page_index,
                    page_number=page_index + 1,
                    text=text,
                    needs_ocr=(
                        bool(region.needs_ocr)
                        if region
                        else page_index in classification.pages_needing_ocr
                    ),
                    ocr_reason=(region.ocr_reason if region else None),
                    metadata={
                        "pdf_type": classification.pdf_type,
                        "pdf_confidence": classification.confidence,
                        "classification_latency_ms": classification.latency_ms,
                    },
                )
            )
        return out


@dataclass
class PageIndex:
    """Persistent BM25 index over page evidence."""

    pages: list[PageEvidence] = field(default_factory=list)
    bm25: BM25Index = field(default_factory=BM25Index)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, pages: Iterable[PageEvidence]) -> "PageIndex":
        page_list = list(pages)
        records = [page.as_record() for page in page_list]
        return cls(pages=page_list, bm25=BM25Index(analyzer_name="auto").build(records))

    def apply_ppocrv5_jsonl(
        self, path: str | Path, *, strict: bool = True
    ) -> dict[str, Any]:
        """Merge notebook PP-OCRv5 page text into the light page index.

        The notebook emits one JSON object per page with a ``unit`` field. The
        loader accepts canonical page IDs as well as ``doc_id`` plus a
        zero-based ``page_index`` so artifacts made by older bundle versions
        remain usable. No Tesseract output is read or used here.
        """
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"PP-OCRv5 JSONL does not exist: {source}")

        aliases: dict[str, PageEvidence] = {}
        locations: dict[tuple[str, int], PageEvidence] = {}
        by_doc: dict[str, list[PageEvidence]] = defaultdict(list)
        for page in self.pages:
            page_source = str(page.source_uri)
            for value in (
                page.page_id,
                f"{page_source}#page={page.page_index}",
                f"{page_source}#page={page.page_number}",
            ):
                aliases[value] = page
            locations[(str(Path(page.file_path).resolve()), page.page_index)] = page
            by_doc[page_source].append(page)

        if source.suffix.lower() == ".zip":
            with zipfile.ZipFile(source) as archive:
                candidates = [
                    name
                    for name in archive.namelist()
                    if name.endswith("light_ocr_ppocrv5.jsonl")
                ]
                if not candidates:
                    raise FileNotFoundError(
                        f"No light_ocr_ppocrv5.jsonl found in {source}"
                    )
                lines = archive.read(candidates[0]).decode("utf-8").splitlines()
        else:
            lines = source.read_text(encoding="utf-8").splitlines()

        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid PP-OCRv5 JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"PP-OCRv5 line {line_number} must contain a JSON object"
                )
            rows.append(value)
        if not rows:
            raise ValueError(f"PP-OCRv5 artifact is empty: {source}")

        matched: dict[str, dict[str, Any]] = {}
        unmatched: list[str] = []
        duplicate: list[str] = []
        for row in rows:
            page = _resolve_ppocr_page(row, aliases, locations, by_doc)
            unit = str(row.get("unit") or row.get("page_id") or "")
            if page is None:
                unmatched.append(unit or "<missing unit>")
                continue
            if page.page_id in matched:
                duplicate.append(unit or page.page_id)
                # Resume checkpoints are append-only; the latest row is the
                # most recent OCR attempt for this page.
            matched[page.page_id] = row

        if strict and unmatched:
            details = []
            if unmatched:
                details.append(f"unmatched={unmatched[:5]}")
            raise ValueError(
                "PP-OCRv5 artifact does not align with the pdf-inspector page "
                + "index (" + ", ".join(details) + ")"
            )

        updated: list[PageEvidence] = []
        appended = 0
        empty_outputs = 0
        for page in self.pages:
            row = matched.get(page.page_id)
            if row is None:
                updated.append(page)
                continue
            ocr_text = str(row.get("text") or "").strip()
            if not ocr_text:
                empty_outputs += 1
            native = page.inspector_text if page.inspector_text is not None else page.text
            canonical = "\n".join(part for part in (native.strip(), ocr_text) if part)
            applied = bool(ocr_text)
            if applied:
                appended += 1
            word_count = row.get("ocr_word_count")
            try:
                word_count = int(word_count) if word_count is not None else len(ocr_text.split())
            except (TypeError, ValueError):
                word_count = len(ocr_text.split())
            confidence = row.get("ocr_mean_confidence")
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None
            seconds = row.get("seconds", row.get("ocr_seconds", 0.0))
            try:
                seconds = float(seconds or 0.0)
            except (TypeError, ValueError):
                seconds = 0.0
            updated.append(
                replace(
                    page,
                    text=canonical,
                    inspector_text=native,
                    ocr_text=ocr_text,
                    text_source="pdf_inspector+ppocrv5" if applied else page.text_source,
                    ocr_applied=applied,
                    ocr_word_count=word_count,
                    ocr_mean_confidence=confidence,
                    ocr_seconds=seconds,
                    ocr_error=str(row.get("error")) if row.get("error") else None,
                )
            )

        self.pages = updated
        self.bm25 = BM25Index(analyzer_name="auto").build(
            [page.as_record() for page in self.pages]
        )
        stats = {
            "artifact": str(source),
            "rows": len(rows),
            "matched_pages": len(matched),
            "appended_pages": appended,
            "empty_outputs": empty_outputs,
            "error_rows": sum(bool(row.get("error")) for row in matched.values()),
            "unmatched_rows": len(unmatched),
            "duplicate_rows": len(duplicate),
            "strict": strict,
            "duplicate_policy": "latest_row_wins",
            "merge_policy": "inspector_text + ppocrv5_text",
        }
        self.metadata["preparation"] = "pdf_inspector -> ppocrv5"
        self.metadata["ppocrv5"] = stats
        return stats

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        allowed_page_ids: set[str] | None = None,
    ) -> list[DiscoveryHit]:
        if top_k <= 0:
            return []
        allowed_positions = None
        if allowed_page_ids is not None:
            allowed_positions = {
                position
                for position, page in enumerate(self.pages)
                if page.page_id in allowed_page_ids
            }
        hits = positions_to_ids(
            self.bm25,
            self.bm25.search(query, top_k, allowed_positions),
        )
        by_id = {page.page_id: page for page in self.pages}
        return [
            DiscoveryHit(evidence=by_id[page_id], score=score, rank=rank)
            for rank, (page_id, score) in enumerate(hits, start=1)
            if page_id in by_id
        ]

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "pages.jsonl").write_text(
            "".join(
                json.dumps(_page_to_dict(page), ensure_ascii=False) + "\n"
                for page in self.pages
            ),
            encoding="utf-8",
        )
        self.bm25.save(root / "bm25.json")
        (root / "metadata.json").write_text(
            json.dumps(
                {
                    "contract_version": "page-discovery-v2",
                    "page_count": len(self.pages),
                    **self.metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "PageIndex":
        root = Path(directory)
        pages = [
            _page_from_dict(json.loads(line))
            for line in (root / "pages.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metadata_path = root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        return cls(
            pages=pages,
            bm25=BM25Index.load(root / "bm25.json"),
            metadata=metadata,
        )


def build_page_index(
    files: Iterable[str | Path],
    *,
    parser: PdfInspectorPageParser | None = None,
    source_uri: Callable[[Path], str] | None = None,
) -> PageIndex:
    """Parse a PDF collection once and return a reusable page BM25 index."""

    page_parser = parser or PdfInspectorPageParser()
    pages: list[PageEvidence] = []
    for path in sorted((Path(item) for item in files), key=lambda value: str(value)):
        if path.suffix.lower() != ".pdf":
            continue
        uri = source_uri(path) if source_uri else str(path)
        pages.extend(page_parser.parse(path, source_uri=uri))
    return PageIndex.build(pages)


@dataclass
class OnDemandResult:
    query: str
    hits: list[DiscoveryHit]
    selected_pages: dict[str, list[int]]
    ingestion: ingestion_runner.IngestionOutput
    cleaned: Any = None
    enriched: Any = None
    chunking_embedding: Any = None


def run_on_demand(
    index: PageIndex,
    query: str,
    *,
    parser_config: dict[str, Any],
    top_k_pages: int = 10,
    chunking_config: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
    work_dir: str | Path | None = None,
) -> OnDemandResult:
    """Discover pages, selectively ingest them, and run main data stages.

    The accurate parser receives a temporary PDF containing only selected
    pages.  Parsed page coordinates are restored to their original document
    coordinates before the result reaches cleaning and enrichment.
    """

    hits = index.search(query, top_k=top_k_pages)
    selected = _group_selected_pages(hits)
    ingestion_output = _ingest_selected_pages(
        selected,
        parser_config=parser_config,
        project_root=project_root,
        work_dir=work_dir,
    )

    from src import cleaning, enrichment

    cleaned = cleaning.run(ingestion_output.parsed_data, ingestion_output.initial_schemas)
    enriched = enrichment.run(cleaned.cleaned_data, cleaned.cleaned_schemas)
    chunking = None
    if chunking_config is not None:
        from src.chunking_embedding.stage import run as run_chunking_embedding

        chunking = run_chunking_embedding(
            [record.__dict__ for record in enriched.enriched_data],
            chunking_config,
        )
    return OnDemandResult(
        query=query,
        hits=hits,
        selected_pages=selected,
        ingestion=ingestion_output,
        cleaned=cleaned,
        enriched=enriched,
        chunking_embedding=chunking,
    )


def run_selected_pages(
    selected: dict[str, Sequence[int]],
    *,
    parser_config: dict[str, Any],
    chunking_config: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
    work_dir: str | Path | None = None,
    one_page_inputs: bool = True,
) -> OnDemandResult:
    """Ingest a preselected page set through the normal pipeline stages.

    ``one_page_inputs`` is enabled by default for page-level data discovery:
    every selected PDF page becomes one logical ingestion document.  This
    preserves the page boundary when the production chunker creates retrieval
    records and makes the result safe to filter back to a query's selected
    pages.
    """

    normalized = {
        str(path): sorted({int(index) for index in indices})
        for path, indices in selected.items()
        if indices
    }
    ingestion_output = _ingest_selected_pages(
        normalized,
        parser_config=parser_config,
        project_root=project_root,
        work_dir=work_dir,
        one_page_inputs=one_page_inputs,
    )

    from src import cleaning, enrichment

    cleaned = cleaning.run(ingestion_output.parsed_data, ingestion_output.initial_schemas)
    enriched = enrichment.run(cleaned.cleaned_data, cleaned.cleaned_schemas)
    chunking = None
    if chunking_config is not None:
        from src.chunking_embedding.stage import run as run_chunking_embedding

        chunking = run_chunking_embedding(
            [record.__dict__ for record in enriched.enriched_data],
            chunking_config,
        )
    return OnDemandResult(
        query="",
        hits=[],
        selected_pages=normalized,
        ingestion=ingestion_output,
        cleaned=cleaned,
        enriched=enriched,
        chunking_embedding=chunking,
    )


def run_from_parse_artifacts(
    selected: dict[str, Sequence[int]],
    *,
    parser_artifacts_dir: str | Path,
    project_root: str | Path | None = None,
    chunking_config: dict[str, Any] | None = None,
) -> OnDemandResult:
    """Continue the pipeline from persisted KDL parser artifacts.

    The artifacts are written one per selected page by KDL's raw-output
    persistence.  This path reconstructs the provider-neutral ``ParsedData``
    objects and runs the downstream stages without rendering PDFs or calling
    the parser endpoint again.
    """
    normalized = {
        str(path): sorted({int(index) for index in indices})
        for path, indices in selected.items()
        if indices
    }
    ingestion_output = _load_parse_artifacts(
        normalized,
        parser_artifacts_dir=parser_artifacts_dir,
        project_root=project_root,
    )

    from src import cleaning, enrichment

    cleaned = cleaning.run(ingestion_output.parsed_data, ingestion_output.initial_schemas)
    enriched = enrichment.run(cleaned.cleaned_data, cleaned.cleaned_schemas)
    chunking = None
    if chunking_config is not None:
        from src.chunking_embedding.stage import run as run_chunking_embedding

        chunking = run_chunking_embedding(
            [record.__dict__ for record in enriched.enriched_data],
            chunking_config,
        )
    return OnDemandResult(
        query="",
        hits=[],
        selected_pages=normalized,
        ingestion=ingestion_output,
        cleaned=cleaned,
        enriched=enriched,
        chunking_embedding=chunking,
    )


def _load_parse_artifacts(
    selected: dict[str, list[int]],
    *,
    parser_artifacts_dir: str | Path,
    project_root: str | Path | None,
) -> Any:
    from src.ingestion import runner as ingestion_runner
    from src.ingestion.parsing import infer_initial_schema
    from src.ingestion.parsing.kdl import _build_extraction, _source_blocks
    from src.models import ParsedData

    artifact_root = Path(parser_artifacts_dir)
    raw_by_ordinal: dict[int, tuple[Path, dict[str, Any]]] = {}
    for raw_path in artifact_root.rglob("result.json"):
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        source_file = str(payload.get("source_file") or "")
        match = re.search(r"(?:^|[\\/])(\d+)-[^\\/]+\.pdf$", source_file)
        if match:
            raw_by_ordinal[int(match.group(1))] = (raw_path, payload)

    ordered_pages = [
        (source_path, page_index)
        for source_path, page_indices in sorted(selected.items())
        for page_index in sorted(page_indices)
    ]
    expected = set(range(len(ordered_pages)))
    if set(raw_by_ordinal) != expected:
        missing = sorted(expected - set(raw_by_ordinal))
        extra = sorted(set(raw_by_ordinal) - expected)
        raise RuntimeError(
            "Parser artifact ordinals do not match selected pages: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    output = ingestion_runner.IngestionOutput()
    for ordinal, (source_path, page_index) in enumerate(ordered_pages):
        raw_path, raw = raw_by_ordinal[ordinal]
        original = Path(source_path).resolve()
        source_uri = f"{original}#page={page_index + 1}"
        input_metadata = {
            "file_name": original.name,
            "discovery_original_path": str(original),
            "discovery_page_indices": [page_index],
            "discovery_page_numbers": [page_index + 1],
        }
        data_object = ingestion_runner._build_data_object(
            original,
            source_uri=source_uri,
            input_metadata=input_metadata,
            project_root=project_root,
        )
        source_blocks = _source_blocks(raw.get("pages") or [])
        markdown = str(raw.get("markdown") or "")
        extraction = _build_extraction(markdown, source_blocks)
        metadata: dict[str, Any] = {
            "parser": "kdl_pdf_inspector",
            "method": "vllm",
            "page_count": len(raw.get("pages") or []),
            "raw_output_path": str(raw_path.with_suffix(".md")),
            "raw_metadata_path": str(raw_path),
            **input_metadata,
        }
        if isinstance(raw.get("usage"), dict):
            metadata["kdl_usage"] = raw["usage"]
        if isinstance(raw.get("_routing"), dict):
            metadata.update(raw["_routing"])
        if isinstance(raw.get("_global_scheduler"), dict):
            metadata["kdl_global_scheduler"] = raw["_global_scheduler"]

        parsed = ParsedData(
            object_id=data_object.object_id,
            source_uri=source_uri,
            source_format=str(data_object.metadata.get("format") or "pdf"),
            rows=[
                {
                    "extraction": extraction,
                    "text": extraction["main_text"],
                    "source_blocks": source_blocks,
                    "reading_order": [block["component_id"] for block in source_blocks],
                }
            ],
            text=markdown,
            metadata=metadata,
        )
        _restore_original_page_coordinates(parsed)
        output.data_objects.append(data_object)
        output.parsed_data.append(parsed)
        output.initial_schemas.append(infer_initial_schema(parsed))
    return output


def page_id(source_uri: str, page_index: int) -> str:
    return f"{source_uri}#page={page_index + 1}"


def _group_selected_pages(hits: Sequence[DiscoveryHit]) -> dict[str, list[int]]:
    grouped: dict[str, set[int]] = defaultdict(set)
    for hit in hits:
        grouped[hit.evidence.file_path].add(hit.evidence.page_index)
    return {path: sorted(indices) for path, indices in sorted(grouped.items())}


def _ingest_selected_pages(
    selected: dict[str, list[int]],
    *,
    parser_config: dict[str, Any],
    project_root: str | Path | None,
    work_dir: str | Path | None,
    one_page_inputs: bool = False,
) -> ingestion_runner.IngestionOutput:
    from src import ingestion as ingestion_runner

    if not selected:
        return ingestion_runner.IngestionOutput()

    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    manager = TemporaryDirectory(prefix="axiom-discovery-", dir=str(work_dir) if work_dir else None)
    try:
        root = Path(manager.name)
        inputs: list[ingestion_runner.IngestionInput] = []
        for source_path, page_indices in selected.items():
            original = Path(source_path).resolve()
            if one_page_inputs:
                input_pages = [[index] for index in page_indices]
            else:
                input_pages = [list(page_indices)]
            for selected_indices in input_pages:
                subset_path = root / f"{len(inputs):06d}-{original.name}"
                _write_page_subset(original, selected_indices, subset_path)
                page_suffix = (
                    f"#page={selected_indices[0] + 1}"
                    if len(selected_indices) == 1
                    else ""
                )
                inputs.append(
                    ingestion_runner.IngestionInput(
                        path=subset_path,
                        source_uri=str(original) + page_suffix,
                        metadata={
                            "file_name": original.name,
                            "discovery_original_path": str(original),
                            "discovery_page_indices": selected_indices,
                            "discovery_page_numbers": [
                                index + 1 for index in selected_indices
                            ],
                        },
                    )
                )
        output = ingestion_runner.run_many(
            inputs,
            parser_config=parser_config,
            project_root=project_root,
        )
        metadata_by_id = {
            item.object_id: item.metadata for item in output.data_objects
        }
        for parsed in output.parsed_data:
            _copy_discovery_metadata(parsed, metadata_by_id.get(parsed.object_id, {}))
            _restore_original_page_coordinates(parsed)
        for quarantined in output.quarantined_documents:
            _copy_discovery_metadata(
                quarantined.parsed,
                quarantined.source.metadata,
            )
            _restore_original_page_coordinates(quarantined.parsed)
        return output
    finally:
        manager.cleanup()


def _write_page_subset(source: Path, page_indices: Sequence[int], target: Path) -> None:
    import pymupdf as fitz

    with fitz.open(str(source)) as document, fitz.open() as subset:
        count = len(document)
        wanted = sorted(set(int(index) for index in page_indices))
        if any(index < 0 or index >= count for index in wanted):
            raise IndexError(f"Selected page is outside {source.name}: {wanted}")
        for index in wanted:
            # Links/annotations/widgets are irrelevant to page parsing and
            # malformed named destinations in some DocBench PDFs can make
            # PyMuPDF fail while copying them into the temporary subset.
            subset.insert_pdf(
                document,
                from_page=index,
                to_page=index,
                links=False,
                annots=False,
                widgets=False,
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        subset.save(str(target))


def _restore_original_page_coordinates(parsed: Any) -> None:
    metadata = getattr(parsed, "metadata", {})
    selected = metadata.get("discovery_page_indices")
    if not isinstance(selected, list):
        return
    mapping = {local: int(original) for local, original in enumerate(selected)}
    metadata["discovery_page_coordinate_system"] = "original_zero_based"
    metadata["discovery_selected_page_indices"] = list(selected)
    metadata["discovery_selected_page_count"] = len(selected)
    metadata["page_count_before_discovery"] = metadata.get("page_count")
    metadata["page_count"] = len(selected)

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if isinstance(value.get("page"), int):
                value["page"] = mapping.get(value["page"], value["page"])
            if isinstance(value.get("reading_order"), list):
                value["reading_order"] = [
                    _rebase_component_id(item, mapping)
                    if isinstance(item, str)
                    else item
                    for item in value["reading_order"]
                ]
            for key in ("component_id",):
                item = value.get(key)
                if isinstance(item, str):
                    value[key] = _rebase_component_id(item, mapping)
            for item in value.values():
                visit(item)

    visit(getattr(parsed, "rows", []))
    if isinstance(parsed.text, str):
        parsed.metadata["text_page_mapping"] = mapping


def _copy_discovery_metadata(parsed: Any, source_metadata: dict[str, Any]) -> None:
    for key in (
        "discovery_original_path",
        "discovery_page_indices",
        "discovery_page_numbers",
    ):
        if key in source_metadata:
            parsed.metadata[key] = source_metadata[key]


def _rebase_component_id(value: str, mapping: dict[int, int]) -> str:
    match = re.search(r"(/page/)(\d+)(/)", value)
    if not match:
        return value
    local = int(match.group(2))
    original = mapping.get(local, local)
    return f"{value[:match.start(2)]}{original}{value[match.end(2):]}"


def _pdf_page_dimensions(path: Path) -> Sequence[tuple[float, float]]:
    import pymupdf as fitz

    with fitz.open(str(path)) as document:
        return [(float(page.rect.width), float(page.rect.height)) for page in document]


def _page_to_dict(page: PageEvidence) -> dict[str, Any]:
    return {
        "page_id": page.page_id,
        "file_path": page.file_path,
        "source_uri": page.source_uri,
        "page_index": page.page_index,
        "page_number": page.page_number,
        "text": page.text,
        "needs_ocr": page.needs_ocr,
        "ocr_reason": page.ocr_reason,
        "inspector_text": page.inspector_text,
        "ocr_text": page.ocr_text,
        "text_source": page.text_source,
        "ocr_applied": page.ocr_applied,
        "ocr_word_count": page.ocr_word_count,
        "ocr_mean_confidence": page.ocr_mean_confidence,
        "ocr_seconds": page.ocr_seconds,
        "ocr_error": page.ocr_error,
        "metadata": page.metadata,
    }


def _page_from_dict(value: dict[str, Any]) -> PageEvidence:
    return PageEvidence(
        page_id=str(value["page_id"]),
        file_path=str(value["file_path"]),
        source_uri=str(value["source_uri"]),
        page_index=int(value["page_index"]),
        page_number=int(value["page_number"]),
        text=str(value.get("text") or ""),
        needs_ocr=bool(value.get("needs_ocr", False)),
        ocr_reason=value.get("ocr_reason"),
        inspector_text=value.get("inspector_text"),
        ocr_text=str(value.get("ocr_text") or ""),
        text_source=str(value.get("text_source") or "pdf_inspector"),
        ocr_applied=bool(value.get("ocr_applied", False)),
        ocr_word_count=int(value.get("ocr_word_count") or 0),
        ocr_mean_confidence=value.get("ocr_mean_confidence"),
        ocr_seconds=float(value.get("ocr_seconds") or 0.0),
        ocr_error=value.get("ocr_error"),
        metadata=dict(value.get("metadata") or {}),
    )


def _resolve_ppocr_page(
    row: dict[str, Any],
    aliases: dict[str, PageEvidence],
    locations: dict[tuple[str, int], PageEvidence],
    by_doc: dict[str, list[PageEvidence]],
) -> PageEvidence | None:
    for key in ("unit", "page_id", "id"):
        value = row.get(key)
        if value is not None and str(value) in aliases:
            return aliases[str(value)]

    doc_id = row.get("doc_id") or row.get("source_uri")
    page_index = row.get("page_index")
    if doc_id is not None and page_index is not None:
        try:
            candidates = by_doc.get(str(doc_id), [])
            for page in candidates:
                if page.page_index == int(page_index):
                    return page
        except (TypeError, ValueError):
            pass

    file_path = row.get("file_path") or row.get("source_path")
    if file_path is not None and page_index is not None:
        try:
            page = locations.get((str(Path(str(file_path)).resolve()), int(page_index)))
            if page is not None:
                return page
        except (TypeError, ValueError):
            pass

    unit = str(row.get("unit") or row.get("page_id") or "")
    match = re.match(r"^(.*)#page=(\d+)$", unit)
    if match:
        doc, number = match.group(1), int(match.group(2))
        candidates = by_doc.get(doc, [])
        for page in candidates:
            if number in {page.page_index, page.page_number}:
                return page
    return None


__all__ = [
    "DiscoveryHit",
    "OnDemandResult",
    "PageEvidence",
    "PageIndex",
    "PdfInspectorPageParser",
    "build_page_index",
    "page_id",
    "run_on_demand",
]
