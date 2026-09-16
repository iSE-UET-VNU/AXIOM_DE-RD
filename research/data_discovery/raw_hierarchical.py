"""Generic raw-input hierarchical retrieval primitives.

This module deliberately keeps the benchmark adapter separate from the old
ViDoRe Physics runner.  It accepts a small JSONL manifest, materialises one
page record per PDF/image, builds the cached BM25/V-SPLADE inputs, and exposes
the fixed cascade used by the earlier Physics experiment.

The module is offline after the optional source-corpus page map has been
downloaded by the runner.  Qrels are never used while parsing or indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.retrieval.sparse import BM25Index


IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".jfif",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}
PDF_SUFFIX = ".pdf"
VECTOR_DIM = 50_368


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json_dump(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


@dataclass(frozen=True)
class RawDocument:
    doc_id: str
    source: str
    path: Path
    relative_path: str
    mime_type: str
    page_count_hint: int | None
    source_doc_id: str | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def is_pdf(self) -> bool:
        return self.path.suffix.lower() == PDF_SUFFIX

    @property
    def is_image(self) -> bool:
        return self.path.suffix.lower() in IMAGE_SUFFIXES

    def manifest_record(self, root: Path) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "path": self.relative_path,
            "absolute_path": str(self.path),
            "mime_type": self.mime_type,
            "page_count_hint": self.page_count_hint,
            "source_doc_id": self.source_doc_id,
            "sha256": sha256_file(self.path),
            "size_bytes": self.path.stat().st_size,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class RawPage:
    page_id: str
    doc_id: str
    source: str
    source_path: str
    relative_path: str
    page_index: int
    page_number: int
    text: str
    visual_only: bool
    needs_ocr: bool = False
    parse_status: str = "ok"
    parse_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_page_id(doc_id: str, page_index: int) -> str:
    return f"{doc_id}#page={int(page_index)}"


def _infer_mime(path: Path) -> str:
    if path.suffix.lower() == PDF_SUFFIX:
        return "application/pdf"
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return "image/" + path.suffix.lower().lstrip(".")
    return "application/octet-stream"


def load_documents(dataset_root: Path, documents_path: Path | None = None) -> list[RawDocument]:
    """Load the explicit benchmark manifest or derive a stable raw inventory."""
    manifest_path = documents_path or dataset_root / "documents.jsonl"
    documents: list[RawDocument] = []
    if manifest_path.is_file():
        for row in read_jsonl(manifest_path):
            raw_path = Path(str(row["path"]))
            path = raw_path if raw_path.is_absolute() else dataset_root / raw_path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Document path from manifest does not exist: {path}")
            metadata = dict(row.get("metadata") or {})
            documents.append(
                RawDocument(
                    doc_id=str(row["doc_id"]),
                    source=str(row.get("source") or "raw"),
                    path=path,
                    relative_path=str(row.get("path") or path.name),
                    mime_type=str(row.get("mime_type") or _infer_mime(path)),
                    page_count_hint=(
                        int(metadata["page_count"])
                        if metadata.get("page_count") is not None
                        else None
                    ),
                    source_doc_id=(
                        str(metadata["source_doc_id"])
                        if metadata.get("source_doc_id") is not None
                        else None
                    ),
                    metadata=metadata,
                )
            )
    else:
        candidates = sorted(
            path
            for path in dataset_root.rglob("*")
            if path.is_file() and (path.suffix.lower() == PDF_SUFFIX or path.suffix.lower() in IMAGE_SUFFIXES)
        )
        for path in candidates:
            relative = path.relative_to(dataset_root).as_posix()
            digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:16]
            documents.append(
                RawDocument(
                    doc_id=f"raw::file_{digest}",
                    source="raw",
                    path=path.resolve(),
                    relative_path=relative,
                    mime_type=_infer_mime(path),
                    page_count_hint=None,
                )
            )
    if not documents:
        raise FileNotFoundError(f"No PDF or image inputs found under {dataset_root}")
    doc_ids = [document.doc_id for document in documents]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("Duplicate doc_id in input inventory")
    return documents


def load_queries(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        qid = str(row.get("query_id") or row.get("qid") or "")
        if not qid or qid in seen:
            raise ValueError(f"Missing or duplicate query_id: {qid!r}")
        query = str(row.get("query") or "").strip()
        if not query:
            raise ValueError(f"Empty query for {qid}")
        seen.add(qid)
        output.append({**row, "query_id": qid, "query": query})
    return output


def build_page_bm25(pages: Sequence[RawPage]) -> BM25Index:
    return BM25Index(analyzer_name="auto").build(
        {
            "chunk_id": page.page_id,
            "doc_id": page.doc_id,
            "text": page.text,
        }
        for page in pages
    )


def build_file_bm25(pages: Sequence[RawPage]) -> BM25Index:
    grouped: dict[str, list[str]] = {}
    for page in pages:
        grouped.setdefault(page.doc_id, []).append(page.text)
    return BM25Index(analyzer_name="auto").build(
        {
            "chunk_id": doc_id,
            "doc_id": doc_id,
            "text": "\n".join(texts).strip(),
        }
        for doc_id, texts in grouped.items()
    )


def normalise_scores(scores: Mapping[str, float]) -> dict[str, float]:
    maximum = max((float(value) for value in scores.values() if float(value) > 0), default=0.0)
    if maximum <= 0:
        return {key: 0.0 for key in scores}
    return {key: max(0.0, float(value)) / maximum for key, value in scores.items()}


def sort_scores(scores: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(
        ((key, float(value)) for key, value in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )


def aggregate_file_scores(
    page_scores: Mapping[str, float],
    page_to_file: Mapping[str, str],
    file_ids: Sequence[str],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for page_id, score in page_scores.items():
        file_id = page_to_file.get(page_id)
        if file_id is not None:
            grouped.setdefault(file_id, []).append(max(0.0, float(score)))
    return {
        file_id: max(grouped.get(file_id, []), default=0.0)
        for file_id in file_ids
    }
