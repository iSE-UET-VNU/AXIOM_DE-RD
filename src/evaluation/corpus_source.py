"""Where retrieval units come from, so the runner need not know.

A benchmark ships its own text; a pipeline run ships parsed blocks. Both reduce
to the same thing -- indexable units plus an identity that names the corpus in
the run cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, runtime_checkable
import hashlib
import json

from src.retrieval.protocol import ChunkRecord


@runtime_checkable
class CorpusSource(Protocol):
    def units(self) -> Iterable[ChunkRecord]: ...

    def corpus_identity(self) -> str: ...


def content_identity(benchmark: Any) -> str:
    """Content-derived name for the corpus an arm ran over."""
    describe = getattr(benchmark, "corpus_identity", None)
    if callable(describe):
        return str(describe())

    path = getattr(benchmark, "corpus_path", None)
    if path is None:
        return "builtin"
    path = Path(path)
    if not path.exists():
        # Stem-qualified, so two absent corpora still differ; not a sentinel.
        return f"{path.stem}-missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"{path.stem}-{digest.hexdigest()[:8]}"


class BenchmarkCorpus:
    """The benchmark's own text, optionally re-chunked."""

    def __init__(self, benchmark: Any, chunker: str = "",
                 params: dict | None = None, prefix: bool = False) -> None:
        self.benchmark = benchmark
        self.chunker = chunker
        self.params = params or {}
        self.prefix = prefix

    def units(self) -> Iterator[ChunkRecord]:
        docs = list(self.benchmark.corpus())
        records = self._chunked(docs) if self.chunker else self._pages(docs)
        return iter([r for r in records if r.text.strip()])

    def corpus_identity(self) -> str:
        return content_identity(self.benchmark)

    def _pages(self, docs: list) -> list[ChunkRecord]:
        return [
            ChunkRecord(
                chunk_id=doc.doc_id,
                doc_id=doc.doc_id,
                text=doc.text or "",
                page=doc.page,
                meta={"modality": doc.modality, **dict(doc.meta)},
            )
            for doc in docs
        ]

    def _chunked(self, docs: list) -> list[ChunkRecord]:
        from .chunking import chunk_corpus

        source = {doc.doc_id: doc for doc in docs}
        documents = [
            {
                "doc_id": doc.doc_id,
                "title": doc.meta.get("title", ""),
                "text": doc.text or "",
                "blocks": doc.meta.get("blocks") or [],
            }
            for doc in docs
        ]
        # Carried through chunking, or the per-modality breakdown silently empties.
        return [
            ChunkRecord(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.index_text,
                page=getattr(source.get(c.doc_id), "page", ""),
                meta={
                    "modality": getattr(source.get(c.doc_id), "modality", ""),
                    **dict(getattr(source.get(c.doc_id), "meta", {}) or {}),
                    "chunk_kind": c.kind,
                },
            )
            for c in chunk_corpus(documents, self.chunker, self.params, self.prefix)
        ]


GRANULARITIES = ("page", "chunk", "content")

# Tags that carry row/column or item association blocks[].text flattens away.
STRUCTURAL_HTML = ("<table", "<ul", "<ol")

# Per-document measurements, not settings; a one-document run would otherwise
# fold them into the identity and never reproduce.
VOLATILE = frozenset({
    "image_files", "image_filtering", "label_counts", "latency_seconds",
    "raw_chandra_outputs", "raw_kdl_outputs", "raw_metadata_path", "raw_output_path",
    "reading_order_complete", "source_block_count", "status",
})


def _parser_metadata(path: Path) -> dict:
    """Settings live in the ingested stage; later stages drop the parsed block."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = ((payload.get("parsed") or {}).get("metadata") or {})
    if not meta:
        # Consolidated output documents retain the complete ingestion payload.
        # Prefer it before requiring a separately downloaded ingested-stage
        # artifact; retrieval only needs these settings to build a collision-safe
        # corpus identity.
        meta = (
            (((payload.get("ingest") or {}).get("data") or {}).get("metadata"))
            or {}
        )
    if meta:
        return meta
    parts = list(path.parts)
    if "output" in parts:
        sibling = Path(*[("ingested" if p == "output" else p) for p in parts])
        if sibling.is_file():
            payload = json.loads(sibling.read_text(encoding="utf-8"))
            return ((payload.get("parsed") or {}).get("metadata") or {})
    return {}


def _projection(block: dict, structured: bool) -> str:
    """HTML only where it adds structure; a <p> wrapper is noise in the index."""
    if not structured:
        return str(block.get("text") or "")
    html = str(block.get("html") or "")
    if any(tag in html for tag in STRUCTURAL_HTML):
        return html
    return str(block.get("text") or "")


def parser_identity(run_dir: Path | str) -> str:
    """SHA over the parser settings recorded in a run, not just the parser name.

    Config is whatever is identical across every document; per-document counters
    vary and are excluded. Keyed on the name alone, chandra2 and kdl runs with
    different configs would share a token and report as the same corpus.
    """
    run_dir = Path(run_dir)
    docs = sorted(run_dir.glob("documents/*.json"))
    if not docs:
        raise FileNotFoundError(f"{run_dir} holds no documents/*.json")
    metas = [_parser_metadata(path) for path in docs]
    if not any(metas):
        # Output-stage documents carry no parser block; without it every run
        # hashes to the SHA of {} and two parsers share one cache token.
        raise ValueError(
            f"{run_dir} records no parser metadata, in its own documents or in the "
            "ingested stage beside it. Corpus identity would be constant across "
            "parsers, so the run is refused rather than silently deduplicated."
        )
    shared = {
        key: metas[0][key]
        for key in metas[0]
        if key not in VOLATILE and not key.endswith(("_count", "_seconds"))
        and all(key in m and m[key] == metas[0][key] for m in metas[1:])
    }
    parser = str(shared.get("parser") or metas[0].get("parser") or "")
    if not parser:
        # A sentinel in an identity key is a collision by construction.
        raise ValueError(f"{run_dir} records no parser name; identity would be shared.")
    digest = hashlib.sha256(json.dumps(shared, sort_keys=True, default=str).encode()).hexdigest()
    return f"{parser}-{digest[:8]}"


class PipelineRunCorpus:
    """Units reassembled from a pipeline run directory."""

    def __init__(self, run_dir: Path | str, subset: str, granularity: str = "page",
                 chunker: str = "", params: dict | None = None, prefix: bool = False) -> None:
        if granularity not in GRANULARITIES:
            raise ValueError(f"Unknown granularity {granularity!r}; expected one of {GRANULARITIES}.")
        if granularity == "chunk" and chunker:
            raise ValueError(
                "Chunks are fixed at parse time, so a chunker cannot apply at 'chunk' "
                "granularity. Use --granularity page to vary the chunker, or drop it."
            )
        self.run_dir = Path(run_dir)
        self.subset = subset
        self.granularity = granularity
        self.chunker = chunker
        self.params = params or {}
        self.prefix = prefix

    def units(self) -> Iterator[ChunkRecord]:
        if not sorted(self.run_dir.glob("documents/*.json")):
            raise FileNotFoundError(f"{self.run_dir} holds no documents/*.json")
        if self.granularity == "chunk":
            pages = self._chunks()
        else:
            pages = self._pages(structured=self.granularity == "content")
        records = self._rechunked(pages) if self.chunker else pages
        return iter([r for r in records if r.text.strip()])

    def corpus_identity(self) -> str:
        # Subset and projection as well as parser: same run, different corpus.
        projection = "" if self.granularity == "page" else f"-{self.granularity}"
        return f"{self.subset}-{parser_identity(self.run_dir)}{projection}"

    def _pages(self, structured: bool = False) -> list[ChunkRecord]:
        from .pipeline_pages import canonical_doc, documents, page_blocks

        parser = parser_identity(self.run_dir).rsplit("-", 1)[0]
        out: list[ChunkRecord] = []
        for document in documents(self.run_dir):
            doc = canonical_doc(document.get("document", {}).get("file_name"))
            for page, blocks in page_blocks(document).items():
                unit = f"{self.subset}::{doc}#page={page}"
                out.append(ChunkRecord(
                    chunk_id=unit, doc_id=unit,
                    text="\n".join(_projection(b, structured) for b in blocks
                                    if _projection(b, structured).strip()),
                    page=str(page),
                    meta={"modality": "page", "subset": self.subset, "doc_id": doc,
                          "parser": parser},
                ))
        return out

    def _chunks(self) -> list[ChunkRecord]:
        """Refused: pipeline chunks carry no page, so page-level gold cannot score them."""
        raise NotImplementedError(
            "retrieval.items are character spans of main_text -- position is "
            "{index, start_char, end_char} with no page -- so a chunk cannot be "
            "attributed to the page its gold label refers to. Use granularity "
            "'page' or 'content'. Enabling this needs a chunk-to-page mapping first."
        )

    def _rechunked(self, pages: list[ChunkRecord]) -> list[ChunkRecord]:
        from .chunking import chunk_corpus

        source = {r.doc_id: r for r in pages}
        documents = [{"doc_id": r.doc_id, "title": r.meta.get("doc_id", ""),
                      "text": r.text, "blocks": []} for r in pages]
        return [
            ChunkRecord(chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.index_text,
                        page=getattr(source.get(c.doc_id), "page", ""),
                        meta={**dict(getattr(source.get(c.doc_id), "meta", {}) or {}),
                              "chunk_kind": c.kind})
            for c in chunk_corpus(documents, self.chunker, self.params, self.prefix)
        ]
