"""Research-only hierarchical retrieval primitives.

The production retrieval service deliberately stays untouched while the
Physics experiment is being developed.  This module turns the existing KDL
output into a stable file -> page -> evidence-atom hierarchy and provides the
small, explainable cascade used by the experiment runner.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.chunking_embedding.fields import sentence_spans
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks
from src.retrieval.sparse import BM25Index


LEVELS = frozenset({"file", "page", "block", "paragraph", "sentence", "evidence_atom"})
# Captions stay atomic along with tables, figures and equations.  A caption can
# contain prose, but splitting it into sentence units can separate the visual
# reference from the claim it describes.
TEXT_KINDS = frozenset({"text", "heading"})
SKIP_KINDS = frozenset({"boilerplate", "page_header", "page_footer", "page_number"})


@dataclass(frozen=True)
class HierarchyNode:
    """One addressable hierarchy node with an explicit parent."""

    node_id: str
    level: str
    parent_id: str | None
    page_id: str | None
    file_id: str
    text: str
    start_char: int | None = None
    end_char: int | None = None
    block_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"Unknown hierarchy level: {self.level!r}")
        if self.level == "file" and self.parent_id is not None:
            raise ValueError("A file node cannot have a parent")
        if self.level != "file" and not self.parent_id:
            raise ValueError(f"Non-file node {self.node_id!r} needs a parent")

    @property
    def index_text(self) -> str:
        return self.text


@dataclass
class HierarchyCorpus:
    """Canonical nodes and parent/child maps for one parsed corpus."""

    subset: str
    files: dict[str, HierarchyNode]
    pages: dict[str, HierarchyNode]
    blocks: dict[str, HierarchyNode]
    paragraphs: dict[str, HierarchyNode]
    sentences: dict[str, HierarchyNode]
    evidence_atoms: dict[str, HierarchyNode]
    page_order: list[str]
    file_to_pages: dict[str, list[str]]
    page_to_children: dict[str, list[str]]
    file_titles: dict[str, str]

    @classmethod
    def from_parsed_run(
        cls,
        run_dir: Path | str,
        *,
        subset: str = "physics",
        page_ids: Sequence[str] | None = None,
    ) -> "HierarchyCorpus":
        """Build nodes from KDL output while preserving every page in inventory.

        ``page_ids`` should normally come from the V-SPLADE page metadata or
        the benchmark corpus.  Empty parser pages are intentionally retained so
        a visual-only page cannot disappear before the page stage.
        """

        parsed_pages: dict[str, list[dict[str, Any]]] = {}
        file_titles: dict[str, str] = {}
        for document in documents(run_dir):
            file_name = str((document.get("document") or {}).get("file_name") or "")
            doc = canonical_doc(file_name)
            file_id = f"{subset}::{doc}"
            file_titles[file_id] = str((document.get("document") or {}).get("title") or doc)
            for page, blocks in page_blocks(document).items():
                page_id = f"{file_id}#page={int(page)}"
                parsed_pages[page_id] = list(blocks)

        ordered_pages = list(page_ids or sorted(parsed_pages))
        for page_id in parsed_pages:
            if page_id not in ordered_pages:
                ordered_pages.append(page_id)

        files: dict[str, HierarchyNode] = {}
        pages: dict[str, HierarchyNode] = {}
        blocks: dict[str, HierarchyNode] = {}
        paragraphs: dict[str, HierarchyNode] = {}
        sentences: dict[str, HierarchyNode] = {}
        evidence_atoms: dict[str, HierarchyNode] = {}
        file_to_pages: dict[str, list[str]] = defaultdict(list)
        page_to_children: dict[str, list[str]] = defaultdict(list)

        for page_id in ordered_pages:
            file_id = page_id.split("#page=", 1)[0]
            if file_id not in files:
                title = file_titles.get(file_id, file_id.rsplit("::", 1)[-1])
                files[file_id] = HierarchyNode(
                    node_id=file_id,
                    level="file",
                    parent_id=None,
                    page_id=None,
                    file_id=file_id,
                    text=title,
                    metadata={"title": title, "subset": subset},
                )

            page_number = _page_number(page_id)
            page_blocks_for_id = parsed_pages.get(page_id, [])
            page_text = "\n".join(
                str(block.get("text") or "").strip()
                for block in page_blocks_for_id
                if str(block.get("text") or "").strip()
            )
            pages[page_id] = HierarchyNode(
                node_id=page_id,
                level="page",
                parent_id=file_id,
                page_id=page_id,
                file_id=file_id,
                text=page_text,
                metadata={"page": page_number, "has_parser_text": bool(page_text)},
            )
            file_to_pages[file_id].append(page_id)

            active_section = ""
            used_block_ids: set[str] = set()
            for position, raw_block in enumerate(page_blocks_for_id):
                kind = _block_kind(raw_block)
                if kind in SKIP_KINDS:
                    continue
                text = str(raw_block.get("text") or "").strip()
                if not text:
                    continue
                raw_index = raw_block.get("block_index")
                block_index = int(raw_index) if raw_index is not None else position
                block_suffix = str(block_index)
                if block_suffix in used_block_ids:
                    block_suffix = f"{block_index}-{position}"
                used_block_ids.add(block_suffix)
                block_id = f"{page_id}#block={block_suffix}"
                if kind == "heading":
                    active_section = " ".join(text.split())[:240]

                block_node = HierarchyNode(
                    node_id=block_id,
                    level="block",
                    parent_id=page_id,
                    page_id=page_id,
                    file_id=file_id,
                    text=text,
                    block_index=block_index,
                    metadata={
                        "block_type": str(raw_block.get("type") or ""),
                        "kind": kind,
                        "section": active_section,
                    },
                )
                blocks[block_id] = block_node
                page_to_children[page_id].append(block_id)

                context = _context_prefix(
                    file_titles.get(file_id, file_id), active_section, page_number
                )
                if kind in TEXT_KINDS:
                    paragraph_id = f"{block_id}#paragraph=0"
                    paragraph_node = HierarchyNode(
                        node_id=paragraph_id,
                        level="paragraph",
                        parent_id=block_id,
                        page_id=page_id,
                        file_id=file_id,
                        text=text,
                        start_char=0,
                        end_char=len(text),
                        block_index=block_index,
                        metadata={**block_node.metadata, "context": context},
                    )
                    paragraphs[paragraph_id] = paragraph_node

                    spans = sentence_spans(text) or [(0, len(text))]
                    for sentence_index, (start, end) in enumerate(spans):
                        body = text[start:end].strip()
                        if not body:
                            continue
                        sentence_id = f"{block_id}#sentence={sentence_index}"
                        sentences[sentence_id] = HierarchyNode(
                            node_id=sentence_id,
                            level="sentence",
                            parent_id=paragraph_id,
                            page_id=page_id,
                            file_id=file_id,
                            text=body,
                            start_char=start,
                            end_char=end,
                            block_index=block_index,
                            metadata={**block_node.metadata, "context": context},
                        )
                else:
                    atom_id = f"{block_id}#atom=0"
                    evidence_atoms[atom_id] = HierarchyNode(
                        node_id=atom_id,
                        level="evidence_atom",
                        parent_id=block_id,
                        page_id=page_id,
                        file_id=file_id,
                        text=text,
                        start_char=0,
                        end_char=len(text),
                        block_index=block_index,
                        metadata={**block_node.metadata, "context": context},
                    )

        return cls(
            subset=subset,
            files=files,
            pages=pages,
            blocks=blocks,
            paragraphs=paragraphs,
            sentences=sentences,
            evidence_atoms=evidence_atoms,
            page_order=ordered_pages,
            file_to_pages={key: value for key, value in file_to_pages.items()},
            page_to_children={key: value for key, value in page_to_children.items()},
            file_titles=file_titles,
        )

    @property
    def file_order(self) -> list[str]:
        return list(self.files)

    def nodes_for_unit(self, unit: str) -> list[HierarchyNode]:
        """Return fine nodes without splitting structural blocks blindly."""
        if unit == "block":
            return list(self.blocks.values())
        if unit == "paragraph":
            return list(self.paragraphs.values()) + list(self.evidence_atoms.values())
        if unit in {"sentence_group3", "sentence_group5"}:
            group_size = int(unit.removeprefix("sentence_group"))
            # Structural evidence has no sentence boundary. Keep it in the
            # same fine index instead of silently dropping formulas, tables or
            # figure/caption blocks from sentence-group arms.
            return self._sentence_groups(group_size) + list(self.evidence_atoms.values())
        if unit in {"sentence", "atom", "evidence_atom"}:
            return list(self.sentences.values()) + list(self.evidence_atoms.values())
        if unit in {"none", ""}:
            return []
        raise ValueError(f"Unknown fine unit: {unit!r}")

    def file_texts(self, representation: str, *, page_stub_tokens: int = 64) -> dict[str, str]:
        """Create deterministic file representations without LLM summaries."""
        if representation not in {"all_text", "headings_stubs"}:
            raise ValueError(f"Unknown file representation: {representation!r}")
        output: dict[str, str] = {}
        for file_id, page_ids in self.file_to_pages.items():
            title = self.file_titles.get(file_id, file_id)
            if representation == "all_text":
                body = "\n".join(self.pages[page_id].text for page_id in page_ids)
                output[file_id] = f"{title}\n{body}".strip()
                continue

            headings: list[str] = []
            stubs: list[str] = []
            for page_id in page_ids:
                for node in self._page_nodes(page_id, self.blocks):
                    if node.metadata.get("kind") == "heading":
                        headings.append(node.text)
                tokens = self.pages[page_id].text.split()
                stubs.append(" ".join(tokens[:page_stub_tokens]))
            output[file_id] = f"{title}\n{' '.join(headings)}\n{' '.join(stubs)}".strip()
        return output

    def indexed_text(self, node: HierarchyNode, *, include_context: bool = True) -> str:
        if not include_context:
            return node.text
        context = str(node.metadata.get("context") or "").strip()
        return f"{context}\n{node.text}".strip() if context else node.text

    def _sentence_groups(self, group_size: int) -> list[HierarchyNode]:
        by_parent: dict[str, list[HierarchyNode]] = defaultdict(list)
        for node in self.sentences.values():
            by_parent[node.parent_id or ""].append(node)
        output: list[HierarchyNode] = []
        for parent_id, nodes in by_parent.items():
            ordered = sorted(nodes, key=lambda node: (node.start_char or 0, node.node_id))
            for start in range(0, len(ordered), group_size):
                group = ordered[start : start + group_size]
                if not group:
                    continue
                first, last = group[0], group[-1]
                group_id = f"{first.parent_id}#sentence_group={start // group_size}"
                output.append(
                    HierarchyNode(
                        node_id=group_id,
                        level="sentence",
                        parent_id=parent_id,
                        page_id=first.page_id,
                        file_id=first.file_id,
                        text=" ".join(node.text for node in group),
                        start_char=first.start_char,
                        end_char=last.end_char,
                        block_index=first.block_index,
                        metadata={**dict(first.metadata), "group_size": group_size},
                    )
                )
        return output

    @staticmethod
    def _page_nodes(
        page_id: str, nodes: Mapping[str, HierarchyNode]
    ) -> Iterable[HierarchyNode]:
        return (node for node in nodes.values() if node.page_id == page_id)


@dataclass(frozen=True)
class CascadeConfig:
    """All query-time choices for one reproducible cascade arm."""

    name: str
    file_representation: str = "all_text"
    file_pool: str = "max"
    k_files: int = 10
    page_depth: int = 100
    final_depth: int = 100
    bm25_weight: float = 0.70
    parent_weight: float = 0.15
    fine_unit: str = "none"
    fine_pool: str = "max"
    fine_weight: float = 0.25
    neighbor_window: int = 0
    file_quota: int = 0
    include_context: bool = True
    file_direct_weight: float = 0.50
    file_pool_source: str = "page_base"


def build_bm25(records: Iterable[tuple[str, str]]) -> BM25Index:
    return BM25Index(analyzer_name="plain").build(
        [{"chunk_id": node_id, "doc_id": node_id, "text": text} for node_id, text in records]
    )


def normalise_scores(scores: Mapping[str, float]) -> dict[str, float]:
    positive = [float(value) for value in scores.values() if float(value) > 0]
    maximum = max(positive, default=0.0)
    if maximum <= 0:
        return {key: 0.0 for key in scores}
    return {key: max(0.0, float(value)) / maximum for key, value in scores.items()}


def aggregate_file_scores(
    page_scores: Mapping[str, float],
    page_to_file: Mapping[str, str],
    file_ids: Sequence[str],
    pool: str,
) -> dict[str, float]:
    """Pool page evidence into a deterministic file score."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for page_id, score in page_scores.items():
        file_id = page_to_file.get(page_id)
        if file_id is not None:
            grouped[file_id].append(max(0.0, float(score)))

    output: dict[str, float] = {}
    for file_id in file_ids:
        values = sorted(grouped.get(file_id, []), reverse=True)
        if pool == "max":
            output[file_id] = values[0] if values else 0.0
        elif pool == "sum_top2":
            output[file_id] = sum(values[:2])
        elif pool == "coverage":
            # Rank-weighted support rewards several independently matching
            # pages, while the tiny max term makes ties deterministic.
            output[file_id] = sum(
                1.0 / (10.0 + rank) for rank in range(1, min(10, len(values)) + 1)
            ) + (values[0] * 1e-3 if values else 0.0)
        else:
            raise ValueError(f"Unknown file pool: {pool!r}")
    return output


def sort_scores(scores: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(
        ((key, float(value)) for key, value in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )


def diversify_pages(
    ranked: Sequence[tuple[str, float]],
    page_to_file: Mapping[str, str],
    *,
    max_pages_per_file: int,
    depth: int,
) -> list[tuple[str, float]]:
    if max_pages_per_file <= 0:
        return list(ranked[:depth])
    selected: list[tuple[str, float]] = []
    counts: dict[str, int] = defaultdict(int)
    deferred: list[tuple[str, float]] = []
    for page_id, score in ranked:
        file_id = page_to_file.get(page_id, page_id)
        if counts[file_id] < max_pages_per_file and len(selected) < depth:
            selected.append((page_id, score))
            counts[file_id] += 1
        else:
            deferred.append((page_id, score))
    if len(selected) < depth:
        selected.extend(deferred[: depth - len(selected)])
    return selected[:depth]


def node_index_records(
    nodes: Iterable[HierarchyNode],
    corpus: HierarchyCorpus,
    *,
    include_context: bool,
) -> list[tuple[str, str]]:
    return [
        (node.node_id, corpus.indexed_text(node, include_context=include_context))
        for node in nodes
        if node.text.strip()
    ]


def _page_number(page_id: str) -> int:
    try:
        return int(page_id.rsplit("#page=", 1)[-1])
    except ValueError:
        return -1


def _block_kind(block: Mapping[str, Any]) -> str:
    raw_type = str(block.get("type") or "").casefold()
    if raw_type in {"pageheader", "page_header"}:
        return "page_header"
    if raw_type in {"pagefooter", "page_footer"}:
        return "page_footer"
    if raw_type in {"pagenumber", "page_number"}:
        return "page_number"
    if raw_type in {"sectionheader", "title", "heading"}:
        return "heading"
    if raw_type in {"table", "tablegroup", "table_row", "table_header"}:
        return "table"
    if raw_type in {"equationblock", "formula", "equation"}:
        return "formula"
    if raw_type in {"figure", "image", "diagram"}:
        return "figure"
    if raw_type == "caption":
        return "caption"
    return "text"


def _context_prefix(title: str, section: str, page_number: int) -> str:
    parts = [part for part in (title, section, f"page {page_number}") if part]
    return " | ".join(parts)
