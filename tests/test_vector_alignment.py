"""The seam between ``chunk_ids.json`` order and ``vectors.npy`` rows.

A row-order mismatch produces a retriever that works, returns results, and is
wrong: ``vectors[i]`` describes a different chunk than ``chunk_ids[i]``, so every
dense score is attached to the wrong text. Nothing errors, every count is
correct, and the numbers look like a weak embedder rather than a broken index.

The spanning assertion is semantic, not structural: **embed a chunk's own text
and its nearest neighbour must be itself.** A length check cannot catch a shift;
this can, because it only holds if the two files agree row for row.

Per the repo testing rules, each test here is demonstrated to fail against the
defect it guards -- see ``test_the_alignment_check_actually_catches_a_shift``.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from src.chunking_embedding.embedders import create_embedder
from src.retrieval.build_artifacts import build
from src.retrieval.index import ArtifactMisaligned, load_artifacts
from src.retrieval.protocol import ChunkRecord
from src.retrieval.index import LocalIndex
from src.retrieval.retrievers import DenseRetriever

TEXTS = [
    "quarterly revenue analysis for the northern region",
    "employee onboarding checklist and probation policy",
    "建筑消防设施故障维修记录表 annual inspection",
    "Đại học Công nghệ thông báo tuyển sinh năm 2026",
    "structural load calculations for reinforced concrete",
]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("vectors")
    documents = root / "out" / "RUN" / "documents"
    documents.mkdir(parents=True)
    for i, text in enumerate(TEXTS):
        (documents / f"d{i}.json").write_text(
            json.dumps(
                {
                    "document": {"document_id": f"doc{i}.pdf", "file_name": f"doc{i}.pdf"},
                    "retrieval": {
                        "items": [
                            {
                                "item_id": f"c{i}",
                                "type": "text",
                                "content": {"text": text},
                                "embeddings": [{"model": "local_hash"}],
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
    return build(
        "RUN", root / "artifacts", "cfg", "auto",
        output_root=root / "out",
        embedder=create_embedder("local_hash", {}),
        embedder_id="local_hash",
    )


def test_chunk_ids_match_chunks_jsonl_order(artifacts: Path):
    ids = json.loads((artifacts / "chunk_ids.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)["chunk_id"]
        for line in (artifacts / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert ids == rows


def test_vector_count_matches_chunk_count(artifacts: Path):
    ids = json.loads((artifacts / "chunk_ids.json").read_text(encoding="utf-8"))
    assert np.load(artifacts / "vectors.npy").shape[0] == len(ids)


def test_each_chunk_is_its_own_nearest_neighbour(artifacts: Path):
    """The spanning assertion. Only true if rows and ids agree, one for one.

    Deliberately semantic: it fails on a shift, which a shape check cannot see.
    """
    embedder = create_embedder("local_hash", {})
    index = load_artifacts(artifacts, embedder=embedder)
    dense = DenseRetriever(index)
    for record in index.records:
        top = dense.retrieve(record.text, 1)
        assert top, f"no hit for {record.chunk_id}"
        assert top[0].chunk_id == record.chunk_id, (
            f"{record.chunk_id} retrieved {top[0].chunk_id} for its own text -- "
            "vectors.npy rows are not aligned with chunk_ids"
        )


def test_the_alignment_check_actually_catches_a_shift(artifacts: Path):
    """Rule 2: demonstrate the guard failing against the defect it guards.

    Rolling the vector matrix by one row is exactly the silent corruption -- same
    shape, same dtype, same count, every structural check still green.
    """
    embedder = create_embedder("local_hash", {})
    index = load_artifacts(artifacts, embedder=embedder)
    shifted = LocalIndex(
        index_id="shifted",
        records=list(index.records),
        bm25=index.bm25,
        vectors=np.roll(index.vectors, 1, axis=0),
        embedder=embedder,
        normalized=True,
    )
    dense = DenseRetriever(shifted)
    wrong = sum(
        1
        for record in shifted.records
        if (hit := dense.retrieve(record.text, 1)) and hit[0].chunk_id != record.chunk_id
    )
    assert wrong == len(shifted.records), (
        "A one-row shift must break every self-retrieval; if it does not, the "
        "check cannot detect misalignment and is not a guard."
    )


def test_truncated_vectors_are_refused_at_load(artifacts: Path, tmp_path: Path):
    broken = tmp_path / "broken"
    broken.mkdir()
    for name in ("bm25.json", "chunks.jsonl", "chunk_ids.json", "manifest.json"):
        (broken / name).write_text((artifacts / name).read_text(encoding="utf-8"), encoding="utf-8")
    np.save(broken / "vectors.npy", np.load(artifacts / "vectors.npy")[:-1])
    with pytest.raises(ArtifactMisaligned):
        load_artifacts(broken)


def test_reordered_chunk_ids_are_refused_at_load(artifacts: Path, tmp_path: Path):
    """A shift that a count check cannot see, caught structurally at load."""
    broken = tmp_path / "reordered"
    broken.mkdir()
    for name in ("bm25.json", "chunks.jsonl", "manifest.json"):
        (broken / name).write_text((artifacts / name).read_text(encoding="utf-8"), encoding="utf-8")
    (broken / "vectors.npy").write_bytes((artifacts / "vectors.npy").read_bytes())
    ids = json.loads((artifacts / "chunk_ids.json").read_text(encoding="utf-8"))
    (broken / "chunk_ids.json").write_text(
        json.dumps(list(reversed(ids)), ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ArtifactMisaligned, match="does not match"):
        load_artifacts(broken)
