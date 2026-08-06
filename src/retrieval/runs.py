
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json

from .protocol import ScoredChunk

RUN_SCHEMA_VERSION = "retrieval-run-v1"


@dataclass
class RunRecord:
    qid: str
    query: str
    retriever_id: str
    index_id: str
    params_hash: str
    chunks: list[dict[str, Any]] = field(default_factory=list)
    scope_doc_ids: list[str] | None = None
    latency_ms: float = 0.0

    @classmethod
    def build(
        cls,
        qid: str,
        query: str,
        retriever_id: str,
        index_id: str,
        params_hash: str,
        hits: Sequence[ScoredChunk],
        *,
        scope_doc_ids: list[str] | None = None,
        latency_ms: float = 0.0,
    ) -> "RunRecord":
        return cls(
            qid=str(qid),
            query=query,
            retriever_id=retriever_id,
            index_id=index_id,
            params_hash=params_hash,
            scope_doc_ids=scope_doc_ids,
            latency_ms=round(latency_ms, 2),
            chunks=[
                {
                    "chunk_id": hit.chunk_id,
                    "doc_id": hit.doc_id,
                    "text": hit.text,
                    "score": round(float(hit.score), 6),
                    "rank": hit.rank,
                    **({"scores": dict(hit.scores)} if hit.scores else {}),
                }
                for hit in hits
            ],
        )


def stable_hash(value: Any, length: int = 12) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def params_hash(params: Mapping[str, Any]) -> str:
    return stable_hash(dict(params))


def query_set_hash(qids: Iterable[str]) -> str:
    return stable_hash(sorted(str(qid) for qid in qids))


def cache_path(
    root: Path, index_id: str, retriever_id: str, params: Mapping[str, Any], qids: Iterable[str]
) -> Path:
    key = f"{retriever_id}__{params_hash(params)}__{query_set_hash(qids)}"
    return Path(root) / index_id / f"{key}.jsonl"


def write(path: Path, records: Iterable[RunRecord]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def read(path: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(RunRecord(**json.loads(line)))
    return records
