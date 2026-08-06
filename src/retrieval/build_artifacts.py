"""Build the service's artifacts from a completed pipeline run.

    python -m src.retrieval.build_artifacts --run-id <RUN_ID>

"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.settings import ARTIFACT_ROOT  # noqa: E402
from src.retrieval.ids import embedding_id, short_hash  # noqa: E402
from src.retrieval.sparse import BM25Index, resolve_analyzer  # noqa: E402

DEFAULT_CONFIG = "configs/pipeline.yaml"
ARTIFACT_SCHEMA_VERSION = "retrieval-artifacts-v2"


def read_run(run_id: str, output_root: Path) -> tuple[list[dict[str, Any]], set[str]]:
    documents_dir = output_root / run_id / "documents"
    if not documents_dir.is_dir():
        raise FileNotFoundError(f"No output documents for run {run_id}: {documents_dir}")

    chunks: list[dict[str, Any]] = []
    models: set[str] = set()
    for path in sorted(documents_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        document = payload.get("document") or {}
        doc_id = str(document.get("document_id") or path.stem)
        source_name = str(document.get("file_name") or doc_id)

        for item in ((payload.get("retrieval") or {}).get("items")) or []:
            item_id = str(item.get("item_id") or "")
            text = _text_of(item.get("content"))
            if not item_id or not text.strip():
                continue
            item_type = str(item.get("type") or "text")
            for embedding in item.get("embeddings") or []:
                if embedding.get("model"):
                    models.add(str(embedding["model"]))
            chunks.append(
                {
                    "chunk_id": item_id,
                    "doc_id": doc_id,
                    "text": text,
                    "source_name": source_name,
                    "embedding_id": embedding_id(run_id, item_id, item_type),
                }
            )
    return chunks, models


def _text_of(content: Any) -> str:
    if not isinstance(content, dict):
        return content if isinstance(content, str) else ""
    for key in ("text", "content", "value"):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            caption = content.get("caption")
            return f"{caption}\n{value}" if isinstance(caption, str) and caption else value
    return ""


def embed_chunks(chunks: list[dict[str, Any]], embedder: Any, batch: int = 128) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch):
        window = [chunk["text"] for chunk in chunks[start : start + batch]]
        vectors.extend(embedder.embed(window))
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape[0] != len(chunks):
        raise ValueError(
            f"Embedder returned {matrix.shape[0]} vectors for {len(chunks)} chunks. "
            "Row order is the alignment key; a partial batch cannot be reconciled."
        )
    return matrix


def build(run_id: str, out_root: Path, config_hash: str, analyzer: str = "auto",
          output_root: Path | None = None, embedder: Any = None,
          embedder_id: str = "", metric: str = "cosine") -> Path:
    chunks, models = read_run(run_id, output_root or (PROJECT_ROOT / "data" / "output"))
    if not chunks:
        raise ValueError(f"Run {run_id} produced no retrievable chunks.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    corpus_hash = short_hash(*chunk_ids, length=12)
    if analyzer == "auto":
        analyzer = resolve_analyzer(chunk["text"] for chunk in chunks)

    out_dir = out_root / config_hash
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    (out_dir / "chunk_ids.json").write_text(
        json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8"
    )

    BM25Index(analyzer_name=analyzer).build(chunks).save(out_dir / "bm25.json")

    dim = 0
    normalized = False
    if embedder is not None:
        matrix = embed_chunks(chunks, embedder)
        if metric == "cosine":
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.clip(norms, 1e-12, None)
            normalized = True
        np.save(out_dir / "vectors.npy", matrix)
        dim = int(matrix.shape[1])
        if embedder_id:
            models.add(embedder_id)

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": ARTIFACT_SCHEMA_VERSION,
                "config_hash": config_hash,
                "run_id": run_id,
                "chunk_count": len(chunks),
                "document_count": len({chunk["doc_id"] for chunk in chunks}),
               
                "analyzer": analyzer,
                "analyzer_id": analyzer,
                "embedder_id": embedder_id,
                "dim": dim,
                "normalized": normalized,
                "metric": metric,
                "corpus_hash": corpus_hash,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                
                "embeddings_models": sorted(models),
                "source": "pipeline_output",
               
                "index_id": index_id(config_hash, analyzer, embedder_id, corpus_hash),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def index_id(config_hash: str, analyzer: str, embedder_id: str, corpus_hash: str) -> str:
    return f"{config_hash}.{analyzer}.{embedder_id or 'noembed'}.{corpus_hash}"


def config_hash_from(config_path: Path) -> str:
    import yaml

    from src.chunking_embedding.contracts import ChunkEmbedConfig

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return ChunkEmbedConfig.from_mapping(config.get("chunking_embedding", {})).config_hash()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Pipeline run whose output to index.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config the run used; names the artifact set.")
    parser.add_argument("--output", default=str(ARTIFACT_ROOT))
    parser.add_argument("--analyzer", default="auto", choices=["plain", "cjk_bigram", "auto"])
    parser.add_argument(
        "--embedder",
        default="",
        help="Embedder name; writes vectors.npy so the dense leg runs locally. "
        "Omit to build sparse-only artifacts.",
    )
    parser.add_argument("--embedder-param", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "ip"])
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_hash = config_hash_from(config_path)

    embedder = None
    if args.embedder:
        from src.chunking_embedding.embedders import create_embedder

        params = dict(
            item.split("=", 1) for item in args.embedder_param if "=" in item
        )
        embedder = create_embedder(args.embedder, params)

    out_dir = build(
        args.run_id, Path(args.output), config_hash, args.analyzer,
        embedder=embedder, embedder_id=args.embedder, metric=args.metric,
    )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"config      : {config_path}")
    print(f"index_id    : {manifest['index_id']}")
    print(f"chunks      : {manifest['chunk_count']} from {manifest['document_count']} document(s)")
    print(f"analyzer    : {manifest['analyzer_id']}")
    print(
        f"vectors     : {manifest['dim']}d {manifest['metric']}"
        f"{' normalized' if manifest['normalized'] else ''}"
        if manifest["dim"]
        else "vectors     : (sparse-only build)"
    )
    print(f"embed model : {', '.join(manifest['embeddings_models']) or '(none recorded)'}")
    print(f"artifacts   : {out_dir}")


if __name__ == "__main__":
    main()
