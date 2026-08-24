"""Load a pipeline run's retrieval payload into corpus-service's tables.

    python scripts/load_corpus_service.py --run-id <RUN_ID>

Stands in for the Spark job in `services/indexing-streaming`, which is the real
writer: it consumes MinIO events, calls us at /v1/dataeng, and persists what we
return. This script performs the same inserts against the same columns so the
dense retrieval path can be exercised without standing up Kafka, MinIO and Spark.

It is a development tool, not a production path. Nothing in the service reads it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.ids import short_hash  # noqa: E402
from src.retrieval.ids import embedding_id  # noqa: E402

CONTENT_ID_LENGTH = 32
CONTAINER = "k8s-postgres-1"
DATABASE = "corpus"
USER = "app_dev"


def _sql(statement: str) -> str:
    result = subprocess.run(
        ["docker", "exec", "-i", CONTAINER, "psql", "-U", USER, "-d", DATABASE, "-v", "ON_ERROR_STOP=1", "-c", statement],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:500])
    return result.stdout


def _lit(value: Any) -> str:
    """Single-quoted SQL literal with quotes doubled."""
    return "'" + json.dumps(value, ensure_ascii=False).replace("'", "''") + "'" if not isinstance(value, str) \
        else "'" + value.replace("'", "''") + "'"


def _has_lexical_column() -> bool:
    """corpus-service gained document_embeddings.lexical with its server-side
    BM25. Older images predate it, so the column is detected rather than assumed:
    omitting it there is required, omitting it here leaves keyword-search dark."""
    out = _sql(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name='document_embeddings' AND column_name='lexical'"
    )
    return "(0 rows)" not in out


def _json_lit(value: Any) -> str:
    """SQL literal for a jsonb column. Unlike _lit, a bare string is encoded as
    JSON text rather than passed through, which ::jsonb would reject."""
    return "'" + json.dumps(value, ensure_ascii=False).replace("'", "''") + "'"


def load(run_id: str, organization_id: str = "dev-org", bucket: str = "dev-bucket") -> int:
    output_dir = PROJECT_ROOT / "data" / "output" / run_id / "documents"
    if not output_dir.is_dir():
        raise FileNotFoundError(f"No output documents for run {run_id}: {output_dir}")

    with_lexical = _has_lexical_column()
    if not with_lexical:
        print("note: this corpus-service schema has no document_embeddings.lexical column; "
              "server-side BM25 will fall back to scanning document_contents.")

    inserted = 0
    for path in sorted(output_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        document = payload.get("document") or {}
        items = ((payload.get("retrieval") or {}).get("items")) or []
        if not items:
            continue

        doc_id = str(document.get("document_id") or path.stem)[:64]
        object_key = str(document.get("source_uri") or document.get("file_name") or path.stem)
        file_name = Path(object_key).name

        # etag and sha256 are NOT NULL: the Spark job gets them from the S3 event.
        # Loading from disk, we derive a stable stand-in from the object key.
        etag = hashlib.md5(object_key.encode()).hexdigest()
        sha256 = hashlib.sha256(object_key.encode()).hexdigest()
        _sql(
            "INSERT INTO documents (document_id, organization_id, source_uri, bucket, object_key, "
            "etag, sha256, file_name, size_bytes, current_status, latest_run_id) VALUES ("
            f"{_lit(doc_id)}, {_lit(organization_id)}, {_lit(object_key)}, {_lit(bucket)}, "
            f"{_lit(object_key)}, {_lit(etag)}, {_lit(sha256)}, {_lit(file_name)}, 0, 'indexed', {_lit(run_id)}) "
            "ON CONFLICT (document_id) DO UPDATE SET latest_run_id = EXCLUDED.latest_run_id"
        )
        _sql(
            "INSERT INTO document_processing_runs (run_id, document_id, batch_id, status) VALUES ("
            f"{_lit(run_id)}, {_lit(doc_id)}, {_lit(run_id)}, 'completed') "
            "ON CONFLICT (run_id) DO UPDATE SET status = EXCLUDED.status"
        )

        # document_contents is what corpus-service's keyword-search reads. The
        # Spark job writes one row per key of the ``content`` block; omitting
        # them made every keyword query return zero results.
        for content_type, content_value in (payload.get("content") or {}).items():
            _sql(
                "INSERT INTO document_contents (content_id, document_id, run_id, type, content) VALUES ("
                f"{_lit(short_hash(run_id, content_type, length=CONTENT_ID_LENGTH))}, {_lit(doc_id)}, "
                f"{_lit(run_id)}, {_lit(content_type)}, {_json_lit(content_value)}::jsonb) "
                "ON CONFLICT (content_id) DO UPDATE SET content = EXCLUDED.content"
            )

        for item in items:
            embeddings = item.get("embeddings") or []
            first = embeddings[0] if embeddings else {}
            values, model = first.get("values"), first.get("model")
            if not values or not model:
                print(f"  skipping item without embedding values/model: {item.get('item_id')}")
                continue
            # Same key the Spark job builds: short_hash(run_id, item_id, item_type).
            item_id = str(item.get("item_id") or "unknown")
            item_type = str(item.get("type") or "unknown")
            emb_id = embedding_id(run_id, item_id, item_type)
            vector = "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"
            lexical_column = ", lexical" if with_lexical else ""
            lexical_value = f", {_json_lit(item.get('lexical'))}::jsonb" if with_lexical else ""
            _sql(
                "INSERT INTO document_embeddings (embedding_id, document_id, run_id, type, position, "
                f"content, embedding, embeddings_model{lexical_column}) VALUES ("
                f"{_lit(emb_id)}, {_lit(doc_id)}, {_lit(run_id)}, {_lit(item_type)}, "
                f"{_json_lit(item.get('position') or {})}::jsonb, {_json_lit(item.get('content') or {})}::jsonb, "
                f"{_lit(vector)}::vector, {_lit(model)}{lexical_value}) "
                # Every mutable column, not just the vector: a re-load after a
                # schema change must be able to backfill, or rows silently keep
                # the values they had when first inserted.
                "ON CONFLICT (embedding_id) DO UPDATE SET embedding = EXCLUDED.embedding, "
                "content = EXCLUDED.content, position = EXCLUDED.position, "
                f"embeddings_model = EXCLUDED.embeddings_model{', lexical = EXCLUDED.lexical' if with_lexical else ''}"
            )
            inserted += 1
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--organization-id", default="dev-org")
    parser.add_argument("--bucket", default="dev-bucket")
    args = parser.parse_args()
    count = load(args.run_id, args.organization_id, args.bucket)
    print(f"inserted {count} embedding row(s) for run {args.run_id}")


if __name__ == "__main__":
    main()
