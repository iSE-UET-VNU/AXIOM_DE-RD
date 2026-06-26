"""Index record builders for database, vector, and graph targets."""

from __future__ import annotations

from ..models import EnrichedData, IndexRecord, make_id


def build_db_index_records(enriched_data: list[EnrichedData]) -> list[IndexRecord]:
    return [
        IndexRecord(
            record_id=make_id(data.source_object_id, "db-index"),
            index_type="database",
            source_object_id=data.source_object_id,
            payload={
                "row_count": len(data.rows),
                "fields": data.annotations.get("fields", []),
            },
            metadata={"todo": "Map payloads to real database tables."},
        )
        for data in enriched_data
    ]


def build_vector_index_records(enriched_data: list[EnrichedData]) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    for data in enriched_data:
        text_values = [str(row.get("text", "")) for row in data.rows if row.get("text")]
        records.append(
            IndexRecord(
                record_id=make_id(data.source_object_id, "vector-index"),
                index_type="vector",
                source_object_id=data.source_object_id,
                payload={
                    "text_preview": " ".join(text_values)[:500],
                    "embedding": None,
                },
                metadata={"todo": "Generate embeddings and upsert into a vector database."},
            )
        )
    return records


def build_graph_index_records(enriched_data: list[EnrichedData]) -> list[IndexRecord]:
    return [
        IndexRecord(
            record_id=make_id(data.source_object_id, "graph-index"),
            index_type="graph",
            source_object_id=data.source_object_id,
            payload={
                "nodes": [{"id": data.source_object_id, "type": "data_object"}],
                "edges": [],
            },
            metadata={"todo": "Create graph nodes and edges from extracted relationships."},
        )
        for data in enriched_data
    ]
