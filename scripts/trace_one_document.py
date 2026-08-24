"""Walk one document through every pipeline stage, printing what crosses each boundary.

    python scripts/trace_one_document.py

A teaching aid, not a pipeline. The input is a real file on disk, but the
embedder is `local_hash` (deterministic, offline, free) so the trace runs
anywhere with no services, no API keys and no cost. Everything else is the code
run_pipeline runs, in the order it runs it.

Each stage prints the TYPE it receives, the TYPE it emits, and the actual data
at that boundary -- the intermediate representations that are otherwise only
visible by reading five files at once.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import argparse
import json
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

W = 84
SAMPLE = (
    "Kế hoạch kết thúc khóa học\n\n"
    "Nhà trường thông báo kế hoạch tổ chức lễ trao bằng tốt nghiệp cho sinh viên "
    "khóa QH-2022 và các khóa cũ.\n\n"
    "Sinh viên phải thanh toán các khoản nợ tại các đơn vị trước ngày 27/06/2026. "
    "Sau thời hạn này, sinh viên sẽ không hoàn thành thủ tục tốt nghiệp và không "
    "được xét trao bằng trong đợt tháng 6.\n\n"
    "Lễ trao bằng diễn ra vào thứ Tư, ngày 01/07/2026 tại hội trường lớn của Nhà trường. "
    "Sinh viên có mặt trước 30 phút để làm thủ tục nhận bằng và nhận giấy mời cho người nhà.\n"
)


def stage(n: str, title: str, where: str) -> None:
    print(f"\n{'=' * W}\n{n}  {title}\n{'-' * W}\n   {where}\n{'=' * W}")


def io(receives: str, emits: str) -> None:
    print(f"  in  <- {receives}")
    print(f"  out -> {emits}\n")


def show(label: str, value: Any, limit: int = 260) -> None:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split()) if "\n" in text and len(text) > limit else text
    if len(text) > limit:
        text = text[:limit] + f" … <{len(text)} chars>"
    print(f"    {label:<26}: {text}")


def note(*lines: str) -> None:
    for line in lines:
        print(f"    │ {line}")


def to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj) if is_dataclass(obj) else dict(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunker", default="fixed_overlap")
    parser.add_argument("--n-words", type=int, default=60)
    parser.add_argument("--overlap", type=int, default=15)
    args = parser.parse_args()

    print(f"\n{'#' * W}\n#  ONE DOCUMENT, EVERY STAGE\n{'#' * W}")

    workdir = Path(tempfile.mkdtemp(prefix="trace-"))
    source = workdir / "notice.txt"
    source.write_text(SAMPLE, encoding="utf-8")

    # =======================================================================
    stage("[0]", "RAW INPUT", "a file on disk / an object in MinIO")
    io("bytes", "a path + source_uri")
    show("path", str(source))
    show("bytes", f"{source.stat().st_size}")

    # =======================================================================
    stage("[1]", "INGESTION + PARSING", "src/ingestion/__init__.py :: run")
    io("path, parser_config", "IngestionResult{data_objects, parsed_data, initial_schemas}")
    from src import ingestion

    ing = ingestion.run(source, source_uri=str(source), input_metadata={}, parser_config={},
                        project_root=PROJECT_ROOT)
    obj = ing.data_objects[0]
    parsed = ing.parsed_data[0]
    print("  DataObject — the routing identity")
    show("object_id", obj.object_id)
    show("content_type", obj.content_type)
    note("object_id is a hash of the source; every later id derives from it.", "")
    print("  ParsedData — the parser's neutral output")
    show("source_format", parsed.source_format)
    show("rows", f"{len(parsed.rows)} row(s)")
    show("tables", f"{len(parsed.tables)}")
    extraction = parsed.rows[0].get("extraction", {})
    show("rows[0].extraction keys", sorted(extraction))
    show("  .main_text", extraction.get("main_text", ""), 150)
    note("Every parser (lift_api, chandra2, local) emits THIS shape.",
         "That is why swapping the parser changes nothing downstream.")

    # =======================================================================
    stage("[2]", "CLEANING", "src/cleaning/__init__.py :: run")
    io("parsed_data, initial_schemas", "cleaned_data, cleaned_schemas")
    from src import cleaning

    cl = cleaning.run(ing.parsed_data, ing.initial_schemas)
    show("cleaned_data", f"{len(cl.cleaned_data)} dataset(s)")
    show("identical to input", cl.cleaned_data[0] == ing.parsed_data[0])
    note("Pass-through scaffold today. It exists so a real cleaning step can be",
         "inserted without any other stage changing.")

    # =======================================================================
    stage("[3]", "ENRICHMENT", "src/enrichment/__init__.py :: run")
    io("cleaned_data, cleaned_schemas", "enriched_data, enriched_schemas")
    from src import enrichment

    en = enrichment.run(cl.cleaned_data, cl.cleaned_schemas)
    show("enriched_data", f"{len(en.enriched_data)} dataset(s)")
    note("Also pass-through. Chunking reads enriched_data, so cleaning and",
         "enrichment stay upstream of it once they stop being scaffolds.")

    # =======================================================================
    stage("[4]", "CHUNKING + EMBEDDING", "src/chunking_embedding/stage.py :: run")
    io("enriched_data", "retrieval_records, vector_records, reports")

    enriched_records = [to_dict(record) for record in en.enriched_data]
    _trace_our_engine(enriched_records[0], args)
    from src.chunking_embedding.stage import run as run_chunking_embedding

    out = run_chunking_embedding(enriched_records, {
        "chunker": args.chunker,
        "chunker_params": {"n_words": args.n_words, "overlap": args.overlap},
        "embedder": "local_hash", "embedder_params": {"dimension": 8},
    })

    print("\n  ── stage output ──")
    show("retrieval_records", f"{len(out.retrieval_records)}")
    show("vector_records", f"{len(out.vector_records)}")
    show("skipped_docs", f"{len(out.skipped_docs)}")
    from src.artifacts.pipeline_output import RETRIEVAL_ITEM_TYPES

    retrievable = [r for r in out.retrieval_records if r.retrieval_type in RETRIEVAL_ITEM_TYPES]
    show("retrieval_types", dict((t, sum(1 for r in out.retrieval_records if r.retrieval_type == t))
                                 for t in {r.retrieval_type for r in out.retrieval_records}))
    if not retrievable:
        print("\n  no retrieval-typed records; nothing reaches corpus-service.")
        return
    record = retrievable[0]
    print(f"\n  RetrievalRecord — first retrieval-typed record ({record.retrieval_type})")
    show("record_id", record.record_id)
    show("retrieval_type", record.retrieval_type)
    show("source_object_id", record.source_object_id)
    show("payload keys", sorted(record.payload))
    lex = record.payload.get("lexical") or {}
    show("payload.lexical", {"analyzer": lex.get("analyzer"), "dl": lex.get("dl"),
                             "tf": f"<{len(lex.get('tf', {}))} terms>"} if lex else "MISSING")
    matching = [v for v in out.vector_records if v.get("record_id") == record.record_id]
    if not matching:
        print("\n  no vector for this record — embeddings disabled or filtered out")
        return
    vec = matching[0]
    print("\n  vector_records — the one whose record_id matches")
    show("record_id", vec["record_id"])
    show("embedding_model", vec["embedding_model"])
    show("embedding", f"[{', '.join(f'{v:.4f}' for v in vec['embedding'][:5])}, …] dim={vec['embedding_dimension']}")
    note("record_id is the join key between retrieval_records and vector_records.",
         "They are separate lists; only this id keeps them aligned.")

    # =======================================================================
    stage("[5]", "INTEGRATION", "src/integration/__init__.py :: run")
    io("retrieval_records", "passed_retrieval_records, schema/entity matches")
    from src import integration

    integ = integration.run(out.retrieval_records)
    show("passed", f"{len(integ.passed_retrieval_records)} / {len(out.retrieval_records)}")
    note("Pass-through gate. A future validation step drops bad records here.")

    # =======================================================================
    stage("[6]", "ARTIFACTS — the wire contract", "src/artifacts/pipeline_output.py")
    io("PipelineState", "one JSON file per document (output-document-v4)")
    from src.artifacts.pipeline_output import _compact_retrieval_item

    # Takes the raw vector_record; _compact_embedding renames its keys to the
    # wire names (embedding_model -> model, embedding -> values).
    item = _compact_retrieval_item(record, [vec], [])
    print("  retrieval.items[0] — exactly what leaves this repo")
    show("item_id", item["item_id"])
    show("type", item["type"])
    show("position", item["position"])
    show("content keys", sorted(item["content"]))
    show("lexical", "present" if item.get("lexical") else "MISSING")
    show("embeddings[0].model", item["embeddings"][0]["model"])
    note("retrieval_type -> type mapping: text_chunk->text, table->table, image->image.",
         "config_hash rides INSIDE position: the Spark job persists position",
         "verbatim and drops item-level keys it does not recognise.")

    print("\n  the full document envelope:")
    print("""    {
      "contract_version": "output-document-v4",
      "document":  {document_id, file_name, source_uri, ...},   -> documents
      "content":   {main_text, blocks, tables, figures, ...},    -> document_contents
      "retrieval": {items: [...], lexical_stats: {...}},         -> document_embeddings
      "lineage":   {run_id, status, completed_stages, ...}          + document_lexical_stats
    }""")

    # =======================================================================
    stage("[7]", "HANDOFF TO PLATFORM", "services/indexing-streaming/.../database.py")
    io("HTTP POST /v1/dataeng -> {documents: [...]}", "rows in Postgres")
    print("    run_id       = short_hash(document_id, kafka_batch_id, kafka_offset)")
    print("    embedding_id = short_hash(run_id, item_id, item_type)")
    print()
    print("    persist_document      : document          -> documents")
    print("    persist_contents      : content{k: v}     -> document_contents  (one row per key)")
    print("    persist_embeddings    : retrieval.items[] -> document_embeddings(one row per item)")
    print("    persist_lexical_stats : lexical_stats     -> document_lexical_stats")
    note("We never compute run_id ourselves. It is derived from Kafka position,",
         "which is why chunk identity has to survive as item_id, not as our id.")

    # =======================================================================
    stage("[8]", "RETRIEVAL — how the rows are read back", "corpus-service")
    io("a query", "ranked chunks")
    print("    vector-search  : cosine over `embedding`,  filtered by `embeddings_model`")
    print("    keyword-search : BM25 computed from `lexical` {tf, dl}, server-side")
    print("    hybrid-search  : vector_weight * dense + keyword_weight * sparse")
    note("Both legs read the SAME row, keyed by embedding_id. That is the whole",
         "reason chunk identity has to be stable end to end.")

    # =======================================================================
    print(f"\n{'#' * W}\n#  WHERE IT BREAKS — all three were real bugs, none raised an error\n{'#' * W}")
    print("""
  1. chunker differs between embedding and building the sparse index
     -> char offsets differ -> chunk_ids differ -> the two legs key on
        disjoint id spaces -> fusion silently degrades to a union
     symptom: results look fine, top score capped at the sparse-only ceiling

  2. `lexical` not emitted
     -> document_embeddings.lexical is NULL -> their BM25 has nothing to score
     symptom: keyword-search returns [] with HTTP 200

  3. embeddings_model holds the gateway ALIAS instead of the model id
     -> equality filter matches no rows
     symptom: vector-search returns [] with HTTP 200

  In all three the SHAPE is correct and only the VALUES are wrong, so every
  schema check passes. Re-run with --chunker recursive to watch every id
  change while the shape stays identical.
""")
    print(f"  scratch dir: {workdir}\n")


def _trace_our_engine(enriched: dict[str, Any], args: argparse.Namespace) -> None:
    """The four sub-steps inside stage.run, made visible."""
    from src.chunking_embedding.contracts import ChunkEmbedConfig
    from src.chunking_embedding.embedders import create_embedder
    from src.chunking_embedding.fields import FieldContext, Resources, route_document

    config = ChunkEmbedConfig.from_mapping({
        "chunker": args.chunker, "chunker_params": {"n_words": args.n_words, "overlap": args.overlap},
        "embedder": "local_hash", "embedder_params": {"dimension": 8},
    })
    print("  4a. ChunkEmbedConfig.from_mapping   (contracts.py)")
    show("chunker", f"{config.chunker} {config.chunker_params}")
    show("config_hash", config.config_hash())

    embedder = create_embedder(config.embedder, config.embedder_params)
    print("\n  4b. create_embedder                 (embedders/__init__.py)")
    show("class", type(embedder).__name__)
    show("embedder.model", embedder.model)
    note("This string becomes document_embeddings.embeddings_model.",
         "It must name the MODEL, never the route used to reach it.")

    extraction = enriched["rows"][0]["extraction"]
    ctx = FieldContext(doc_id=enriched["source_object_id"], chunker_name=config.chunker,
                       chunker_params=config.chunker_params,
                       max_rows_per_chunk=config.max_rows_per_chunk,
                       language=extraction.get("language"))
    chunks = route_document(extraction, ctx, Resources(embedder=embedder, llm=None))
    print(f"\n  4c. route_document                  (fields.py)  -> {len(chunks)} Chunk(s)")
    for chunk in chunks:
        print(f"      • {chunk.chunk_type:<12} {chunk.field_path:<12} [{chunk.start}:{chunk.end}]")
        print(f"        chunk_id {chunk.chunk_id[:32]}…")
        print(f"        text     {' '.join(chunk.text.split())[:58]}…")
    note("", "Each FIELD is routed to its own rule: prose is windowed by the",
         "chunker, tables are kept whole (or split by rows), figures by caption.",
         "chunk_id = hash(doc_id, field_path, start, end, #i) -- offsets, so any",
         "chunker change rewrites every id in the corpus.")

    vectors = embedder.embed([c.embedding_text() for c in chunks])
    print(f"\n  4d. embedder.embed(...)             -> {len(vectors)} x {len(vectors[0])} dims")
    first = chunks[0]
    same = first.embedding_text() == first.text
    show("embedding_text == text", same)
    if not same:
        show("embedding_text", first.embedding_text(), 120)
    note("embedding_text() may prepend a title/caption so the vector sees context",
         "the raw text lacks. The STORED content stays unprefixed.")


if __name__ == "__main__":
    main()
