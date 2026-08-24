"""Flatten the ViDoRe V3 parquets into two readable CSVs.

queries.csv -- one row per (subset, language, query), gold joined in.
pages.csv   -- one row per corpus page, document metadata joined in.
"""
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data/benchmark/vidore_v3"
PREVIEW = 300


def joined(values):
    return "; ".join(str(v) for v in values) if values is not None and len(values) else ""


query_rows, page_rows = [], []

for directory in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    subset = directory.name
    queries = pq.read_table(directory / "queries.parquet").to_pandas()
    qrels = pq.read_table(directory / "qrels.parquet").to_pandas()
    corpus = pq.read_table(directory / "corpus.parquet").to_pandas()
    meta = pq.read_table(directory / "documents_metadata.parquet").to_pandas()

    unit = {row.corpus_id: f"{subset}::{row.doc_id}#page={row.page_number_in_doc}"
            for row in corpus.itertuples()}
    doc_of = dict(zip(corpus.corpus_id, corpus.doc_id))

    gold = {}
    for row in qrels.itertuples():
        boxes = row.bounding_boxes
        gold.setdefault(row.query_id, []).append(
            (unit.get(row.corpus_id, f"{subset}::MISSING_corpus_id={row.corpus_id}"),
             doc_of.get(row.corpus_id), row.score,
             0 if boxes is None else len(boxes), joined(row.content_type))
        )

    for row in queries.itertuples():
        g = sorted(gold.get(row.query_id, []))
        query_rows.append({
            "subset": subset,
            "language": row.language,
            "query_id": row.query_id,
            "query": row.query,
            "answer": row.answer,
            "n_gold_pages": len(g),
            "n_gold_docs": len({d for _, d, _, _, _ in g}),
            "n_gold_regions": sum(n for _, _, _, n, _ in g),
            "gold_pages": " | ".join(f"{u}(score={s})" for u, _, s, _, _ in g),
            "gold_modalities": joined(sorted({m for _, _, _, _, c in g for m in c.split("; ") if m})),
            "query_format": row.query_format,
            "query_types": joined(row.query_types),
            "query_content_type": joined(row.content_type),
            "source_type": row.source_type,
            "n_raw_answers": 0 if row.raw_answers is None else len(row.raw_answers),
        })

    refs = {}
    for row in qrels.itertuples():
        refs[row.corpus_id] = refs.get(row.corpus_id, 0) + 1
    by_doc = meta.set_index("doc_id").to_dict("index")

    for row in corpus.itertuples():
        info = by_doc.get(row.doc_id, {})
        text = row.markdown or ""
        page_rows.append({
            "subset": subset,
            "unit_id": unit[row.corpus_id],
            "doc_id": row.doc_id,
            "page_number_in_doc": row.page_number_in_doc,
            "corpus_id": row.corpus_id,
            "n_chars": len(text),
            "n_query_refs": refs.get(row.corpus_id, 0),
            "is_distractor": refs.get(row.corpus_id, 0) == 0,
            "doc_file_name": info.get("file_name", ""),
            "doc_type": info.get("doc_type", ""),
            "doc_language": info.get("doc_language", ""),
            "doc_year": info.get("doc_year", ""),
            "doc_n_pages": info.get("page_number", ""),
            "doc_visual_types": joined(info.get("visual_types")),
            "doc_url": info.get("url", ""),
            "text_preview": " ".join(text[:PREVIEW].split()),
        })

q = pd.DataFrame(query_rows)
p = pd.DataFrame(page_rows)
q.to_csv(ROOT / "queries.csv", index=False)
p.to_csv(ROOT / "pages.csv", index=False)

print(f"queries.csv  {len(q):>6} rows  {(ROOT/'queries.csv').stat().st_size/1e6:.1f} MB")
print(f"pages.csv    {len(p):>6} rows  {(ROOT/'pages.csv').stat().st_size/1e6:.1f} MB")
print()
print(q.groupby(["subset", "language"]).size().unstack().to_string())
print()
print(p.groupby("subset").agg(pages=("unit_id", "size"), docs=("doc_id", "nunique"),
                              distractors=("is_distractor", "sum"),
                              mean_chars=("n_chars", "mean")).round(0).to_string())
print()
print("gold pages per query :", q.n_gold_pages.mean().round(2),
      "| queries with >1 gold doc:", f"{100*(q.n_gold_docs > 1).mean():.1f}%",
      "| queries with 0 gold:", int((q.n_gold_pages == 0).sum()))
print("unresolved corpus_id in gold:", int(q.gold_pages.str.contains("MISSING_corpus_id").sum()))
