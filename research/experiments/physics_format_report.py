"""Format report for the externally parsed ViDoRe physics subset. Read-only."""
import json
import glob
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

EXT = str(Path(__file__).resolve().parents[2] / "data_vidore_parsed_physics")
REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).parent

docs = [json.load(open(p)) for p in sorted(glob.glob(f"{EXT}/output/benchmarks/*/*/documents/*.json"))]
print(f"documents parsed: {len(docs)}")

# -- 1. config parity ---------------------------------------------------------
print("\n=== 1. refine_tables / table_refinement parity ===")
ref = Counter()
attempts = Counter()
for d in docs:
    m = d["ingest"]["data"]["metadata"]
    tr = m.get("table_refinement") or {}
    ref[tr.get("enabled")] += 1
    attempts[tr.get("attempted", 0)] += 1
print(f"  table_refinement.enabled : {dict(ref)}")
print(f"  attempted counts         : {dict(attempts)}")
print(f"  our configs/pipeline.yaml refine_tables: "
      f"{json.load(open(REPO/'configs/pipeline.yaml'))['parsing']['chandra2']['refine_tables']}")
print(f"  parser aliases seen      : {Counter(d['ingest']['data']['metadata'].get('parser') for d in docs)}")
print(f"  model_name               : {Counter(d['ingest']['data']['metadata'].get('model_name') for d in docs)}")
print(f"  status                   : {Counter(d['ingest']['data']['metadata'].get('status') for d in docs)}")

# -- 2. doc identity ----------------------------------------------------------
print("\n=== 2. document identity ===")
meta = pq.read_table(SCRATCH/"vidore"/"physics__documents_metadata.parquet").to_pylist()
vid_docs = {m["doc_id"] for m in meta}
print(f"  ViDoRe physics doc_ids : {len(vid_docs)}")
print(f"  their file_name sample : {docs[0]['document']['file_name']!r}")
print(f"  their document_id      : {docs[0]['document']['document_id']!r}")


def stem(file_name: str) -> str:
    base = file_name.rsplit("/", 1)[-1]
    return base[:-4] if base.lower().endswith(".pdf") else base


naive_suffix_only = {d["document"]["file_name"][:-4] if d["document"]["file_name"].lower().endswith(".pdf")
                     else d["document"]["file_name"] for d in docs}
proper = {stem(d["document"]["file_name"]) for d in docs}
print(f"  strip .pdf only -> matches ViDoRe doc_id: {len(naive_suffix_only & vid_docs)}/{len(docs)}")
print(f"  basename + strip .pdf                   : {len(proper & vid_docs)}/{len(docs)}")
print(f"  document_id matches doc_id              : "
      f"{len({d['document']['document_id'] for d in docs} & vid_docs)}/{len(docs)}")
missing = vid_docs - proper
extra = proper - vid_docs
if missing:
    print(f"  ViDoRe docs NOT parsed ({len(missing)}): {sorted(missing)[:5]}")
if extra:
    print(f"  parsed docs not in ViDoRe ({len(extra)}): {sorted(extra)[:5]}")

# -- 3. page identity and coverage -------------------------------------------
print("\n=== 3. page identity and coverage ===")
idx = pq.read_table(REPO/"data/benchmark/vidore_v3/page_index.parquet").to_pylist()
vid_pages = {(r["doc_id"], r["page_number_in_doc"]) for r in idx if r["subset"] == "physics"}
print(f"  ViDoRe physics pages: {len(vid_pages)}")

their_pages = set()
page_mins, page_source = Counter(), Counter()
for d in docs:
    s = stem(d["document"]["file_name"])
    pages = {b.get("page") for b in d["content"]["blocks"] if b.get("page") is not None}
    if pages:
        page_mins[min(pages)] += 1
    for p in pages:
        their_pages.add((s, p))
    page_source[d["ingest"]["data"]["metadata"].get("reading_order_source")] += 1
print(f"  their (doc,page) pairs from blocks: {len(their_pages)}")
print(f"  min page per document histogram   : {dict(page_mins)}  -> 0-based" if 0 in page_mins else
      f"  min page per document histogram   : {dict(page_mins)}  -> NOT 0-based")
print(f"  reading_order_source              : {dict(page_source)}")

declared = sum(d["ingest"]["data"]["metadata"].get("page_count", 0) for d in docs)
print(f"  sum of declared page_count        : {declared}")
print(f"  matched pages   : {len(their_pages & vid_pages)}")
print(f"  ViDoRe-only     : {len(vid_pages - their_pages)}  (coverage loss -> unreachable_n)")
print(f"  theirs-only     : {len(their_pages - vid_pages)}")
sample = sorted(vid_pages - their_pages)[:5]
if sample:
    print(f"  sample ViDoRe-only: {sample}")
by_doc = Counter(d for d, _ in (vid_pages - their_pages))
if by_doc:
    print(f"  worst docs by missing pages: {by_doc.most_common(5)}")

# -- 4. qrels join ------------------------------------------------------------
print("\n=== 4. qrels corpus_id -> their output ===")
cid_to_page = {r["corpus_id"]: (r["doc_id"], r["page_number_in_doc"])
               for r in idx if r["subset"] == "physics"}
qr = pq.read_table(SCRATCH/"vidore"/"physics__qrels.parquet").to_pylist()
qs = pq.read_table(SCRATCH/"vidore"/"physics__queries.parquet").to_pylist()
for lang in ("french", "english"):
    keep = {q["query_id"] for q in qs if q["language"] == lang}
    rows = [r for r in qr if r["query_id"] in keep]
    stats = Counter()
    lost_q = set()
    for r in rows:
        stats["attempted"] += 1
        key = cid_to_page.get(r["corpus_id"])
        if key is None:
            stats["no_page"] += 1
        elif key in their_pages:
            stats["matched"] += 1
        else:
            stats["unreachable"] += 1
            lost_q.add(r["query_id"])
    print(f"  {lang:8s} attempted={stats['attempted']:5d} matched={stats['matched']:5d} "
          f"unreachable={stats['unreachable']:4d} no_page={stats['no_page']:3d} "
          f"queries affected={len(lost_q)}/{len(keep)}")

# -- 5. retrieval granularity -------------------------------------------------
print("\n=== 5. what granularity did they emit? ===")
types = Counter()
page_fields = Counter()
for d in docs:
    for item in d.get("retrieval", {}).get("items", []) or []:
        types[item.get("type")] += 1
print(f"  retrieval item types: {dict(types)}")
one = docs[0]["retrieval"]["items"][0] if docs[0].get("retrieval", {}).get("items") else None
if one:
    print(f"  item keys: {list(one.keys())}")
    print(f"  item_id  : {one.get('item_id')!r}  component_type={one.get('component_type')!r}")
    emb = (one.get("embeddings") or [{}])[0]
    print(f"  embedding: model={emb.get('model')!r} dim={emb.get('dimension')}")
print(f"  main_text present: {sum(1 for d in docs if (d['content'].get('main_text') or '').strip())}/{len(docs)}")
print(f"  blocks total     : {sum(len(d['content']['blocks']) for d in docs)}")
