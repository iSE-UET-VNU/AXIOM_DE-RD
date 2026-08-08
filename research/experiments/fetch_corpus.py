"""Materialise the ViDoRe V3 corpus the adapter reads, text columns only.
Provenance is recorded in the sidecar so a reader knows this file was assembled
rather than downloaded whole.
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCRATCH = Path(__file__).parent
DEST = Path("/Users/khoatran/Desktop/ISE/Axiom-DE-RnD/AXIOM_DE-RD/data/benchmark/vidore_v3")

# The five public subsets whose source documents are English. The three French
# ones (physics, energy, finance_fr) are a separate acceptance target.
ENGLISH = ["physics"]
COLUMNS = ["corpus_id", "doc_id", "markdown", "page_number_in_doc"]


def get(url, tries=10):
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=90))
        except urllib.error.HTTPError as exc:
            if (exc.code != 429 and exc.code < 500) or i == tries - 1:
                raise
            time.sleep(20 * (i + 1))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def fetch(sub):
    ds = f"vidore%2Fvidore_v3_{sub}"
    n = get(f"https://datasets-server.huggingface.co/info?dataset={ds}"
            )["dataset_info"]["corpus"]["splits"]["test"]["num_examples"]
    rows, off = [], 0
    while off < n:
        d = get(f"https://datasets-server.huggingface.co/rows?dataset={ds}"
                f"&config=corpus&split=test&offset={off}&length=100"
                f"&columns={','.join(COLUMNS)}")
        batch = [r["row"] for r in d["rows"]]
        if not batch:
            break
        rows.extend(batch)
        off += len(batch)
        if off % 1000 == 0:
            print(f"    {sub}: {off}/{n}", flush=True)
    assert len(rows) == n, f"{sub}: got {len(rows)} of {n}"
    return rows


for sub in ENGLISH:
    out_dir = DEST / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "corpus.parquet"

    # The annotation tables were already pulled whole; copy them into place.
    for cfg in ("queries", "qrels", "documents_metadata"):
        src = SCRATCH / "vidore" / f"{sub}__{cfg}.parquet"
        dst = out_dir / f"{cfg}.parquet"
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())

    if target.exists():
        print(f"  {sub}: corpus.parquet already present ({pq.read_metadata(target).num_rows} rows)")
        continue

    rows = fetch(sub)
    pq.write_table(
        pa.table({
            "corpus_id": [r["corpus_id"] for r in rows],
            "doc_id": [r["doc_id"] for r in rows],
            "markdown": [r["markdown"] or "" for r in rows],
            "page_number_in_doc": [r["page_number_in_doc"] for r in rows],
        }),
        target, compression="zstd",
    )
    (out_dir / "PROVENANCE.json").write_text(json.dumps({
        "source": f"https://huggingface.co/datasets/vidore/vidore_v3_{sub}",
        "config": "corpus",
        "transport": "datasets-server /rows API",
        "columns": COLUMNS,
        "omitted": ["image"],
        "why": "image holds page renders (~12 GB across 8 subsets); no text arm reads it",
        "rows": len(rows),
    }, indent=1), encoding="utf-8")
    print(f"  {sub}: wrote {len(rows)} rows -> {target}", flush=True)

print("\ndone")
for sub in ENGLISH:
    p = DEST / sub / "corpus.parquet"
    if p.exists():
        print(f"  {sub:18s} {pq.read_metadata(p).num_rows:5d} pages  {p.stat().st_size/1e6:.1f} MB")
