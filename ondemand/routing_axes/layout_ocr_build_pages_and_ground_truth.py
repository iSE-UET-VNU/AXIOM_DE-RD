import glob
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ondemand.bench import BENCH, FLAWED_QIDS, load_jsonl, work

W = work()
RUN = W / "results/ungated_evie_m10"
K3 = W / "routing_axes/k3/k3_input"
OUT = W / "routing_axes/k14"
PAGES = REPO / "kaggle/layout_ocr_pages.json"
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
TOP, SEED = 10, 0
NAMED = ([f"ohrbench::file_{h}#page=0" for h in ("16c197e1787e258c", "3022faa6428bc618", "44a0f44f0d18fc1e",
                                                  "e1357cf04ffdffbc", "8a511cfbc0889bf9", "660d716b738ce54b",
                                                  "82c69560f2e0ad55")]
         + ["ohrbench::file_0f0ac1ad8ed6dcc5#page=1", "ohrbench::file_7acf81f23d37fbcb#page=13"])
SMOKE_MORE = {"ohrbench": 10, "vidore_industrial": 6, "vidore_physics": 5}
TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$|\\begin\{tabular", re.M)


def context_pages():
    rows = [r for r in load_jsonl(RUN / "per_query.jsonl") if r["query_id"] not in FLAWED_QIDS]
    if len(rows) != 214:
        raise SystemExit(f"{RUN.name} has {len(rows)} queries after excluding flawed, expected 214")
    return sorted({p["unit"] for r in rows for p in r["branch_pages"][:TOP]})


def references():
    snap = lambda repo, pattern: sorted(glob.glob(str(HUB / f"datasets--{repo}/snapshots/*/{pattern}")))
    ohr = {(r["doc_name"], int(r["page_idx"])): r["gt_text"]
           for f in snap("opendatalab--OHR-Bench", "OHR-Bench_v2.parquet")
           for r in pq.read_table(f, columns=["doc_name", "page_idx", "gt_text"]).to_pylist()}
    vidore = {(name, r["doc_id"], int(r["page_number_in_doc"])): r["markdown"]
              for name in ("industrial", "physics") for f in snap(f"vidore--vidore_v3_{name}", "corpus/*.parquet")
              for r in pq.read_table(f, columns=["doc_id", "page_number_in_doc", "markdown"]).to_pylist()}
    if not ohr or not vidore:
        raise SystemExit(f"reference parquets not found under {HUB}")
    return ohr, vidore


def build():
    pages = context_pages()
    docs = {d["doc_id"]: d for d in load_jsonl(BENCH / "documents.jsonl")}
    where = {p["page_id"] for p in json.loads((K3 / "pages.json").read_text())}
    ohr, vidore = references()
    rows, missing = [], []
    for page in pages:
        doc, index = page.split("#page=")
        source = doc.split("::")[0]
        meta = docs[doc]["metadata"]
        name = meta["source_doc_id"].removesuffix(".pdf")
        if source == "ohrbench":
            text, kind, group = ohr.get((name, int(index))), "ohr_gt_text_v2", f"ohrbench:{meta['visual_classes'][0]}"
        else:
            text, kind, group = vidore.get((source.split("_")[1], name, int(index))), "vidore_markdown", source
        if text is None or page not in where:
            missing.append(page)
            continue
        rows.append({"page_id": page, "group": group, "reference": kind, "has_table": bool(TABLE_LINE.search(text)),
                     "text": text})
    if missing:
        raise SystemExit(f"{len(missing)} pages lack a reference or a k3_input PDF, e.g. {missing[:3]}")
    if not set(NAMED) <= set(pages):
        raise SystemExit(f"named smoke pages outside the context list: {sorted(set(NAMED) - set(pages))}")
    rng = random.Random(SEED)
    smoke = list(NAMED)
    for source, n in SMOKE_MORE.items():
        pool = [r["page_id"] for r in rows if r["page_id"].startswith(source) and r["page_id"] not in smoke
                and (r["has_table"] or source != "vidore_industrial")]
        smoke += sorted(rng.sample(pool, n))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "reference.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    PAGES.write_text(json.dumps({"pages": pages, "smoke": smoke, "top": TOP, "run": RUN.name, "seed": SEED}, indent=1))
    print(f"{len(pages)} pages {dict(Counter(r['group'] for r in rows))}, tables "
          f"{dict(Counter(r['group'] for r in rows if r['has_table']))}, smoke {len(smoke)}")
    print(f"-> {PAGES.relative_to(REPO)}, {OUT / 'reference.jsonl'}")


if __name__ == "__main__":
    build()
