import json
import zipfile
from collections import defaultdict

import fitz

from .bench import BENCH, documents, gold as load_gold, group_of, load_jsonl, queries as load_queries, source_of, work
from .light_prep import wants_ocr
from .text import real_text

CONF = 60


def weak(row):
    return row["ocr_applied"] and (row["ocr_mean_confidence"] < CONF or row["ocr_word_count"] == 0)


SELECTORS = {"weak": (weak, "pages_ocr_sparse.jsonl"), "targets": (wants_ocr, "pages_inspector.jsonl")}


def build(bench=BENCH, select="weak", name=None, pages_name=None):
    keep, default_pages = SELECTORS[select]
    source = work("light_prep", bench=bench) / (pages_name or default_pages)
    if not source.exists():
        raise SystemExit(f"{source} is missing; build it first (light-prep --ocr none writes pages_inspector.jsonl)")
    rows = [r for r in load_jsonl(source) if keep(r)]
    kdl = {r["page_id"]: r["text"] for r in load_jsonl(work("kdl", bench=bench) / "kdl_pages.jsonl")}
    gold, queries = load_gold(bench), load_queries(bench)
    evidence = defaultdict(list)
    for q, row in gold.items():
        if source_of(q) != "ohrbench":
            continue
        meta = queries[q]["metadata"]
        for u, g in row.items():
            if g > 0:
                evidence[u].append({"query_id": q, "group": group_of(q, queries[q]),
                                    "evidence_context": meta.get("evidence_context", ""), "answers": queries[q]["answers"]})
    path_of = {d["doc_id"]: bench / d["path"] for d in documents(bench)}
    out = work("upload", bench=bench) / (name or f"light_ocr_bundle_{select}.zip")
    out.unlink(missing_ok=True)
    manifest = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in rows:
            name = f"pages/{r['doc_id'].replace('::', '__')}__p{r['page_index']}.pdf"
            with fitz.open(path_of[r["doc_id"]]) as src, fitz.open() as one:
                one.insert_pdf(src, from_page=r["page_index"], to_page=r["page_index"])
                zf.writestr(name, one.tobytes())
            manifest.append({"unit": r["page_id"], "doc_id": r["doc_id"], "source": r["source"],
                             "page_index": r["page_index"], "file": name,
                             "tesseract_text": real_text(r["text"]), "tesseract_conf": r["ocr_mean_confidence"],
                             "tesseract_words": r["ocr_word_count"], "kdl_text": real_text(kdl.get(r["page_id"], "")),
                             "evidence": evidence.get(r["page_id"], [])})
        zf.writestr("pages.jsonl", "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest))
    print(len(manifest), "pages,", sum(bool(m["evidence"]) for m in manifest), "with OHR-Bench evidence ->", out)
    return out


if __name__ == "__main__":
    build()
