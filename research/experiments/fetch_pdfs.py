"""Download a ViDoRe V3 subset's source PDFs, the ones the parse arms run over.

The dataset repo carries the documents themselves under ``pdfs/``; that folder
is not a datasets-server config, so it is reachable only through the hub file
API. Downloading them from the publisher's own site instead does not reproduce
the corpus -- documents get revised in place, and a revision that gains a page
shifts every page-level gold label after it.

    python research/experiments/fetch_pdfs.py physics data/raw/benchmarks/vidore_v3_physics
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq

SUBSET = sys.argv[1] if len(sys.argv) > 1 else "physics"
DEST = Path(sys.argv[2] if len(sys.argv) > 2 else f"data/raw/benchmarks/vidore_v3_{SUBSET}")
REPO = f"vidore/vidore_v3_{SUBSET}"


def page_count(path):
    """None when pdfinfo is absent, so a missing tool skips the check not the file."""
    if not shutil.which("pdfinfo"):
        return None
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    return next((int(l.split(":")[1]) for l in out.splitlines() if l.startswith("Pages:")), None)


def expected_pages():
    path = hf_hub_download(REPO, "documents_metadata/test-00000-of-00001.parquet",
                           repo_type="dataset")
    table = pq.read_table(path, columns=["file_name", "page_number"]).to_pydict()
    return dict(zip(table["file_name"], table["page_number"]))


want = expected_pages()
remote = [f for f in list_repo_files(REPO, repo_type="dataset")
          if f.startswith("pdfs/") and f.lower().endswith(".pdf")]
if not remote:
    raise SystemExit(f"{REPO} exposes no pdfs/; the subset layout changed.")

DEST.mkdir(parents=True, exist_ok=True)
failed, mismatched = [], []

for i, name in enumerate(sorted(remote), 1):
    out = DEST / Path(name).name
    try:
        cached = hf_hub_download(REPO, name, repo_type="dataset")
        if not Path(cached).open("rb").read(4) == b"%PDF":
            raise ValueError("not a PDF")
        out.write_bytes(Path(cached).read_bytes())
    except Exception as exc:
        failed.append({"file_name": out.name, "error": str(exc)})
        print(f"  [{i}/{len(remote)}] FAILED {out.name}: {exc}", flush=True)
        continue

    pages, exp = page_count(out), want.get(out.name)
    if pages is not None and exp is not None and pages != exp:
        mismatched.append({"file_name": out.name, "pages": pages, "expected": exp})
    note = "" if pages is None else f" {pages}p"
    flag = " MISMATCH" if mismatched and mismatched[-1]["file_name"] == out.name else ""
    print(f"  [{i}/{len(remote)}] {out.name}{note}{flag}", flush=True)

(DEST / "PROVENANCE.json").write_text(json.dumps({
    "source": f"https://huggingface.co/datasets/{REPO}",
    "path": "pdfs/",
    "transport": "huggingface_hub file API",
    "documents": len(remote),
    "expected_pages": sum(want.values()),
    "page_number_by_file": want,
    "failed": failed,
    "page_count_mismatched": mismatched,
}, indent=2, ensure_ascii=False), encoding="utf-8")

have = sorted(DEST.glob("*.pdf"))
print(f"\n{len(have)}/{len(remote)} PDFs in {DEST}")
print(f"pages: {sum(want.values())} expected across {len(want)} documents")
if failed or mismatched:
    # A short or drifted corpus reports as a parser difference; refuse instead.
    raise SystemExit(f"{len(failed)} failed, {len(mismatched)} page-count mismatch; see PROVENANCE.json")
