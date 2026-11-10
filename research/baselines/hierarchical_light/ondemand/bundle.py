import hashlib
import json
import subprocess
import zipfile

from .bench import BENCH, ROOT, documents, fingerprint, queries, work

CODE = ("pyproject.toml", "ondemand")
BENCH_NAME = "0.1. BENCHMARK"
SKIP = ("__pycache__", ".pyc", ".DS_Store")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def build(target, bench=BENCH):
    docs = documents(bench)
    entries = []
    for root in CODE:
        path = ROOT / root
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        entries += [(p, p.relative_to(ROOT).as_posix()) for p in files if not any(s in str(p) for s in SKIP)]
    for name in ("documents.jsonl", "queries.jsonl", "qrels.jsonl", "summary.json"):
        entries.append((bench / name, f"{BENCH_NAME}/{name}"))
    entries += [(bench / d["path"], f"{BENCH_NAME}/{d['path']}") for d in docs]
    manifest = {"target": target, "smoke": False, "bundle_fingerprint": fingerprint(bench),
                "n_documents": len(docs), "n_queries": len(queries(bench)),
                "n_pages": sum(d["metadata"]["page_count"] for d in docs),
                "git": {"sha": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                        "uncommitted_files": len(git("status", "--porcelain").splitlines())},
                "files": {arc: sha256(src) for src, arc in entries}, "has_reference_scores": False}
    out = work("upload", bench=bench) / f"{target}_bundle.zip"
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w") as zf:
        for src, arc in entries:
            stored = arc.lower().endswith(".pdf")
            zf.write(src, arc, compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,
                     compresslevel=None if stored else 1)
        zf.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=1))
    return out
