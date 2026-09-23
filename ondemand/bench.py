import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "0.1. BENCHMARK"
WORK_ROOT = ROOT / "data/work/ondemand_v2"


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def fingerprint(bench=BENCH):
    digest = hashlib.sha256()
    for name in ("documents.jsonl", "queries.jsonl", "qrels.jsonl"):
        digest.update(name.encode())
        digest.update((Path(bench) / name).read_bytes())
    return digest.hexdigest()[:12]


def work(*parts, bench=BENCH):
    path = WORK_ROOT / fingerprint(bench) / Path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def unit_id(doc_id, page0):
    return f"{doc_id}#page={page0}"


def doc_of(unit):
    return unit.split("#page=")[0]


def source_of(key):
    return key.split("::", 1)[0]


def documents(bench=BENCH):
    return load_jsonl(Path(bench) / "documents.jsonl")


def queries(bench=BENCH):
    return {q["query_id"]: q for q in load_jsonl(Path(bench) / "queries.jsonl")}


def gold(bench=BENCH):
    out = defaultdict(dict)
    for row in load_jsonl(work("light_prep", bench=bench) / "qrels_page_level.jsonl"):
        out[row["query_id"]][row["page_id"]] = max(out[row["query_id"]].get(row["page_id"], 0), int(row["relevance"]))
    return dict(out)


def group_of(qid, query):
    if source_of(qid) == "ohrbench":
        return "ohrbench:" + query["metadata"].get("visual_class", "?")
    return source_of(qid)
