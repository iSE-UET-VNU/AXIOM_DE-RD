"""Move every result artifact out of the session scratchpad and add what was missing.

The end-to-end rows recorded how many gold pages landed in context but not which
pages were retrieved, so a verdict could not be traced back to its evidence.
Retrieval is deterministic given the cached embeddings, so the ranking is
re-derived here rather than re-generated.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.utils.env import load_dotenv_file

load_dotenv_file(Path(__file__).resolve().parents[2])

from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks
from src.evaluation.retrieval import alpha_fuse
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.retrieval.sparse import BM25Index

SP = Path(__file__).parent
DEST = Path(__file__).resolve().parents[2] / "data/benchmark/vidore_v3/results"
RUN = (str(Path(__file__).resolve().parents[2] / "data_vidore_parsed_physics/output/benchmarks/vidore-v3-physics-chandra2/c6e049e4fd9c4b9b"))
TOP_K, DEPTH, ALPHA = 10, 100, 0.7

DEST.mkdir(parents=True, exist_ok=True)
bench = load("vidore_v3", subset="physics", language="french")
qrels = bench.qrels()
questions = [q for q in bench.questions() if qrels.get(q.qid)]
vidore_pages = {d.doc_id: (d.text or "") for d in bench.corpus()}
chandra_pages = {}
for document in documents(RUN):
    doc = canonical_doc(document.get("document", {}).get("file_name"))
    for page, blocks in page_blocks(document).items():
        chandra_pages[unit_id("physics", doc, page)] = "\n".join(
            b["text"] for b in blocks if b["text"].strip())

embedder = OpenRouterEmbedder(cache_dir=SP / "physics_emb", batch_size=64)
qvec = np.asarray(embedder.embed([q.query for q in questions]), dtype=np.float32)
qvec /= np.clip(np.linalg.norm(qvec, axis=1, keepdims=True), 1e-12, None)


def ranking(pages):
    units = [u for u, t in pages.items() if t.strip()]
    texts = [pages[u] for u in units]
    bm25 = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": u, "doc_id": u, "text": t} for u, t in zip(units, texts)])
    matrix = np.asarray(embedder.embed(texts), dtype=np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    out = {}
    for question, vector in zip(questions, qvec):
        lexical = bm25.search(question.query, DEPTH)
        scores = matrix @ vector
        top = np.argpartition(-scores, DEPTH)[:DEPTH]
        dense = sorted(((int(i), float(scores[i])) for i in top), key=lambda p: -p[1])
        out[question.qid] = [(units[p], round(float(s), 6))
                             for p, s in alpha_fuse(lexical, dense, ALPHA, DEPTH)[:TOP_K]]
    return out


rankings = {"retrieved_vidore": ranking(vidore_pages),
            "retrieved_chandra2": ranking(chandra_pages),
            "oracle": {q.qid: [(u, float(g)) for u, g in sorted(qrels[q.qid].items())]
                       for q in questions}}

for name, ranked in rankings.items():
    path = SP / "physics_e2e" / f"{name}.json"
    rows = json.loads(path.read_text())
    gold = {q.qid: set(qrels[q.qid]) for q in questions}
    for row in rows:
        hits = ranked[row["qid"]]
        row["retrieved"] = [{"unit_id": u, "score": s, "is_gold": u in gold[row["qid"]]}
                            for u, s in hits]
    out = DEST / f"physics_e2e_{name}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    check = sum(r["gold_hit"] == sum(1 for h in r["retrieved"] if h["is_gold"]) for r in rows)
    print(f"{name:20s} {len(rows):4d} rows  gold_hit reconciles {check}/{len(rows)}")

for src, dst in [(SP / "oracle_gen", DEST / "english_oracle_generation"),
                 (SP / "physics_ladder.json", DEST / "physics_retrieval_ladder.json"),
                 (SP / "physics_e2e" / "summary.json", DEST / "physics_e2e_summary.json")]:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    print(f"saved {dst.relative_to(DEST.parent.parent.parent)}")
