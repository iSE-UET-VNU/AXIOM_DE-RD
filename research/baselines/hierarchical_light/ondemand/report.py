import csv
import json
from time import perf_counter

import numpy as np
import pytrec_eval

from .bench import BENCH, gold as load_gold, work
from .text import ChunkBM25
from .timing import rows

COLUMNS = ["Method", "Correct_only", "Correct+partial", "File Recall@20", "Light NDCG@10", "Light Recall@10",
           "Light NDCG@20", "Light Recall@20", "Light NDCG@AdaptiveK", "Light Recall@AdaptiveK", "Accurate NDCG@10",
           "Accurate Recall@10", "Time Light Preparation (s)", "Latency Light Retrieval (s/q)", "Latency Parsing (s/q)",
           "Latency Chunk&Embed&Index (s/q)", "Latency Retrieval (s/q)", "Infer Time (s/q)"]


def latest_per_doc(records):
    last = {}
    for r in records:
        last[r.get("doc_id")] = r
    return list(last.values())


def adaptive_light(gate, gold):
    ndcg, recall = [], []
    for q, k in gate["adaptive_k"].items():
        ranked = gate["light_run_top100"][q][:k]
        ev = pytrec_eval.RelevanceEvaluator({q: gold[q]}, {f"ndcg_cut_{k}", f"recall_{k}"})
        r = ev.evaluate({q: {u: float(100 - i) for i, u in enumerate(ranked)}})[q]
        ndcg.append(r[f"ndcg_cut_{k}"])
        recall.append(r[f"recall_{k}"])
    return round(100 * float(np.mean(ndcg)), 2), round(100 * float(np.mean(recall)), 2)


def times(gate, bench):
    k = gate["gate_k"]
    prep = work("light_prep", bench=bench)
    lp = rows(prep / "timings.jsonl")
    pdf_inspector = sum(r["seconds"] for r in latest_per_doc(r for r in lp if r["stage"] == "pdf_inspector_parse"))
    tesseract = sum(r["seconds"] for r in lp if r["stage"] == "tesseract_ocr_wall")
    qwen = json.loads((work("light_v2", bench=bench) / "dense_qwen8b_meta.json").read_text())
    light_query = [r["seconds"] for r in rows(prep / "gate_timings.jsonl") if r["stage"] == "light_query"]
    cv = [r["seconds"] for r in rows(work("colvec", bench=bench) / "timings.jsonl") if r["stage"] == "colvec_page"]
    kdl_wall = [r for r in rows(work("kdl", bench=bench) / "timings.jsonl") if r["stage"] == "kdl_corpus_wall"][-1]
    emb = rows(work("embeddings", "kdl", bench=bench) / "timings.jsonl")
    per_chunk = sum(r["seconds"] for r in emb if r["stage"] == "embed_chunks") / sum(r["n"] for r in emb if r["stage"] == "embed_chunks")
    per_query_embed = sum(r["seconds"] for r in emb if r["stage"] == "embed_queries") / sum(r["n"] for r in emb if r["stage"] == "embed_queries")
    folder = work("embeddings", "kdl", bench=bench)
    owners = json.loads((folder / "chunk_owners.json").read_text())
    texts = json.loads((folder / "chunk_texts.json").read_text())
    by_page = {}
    for i, u in enumerate(owners):
        by_page.setdefault(u, []).append(i)
    n_chunks, index_seconds = [], []
    for units in gate["gate"].values():
        ids = [i for u in units for i in by_page.get(u, [])]
        n_chunks.append(len(ids))
        started = perf_counter()
        ChunkBM25([texts[i] for i in ids])
        index_seconds.append(perf_counter() - started)
    return {"light_prep": pdf_inspector + tesseract + qwen["page_seconds"],
            "light_retrieval": float(np.mean(light_query)) + qwen["query_seconds_each"],
            "parsing": kdl_wall["seconds"] / kdl_wall["n"] * k,
            "chunk_embed_index": float(np.mean(n_chunks)) * per_chunk + per_query_embed + float(np.mean(index_seconds)),
            "retrieval": float(np.mean(cv)) * k}


def sheet(tag, gate_name="gate_k20_best.json", bench=BENCH):
    gate = json.loads((work("light_prep", bench=bench) / gate_name).read_text())
    res = work("results", tag, bench=bench)
    metrics = json.loads((res / "metrics.json").read_text())
    b_query = [r["seconds"] for r in rows(res / "timings.jsonl") if r["stage"] == "arm_b_query"]
    t = times(gate, bench)
    light = gate["metrics"]["all"]
    adaptive = adaptive_light(gate, load_gold(bench))
    out = []
    for arm, label in (("A", "Arm A: light gate -> ColVec1.1-8B, no parsing; QA reads KDL text of its top pages"),
                       ("B", "Arm B: light gate -> KDL -> chunk hybrid fused with ColVec1.1-8B; QA reads KDL text")):
        path = res / f"qa_{arm}_summary.json"
        qa = json.loads(path.read_text()) if path.exists() else {}
        acc = metrics["arms"][arm]["all"]
        out.append([label, qa.get("correct_only", ""), qa.get("correct_plus_partial", ""), light["file_recall@20"],
                    light["light_ndcg@10"], light["light_recall@10"], light["light_ndcg@20"], light["light_recall@20"],
                    adaptive[0], adaptive[1], acc["ndcg@10"], acc["recall@10"], round(t["light_prep"], 1),
                    round(t["light_retrieval"], 3), "-" if arm == "A" else round(t["parsing"], 2),
                    "-" if arm == "A" else round(t["chunk_embed_index"], 2),
                    round(t["retrieval"] + (float(np.mean(b_query)) if arm == "B" else 0.0), 2),
                    qa.get("infer_seconds_per_query", "")])
    with (res / "sheet.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(out)
    return res / "sheet.csv"
