"""QA for fixed top-50 and nested-OOF reranker+XGBoost K policies.

EVIE scores are cached; QA prompt/judge are the pinned existing benchmark
implementations. Answer and judgment calls resume by exact prompt hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/adaptive/llm/scripts"))
import run_adaptive_accurate_qa as qa  # noqa: E402

RERANK_OOF = ROOT / "data/work/reranker_adaptive_k_20261026/budget_58/oof_predictions.jsonl"
RERANK_CACHE = ROOT / "data/work/frozen_top100_reranker_evie_20261026/rerank"
OUT = ROOT / "data/work/reranker_adaptive_k_qa_20261026"
ALLOWED = ("fixed50", "cohere4pro_fixed50", "qwen3_8b_fixed50", "bge_m3_local_fixed50",
           "original_xgb", "cohere4pro_xgb", "qwen3_8b_xgb", "bge_m3_local_xgb")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_rows(arms: list[str]) -> tuple[list[dict], dict]:
    gate = json.loads(qa.GATE_PATH.read_text(encoding="utf-8"))
    queries = {r["query_id"]: r for r in read_jsonl(qa.QA_RECORDS_PATH)}
    kdl = {r["page_id"]: r.get("text") or "" for r in read_jsonl(qa.KDL_PATH)}
    oof = {(r["arm"], r["qid"]): r for r in read_jsonl(RERANK_OOF)}
    q_index = {q:i for i,q in enumerate(json.loads((qa.COLVEC_DIR/"colvec_qids.json").read_text(encoding="utf-8")))}
    p_index = {p:i for i,p in enumerate(json.loads((qa.COLVEC_DIR/"colvec_keys.json").read_text(encoding="utf-8")))}
    evie = np.load(qa.EVIE_DIR/"evie_scores_2048.npy", mmap_mode="r")
    assert len(queries) == 220 and evie.shape == (220,5091)
    rows = []
    for arm in arms:
        for qid in sorted(queries):
            if arm == "fixed50":
                k = 50
                ranking = gate["light_run_top100"][qid]
            elif arm.endswith("_fixed50"):
                k = 50
                source_arm = arm.removesuffix("_fixed50")
                path = RERANK_CACHE/source_arm/(hashlib.sha256(qid.encode()).hexdigest()+".json")
                record = json.loads(path.read_text(encoding="utf-8"))
                candidates = gate["light_run_top100"][qid]
                assert record["candidate_page_ids"] == candidates
                ranking = [candidates[i] for i in sorted(range(100), key=lambda i:(-float(record["scores"][i]),i))]
            else:
                source_arm = arm.removesuffix("_xgb")
                row = oof[(source_arm, qid)]
                k = int(row["prediction_k_oof"])
                ranking = row["ranking_top100"]
            prefix = ranking[:k]
            accurate = sorted(prefix, key=lambda pid: -float(evie[q_index[qid],p_index[pid]]))[:10]
            relevant = set(gate["gold_rank"][qid])
            dcg = sum(1/math.log2(i+2) for i,p in enumerate(accurate) if p in relevant)
            ideal = sum(1/math.log2(i+2) for i in range(min(len(relevant),len(accurate))))
            ctx = qa.get_context(accurate,kdl)
            prompt = qa.ANSWER_PROMPT.format(documents="\n\n".join(f"[{i+1}] {s}" for i,s in enumerate(ctx)),query=queries[qid]["query"])
            key = qa.stable_hash(qa.ANSWER_MODEL+"\n"+prompt)
            rows.append({"arm":arm,"qid":qid,"source":queries[qid]["source"],"chosen_k":k,
                         "prefix_recall":len(set(prefix)&relevant)/len(relevant),"accurate_recall_at_10":len(set(accurate)&relevant)/len(relevant),
                         "accurate_ndcg_at_10":dcg/ideal if ideal else 0.,"accurate_top_pages":accurate,
                         "answer_prompt_hash":key,"answer_prompt":prompt,"ctx":ctx})
    return rows,queries


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--arms",nargs="+",choices=ALLOWED,default=["fixed50","original_xgb","cohere4pro_xgb","qwen3_8b_xgb"])
    parser.add_argument("--workers",type=int,default=16)
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    qa.OUT=OUT
    rows,queries=build_rows(args.arms)
    assert len(rows)==220*len(args.arms)
    jobs={r["answer_prompt_hash"]:{"hash":r["answer_prompt_hash"],"qid":r["qid"],"prompt":r["answer_prompt"],
                                    "ctx":r["ctx"],"legacy_key":qa.answer_cache_key(r["qid"],r["ctx"])} for r in rows}
    print(f"Preflight: {len(rows)} rows, {len(jobs)} unique QA prompts",flush=True)
    if args.preflight:
        return
    started=perf_counter()
    answers={}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(qa.cached_or_call_answer,job,OUT):key for key,job in jobs.items()}
        for count,future in enumerate(as_completed(futures),1):
            key=futures[future]
            answers[key]=future.result()
            if count%50==0 or count==len(futures):
                print(f"answers {count}/{len(futures)}",flush=True)
    for row in rows:
        item=answers[row["answer_prompt_hash"]]
        row.update(answer=item.get("answer",""),answer_cache=item.get("cache"),answer_error=item.get("error"),
                   answer_latency_seconds=item.get("latency_seconds"),answer_cost_usd=item.get("cost_usd"))
        row.pop("answer_prompt");row.pop("ctx")
    if any(r["answer_error"] for r in rows):
        raise RuntimeError(f"{sum(bool(r['answer_error']) for r in rows)} QA answer failures; resume after fixing")
    qa.run_judges(rows,{"qa_rows":queries},OUT,args.workers)
    if any(r.get("judge_error") for r in rows):
        raise RuntimeError(f"{sum(bool(r.get('judge_error')) for r in rows)} judge failures; resume after fixing")
    report={"protocol":{"queries":220,"evie":"cached scores; model inference time excluded", "qa_model":qa.ANSWER_MODEL,
                        "judge_model":qa.JUDGE_MODEL,"adaptive":"nested 5-fold OOF, budget 58; fixed50 no model"},
            "wall_seconds":perf_counter()-started,"arms":{}}
    for arm in args.arms:
        subset=[r for r in rows if r["arm"]==arm]
        def aggregate(part):
            return {"n":len(part),"mean_k":float(np.mean([r["chosen_k"] for r in part])),
                    "prefix_recall":float(np.mean([r["prefix_recall"] for r in part])),
                    "evie_recall_at_10":float(np.mean([r["accurate_recall_at_10"] for r in part])),
                    "evie_ndcg_at_10":float(np.mean([r["accurate_ndcg_at_10"] for r in part])),
                    "qa_correct":float(np.mean([r["correct"] for r in part])),
                    "qa_correct_or_partial":float(np.mean([r["credited"] for r in part]))}
        report["arms"][arm]={"all":aggregate(subset),"by_source":{source:aggregate([r for r in subset if r["source"]==source])
                                                           for source in sorted({r["source"] for r in subset})}}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"per_query.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
