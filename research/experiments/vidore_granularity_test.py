"""Does page-level pooling cost us? BM25-only, free, no re-embed.

Compares scoring the pooled page against MaxSim over sub-page units under
several groupings, plus a random-grouping control at matched unit count to
separate "better units" from "more chances to score high".
"""
import json,sys,random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,'.')
import pytrec_eval
from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import documents, page_blocks, canonical_doc
from src.retrieval.sparse import BM25Index

SUB="physics"
run=next(Path(f"data_vidore_parsed_physics/output/benchmarks/vidore-v3-{SUB}-kdl").iterdir())
bench=load("vidore_v3",subset=SUB,language="french"); qrels=bench.qrels()
questions=[q for q in bench.questions() if qrels.get(q.qid)]

page_blocks_map={}
for d in documents(run):
    doc=canonical_doc(d.get("document",{}).get("file_name"))
    for pg,blocks in page_blocks(d).items():
        page_blocks_map[unit_id(SUB,doc,pg)]=[b for b in blocks if (b.get("text") or "").strip()]

def group_page(blocks, mode, rng=None):
    if mode=="page":    return ["\n".join(b["text"] for b in blocks)]
    if mode=="block":   return [b["text"] for b in blocks]
    if mode=="section":
        out,cur=[],[]
        for b in blocks:
            if b.get("type")=="SectionHeader" and cur: out.append(cur); cur=[]
            cur.append(b)
        if cur: out.append(cur)
        return ["\n".join(x["text"] for x in g) for g in out]
    if mode=="typed":
        out,cur=[],[]
        for b in blocks:
            t=b.get("type")
            if t in ("Table","Figure","Caption"): 
                if cur: out.append(cur); cur=[]
                out.append([b])
            else: cur.append(b)
        if cur: out.append(cur)
        return ["\n".join(x["text"] for x in g) for g in out]
    if mode.startswith("fixed"):
        k=int(mode[5:]); return ["\n".join(b["text"] for b in blocks[i:i+k]) for i in range(0,len(blocks),k)]
    if mode.startswith("rand"):
        k=int(mode[4:]); idx=list(range(len(blocks))); rng.shuffle(idx)
        groups=[idx[i:i+k] for i in range(0,len(idx),k)]
        return ["\n".join(blocks[j]["text"] for j in sorted(g)) for g in groups]
    raise ValueError(mode)

ev=pytrec_eval.RelevanceEvaluator(qrels,{"ndcg_cut_10"})
print(f"{'grouping':12s} {'units':>7s} {'u/page':>7s} {'NDCG@10':>8s} {'delta':>7s}")
base=None
for mode in ("page","section","typed","fixed3","fixed5","block","rand3","rand5"):
    rng=random.Random(0)
    recs=[];owner=[]
    for uid,blocks in page_blocks_map.items():
        for j,txt in enumerate(group_page(blocks,mode,rng)):
            if txt.strip(): recs.append({"chunk_id":f"{uid}##{j}","doc_id":uid,"text":txt}); owner.append(uid)
    bm=BM25Index(analyzer_name="plain").build(recs)
    runres={}
    for q in questions:
        hits=bm.search(q.query,500)
        best=defaultdict(lambda:-1e9)
        for pos,sc in hits:
            best[owner[pos]]=max(best[owner[pos]],sc)   # MaxSim over units of a page
        runres[q.qid]={u:float(s) for u,s in sorted(best.items(),key=lambda kv:-kv[1])[:100]}
    sc=ev.evaluate(runres); n=100*sum(v["ndcg_cut_10"] for v in sc.values())/len(sc)
    if base is None: base=n
    print(f"{mode:12s} {len(recs):7d} {len(recs)/len(page_blocks_map):7.1f} {n:8.2f} {n-base:+7.2f}")
