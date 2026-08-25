"""Trace SEP on one real query over the real KDL parse, printing every intermediate."""
import json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,'.')
import numpy as np, pytrec_eval
from src.utils.env import load_dotenv_file; load_dotenv_file(Path('.'))
from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import documents, page_blocks, canonical_doc
from src.evaluation.retrieval import alpha_fuse
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.chunking_embedding.chunkers.builtin import fixed_overlap
from src.retrieval.sparse import BM25Index
from research.experiments.physics_sep import propagate, split, minmax
from research.experiments.physics_sep_test import W,GAMMA,BETA,TOPM

norm=lambda m: m/np.clip(np.linalg.norm(m,axis=-1,keepdims=True),1e-12,None)
run=next(Path("data_vidore_parsed_physics/output/benchmarks/vidore-v3-physics-kdl").iterdir())
raw={}   # keep the blocks so we can show real KDL structure
pages={}
for d in documents(run):
    doc=canonical_doc(d.get("document",{}).get("file_name"))
    for pg,bl in page_blocks(d).items():
        u=unit_id("physics",doc,pg)
        raw[u]=bl
        pages[u]="\n".join(b["text"] for b in bl if (b.get("text") or "").strip())
bench=load("vidore_v3",subset="physics",language="french"); qrels=bench.qrels()
qs=[q for q in bench.questions() if qrels.get(q.qid)]
emb=OpenRouterEmbedder(cache_dir=Path("data/work/vidore_physics_kdl_chunk_emb"),batch_size=64)
qv=norm(np.asarray(emb.embed([q.query for q in qs]),dtype=np.float32))
recs=[];owner=[]
for u,t in pages.items():
    if not t.strip(): continue
    for sp in fixed_overlap(t,n_words=512,overlap=128):
        seg=t[sp[0]:sp[1]]
        if seg.strip(): recs.append(seg); owner.append(u)
M=norm(np.asarray(emb.embed(recs),dtype=np.float32)); owner=np.array(owner)
bm=BM25Index(analyzer_name="plain").build([{"chunk_id":f"c{i}","doc_id":owner[i],"text":recs[i]} for i in range(len(recs))])
def maxp(sc):
    best=defaultdict(lambda:-1e9)
    for p,s in sc: best[owner[p]]=max(best[owner[p]],s)
    return sorted(best.items(),key=lambda kv:-kv[1])
ev=pytrec_eval.RelevanceEvaluator(qrels,{"ndcg_cut_10"})
def nd(qid,r): return 100*ev.evaluate({qid:r})[qid]["ndcg_cut_10"]

pools={}
for q,v in zip(qs,qv):
    lex=bm.search(q.query,1000); ds=M@v
    t=np.argpartition(-ds,min(1000,len(ds)-1))[:1000]
    dn=sorted(((int(i),float(ds[i])) for i in t),key=lambda p:-p[1])
    lp,dp=maxp(list(lex)),maxp(dn)
    uids=list(dict.fromkeys([u for u,_ in lp]+[u for u,_ in dp])); idx={u:i for i,u in enumerate(uids)}
    pools[q.qid]={uids[p]:float(s) for p,s in alpha_fuse([(idx[u],s) for u,s in lp],[(idx[u],s) for u,s in dp],0.7,100)}

gains=[]
for q in qs:
    b=nd(q.qid,pools[q.qid]); a=nd(q.qid,propagate(pools[q.qid],0.5,W,GAMMA,BETA,TOPM))
    gains.append((a-b,b,a,q))
gains.sort(key=lambda x:-x[0])
d,b,a,q=gains[3]           # a clear but not freak example
gold={u for u,v in qrels[q.qid].items() if v>0}
pool=pools[q.qid]
print(json.dumps({"qid":q.qid,"query":q.query,"ndcg_before":round(b,2),"ndcg_after":round(a,2),
                  "gain":round(d,2),"n_gold":len(gold)},ensure_ascii=False,indent=1))
base=minmax(pool)
by_file=defaultdict(dict)
for u,v in base.items():
    p=split(u)
    if p: by_file[p[0]][p[1]]=v
after=propagate(pool,0.5,W,GAMMA,BETA,TOPM)
order_b=[u for u,_ in sorted(pool.items(),key=lambda kv:-kv[1])]
order_a=[u for u,_ in sorted(after.items(),key=lambda kv:-kv[1])]
rank_b={u:i+1 for i,u in enumerate(order_b)}; rank_a={u:i+1 for i,u in enumerate(order_a)}
rows=[]
for u in order_b[:12]+[u for u in order_a[:12] if u not in order_b[:12]]:
    f,pg=split(u)
    pages_in=by_file[f]
    nb=sum((GAMMA**abs(dd))*pages_in[pg+dd] for dd in range(-W,W+1) if dd and (pg+dd) in pages_in)
    top=sorted(pages_in.values(),reverse=True)[:TOPM]
    rows.append({"unit":u,"file":f.split('_Ch-')[0][:26],"page":pg,"gold":u in gold,
        "s_norm":round(base[u],4),"A_file":round(sum(top)/len(top),4),"N_raw":round(nb,4),
        "pages_of_file_in_pool":len(pages_in),
        "rank_before":rank_b[u],"rank_after":rank_a.get(u,999),
        "score_after":round(after[u],4)})
print(json.dumps(rows,ensure_ascii=False,indent=1))
# real KDL blocks for the top page, to show the structure the unit_id comes from
top_unit=order_a[0]
print(json.dumps({"top_unit_after":top_unit,"blocks":[
    {"component_id":x["component_id"],"type":x["type"],"page":x["page"],
     "text":(x.get("text") or "")[:70]} for x in raw[top_unit][:6]]},ensure_ascii=False,indent=1))

print("\n=== NORMALISED INTERMEDIATES (so the arithmetic is checkable) ===")
lam=0.5
nb={}; ag={}
for u in base:
    f,pg=split(u); pi=by_file[f]
    nb[u]=sum((GAMMA**abs(dd))*pi[pg+dd] for dd in range(-W,W+1) if dd and (pg+dd) in pi)
    top=sorted(pi.values(),reverse=True)[:TOPM]
    ag[u]=sum(top)/len(top) if top else 0.0
nbn, agn = minmax(nb), minmax(ag)
for u in (order_b[0], order_a[0]):
    f,pg=split(u)
    lhs=lam*base[u]+(1-lam)*(BETA*agn[u]+(1-BETA)*nbn[u])
    print(json.dumps({"unit":u,"gold":u in gold,
        "s_norm":round(base[u],4),
        "A_file_raw":round(ag[u],4),"A_file_norm":round(agn[u],4),
        "N_raw":round(nb[u],4),"N_norm":round(nbn[u],4),
        "formula":f"0.5*{base[u]:.4f} + 0.5*(0.75*{agn[u]:.4f} + 0.25*{nbn[u]:.4f})",
        "computed":round(lhs,4),"actual_after":round(after[u],4),
        "rank_before":rank_b[u],"rank_after":rank_a[u]},ensure_ascii=False))
