"""Long-distance evidence: how far does same-file evidence stay useful,
and does propagating it MORE (wider window, multi-hop) help?"""
import sys
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
from research.experiments.physics_sep_test import permutation, W,GAMMA,BETA,TOPM
norm=lambda m: m/np.clip(np.linalg.norm(m,axis=-1,keepdims=True),1e-12,None)
run=next(Path("data_vidore_parsed_physics/output/benchmarks/vidore-v3-physics-kdl").iterdir())
pages={}
for d in documents(run):
    doc=canonical_doc(d.get("document",{}).get("file_name"))
    for pg,bl in page_blocks(d).items():
        pages[unit_id("physics",doc,pg)]="\n".join(b["text"] for b in bl if (b.get("text") or "").strip())
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
pools={}
for q,v in zip(qs,qv):
    lex=bm.search(q.query,1000); ds=M@v
    t=np.argpartition(-ds,min(1000,len(ds)-1))[:1000]
    dn=sorted(((int(i),float(ds[i])) for i in t),key=lambda p:-p[1])
    lp,dp=maxp(list(lex)),maxp(dn)
    uids=list(dict.fromkeys([u for u,_ in lp]+[u for u,_ in dp])); idx={u:i for i,u in enumerate(uids)}
    pools[q.qid]={uids[p]:float(s) for p,s in alpha_fuse([(idx[u],s) for u,s in lp],[(idx[u],s) for u,s in dp],0.7,100)}
ev=pytrec_eval.RelevanceEvaluator(qrels,{"ndcg_cut_10"})
def sc(r):
    s=ev.evaluate(r); return (100*sum(v["ndcg_cut_10"] for v in s.values())/len(s),
                              {k:100*v["ndcg_cut_10"] for k,v in s.items()})
base,bpq=sc(pools); print(f"baseline alpha0.7 {base:.2f}")
cur,cpq=sc({q:propagate(p,0.5,W,GAMMA,BETA,TOPM) for q,p in pools.items()})
print(f"SEP hien tai (w=2, gamma=0.5) {cur:.2f}  (+{cur-base:.2f})\n")

print("=== 1. Mo rong cua so lan can w (long-distance truc tiep) ===")
print(f"{'w':>3s} {'gamma':>6s} {'NDCG':>7s} {'vs SEP':>8s} {'p':>8s}")
for w in (2,3,5,8,12,20):
    for g in (0.5,0.8):
        n,pq=sc({q:propagate(p,0.5,w,g,BETA,TOPM) for q,p in pools.items()})
        d,pv,*_=permutation(cpq,pq,4000)
        print(f"{w:3d} {g:6.1f} {n:7.2f} {n-cur:+8.2f} {pv:8.4f}{'' if pv<0.05 else '  n.s.'}")

print("\n=== 2. Truyen nhieu buoc (multi-hop diffusion) ===")
print(f"{'hops':>5s} {'NDCG':>7s} {'vs SEP':>8s} {'p':>8s}")
for hops in (1,2,3):
    out={}
    for q,p in pools.items():
        cur_p=p
        for _ in range(hops): cur_p=propagate(cur_p,0.5,W,GAMMA,BETA,TOPM)
        out[q]=cur_p
    n,pq=sc(out); d,pv,*_=permutation(cpq,pq,4000)
    print(f"{hops:5d} {n:7.2f} {n-cur:+8.2f} {pv:8.4f}{'' if pv<0.05 else '  n.s.'}")

print("\n=== 3. Bang chung LIEN FILE (cross-document) ===")
multi=sum(1 for q in qs if len({split(u)[0] for u,v in qrels[q.qid].items() if v>0 and split(u)})>1)
print(f"  cau hoi co gold tren >1 file: {multi}/{len(qs)} = {100*multi/len(qs):.1f}%")
sub=[q.qid for q in qs if len({split(u)[0] for u,v in qrels[q.qid].items() if v>0 and split(u)})>1]
b_sub=sum(bpq[q] for q in sub)/len(sub); c_sub=sum(cpq[q] for q in sub)/len(sub)
oth=[q.qid for q in qs if q.qid not in set(sub)]
print(f"  SEP tren nhom multi-file : {b_sub:.2f} -> {c_sub:.2f}  ({c_sub-b_sub:+.2f})")
print(f"  SEP tren nhom single-file: {sum(bpq[q] for q in oth)/len(oth):.2f} -> "
      f"{sum(cpq[q] for q in oth)/len(oth):.2f} "
      f"({sum(cpq[q] for q in oth)/len(oth)-sum(bpq[q] for q in oth)/len(oth):+.2f})")
