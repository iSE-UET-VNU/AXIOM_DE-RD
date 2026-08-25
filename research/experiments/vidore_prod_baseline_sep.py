"""Reproduce the CSV 'Baseline Legacy' arm: KDL -> fixed_512/128 chunks -> MaxP to pages,
then test whether SEP adds anything ON TOP of it."""
import sys
from pathlib import Path
from collections import defaultdict
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
from research.experiments.physics_sep import propagate
from research.experiments.physics_sep_test import permutation,W,GAMMA,BETA,TOPM

SUB,LANG,ALPHA,POOL="physics","french",0.7,100
norm=lambda m: m/np.clip(np.linalg.norm(m,axis=-1,keepdims=True),1e-12,None)
run=next(Path(f"data_vidore_parsed_physics/output/benchmarks/vidore-v3-{SUB}-kdl").iterdir())
pages={}
for d in documents(run):
    doc=canonical_doc(d.get("document",{}).get("file_name"))
    for pg,bl in page_blocks(d).items():
        pages[unit_id(SUB,doc,pg)]="\n".join(b["text"] for b in bl if (b.get("text") or "").strip())
bench=load("vidore_v3",subset=SUB,language=LANG); qrels=bench.qrels()
qs=[q for q in bench.questions() if qrels.get(q.qid)]
ev=pytrec_eval.RelevanceEvaluator(qrels,{"ndcg_cut_10","recall_10"})
emb=OpenRouterEmbedder(cache_dir=Path("data/work/vidore_physics_kdl_chunk_emb"),batch_size=64)
qv=norm(np.asarray(emb.embed([q.query for q in qs]),dtype=np.float32))

# fixed_512/128 chunks, carrying their page owner
recs=[];owner=[]
for u,t in pages.items():
    if not t.strip(): continue
    spans=fixed_overlap(t,n_words=512,overlap=128)
    for i,sp in enumerate(spans):
        seg=t[sp[0]:sp[1]]
        if seg.strip(): recs.append(seg); owner.append(u)
print(f"pages {len(pages)}  chunks {len(recs)}  ({len(recs)/len(pages):.2f} chunks/page)")
bm=BM25Index(analyzer_name="plain").build(
    [{"chunk_id":f"c{i}","doc_id":owner[i],"text":recs[i]} for i in range(len(recs))])
M=norm(np.asarray(emb.embed(recs),dtype=np.float32))
owner=np.array(owner)

def maxp(scored):  # chunk scores -> best chunk per page
    best=defaultdict(lambda:-1e9)
    for pos,s in scored: best[owner[pos]]=max(best[owner[pos]],s)
    return sorted(best.items(),key=lambda kv:-kv[1])
pool={}
for q,v in zip(qs,qv):
    lex=bm.search(q.query,1000); ds=M@v
    top=np.argpartition(-ds,min(1000,len(ds)-1))[:1000]
    dn=sorted(((int(i),float(ds[i])) for i in top),key=lambda p:-p[1])
    lp=maxp([(p,s) for p,s in lex]); dp=maxp(dn)
    lu={u:s for u,s in lp}; du={u:s for u,s in dp}
    uids=list(dict.fromkeys([u for u,_ in lp]+[u for u,_ in dp]))
    idx={u:i for i,u in enumerate(uids)}
    fl=[(idx[u],s) for u,s in lp]; fd=[(idx[u],s) for u,s in dp]
    pool[q.qid]={uids[p]:float(s) for p,s in alpha_fuse(fl,fd,ALPHA,POOL)}
def full(r):
    s=ev.evaluate(r)
    return (100*sum(v["ndcg_cut_10"] for v in s.values())/len(s),
            100*sum(v["recall_10"] for v in s.values())/len(s),
            {k:100*v["ndcg_cut_10"] for k,v in s.items()})
nd,rc,bpq=full(pool)
print(f"\n=== CSV 'Baseline Legacy' reproduction (KDL + fixed_512/128 + MaxP + alpha0.7) ===")
print(f"  NDCG@10 {nd:.2f}   Recall@10 {rc:.2f}      [CSV says 44.2 / 47.47]")
print(f"\n{'arm':26s} {'NDCG@10':>8s} {'R@10':>7s} {'delta':>7s} {'p':>8s}")
for lam in (0.5,0.6,0.7,0.8):
    r={q:propagate(s,lam,W,GAMMA,BETA,TOPM) for q,s in pool.items()}
    n2,r2,pq=full(r); d,p,*_=permutation(bpq,pq,10000)
    print(f"{'+ SEP lambda='+str(lam):26s} {n2:8.2f} {r2:7.2f} {n2-nd:+7.2f} {p:8.4f}{'' if p<0.05 else '  n.s.'}")
