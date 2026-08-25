"""Was the French analyzer rejected under a confound?

It gives BM25 +2.71 standalone but was dropped for -0.13 "on the fusion" -- a
fusion at alpha=0.7, i.e. dense-dominant, which is where a better BM25 leg gets
washed out. The headroom scan then showed 37% of queries want pure BM25. So
re-test it across alpha, on the production config, with and without SEP.
"""
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

SUB,LANG,POOL="physics","french",100
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

# production chunking: fixed_512/128 + MaxP
recs=[];owner=[]
for u,t in pages.items():
    if not t.strip(): continue
    for sp in fixed_overlap(t,n_words=512,overlap=128):
        seg=t[sp[0]:sp[1]]
        if seg.strip(): recs.append(seg); owner.append(u)
M=norm(np.asarray(emb.embed(recs),dtype=np.float32))
owner=np.array(owner)
def maxp(scored):
    best=defaultdict(lambda:-1e9)
    for pos,s in scored: best[owner[pos]]=max(best[owner[pos]],s)
    return sorted(best.items(),key=lambda kv:-kv[1])
def full(r):
    s=ev.evaluate(r)
    return (100*sum(v["ndcg_cut_10"] for v in s.values())/len(s),
            100*sum(v["recall_10"] for v in s.values())/len(s),
            {k:100*v["ndcg_cut_10"] for k,v in s.items()})

ALPHAS=[0.3,0.4,0.5,0.6,0.7]
results={}
for analyzer in ("plain","french"):
    bm=BM25Index(analyzer_name=analyzer).build(
        [{"chunk_id":f"c{i}","doc_id":owner[i],"text":recs[i]} for i in range(len(recs))])
    pools={a:{} for a in ALPHAS}; lexonly={}
    for q,v in zip(qs,qv):
        lex=bm.search(q.query,1000); ds=M@v
        top=np.argpartition(-ds,min(1000,len(ds)-1))[:1000]
        dn=sorted(((int(i),float(ds[i])) for i in top),key=lambda p:-p[1])
        lp=maxp(list(lex)); dp=maxp(dn)
        uids=list(dict.fromkeys([u for u,_ in lp]+[u for u,_ in dp]))
        idx={u:i for i,u in enumerate(uids)}
        fl=[(idx[u],s) for u,s in lp]; fd=[(idx[u],s) for u,s in dp]
        lexonly[q.qid]={u:float(s) for u,s in lp[:POOL]}
        for a in ALPHAS:
            pools[a][q.qid]={uids[p]:float(s) for p,s in alpha_fuse(fl,fd,a,POOL)}
    results[analyzer]={"pools":pools,"lex":lexonly}
    print(f"\n=== analyzer={analyzer} (production KDL + fixed_512/128 + MaxP) ===")
    n,_,_=full(lexonly); print(f"  bm25 alone          {n:6.2f}")
    for a in ALPHAS:
        n,r,_=full(pools[a]); print(f"  alpha={a:.1f}            {n:6.2f}  R@10 {r:5.2f}")

print("\n=== french vs plain, paired ===")
n,_,lp_=full(results['plain']['lex']); n2,_,lf_=full(results['french']['lex'])
d,p,*_=permutation(lp_,lf_,10000)
print(f"  bm25 alone   plain {n:.2f} -> french {n2:.2f}   {d:+.2f}  p={p:.4f}")
for a in ALPHAS:
    _,_,pp=full(results['plain']['pools'][a]); nn,rr,ff=full(results['french']['pools'][a])
    d,p,*_=permutation(pp,ff,10000)
    base=sum(pp.values())/len(pp)
    print(f"  alpha={a:.1f}      plain {base:.2f} -> french {nn:.2f}   {d:+.2f}  p={p:.4f}"
          f"{'' if p<0.05 else '  n.s.'}")

print("\n=== best combination: french + SEP ===")
_,_,prod=full(results['plain']['pools'][0.7])
prod_nd=sum(prod.values())/len(prod)
best=None
for a in ALPHAS:
    for lam in (0.5,0.6,0.7):
        r={q:propagate(s,lam,W,GAMMA,BETA,TOPM) for q,s in results['french']['pools'][a].items()}
        nd,rc,pq=full(r); d,p,*_=permutation(prod,pq,10000)
        if best is None or nd>best[0]: best=(nd,rc,a,lam,d,p)
        print(f"  french alpha={a:.1f} + SEP lam={lam}   {nd:6.2f}  R@10 {rc:5.2f}   "
              f"vs production {d:+6.2f}  p={p:.4f}{'' if p<0.05 else '  n.s.'}")
nd,rc,a,lam,d,p=best
print(f"\n  BEST: french alpha={a} + SEP lambda={lam} -> NDCG@10 {nd:.2f} R@10 {rc:.2f}"
      f"   vs production 43.86/46.73  ({d:+.2f}, p={p:.4f})")

print("\n=== HELD-OUT: select (analyzer, alpha, lambda) on one fold, score the other ===")
import hashlib
fold=lambda q:int(hashlib.sha256(q.encode()).hexdigest(),16)%2
F={0:[q.qid for q in qs if fold(q.qid)==0],1:[q.qid for q in qs if fold(q.qid)==1]}
LAMS=(0.0,0.5,0.6,0.7)   # 0.0 == no SEP
cache={}
for an in ("plain","french"):
    for a in ALPHAS:
        for lam in LAMS:
            pool=results[an]['pools'][a]
            r=pool if lam==0.0 else {q:propagate(s,lam,W,GAMMA,BETA,TOPM) for q,s in pool.items()}
            _,_,pq=full(r); cache[(an,a,lam)]=pq
def mean(pq,ids): return sum(pq[q] for q in ids)/len(ids)
prod_pq=cache[("plain",0.7,0.0)]
for tune,test in ((0,1),(1,0)):
    best=max(cache, key=lambda k: mean(cache[k],F[tune]))
    held=mean(cache[best],F[test]); prod=mean(prod_pq,F[test])
    d,p,*_=permutation({q:prod_pq[q] for q in F[test]},
                       {q:cache[best][q] for q in F[test]},10000)
    print(f"  tune fold{tune} -> pick {best}  | fold{test}: {held:.2f} vs production {prod:.2f}"
          f"   {d:+.2f}  p={p:.4f}{'' if p<0.05 else '  n.s.'}")
# and the single config that is best on BOTH folds independently
common=[k for k in cache if k!=("plain",0.7,0.0)]
rank=sorted(common,key=lambda k:-(min(mean(cache[k],F[0]),mean(cache[k],F[1]))))
print("\n  most robust configs (ranked by WORST fold, guards against fold-luck):")
for k in rank[:5]:
    d,p,*_=permutation(prod_pq,cache[k],10000)
    print(f"    {str(k):26s} f0 {mean(cache[k],F[0]):5.2f}  f1 {mean(cache[k],F[1]):5.2f}"
          f"  all {sum(cache[k].values())/len(qs):5.2f}  vs prod {d:+5.2f} p={p:.4f}")
