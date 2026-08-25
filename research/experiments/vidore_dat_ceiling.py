"""Ceiling test for DAT (Dynamic Alpha Tuning, arXiv 2503.23013) on our data.

DAT asks an LLM to score the effectiveness of the top-1 result from each leg,
then normalises the two scores into a per-query alpha. That is a POST-retrieval
signal, which is why our seven query-surface features (sec 12) all failed.

This replaces the LLM judge with the true qrels grade of each leg's top-1, i.e.
a perfect judge. If a perfect judge does not recover the +8.60 oracle headroom,
an actual LLM judge cannot either -- and we save the API spend.
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
from research.experiments.physics_sep_test import permutation

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
ev=pytrec_eval.RelevanceEvaluator(qrels,{"ndcg_cut_10"})
emb=OpenRouterEmbedder(cache_dir=Path("data/work/vidore_physics_kdl_chunk_emb"),batch_size=64)
qv=norm(np.asarray(emb.embed([q.query for q in qs]),dtype=np.float32))
recs=[];owner=[]
for u,t in pages.items():
    if not t.strip(): continue
    for sp in fixed_overlap(t,n_words=512,overlap=128):
        seg=t[sp[0]:sp[1]]
        if seg.strip(): recs.append(seg); owner.append(u)
M=norm(np.asarray(emb.embed(recs),dtype=np.float32))
owner=np.array(owner)
bm=BM25Index(analyzer_name="plain").build(
    [{"chunk_id":f"c{i}","doc_id":owner[i],"text":recs[i]} for i in range(len(recs))])
def maxp(scored):
    best=defaultdict(lambda:-1e9)
    for pos,s in scored: best[owner[pos]]=max(best[owner[pos]],s)
    return sorted(best.items(),key=lambda kv:-kv[1])
legs={}
for q,v in zip(qs,qv):
    lex=bm.search(q.query,1000); ds=M@v
    top=np.argpartition(-ds,min(1000,len(ds)-1))[:1000]
    dn=sorted(((int(i),float(ds[i])) for i in top),key=lambda p:-p[1])
    legs[q.qid]=(maxp(list(lex)),maxp(dn))
def fuse(qid,a):
    lp,dp=legs[qid]
    uids=list(dict.fromkeys([u for u,_ in lp]+[u for u,_ in dp]))
    idx={u:i for i,u in enumerate(uids)}
    fl=[(idx[u],s) for u,s in lp]; fd=[(idx[u],s) for u,s in dp]
    return {uids[p]:float(s) for p,s in alpha_fuse(fl,fd,a,POOL)}
def nd(run_):
    s=ev.evaluate(run_); return 100*sum(v["ndcg_cut_10"] for v in s.values())/len(s), \
        {k:100*v["ndcg_cut_10"] for k,v in s.items()}

ALPHAS=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
per={a:{q.qid:nd({q.qid:fuse(q.qid,a)})[0] for q in qs} for a in ALPHAS}
prod=sum(per[0.7].values())/len(qs); bestfix=max(sum(per[a].values())/len(qs) for a in ALPHAS)
oracle=sum(max(per[a][q.qid] for a in ALPHAS) for q in qs)/len(qs)
print(f"production a=0.7 {prod:.2f}   best fixed {bestfix:.2f}   full oracle {oracle:.2f}")

# DAT with a PERFECT judge: grade of each leg's top-1
def grade(qid,u): return qrels[qid].get(u,0)
rows=[]
for name,mapfn in (
    ("DAT perfect judge (graded 0/1/2)", lambda sl,sd: 0.5 if (sl+sd)==0 else sd/(sl+sd)),
    ("DAT perfect judge (binary)",       lambda sl,sd: 0.5 if (sl+sd)==0 else (1.0*(sd>0))/((sl>0)+(sd>0))),
):
    tot={}
    for q in qs:
        lp,dp=legs[q.qid]
        sl=grade(q.qid,lp[0][0]) if lp else 0
        sd=grade(q.qid,dp[0][0]) if dp else 0
        a=min(ALPHAS,key=lambda x:abs(x-mapfn(sl,sd)))
        tot[q.qid]=per[a][q.qid]
    m=sum(tot.values())/len(qs)
    rows.append((name,m,tot))
    print(f"  {name:36s} {m:6.2f}   vs production {m-prod:+.2f}   "
          f"recovers {100*(m-bestfix)/(oracle-bestfix):5.1f}% of oracle headroom")
# how often does the perfect judge even have signal?
both0=sum(1 for q in qs if grade(q.qid,legs[q.qid][0][0][0])==0 and grade(q.qid,legs[q.qid][1][0][0])==0)
agree=sum(1 for q in qs if (grade(q.qid,legs[q.qid][0][0][0])>0)==(grade(q.qid,legs[q.qid][1][0][0])>0))
print(f"\n  queries where BOTH top-1 are non-gold (judge gives no signal): {both0}/{len(qs)} "
      f"= {100*both0/len(qs):.0f}%")
print(f"  queries where both legs' top-1 agree in relevance:             {agree}/{len(qs)} "
      f"= {100*agree/len(qs):.0f}%  -> alpha falls back to 0.5")
for name,m,tot in rows:
    d,p,*_=permutation(per[0.7],tot,10000)
    print(f"  {name:36s} vs production: {d:+.2f}  p={p:.4f}{'' if p<0.05 else '  n.s.'}")

print("\n=== CONTROLS: is the gain the judge, or just the 0.5 fallback? ===")
import random
rng=random.Random(0)
def dat_alpha(sl,sd): return 0.5 if (sl+sd)==0 else sd/(sl+sd)
def snap(x): return min(ALPHAS,key=lambda a:abs(a-x))
grades={q.qid:(grade(q.qid,legs[q.qid][0][0][0]),grade(q.qid,legs[q.qid][1][0][0])) for q in qs}
true={q.qid:per[snap(dat_alpha(*grades[q.qid]))][q.qid] for q in qs}
for a in (0.5,0.6):
    fixed={q.qid:per[a][q.qid] for q in qs}
    d,p,*_=permutation(fixed,true,10000)
    print(f"  DAT vs FIXED alpha={a}  ({sum(fixed.values())/len(qs):.2f} -> "
          f"{sum(true.values())/len(qs):.2f})  {d:+.2f}  p={p:.4f}"
          f"{'' if p<0.05 else '  n.s.'}")
sh=[]
for _ in range(20):
    perm=[q.qid for q in qs]; rng.shuffle(perm)
    v=sum(per[snap(dat_alpha(*grades[donor]))][q] for q,donor in zip([x.qid for x in qs],perm))/len(qs)
    sh.append(v)
print(f"  shuffled judge (another query's grades): {sum(sh)/len(sh):6.2f} (sd {np.std(sh):.2f})"
      f"   vs true judge {sum(true.values())/len(qs):.2f}")

print("\n=== DAT + SEP: do they compose? ===")
from research.experiments.physics_sep import propagate
from research.experiments.physics_sep_test import W,GAMMA,BETA,TOPM
sep_per={}
for a in ALPHAS:
    for lam in (0.5,0.6):
        pools={q.qid:propagate(fuse(q.qid,a),lam,W,GAMMA,BETA,TOPM) for q in qs}
        sep_per[(a,lam)]={q.qid:nd({q.qid:pools[q.qid]})[0] for q in qs}
prodpq=per[0.7]
for lam in (0.5,0.6):
    sep_only={q.qid:sep_per[(0.7,lam)][q.qid] for q in qs}
    datsep={q.qid:sep_per[(snap(dat_alpha(*grades[q.qid])),lam)][q.qid] for q in qs}
    m_s=sum(sep_only.values())/len(qs); m_d=sum(datsep.values())/len(qs)
    d1,p1,*_=permutation(prodpq,sep_only,10000)
    d2,p2,*_=permutation(prodpq,datsep,10000)
    d3,p3,*_=permutation(sep_only,datsep,10000)
    print(f"  lambda={lam}:  SEP alone {m_s:.2f} ({d1:+.2f})   DAT+SEP {m_d:.2f} ({d2:+.2f}, p={p2:.4f})"
          f"   DAT adds {d3:+.2f} on top of SEP  p={p3:.4f}{'' if p3<0.05 else '  n.s.'}")
