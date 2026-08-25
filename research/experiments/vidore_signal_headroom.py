"""Headroom scan: oracle ceilings for unexploited signals, KDL physics.

Measures what each signal could give if we predicted it perfectly, so we
prioritise by available headroom rather than intuition. An oracle is never a
result -- it is an upper bound that tells us whether a predictor is worth
building.
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
from src.retrieval.sparse import BM25Index

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
emb=OpenRouterEmbedder(cache_dir=Path("data/work/vidore_physics_kdl_emb"),batch_size=64)
qv=norm(np.asarray(emb.embed([q.query for q in qs]),dtype=np.float32))
ids=[u for u,t in pages.items() if t.strip()]
bm=BM25Index(analyzer_name="plain").build([{"chunk_id":u,"doc_id":u,"text":pages[u]} for u in ids])
M=norm(np.asarray(emb.embed([pages[u] for u in ids]),dtype=np.float32))

ALPHAS=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
legs={}
for q,v in zip(qs,qv):
    lex=bm.search(q.query,POOL); ds=M@v
    top=np.argpartition(-ds,min(POOL,len(ds)-1))[:POOL]
    dn=sorted(((int(i),float(ds[i])) for i in top),key=lambda p:-p[1])
    legs[q.qid]=(lex,dn)
def nd_one(qid,run1):
    return 100*ev.evaluate({qid:run1})[qid]["ndcg_cut_10"]
per_alpha={a:{} for a in ALPHAS}
for q in qs:
    lex,dn=legs[q.qid]
    for a in ALPHAS:
        r={ids[p]:float(s) for p,s in alpha_fuse(lex,dn,a,POOL)}
        per_alpha[a][q.qid]=nd_one(q.qid,r)
print("=== fixed alpha (what we do today) ===")
for a in ALPHAS:
    m=sum(per_alpha[a].values())/len(qs)
    print(f"  alpha={a:.1f}  NDCG@10 {m:6.2f}{'   <- production' if abs(a-0.7)<1e-9 else ''}")
best_fixed=max(sum(per_alpha[a].values())/len(qs) for a in ALPHAS)
oracle=sum(max(per_alpha[a][q.qid] for a in ALPHAS) for q in qs)/len(qs)
print(f"\n=== ORACLE per-query alpha ===")
print(f"  best fixed alpha      {best_fixed:6.2f}")
print(f"  oracle per-query      {oracle:6.2f}   headroom +{oracle-best_fixed:.2f}")
# how often is 0.7 the right choice?
wins=defaultdict(int)
for q in qs:
    b=max(ALPHAS,key=lambda a:per_alpha[a][q.qid]); wins[b]+=1
print("  best-alpha distribution:", {f"{k:.1f}":v for k,v in sorted(wins.items())})
# how much does a 2-way choice (pure bm25 vs pure dense vs 0.7) get?
two=sum(max(per_alpha[0.0][q.qid],per_alpha[1.0][q.qid],per_alpha[0.7][q.qid]) for q in qs)/len(qs)
print(f"  oracle over just {{0.0, 0.7, 1.0}}: {two:6.2f}   headroom +{two-best_fixed:.2f}")

# --- ADVERSARIAL CONTROL: is the oracle real, or winner's curse over 11 noisy options? ---
import random
rng=random.Random(0)
qids=[q.qid for q in qs]
shuf=[]
for _ in range(20):
    perm=qids[:]; rng.shuffle(perm)
    # give each query the best alpha OF ANOTHER query -> destroys any real per-query signal
    val=0.0
    for q,donor in zip(qids,perm):
        a=max(ALPHAS,key=lambda a:per_alpha[a][donor])
        val+=per_alpha[a][q]
    shuf.append(val/len(qids))
rand=[]
for _ in range(20):
    val=sum(per_alpha[rng.choice(ALPHAS)][q] for q in qids)/len(qids)
    rand.append(val)
print(f"\n=== control ===")
print(f"  shuffled-oracle (best alpha of a RANDOM other query): {sum(shuf)/len(shuf):6.2f} "
      f"(sd {np.std(shuf):.2f})")
print(f"  random alpha per query:                               {sum(rand)/len(rand):6.2f} "
      f"(sd {np.std(rand):.2f})")
print(f"  true oracle:                                          {oracle:6.2f}")
print(f"  => genuine per-query signal = {oracle - sum(shuf)/len(shuf):.2f} of the {oracle-best_fixed:.2f} headroom")

# --- can any cheap feature predict the best alpha? ---
import math
feats={}
for q in qs:
    lex,dn=legs[q.qid]
    ls=[s for _,s in lex]; dsc=[s for _,s in dn]
    lex_ids={ids[p] for p,_ in lex[:10]}; dn_ids={ids[p] for p,_ in dn[:10]}
    feats[q.qid]={
        "bm25_max": ls[0] if ls else 0.0,
        "bm25_gap": (ls[0]-ls[1]) if len(ls)>1 else 0.0,
        "bm25_gap_rel": ((ls[0]-ls[1])/ls[0]) if len(ls)>1 and ls[0]>0 else 0.0,
        "dense_max": dsc[0] if dsc else 0.0,
        "dense_gap": (dsc[0]-dsc[1]) if len(dsc)>1 else 0.0,
        "overlap@10": len(lex_ids&dn_ids)/10.0,
        "qlen": len(q.query.split()),
        "n_lex_hits": len(lex),
    }
best_a={q.qid:max(ALPHAS,key=lambda a:per_alpha[a][q.qid]) for q in qs}
print(f"\n=== correlation of cheap features with the best alpha ===")
ys=np.array([best_a[q.qid] for q in qs])
for k in list(feats[qs[0].qid]):
    xs=np.array([feats[q.qid][k] for q in qs])
    if xs.std()<1e-12: continue
    r=np.corrcoef(xs,ys)[0,1]
    print(f"  {k:14s} pearson r = {r:+.3f}")

print("\n=== what distinguishes BM25-favouring from dense-favouring queries? ===")
lexq=[q for q in qs if best_a[q.qid]<=0.1]
denq=[q for q in qs if best_a[q.qid]>=0.9]
import re
def props(group,label):
    lens=[len(q.query.split()) for q in group]
    digits=sum(1 for q in group if re.search(r'\d',q.query))
    caps=sum(1 for q in group if re.search(r'\b[A-Z]{2,}\b',q.query))
    quest=sum(1 for q in group if q.query.strip().endswith('?'))
    # rare-term mass: min document frequency among query terms
    dfs=[]
    for q in group:
        toks=[t for t in re.findall(r"\w+",q.query.lower()) if len(t)>3]
        d=[bm.document_frequency(t) if hasattr(bm,'document_frequency') else None for t in toks]
        d=[x for x in d if x]
        if d: dfs.append(min(d))
    print(f"  {label:22s} n={len(group):3d}  qlen {sum(lens)/len(lens):4.1f}  "
          f"has-digit {100*digits/len(group):4.1f}%  has-ACRONYM {100*caps/len(group):4.1f}%  "
          f"min-df {sum(dfs)/len(dfs) if dfs else float('nan'):6.1f}")
props(lexq,"BM25-favouring")
props(denq,"dense-favouring")
print("\n  sample BM25-favouring queries:")
for q in lexq[:5]: print("    ",q.query[:110])
print("\n  sample dense-favouring queries:")
for q in denq[:5]: print("    ",q.query[:110])
# does the ORACLE gain concentrate on queries the baseline already fails?
base={q.qid:per_alpha[0.7][q.qid] for q in qs}
gains=sorted(((max(per_alpha[a][q.qid] for a in ALPHAS)-base[q.qid],q) for q in qs),key=lambda x:-x[0])
zero=[g for g,q in gains if base[q.qid]==0]
print(f"\n  queries where production scores 0: {sum(1 for q in qs if base[q.qid]==0)}"
      f"   mean oracle gain on them {sum(zero)/len(zero) if zero else 0:.1f}")
print(f"  total oracle gain concentrated in top-50 queries: "
      f"{100*sum(g for g,_ in gains[:50])/sum(g for g,_ in gains):.0f}%")

print("\n=== query STYLE as the predictor: function-word density ===")
from src.retrieval.sparse import FRENCH_STOPWORDS as SW
INTERROG={"quel","quelle","quels","quelles","comment","pourquoi","combien","quand",
          "où","qui","que","quoi","est-ce","sont","expliquer","décrire","analyser"}
def style(qtext):
    toks=[t.lower() for t in re.findall(r"\w+",qtext)]
    if not toks: return 0.0,0.0
    sw=sum(1 for t in toks if t in SW)/len(toks)
    it=1.0 if any(t in INTERROG for t in toks) else 0.0
    return sw,it
ys=np.array([best_a[q.qid] for q in qs])
sws=np.array([style(q.query)[0] for q in qs]); its=np.array([style(q.query)[1] for q in qs])
qm=np.array([1.0 if q.query.strip().endswith("?") else 0.0 for q in qs])
for name,xs in (("stopword_ratio",sws),("has_interrogative",its),("ends_with_?",qm)):
    print(f"  {name:18s} pearson r with best-alpha = {np.corrcoef(xs,ys)[0,1]:+.3f}")
print(f"\n  BM25-favouring  mean stopword ratio {sws[ys<=0.1].mean():.3f}")
print(f"  dense-favouring mean stopword ratio {sws[ys>=0.9].mean():.3f}")

# Simple two-bucket router, tuned on one half and scored on the other (both directions)
import hashlib
fold=lambda q:int(hashlib.sha256(q.encode()).hexdigest(),16)%2
f0=[q for q in qs if fold(q.qid)==0]; f1=[q for q in qs if fold(q.qid)==1]
def route_score(group,thr,a_lo,a_hi):
    return sum(per_alpha[a_lo if style(q.query)[0]<thr else a_hi][q.qid] for q in group)/len(group)
print("\n=== held-out router: alpha = a_lo if stopword_ratio < thr else a_hi ===")
best=None
for thr in (0.20,0.25,0.30,0.35,0.40,0.45,0.50):
    for a_lo in (0.0,0.1,0.2,0.3,0.4):
        for a_hi in (0.5,0.6,0.7,0.8,0.9,1.0):
            s=route_score(f0,thr,a_lo,a_hi)
            if best is None or s>best[0]: best=(s,thr,a_lo,a_hi)
s0,thr,a_lo,a_hi=best
held=route_score(f1,thr,a_lo,a_hi)
b1=sum(per_alpha[0.7][q.qid] for q in f1)/len(f1)
b1b=sum(per_alpha[0.5][q.qid] for q in f1)/len(f1)
print(f"  tuned on fold0 (thr={thr} a_lo={a_lo} a_hi={a_hi}) -> fold1 {held:.2f} "
      f"vs prod a=0.7 {b1:.2f} ({held-b1:+.2f}) vs best-fixed a=0.5 {b1b:.2f} ({held-b1b:+.2f})")
best=None
for thr in (0.20,0.25,0.30,0.35,0.40,0.45,0.50):
    for a_lo in (0.0,0.1,0.2,0.3,0.4):
        for a_hi in (0.5,0.6,0.7,0.8,0.9,1.0):
            s=route_score(f1,thr,a_lo,a_hi)
            if best is None or s>best[0]: best=(s,thr,a_lo,a_hi)
s1,thr,a_lo,a_hi=best
held=route_score(f0,thr,a_lo,a_hi)
b0=sum(per_alpha[0.7][q.qid] for q in f0)/len(f0)
b0b=sum(per_alpha[0.5][q.qid] for q in f0)/len(f0)
print(f"  tuned on fold1 (thr={thr} a_lo={a_lo} a_hi={a_hi}) -> fold0 {held:.2f} "
      f"vs prod a=0.7 {b0:.2f} ({held-b0:+.2f}) vs best-fixed a=0.5 {b0b:.2f} ({held-b0b:+.2f})")

print("\n=== the disagreement signal: what does a fixed-alpha fusion DISCARD? ===")
def gold_of(qid): return {u for u,v in qrels[qid].items() if v>0}
import itertools
for K in (10,20):
    fused_r=union_r=lex_only=den_only=0.0
    extra=0
    for q in qs:
        lex,dn=legs[q.qid]; g=gold_of(q.qid)
        f=[ids[p] for p,_ in alpha_fuse(lex,dn,0.7,POOL)][:K]
        L=[ids[p] for p,_ in lex[:K//2]]; D=[ids[p] for p,_ in dn[:K//2]]
        u=list(dict.fromkeys(L+D))
        fused_r+=len(g&set(f))/len(g); union_r+=len(g&set(u))/len(g)
        lex_only+=len(g&(set(L)-set(D)))/len(g); den_only+=len(g&(set(D)-set(L)))/len(g)
        extra+=len(set(u)-set(f))
    n=len(qs)
    print(f"  K={K:2d}  recall  fused(alpha0.7 top-{K}) {100*fused_r/n:5.2f}   "
          f"union(bm25 top-{K//2} + dense top-{K//2}) {100*union_r/n:5.2f}   "
          f"delta {100*(union_r-fused_r)/n:+5.2f}")
    print(f"        gold found ONLY by bm25 {100*lex_only/n:5.2f}   ONLY by dense {100*den_only/n:5.2f}"
          f"   |  union brings {extra/n:.1f} candidates/query the fusion dropped")
