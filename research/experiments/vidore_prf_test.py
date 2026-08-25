"""PRF/Rocchio in embedding space -- query-side aggregation.

Every aggregating intervention has won on this benchmark and every localising
one has lost. PRF is the untested aggregating lever, and the only one that can
touch failure case A (gold file absent from top-10), which nothing else targets.

    q' = a*q + b*mean(top-k pooled page vectors)      (renormalised)
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
from research.experiments.physics_sep_test import permutation

ALPHA,POOL=0.7,100
n=lambda m: m/np.clip(np.linalg.norm(m,axis=-1,keepdims=True),1e-12,None)
for SUB,LANG in (("physics","french"),("pharmaceuticals","english")):
    run=next(Path(f"data_vidore_parsed_physics/output/benchmarks/vidore-v3-{SUB}-kdl").iterdir())
    bench=load("vidore_v3",subset=SUB,language=LANG); qrels=bench.qrels()
    questions=[q for q in bench.questions() if qrels.get(q.qid)]
    pages={}
    for d in documents(run):
        doc=canonical_doc(d.get("document",{}).get("file_name"))
        for pg,bl in page_blocks(d).items():
            pages[unit_id(SUB,doc,pg)]="\n".join(b["text"] for b in bl if (b.get("text") or "").strip())
    emb=OpenRouterEmbedder(cache_dir=Path(f"data/work/vidore_{SUB}_kdl_emb"),batch_size=64)
    qv=n(np.asarray(emb.embed([q.query for q in questions]),dtype=np.float32))
    ids=[u for u,t in pages.items() if t.strip()]
    bm=BM25Index(analyzer_name="plain").build([{"chunk_id":u,"doc_id":u,"text":pages[u]} for u in ids])
    M=n(np.asarray(emb.embed([pages[u] for u in ids]),dtype=np.float32))
    ev=pytrec_eval.RelevanceEvaluator(qrels,{"ndcg_cut_10"})
    def score(qvecs):
        run_={}
        for q,v in zip(questions,qvecs):
            lex=bm.search(q.query,POOL); ds=M@v
            top=np.argpartition(-ds,min(POOL,len(ds)-1))[:POOL]
            dn=sorted(((int(i),float(ds[i])) for i in top),key=lambda p:-p[1])
            run_[q.qid]={ids[p]:float(s) for p,s in alpha_fuse(lex,dn,ALPHA,POOL)}
        s=ev.evaluate(run_)
        pq={k:100*v["ndcg_cut_10"] for k,v in s.items()}
        return sum(pq.values())/len(pq), pq, run_
    base,bpq,_=score(qv)
    # recall@100 of the baseline, to see if PRF can even help case A
    def rec100(r):
        tot=0
        for q in questions:
            g={u for u,v in qrels[q.qid].items() if v>0}
            tot+= len(g & set(r[q.qid]))/len(g)
        return 100*tot/len(questions)
    _,_,br=score(qv); print(f"\n=== KDL {SUB}/{LANG}  baseline {base:.2f}  R@100 {rec100(br):.1f} ===")
    print(f"{'k':>3s} {'b':>5s} {'ndcg':>7s} {'delta':>7s} {'R@100':>7s} {'p':>8s}")
    for k in (3,5,10):
        dv=np.zeros_like(qv)
        for i,v in enumerate(qv):
            ds=M@v; top=np.argpartition(-ds,k)[:k]
            top=sorted(top,key=lambda j:-ds[j])[:k]
            dv[i]=M[top].mean(axis=0)
        for b in (0.2,0.3,0.5):
            q2=n((1-b)*qv+b*dv)
            o,pq,r=score(q2); d,p,*_=permutation(bpq,pq,2000)
            print(f"{k:3d} {b:5.2f} {o:7.2f} {d:+7.2f} {rec100(r):7.1f} {p:8.4f}{'' if p<0.05 else '  n.s.'}")
