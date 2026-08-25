import json,sys
from pathlib import Path
sys.path.insert(0,'.')
import numpy as np, pytrec_eval
from src.utils.env import load_dotenv_file; load_dotenv_file(Path('.'))
from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import documents, page_blocks, canonical_doc
from src.evaluation.retrieval import alpha_fuse
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
from src.retrieval.sparse import BM25Index
from research.experiments.physics_sep import propagate, ndcg
from research.experiments.physics_sep_test import permutation, W,GAMMA,BETA,TOPM

ALPHA,POOL=0.7,100
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
    n=lambda m: m/np.clip(np.linalg.norm(m,axis=-1,keepdims=True),1e-12,None)
    qv=n(np.asarray(emb.embed([q.query for q in questions]),dtype=np.float32))
    ids=[u for u,t in pages.items() if t.strip()]
    bm=BM25Index(analyzer_name="plain").build([{"chunk_id":u,"doc_id":u,"text":pages[u]} for u in ids])
    M=n(np.asarray(emb.embed([pages[u] for u in ids]),dtype=np.float32))
    pool={}
    for q,v in zip(questions,qv):
        lex=bm.search(q.query,POOL); ds=M@v
        top=np.argpartition(-ds,min(POOL,len(ds)-1))[:POOL]
        dn=sorted(((int(i),float(ds[i])) for i in top),key=lambda p:-p[1])
        pool[q.qid]={ids[p]:float(s) for p,s in alpha_fuse(lex,dn,ALPHA,POOL)}
    ev=pytrec_eval.RelevanceEvaluator(qrels,{"ndcg_cut_10"})
    base,bpq=ndcg(ev,pool)
    print(f"\n=== KDL {SUB}/{LANG}  baseline alpha0.7 = {base:.2f} ===")
    for lam in (0.4,0.5,0.6,0.7,0.8):
        r={q:propagate(s,lam,W,GAMMA,BETA,TOPM) for q,s in pool.items()}
        o,pq=ndcg(ev,r); d,p,b,w,t=permutation(bpq,pq,10000)
        print(f"  lam {lam:.1f}  {o:6.2f}  {d:+6.2f}  p={p:.4f}  {b}/{w}/{t}{'' if p<0.05 else '  n.s.'}")
