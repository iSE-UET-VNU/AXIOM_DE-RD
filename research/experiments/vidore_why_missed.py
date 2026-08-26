"""First principles: khi ta truot mot trang gold, LY DO la text khong dien dat duoc,
hay la ranking khong tim ra? Neu trang truot nhieu hinh it chu -> visual co cho."""
import sys
from collections import defaultdict, Counter
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
norm=lambda m: m/np.clip(np.linalg.norm(m,axis=-1,keepdims=True),1e-12,None)
run=next(Path("data_vidore_parsed_physics/output/benchmarks/vidore-v3-physics-kdl").iterdir())
blocks={};pages={}
for d in documents(run):
    doc=canonical_doc(d.get("document",{}).get("file_name"))
    for pg,bl in page_blocks(d).items():
        u=unit_id("physics",doc,pg); blocks[u]=bl
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
ranked={}
for q,v in zip(qs,qv):
    lex=bm.search(q.query,1000); ds=M@v
    t=np.argpartition(-ds,min(1000,len(ds)-1))[:1000]
    dn=sorted(((int(i),float(ds[i])) for i in t),key=lambda p:-p[1])
    lp,dp=maxp(list(lex)),maxp(dn)
    uids=list(dict.fromkeys([u for u,_ in lp]+[u for u,_ in dp])); idx={u:i for i,u in enumerate(uids)}
    ranked[q.qid]=[uids[p] for p,_ in alpha_fuse([(idx[u],s) for u,s in lp],[(idx[u],s) for u,s in dp],0.7,1000)]
def props(u):
    bl=blocks.get(u,[])
    tot=sum(len(b.get("text") or "") for b in bl) or 1
    fig=sum(len(b.get("text") or "") for b in bl if b.get("type") in ("Figure","Caption"))
    txt=sum(len(b.get("text") or "") for b in bl if b.get("type") in ("Text","SectionHeader"))
    return {"chars":tot,"fig_ratio":fig/tot,"text_ratio":txt/tot,
            "n_fig":sum(1 for b in bl if b.get("type")=="Figure")}
hit=[];miss=[];deep=[]
for q in qs:
    g={u for u,v in qrels[q.qid].items() if v>0}
    r=ranked[q.qid]; pos={u:i for i,u in enumerate(r)}
    for u in g:
        p=pos.get(u,9999)
        (hit if p<10 else (miss if p<100 else deep)).append(props(u))
def summ(rows,label):
    if not rows: print(f"  {label}: 0"); return
    ch=sorted(r["chars"] for r in rows)
    print(f"  {label:34s} n={len(rows):4d}  chars(med) {ch[len(ch)//2]:5d}  "
          f"fig_ratio {sum(r['fig_ratio'] for r in rows)/len(rows):.3f}  "
          f"text_ratio {sum(r['text_ratio'] for r in rows)/len(rows):.3f}  "
          f"n_fig {sum(r['n_fig'] for r in rows)/len(rows):.2f}")
print("Trang GOLD, phan theo thu hang ta gan cho no:")
summ(hit ,"tim thay (rank < 10)")
summ(miss,"truot nhung con trong 100")
summ(deep,"truot han (rank >= 100)")
allp=[props(u) for u in pages]
summ(allp,"MOI trang trong corpus (tham chieu)")
# co bao nhieu trang gold gan nhu khong co chu?
low=[r for r in hit+miss+deep if r["chars"]<200]
print(f"\n  trang gold co <200 ky tu: {len(low)}/{len(hit)+len(miss)+len(deep)} "
      f"= {100*len(low)/(len(hit)+len(miss)+len(deep)):.1f}%")
lowmiss=[r for r in miss+deep if r["chars"]<200]
print(f"  trong so do, bi truot: {len(lowmiss)} "
      f"({100*len(lowmiss)/max(len(low),1):.0f}% cua nhom it chu)")
