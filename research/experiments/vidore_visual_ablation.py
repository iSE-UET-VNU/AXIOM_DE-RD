"""Visual ablation using BM25 only -- no embeddings, so no API spend."""
import sys
from collections import defaultdict, Counter
from pathlib import Path
sys.path.insert(0,'.')
import pytrec_eval
from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3 import unit_id
from src.evaluation.pipeline_pages import documents, page_blocks, canonical_doc
from src.retrieval.sparse import BM25Index
from research.experiments.physics_sep_test import permutation
run=next(Path("data_vidore_parsed_physics/output/benchmarks/vidore-v3-physics-kdl").iterdir())
blocks={}
for d in documents(run):
    doc=canonical_doc(d.get("document",{}).get("file_name"))
    for pg,bl in page_blocks(d).items():
        blocks[unit_id("physics",doc,pg)]=bl
bench=load("vidore_v3",subset="physics",language="french"); qrels=bench.qrels()
qs=[q for q in bench.questions() if qrels.get(q.qid)]
ev=pytrec_eval.RelevanceEvaluator(qrels,{"ndcg_cut_10"})
def build(drop):
    return {u:"\n".join(b["text"] for b in bl
            if (b.get("text") or "").strip() and b.get("type") not in drop)
            for u,bl in blocks.items()}
def arm(pages,label,ref=None):
    ids=[u for u,t in pages.items() if t.strip()]
    bm=BM25Index(analyzer_name="plain").build([{"chunk_id":u,"doc_id":u,"text":pages[u]} for u in ids])
    runr={}
    for q in qs:
        runr[q.qid]={ids[p]:float(s) for p,s in bm.search(q.query,100)}
    s=ev.evaluate(runr); pq={k:100*v["ndcg_cut_10"] for k,v in s.items()}
    n=sum(pq.values())/len(pq)
    ch=[len(t) for t in pages.values() if t.strip()]
    line=f"  {label:32s} {sum(ch)//len(ch):6d} chars  BM25 {n:6.2f}"
    if ref is not None:
        d,p,*_=permutation(ref,pq,10000)
        line+=f"  {n-sum(ref.values())/len(ref):+6.2f}  p={p:.4f}{'' if p<0.05 else '  n.s.'}"
    print(line,flush=True)
    return pq
c=Counter(b.get("type") for bl in blocks.values() for b in bl if (b.get("text") or "").strip())
print("Block co text, theo loai:",dict(c))
tot=sum(len(b.get("text") or "") for bl in blocks.values() for b in bl)
fig=sum(len(b.get("text") or "") for bl in blocks.values() for b in bl if b.get("type")=="Figure")
tab=sum(len(b.get("text") or "") for bl in blocks.values() for b in bl if b.get("type")=="Table")
print(f"Ty le ky tu: Figure {100*fig/tot:.1f}%  Table {100*tab/tot:.1f}%  con lai {100*(tot-fig-tab)/tot:.1f}%\n")
full=arm(build(set()),"day du (hien tai)")
arm(build({"Figure"}),"bo Figure",full)
arm(build({"Figure","Caption"}),"bo Figure + Caption",full)
arm(build({"Table"}),"bo Table",full)
arm(build({"Figure","Caption","Table","EquationBlock"}),"chi Text + SectionHeader",full)
