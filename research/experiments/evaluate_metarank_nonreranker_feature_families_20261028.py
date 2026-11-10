"""Isolate non-reranker features for query-page LTR ablations."""
from __future__ import annotations

import json, math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRanker

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/work/metarank_feature_ablation_20261027'
GATE=ROOT/'data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/light_prep/gate_ppocrv5_ungated.json'
OUT=ROOT/'data/work/metarank_nonreranker_feature_families_20261028'
CAP=100
PARAMS=dict(objective='rank:pairwise',eval_metric='ndcg@20',n_estimators=180,max_depth=3,learning_rate=.05,
            min_child_weight=5,subsample=.85,colsample_bytree=.9,reg_lambda=5.,tree_method='hist',n_jobs=4,verbosity=0)
SCORE=['discovery_score_minmax','discovery_score_z','discovery_rank_pct']
DENSE=['te3_cosine','te3_score_minmax','te3_rank_pct','qwen8b_cosine','qwen8b_score_minmax','qwen8b_rank_pct']
LEX=['bm25_raw','bm25_score_minmax','bm25_rank_pct','lex_query_coverage','lex_query_precision','lex_matched_terms','lex_phrase_match','lex_numeric_overlap']
PAGE=['page_log_token_count','page_unique_token_ratio','page_numeric_density','page_table_marker_count','page_ocr_marker_count']
ARMS={'discovery_only':SCORE,'dense_only':DENSE,'lexical_only':LEX,'page_shape_only':PAGE,
      'pair_signals_dense_lexical':DENSE+LEX,'pair_signals_plus_page_shape':DENSE+LEX+PAGE,
      'discovery_plus_dense_lexical':SCORE+DENSE+LEX,'discovery_plus_page_shape':SCORE+PAGE,
      'discovery_plus_nonreranker_all':SCORE+DENSE+LEX+PAGE}

def read(path:Path): return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
def ndcg(ranked,gold,k):
 d=sum(1/math.log2(i+2) for i,p in enumerate(ranked[:k]) if p in gold)
 z=sum(1/math.log2(i+2) for i in range(min(k,len(gold))))
 return d/z if z else 0.

def main():
 gate=json.loads(GATE.read_text(encoding='utf-8')); qids=sorted(gate['light_run_top100']); src=np.asarray([q.split('::',1)[0] for q in qids])
 cand={q:gate['light_run_top100'][q] for q in qids}; gold={q:set(gate['gold_rank'][q]) for q in qids}
 meta=json.loads((DATA/'feature_matrix_metadata.json').read_text(encoding='utf-8')); names=meta['feature_names']
 if meta['qids']!=qids: raise ValueError('qid mismatch')
 full=np.load(DATA/'all_engineered_feature_matrix.npy',mmap_mode='r')
 labels=np.asarray([int(p in gold[q]) for q in qids for p in cand[q]],dtype=np.int32)
 splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
 rankings={}
 fold_times={}
 for arm,features in ARMS.items():
  ix=[names.index(n) for n in features]; x=np.asarray(full[:,ix],dtype=np.float32); oof={}; times=[]
  for f,(tr,te) in enumerate(splitter.split(np.zeros(len(qids)),src),1):
   tr_rows=np.concatenate([np.arange(i*CAP,(i+1)*CAP) for i in tr]); te_rows=np.concatenate([np.arange(i*CAP,(i+1)*CAP) for i in te])
   model=XGBRanker(**PARAMS,random_state=42+f); model.fit(x[tr_rows],labels[tr_rows],group=[CAP]*len(tr),verbose=False)
   pred=model.predict(x[te_rows]); times.append(0)
   for j,qi in enumerate(te):
    q=qids[int(qi)]; s=pred[j*CAP:(j+1)*CAP]; order=sorted(range(CAP),key=lambda z:(-float(s[z]),z)); oof[q]=[cand[q][z] for z in order]
  rankings[arm]=oof; fold_times[arm]=times; print(f'{arm}: complete',flush=True)
 per=[]; summaries={}
 for arm,run in rankings.items():
  rs=[]
  for q in qids:
   row={'qid':q,'source':q.split('::',1)[0],'arm':arm}
   for k in (20,50):
    h=len(set(run[q][:k])&gold[q]); r=h/len(gold[q]); p=h/k
    row.update({f'recall@{k}':r,f'precision@{k}':p,f'f1@{k}':2*r*p/(r+p) if r+p else 0.,f'ndcg@{k}':ndcg(run[q],gold[q],k)})
   rs.append(row); per.append(row)
  summaries[arm]={m:float(np.mean([r[m] for r in rs])) for m in rs[0] if m not in ('qid','source','arm')}
 base={r['qid']:r for r in per if r['arm']=='discovery_only'}; rng=np.random.default_rng(20261028); boot={}
 for arm in ARMS:
  if arm=='discovery_only': continue
  cur={r['qid']:r for r in per if r['arm']==arm}; boot[arm]={}
  for k in (20,50):
   d=np.asarray([cur[q][f'recall@{k}']-base[q][f'recall@{k}'] for q in qids]); ix=rng.integers(0,len(qids),(5000,len(qids))); bd=d[ix].mean(axis=1)*100; lo,hi=np.quantile(bd,[.025,.975])
   boot[arm][str(k)]={'delta_pp':float(d.mean()*100),'ci95_low_pp':float(lo),'ci95_high_pp':float(hi)}
 OUT.mkdir(parents=True,exist_ok=True)
 report={'protocol':{'queries':len(qids),'candidate_depth':CAP,'folds':5,'split':'source-stratified query-level OOF','features':ARMS,'no_cross_encoder_reranker_features':True,'xgboost_params':PARAMS},'arms':summaries,'paired_bootstrap_vs_discovery_only':boot}
 (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (OUT/'per_query.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in per),encoding='utf-8')
 lines=['# Non-reranker feature family ablation','','Five-fold source-stratified query OOF; frozen top-100 candidate set. No Cohere, Qwen reranker, or BGE reranker scores are used.','',
        '| Feature set | Recall@20 | nDCG@20 | Recall@50 | nDCG@50 | Δ Recall@20 vs discovery-only | Δ Recall@50 vs discovery-only |','|---|---:|---:|---:|---:|---:|---:|']
 for arm,m in summaries.items():
  d20=boot.get(arm,{}).get('20'); d50=boot.get(arm,{}).get('50')
  s20='—' if not d20 else f"{d20['delta_pp']:+.2f} [{d20['ci95_low_pp']:+.2f}, {d20['ci95_high_pp']:+.2f}]"
  s50='—' if not d50 else f"{d50['delta_pp']:+.2f} [{d50['ci95_low_pp']:+.2f}, {d50['ci95_high_pp']:+.2f}]"
  lines.append(f"| {arm} | {m['recall@20']*100:.2f}% | {m['ndcg@20']*100:.2f} | {m['recall@50']*100:.2f}% | {m['ndcg@50']*100:.2f} | {s20} | {s50} |")
 (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(summaries,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
