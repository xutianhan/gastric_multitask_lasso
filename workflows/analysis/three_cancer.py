#!/usr/bin/env python3
"""Recompute all repeated-partition metrics and pairwise conditional intervals."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,json,csv,collections,datetime
from pathlib import Path
import numpy as np
from gastric_mtl.paths import WORKSPACE, REFERENCE_SHARING
from gastric_mtl.data import load_submission_cohorts
from workflows.experiments.initial_evaluation import dump,digest
from workflows.analysis.shared_metrics import aggregate
from workflows.experiments.three_cancer import OUT,CTS,SEEDS,GRAPHS
BASE=['single',*GRAPHS,'global_selected','target_selected','clinical']
AUG=['clinical_plus_single','clinical_plus_pair_ESCA_STAD','clinical_plus_triple']
FORCED=['forced_'+g for g in GRAPHS]
OLD=REFERENCE_SHARING

def main():
 assert json.loads((OUT/'COMPLETED.json').read_text())['completed_jobs']==10
 assert json.loads((OUT/'validation.json').read_text())['passed']
 co=load_submission_cohorts();results={};historical={};checks=[];hashes={};selections=[];comparisons=[];foldrows=[];curves=[]
 for seed in SEEDS:
  records=[];old=[]
  for fo in range(5):
   p=OUT/f'seed_{seed}/fold_{fo}/outer_predictions.json';done=json.loads((p.parent/'COMPLETED.json').read_text());assert digest(p)==done['prediction_sha256']
   r=json.loads(p.read_text());records.append(r);hashes[str(p.relative_to(WORKSPACE))]=digest(p)
   op=OLD/f'seed_{seed}/fold_{fo}/outer_predictions.json';o=json.loads(op.read_text());old.append(o)
   d=json.loads((p.parent/'selection_before_test.json').read_text());od=json.loads((op.parent/'selection_before_test.json').read_text())
   for name,s in d['selection'].items():
    selections.append(dict(seed=seed,fold=fo,procedure=name,target='all',graph=s['graph'],candidate=s['id'],mu=s['mu'],fractions=s['fractions'],changed_from_original=s['id']!=od['selection'][name]['id'] or s['fraction_indices']!=od['selection'][name]['fraction_indices']))
   for c,s in d['target_selection'].items():
    selections.append(dict(seed=seed,fold=fo,procedure='target_selected',target=c,graph=s['graph'],candidate=s['id'],mu=s['mu'],fractions=s['fractions'],changed_from_original=s['id']!=od['target_selection'][c]['id'] or s['fraction_indices']!=od['target_selection'][c]['fraction_indices']))
   for c in CTS:
    for m in BASE+AUG+FORCED:
     v=r['cancers'][c]['models'][m]
     foldrows.append(dict(seed=seed,fold=fo,cancer=c,model=m,c_index=v['test_c'],difference_vs_single=v['test_c']-r['cancers'][c]['models']['single']['test_c'],brier_1y=v['survival_metrics']['brier_365.25'],brier_3y=v['survival_metrics']['brier_1095.75']))
  results[str(seed)],check=aggregate(records,co,seed,BASE+AUG+FORCED);checks.append(check)
  historical[str(seed)],check=aggregate(old,co,seed,BASE+FORCED);checks.append(check)
  for table in [results[str(seed)],historical[str(seed)]]:
   for c in CTS:
    for name,v in table[c].items():
     graph=name[len('forced_'):] if name.startswith('forced_') else name
     if graph in GRAPHS:v['eligible_sharing_target']=CTS.index(c) in GRAPHS[graph]
  for g in GRAPHS:
   for mu in ['0.0001','0.001','0.01','0.1','1.0']:
    rr=[]
    for r in records:
     b={'cancers':{}}
     for c in CTS:
      block={k:v for k,v in r['cancers'][c].items() if k!='models'}
      block['models']={'single':r['cancers'][c]['models']['single'],'clinical':r['cancers'][c]['models']['clinical'],'curve':r['strength_curves'][g][mu]['cancers'][c]};b['cancers'][c]=block
     rr.append(b)
    vals,_=aggregate(rr,co,seed,['single','clinical','curve'])
    for c in CTS:curves.append(dict(seed=seed,cancer=c,graph=g,mu=float(mu),**{key:vals[c]['curve'][key] for key in ['c_index','brier_1y','brier_3y']}))
 fitrecords=[]
 for path in OUT.glob('seed_*/fold_*/*/*.json'):
  if not path.parent.name.startswith(('inner_','final_fits')) or not path.stem.startswith(('single_','pair_','triple_')):continue
  v=json.loads(path.read_text());fit=v.get('fit',v)
  assert fit['gradient_norm']<=1e-5
  fitrecords.append({'file':str(path.relative_to(OUT)),'phase':'inner' if path.parent.name.startswith('inner_') else 'outer','kind':'single' if path.stem.startswith('single_') else 'joint','gradient_norm':fit['gradient_norm'],'seconds':fit['total_seconds'],'n_iter':sum(a['n_iter'] for a in fit['attempts'])})
 clinical_audits=[]
 for path in OUT.glob('seed_*/fold_*/clinical_fit_audit.json'):clinical_audits+=json.loads(path.read_text())
 assert all(v['gradient_norm']<=1e-5 for v in clinical_audits)
 rows=[]
 for seed,byc in results.items():
  for c,bym in byc.items():
   for m,v in bym.items():
    row={'seed':int(seed),'cancer':c,'model':m}
    for k,x in v.items():
     if isinstance(x,dict):row[k]=x['difference'];row[k+'_low']=x['ci95'][0];row[k+'_high']=x['ci95'][1]
     elif isinstance(x,list):row[k+'_low']=x[0];row[k+'_high']=x[1]
     else:row[k]=x
    rows.append(row)
    if m in historical[seed][c]:
     ov=historical[seed][c][m]
     comparisons.append(dict(seed=int(seed),cancer=c,model=m,old_c=ov['c_index'],new_c=v['c_index'],c_change=v['c_index']-ov['c_index'],old_delta_c=ov['c_index_vs_single']['difference'],new_delta_c=v['c_index_vs_single']['difference']))
 folders=collections.Counter((v['phase'],v['kind']) for v in fitrecords)
 result={'complete':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':digest(OUT/'protocol.json'),'summary_script_sha256':digest(__file__),'source_sha256':hashes,'results':results,'historical_results':historical,'selection_records':selections,'fold_metrics':foldrows,'strength_curves':curves,'old_new_comparison':comparisons,'fit_diagnostics':{'counts':{':'.join(k):v for k,v in folders.items()},'total':len(fitrecords),'strict':len(fitrecords),'maximum_raw_gradient':max(v['gradient_norm'] for v in fitrecords),'clinical_total':len(clinical_audits),'clinical_strict':len(clinical_audits)},'validation':{'passed':True,'fold_brier_checks':sum(v['fold_brier_checks'] for v in checks),'max_brier_error':max(v['max_abs_brier_error'] for v in checks)},'uncertainty':'1000 paired patient bootstrap replicates within each test fold, predictions and censoring estimates fixed. Two partitions reuse patients; no full-refit uncertainty, multiplicity adjustment, equivalence test or external validation.'}
 dump(OUT.parent/'rerun_facts.json',result);dump(OUT.parent/'fit_diagnostics.json',fitrecords)
 folder=OUT.parent/'tables';folder.mkdir(exist_ok=True)
 for name,items in [('all_metrics',rows),('old_new_comparison',comparisons),('fold_metrics',foldrows),('strength_curves',curves),('selection_records',selections)]:
  keys=list(dict.fromkeys(k for row in items for k in row))
  with (folder/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(items)
 print(json.dumps(result['fit_diagnostics'],indent=2))
if __name__=='__main__':main()
