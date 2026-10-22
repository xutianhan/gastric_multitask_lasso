#!/usr/bin/env python3
"""Paired conditional metrics for the frozen EAC/STAD histology extension."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,json,csv,collections,datetime
from pathlib import Path
import numpy as np
from gastric_mtl.paths import ROOT, WORKSPACE
from gastric_mtl.eac import load_eac_cohorts,CTS
from workflows.experiments.initial_evaluation import dump,digest
from workflows.experiments.eac_stad import OUT,SEEDS,MUS
ACTIVE={'pair_EAC_STAD':CTS,'forced_pair_EAC_STAD':CTS}
MODELS=['single','pair_EAC_STAD','forced_pair_EAC_STAD','target_selected',
        'clinical','clinical_plus_single','clinical_plus_pair_EAC_STAD']
EXTRA={'clinical_plus_pair_EAC_STAD':['clinical_plus_single']}

def reverse_km(t,e,q,left=False):
 times=np.unique(t);values=[];g=1.
 for v in times:
  cens=int(((t==v)&~e).sum());den=int((t>=v).sum())-int(((t==v)&e).sum())
  if cens:g*=1.-cens/den
  values.append(g)
 ind=np.searchsorted(times,q,side='left' if left else 'right')-1
 return np.array([1. if j<0 else values[j] for j in ind])

def count_arrays(t,e,r):
 valid=e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:]));d=r[:,None]-r[None,:]
 return valid.astype(float),(valid*((d>0)+.5*(d==0))).astype(float)

def aggregate(records,coh,seed,models,extra_references=None):
 output={};maxerror=0.;checked=0
 for c in CTS:
  all_ids=sum([r['cancers'][c]['patient_ids'] for r in records],[]);assert len(all_ids)==len(set(all_ids))==len(coh[c]['patient_ids'])
  lookup={str(v):i for i,v in enumerate(coh[c]['patient_ids'])};nm=len(models)
  num=np.zeros(nm);den=0.;bn=np.zeros((nm,1000));bd=np.zeros(1000);brier=np.zeros((nm,2));bb=np.zeros((nm,2,1000));n=0
  rng=np.random.RandomState(20260917+seed+100*CTS.index(c))
  for rec in records:
   b=rec['cancers'][c];pid=b['patient_ids'];train=b['train_patient_ids'];assert set(pid).isdisjoint(train) and set(pid)|set(train)==set(lookup)
   ix=np.array([lookup[x] for x in pid]);itr=np.array([lookup[x] for x in train]);t=np.array(b['time']);e=np.array(b['event'],bool)
   np.testing.assert_array_equal(t,coh[c]['time'][ix]);np.testing.assert_array_equal(e,coh[c]['event'][ix].astype(bool))
   tt=coh[c]['time'][itr];ee=coh[c]['event'][itr].astype(bool);nf=len(t);n+=nf
   w=rng.multinomial(nf,np.full(nf,1./nf),size=1000).astype(float)
   valid,_=count_arrays(t,e,np.zeros(nf));den+=valid.sum();bd+=np.sum((w@valid)*w,axis=1)
   for mi,m in enumerate(models):
    model=b['models'][m];r=np.array(model['risk']);assert np.isfinite(r).all()
    _,credit=count_arrays(t,e,r);num[mi]+=credit.sum();bn[mi]+=np.sum((w@credit)*w,axis=1)
    assert abs(float(credit.sum()/valid.sum())-model['test_c'])<1e-12,(c,m,'cindex')
    probs=np.array(model['survival_1y_3y']);assert probs.shape==(nf,2) and np.isfinite(probs).all() and (probs>=0).all() and (probs<=1).all()
    for hj,horizon in enumerate([365.25,1095.75]):
     deaths=e&(t<=horizon);survivors=t>horizon;g_h=reverse_km(tt,ee,[horizon])[0];g_d=reverse_km(tt,ee,t[deaths],left=True)
     assert horizon<tt.max() and g_h>1e-8 and (g_d>1e-8).all()
     x=np.zeros(nf);x[deaths]=probs[deaths,hj]**2/g_d;x[survivors]=(1-probs[survivors,hj])**2/g_h
     error=abs(float(x.mean())-model['survival_metrics']['brier_'+str(horizon)]);maxerror=max(maxerror,error);checked+=1;assert error<1e-12,(c,m,horizon,error)
     brier[mi,hj]+=x.sum();bb[mi,hj]+=w@x
    if m in ACTIVE and c not in ACTIVE[m]:
     assert model['risk']==b['models']['single']['risk'] and model['survival_1y_3y']==b['models']['single']['survival_1y_3y']
  assert (bd>0).all();points=np.column_stack([num/den,brier/n]);samples=np.concatenate([(bn/bd)[:,None,:],bb/n],axis=1)
  result={}
  for mi,m in enumerate(models):
   result[m]={'eligible_sharing_target':m not in ACTIVE or c in ACTIVE[m]}
   for ki,key in enumerate(['c_index','brier_1y','brier_3y']):
    result[m][key]=float(points[mi,ki]);result[m][key+'_ci95']=np.quantile(samples[mi,ki],[.025,.975]).tolist()
    for ref in ['single','clinical']+(extra_references or {}).get(m,[]):
     ri=models.index(ref);dif=samples[mi,ki]-samples[ri,ki]
     result[m][key+'_vs_'+ref]={'difference':float(points[mi,ki]-points[ri,ki]),'ci95':np.quantile(dif,[.025,.975]).tolist()}
  output[c]=result
 return output,{'fold_brier_checks':checked,'max_abs_brier_error':maxerror}

def main():
 assert json.loads((OUT/'COMPLETED.json').read_text())['completed_jobs']==10
 audit=json.loads((OUT/'validation.json').read_text())
 assert audit['passed'] and audit['complete_experiment_validation']
 co=load_eac_cohorts();results={};checks=[];hashes={};selections=[];foldrows=[];curves=[];support=[]
 for seed in SEEDS:
  records=[]
  for fo in range(5):
   path=OUT/f'seed_{seed}/fold_{fo}/outer_predictions.json'
   done=json.loads((path.parent/'COMPLETED.json').read_text());assert digest(path)==done['prediction_sha256']
   r=json.loads(path.read_text());records.append(r);hashes[str(path.relative_to(WORKSPACE))]=digest(path)
   d=json.loads((path.parent/'selection_before_test.json').read_text())
   for name,s in d['selection'].items():
    if name=='global_selected':continue
    selections.append(dict(seed=seed,fold=fo,procedure=name,target='both',candidate=s['id'],mu=s['mu'],fractions=s['fractions']))
   for c,s in d['target_selection'].items():
    selections.append(dict(seed=seed,fold=fo,procedure='target_selected',target=c,candidate=s['id'],mu=s['mu'],fractions=s['fractions']))
   for c in CTS:
    for m in MODELS:
     v=r['cancers'][c]['models'][m];base=r['cancers'][c]['models']['single']
     foldrows.append(dict(seed=seed,fold=fo,cancer=c,model=m,c_index=v['test_c'],difference_vs_single=v['test_c']-base['test_c'],brier_1y=v['survival_metrics']['brier_365.25'],brier_3y=v['survival_metrics']['brier_1095.75']))
    for h,v in r['cancers'][c]['models']['single']['survival_metrics']['details'].items():
     support.append(dict(seed=seed,fold=fo,cancer=c,**v))
  results[str(seed)],check=aggregate(records,co,seed,MODELS,EXTRA);checks.append(check)
  for mu in MUS:
   rr=[]
   for r in records:
    b={'cancers':{}}
    for c in CTS:
     block={k:v for k,v in r['cancers'][c].items() if k!='models'}
     block['models']={'single':r['cancers'][c]['models']['single'],'clinical':r['cancers'][c]['models']['clinical'],'curve':r['strength_curves']['pair_EAC_STAD'][str(mu)]['cancers'][c]};b['cancers'][c]=block
    rr.append(b)
   vals,check=aggregate(rr,co,seed,['single','clinical','curve']);checks.append(check)
   for c in CTS:curves.append(dict(seed=seed,cancer=c,mu=mu,**vals[c]['curve']))
 fitrecords=[]
 for path in OUT.glob('seed_*/fold_*/*/*.json'):
  if not path.parent.name.startswith(('inner_','final_fits')) or not path.stem.startswith(('single_','pair_')):continue
  v=json.loads(path.read_text());fit=v.get('fit',v)
  assert fit['gradient_norm']<=1e-5
  fitrecords.append({'file':str(path.relative_to(OUT)),'phase':'inner' if path.parent.name.startswith('inner_') else 'outer','kind':'single' if path.stem.startswith('single_') else 'joint','gradient_norm':fit['gradient_norm'],'seconds':fit['total_seconds'],'n_iter':sum(a['n_iter'] for a in fit['attempts'])})
 counts=collections.Counter((v['phase'],v['kind']) for v in fitrecords)
 assert counts[('inner','single')]==240 and counts[('inner','joint')]==1200 and counts[('outer','single')]==80
 assert len(fitrecords)==audit['total_fits_with_objective_and_gradient_recomputed']
 clinical=[]
 for path in OUT.glob('seed_*/fold_*/clinical_fit_audit.json'):clinical+=json.loads(path.read_text())
 assert len(clinical)==300 and all(v['gradient_norm']<=1e-5 for v in clinical)
 stability={}
 for c in CTS:
  stability[c]={}
  for model,reference in [('pair_EAC_STAD','single'),('forced_pair_EAC_STAD','single'),('target_selected','single'),('clinical_plus_pair_EAC_STAD','clinical'),('clinical_plus_pair_EAC_STAD','clinical_plus_single')]:
   stability[c][model+'_vs_'+reference]={}
   for metric in ['c_index','brier_1y','brier_3y']:
    values=[results[str(s)][c][model][metric+'_vs_'+reference] for s in SEEDS]
    positive=metric=='c_index'
    stability[c][model+'_vs_'+reference][metric]={'differences':[v['difference'] for v in values],
     'better_point_estimate_in_both_repeats':all(v['difference']>0 if positive else v['difference']<0 for v in values),
     'ci_excludes_no_gain_in_both_repeats':all(v['ci95'][0]>0 if positive else v['ci95'][1]<0 for v in values)}
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
 diagnostics={'counts':{':'.join(k):v for k,v in counts.items()},'total':len(fitrecords),'strict':len(fitrecords),'maximum_raw_gradient':max(v['gradient_norm'] for v in fitrecords),'clinical_total':len(clinical),'clinical_strict':len(clinical)}
 result={'complete':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'protocol_sha256':digest(OUT/'protocol.json'),'summary_script_sha256':digest(__file__),
  'source_sha256':hashes,'results':results,'selection_records':selections,'fold_metrics':foldrows,
  'strength_curves':curves,'horizon_support':support,'stability':stability,
  'preflight':json.loads((OUT.parent/'preflight.json').read_text()),
  'fit_diagnostics':diagnostics,'validation':{'passed':True,'fold_brier_checks':sum(v['fold_brier_checks'] for v in checks),'max_brier_error':max(v['max_abs_brier_error'] for v in checks)},
  'uncertainty':'1000 paired within-test-fold patient bootstrap replicates with fixed predictions and training censoring estimates; RNG 20260917+partition_seed+100*task_index as in the original analysis. Repeats reuse patients, no full refitting uncertainty, multiplicity correction, equivalence test or external validation.',
  'scope':'Post-hoc histology-defined EAC/STAD extension. Original three-cancer results retained separately; no direct EAC-versus-mixed-ESCA effectiveness comparison.'}
 dump(OUT.parent/'eac_stad_facts.json',result);dump(OUT.parent/'fit_diagnostics.json',fitrecords)
 folder=OUT.parent/'tables';folder.mkdir(exist_ok=True)
 for name,items in [('all_metrics',rows),('fold_metrics',foldrows),('strength_curves',curves),('selection_records',selections),('horizon_support',support)]:
  keys=list(dict.fromkeys(k for row in items for k in row))
  with (folder/(name+'.csv')).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
   writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in items)
 print(json.dumps({'fit_diagnostics':diagnostics,'stability':stability},indent=2))
if __name__=='__main__':main()

