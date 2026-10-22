#!/usr/bin/env python3
"""Independently recompute held-out metrics and paired conditional uncertainty."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import numpy as np

CTS=['ESCA','STAD','COAD']
BASE=['single','pair_ESCA_STAD','pair_ESCA_COAD','pair_STAD_COAD','triple','global_selected','target_selected','clinical']
ACTIVE={'pair_ESCA_STAD':['ESCA','STAD'],'pair_ESCA_COAD':['ESCA','COAD'],'pair_STAD_COAD':['STAD','COAD']}

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

