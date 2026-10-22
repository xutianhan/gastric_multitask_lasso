#!/usr/bin/env python3
"""Reproducible nested evaluation; results are never imported from legacy runs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, argparse, hashlib, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from gastric_mtl.cox import fit_cox, fit_multitask, lambda_max, c_index
from gastric_mtl.metrics import evaluate_survival, predict_survival

from gastric_mtl.paths import ROOT, PATHWAYS, REFERENCE_INITIAL
OUT = REFERENCE_INITIAL
CTS = ['ESCA','STAD','COAD']
FRACTIONS = [1.0, 0.3, 0.1, 0.03]
RIDGES = [1000.0,100.0,10.0,1.0]
CLIN_RIDGES = [1.0,0.1,0.01,0.0]
FIT_LOG=[]

def checked_fit(task,group,penalty,lam,max_iter,beta0=None):
    fit=fit_cox(**task,groups=group,penalty=penalty,lam=lam,max_iter=max_iter,beta0=beta0,tol=1e-8,gtol=1e-5)
    if not fit['converged']:
        previous=metadata(fit)
        fit=fit_cox(**task,groups=group,penalty=penalty,lam=lam,max_iter=max_iter*2,beta0=fit['beta'],tol=1e-8,gtol=1e-5)
        fit['initial_attempt']=previous
    FIT_LOG.append({'method':penalty,'n':len(task['X']),'p':task['X'].shape[1],'lambda':float(lam),**metadata(fit)})
    return fit

def checked_multi(tasks,gl,lam,mu,beta0,max_iter):
    fit=fit_multitask(tasks,gl,lam,mu=mu,beta0=beta0,max_iter=max_iter)
    if not fit['converged']:
        previous=metadata(fit)
        fit=fit_multitask(tasks,gl,lam,mu=mu,beta0=fit['betas'],max_iter=max_iter*2)
        fit['initial_attempt']=previous
    FIT_LOG.append({'method':'multitask','n':[len(t['X']) for t in tasks],'p':len(fit['betas'][0]),'mu':mu,**metadata(fit)})
    return fit

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
    temp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def pathway_labels(genes):
    mapping={};names=[];raw_sizes=[]
    for filename in ['h.all.v2023.2.Hs.symbols.gmt','c2.cp.kegg_legacy.v2023.2.Hs.symbols.gmt']:
        for line in (PATHWAYS/filename).read_text().splitlines():
            fields=line.split('\t');name=fields[0];i=len(names);names.append(name)
            members=set(fields[2:]);raw_sizes.append(len(members))
            for gene in members:mapping.setdefault(gene,i)
    labels=np.array([mapping.get(str(g),len(names)) for g in genes])
    return labels,names+['Unassigned'],raw_sizes+[0]

def clinical_matrix(clin, train):
    """Fixed categories; only age imputation and scaling are learned."""
    age=np.asarray(clin['age'],dtype=float)
    median=float(np.nanmedian(age[train])) if np.isfinite(age[train]).any() else 0.
    age_missing=~np.isfinite(age);age=np.where(age_missing,median,age)
    age=(age-age[train].mean())/max(age[train].std(),1.)
    sex=np.asarray(clin['sex']).astype(str)
    stage=np.asarray(clin['stage'],dtype=float)
    return np.column_stack([age,age_missing.astype(float),(sex=='MALE').astype(float),
        (~np.isin(sex,['MALE','FEMALE'])).astype(float),
        *[(stage==s).astype(float) for s in [2.,3.,4.]],
        (~np.isin(stage,[1.,2.,3.,4.])).astype(float)])

def prepare(cohorts, tr, te, labels, include_background=False):
    """Fit feature filters and scaling using training patients in all tasks."""
    base=np.ones(len(labels),bool) if include_background else labels<labels.max()
    sd=[cohorts[c]['X_log2'][tr[k]].std(axis=0) for k,c in enumerate(CTS)]
    mask=base & np.logical_and.reduce([s>1e-8 for s in sd])
    idx=np.flatnonzero(mask)
    tasks=[];tests=[];scalers=[]
    for k,c in enumerate(CTS):
        raw=cohorts[c]['X_log2'][:,idx].astype(float)
        mean=raw[tr[k]].mean(axis=0);scale=raw[tr[k]].std(axis=0)
        Xtr=np.ascontiguousarray((raw[tr[k]]-mean)/scale)
        Xte=np.ascontiguousarray((raw[te[k]]-mean)/scale)
        tasks.append({'X':Xtr,'time':cohorts[c]['time'][tr[k]],'event':cohorts[c]['event'][tr[k]]})
        tests.append({'X':Xte,'time':cohorts[c]['time'][te[k]],'event':cohorts[c]['event'][te[k]]})
        scalers.append((mean,scale))
    return tasks,tests,labels[idx],idx,scalers

def metadata(fit):
    return {k:v for k,v in fit.items() if k not in ['beta','betas']}

def fit_path(task, group, penalty, grid, max_iter):
    scale=1. if penalty=='ridge' else lambda_max(task['X'],task['time'],task['event'],groups=group,penalty=penalty)
    fits=[];b=None
    for value in grid:
        fit=checked_fit(task,group,penalty,value*scale,max_iter,b)
        b=fit['beta'];fits.append(fit)
    return fits,scale

def counts(t,e,r):
    t=np.asarray(t);e=np.asarray(e,dtype=bool);r=np.asarray(r)
    valid=e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:]))
    diff=r[:,None]-r[None,:]
    return float(((diff>0)*valid).sum()+.5*((diff==0)*valid).sum()),int(valid.sum())

def aggregate(records, bootstrap=1000):
    """Compare only patients scored by the same outer-fold fitted model."""
    output={};rng=np.random.RandomState(20260914)
    for c in CTS:
        output[c]={}
        methods=list(records[0]['cancers'][c]['models'])
        for method in methods:
            cs=[];folds=[];total_a=total_b=0.
            for rec in records:
                block=rec['cancers'][c];risk=block['models'][method]['risk']
                a,b=counts(block['time'],block['event'],risk);total_a+=a;total_b+=b
                cs.append(a/b if b else .5);folds.append((np.array(block['time']),np.array(block['event']),np.array(risk)))
            draws=[]
            for _ in range(bootstrap):
                a=b=0.
                for t,e,r in folds:
                    ix=rng.randint(len(t),size=len(t));aa,bb=counts(t[ix],e[ix],r[ix]);a+=aa;b+=bb
                if b:draws.append(a/b)
            output[c][method]={'c_index':total_a/total_b,'fold_mean':float(np.mean(cs)),
                'fold_sd':float(np.std(cs,ddof=1)),'fold_c':cs,'ci95':np.quantile(draws,[.025,.975]).tolist(),
                'comparable_pairs':int(total_b),'ci_method':'1000 stratified-within-outer-fold patient bootstrap draws; fixed predictions, no refitting'}
        for other in ['group','clinical','ridge']:
            if other not in methods or 'multitask' not in methods:continue
            draws=[]
            for _ in range(bootstrap):
                numer=[0.,0.];denom=0.
                for rec in records:
                    block=rec['cancers'][c];t=np.array(block['time']);e=np.array(block['event'])
                    ix=rng.randint(len(t),size=len(t))
                    for j,m in enumerate(['multitask',other]):
                        a,b=counts(t[ix],e[ix],np.array(block['models'][m]['risk'])[ix]);numer[j]+=a
                    denom+=b
                if denom:draws.append((numer[0]-numer[1])/denom)
            output[c]['multitask_vs_'+other]={'difference':output[c]['multitask']['c_index']-output[c][other]['c_index'],
                'paired_ci95':np.quantile(draws,[.025,.975]).tolist()}
    return output

def run(args):
    from gastric_mtl.data import load_submission_cohorts
    cohorts=load_submission_cohorts()
    labels,names,raw_sizes=pathway_labels(cohorts[CTS[0]]['genes'])
    folds=[list(StratifiedKFold(args.outer,shuffle=True,random_state=args.seed+k).split(np.zeros(len(cohorts[c]['event'])),cohorts[c]['event'])) for k,c in enumerate(CTS)]
    base=OUT/'experiments'/args.name;base.mkdir(parents=True,exist_ok=True)
    protocol={'outer':args.outer,'inner':args.inner,'seed':args.seed,'fractions':FRACTIONS,'ridge_penalties':RIDGES,
        'clinical_penalties':CLIN_RIDGES,'mu':[0.,.1],'max_iter':args.max_iter,'feature_universe':'all_common_genes' if args.background else 'MSigDB_annotated_genes',
        'gene_filter':'training-only variance >1e-8 in all three tasks','scaling':'training-only mean and population SD per task',
        'optimizer':'L-BFGS-B','group_penalty':'smooth epsilon=1e-4; no hard zeroing','score_aggregation':'concordant/comparable pairs summed within held-out folds',
        'selection':'mean inner-fold C; ties choose stronger regularization; group tuning per cancer; coupling selected by macro inner C after per-cancer group tuning',
        'sources':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'src/gastric_mtl/paths.py',ROOT/'src/gastric_mtl/cox.py',ROOT/'src/gastric_mtl/data.py',ROOT/'src/gastric_mtl/metrics.py']},
        'cohort_hash':hashlib.sha256(json.dumps({c:{'ids':list(map(str,cohorts[c]['patient_ids'])),'time':cohorts[c]['time'].tolist(),'event':cohorts[c]['event'].tolist()} for c in CTS},sort_keys=True).encode()).hexdigest()}
    if (base/'protocol.json').exists() and json.loads((base/'protocol.json').read_text()) != protocol:
        raise RuntimeError('Protocol or source changed; use a new --name to avoid stale result reuse.')
    if not (base/'protocol.json').exists() and any(base.glob('fold_*.json')):
        raise RuntimeError('Existing reference records are read-only without their original protocol; use a fresh workspace.')
    dump(base/'protocol.json',protocol)
    for fo in range(args.outer):
        target=base/f'fold_{fo}.json'
        if target.exists():print('reusing',target,flush=True);continue
        tr=[f[fo][0] for f in folds];te=[f[fo][1] for f in folds]
        inners=[list(StratifiedKFold(args.inner,shuffle=True,random_state=args.seed+100+fo*7+k).split(np.zeros(len(tr[k])),cohorts[c]['event'][tr[k]])) for k,c in enumerate(CTS)]
        score={m:np.zeros((3,len(FRACTIONS),args.inner)) for m in ['group','elasticnet','ridge','combined_ridge','clinical']}
        inner_cache=[]
        print('outer',fo,'inner tuning',flush=True)
        for fi in range(args.inner):
            itr=[tr[k][inners[k][fi][0]] for k in range(3)];iva=[tr[k][inners[k][fi][1]] for k in range(3)]
            tasks,tests,gl,idx,_=prepare(cohorts,itr,iva,labels,args.background)
            cache={'itr':itr,'iva':iva,'group_fits':[],'scales':[]}
            for k,c in enumerate(CTS):
                cl=clinical_matrix(cohorts[c]['clinical'],itr[k])
                for method in score:
                    grid=CLIN_RIDGES if method=='clinical' else RIDGES if 'ridge' in method else FRACTIONS
                    penalty='ridge' if method in ['clinical','ridge','combined_ridge'] else method
                    task=tasks[k].copy();test=tests[k]['X'];group=gl
                    if method=='clinical':task['X']=cl[itr[k]];test=cl[iva[k]];group=None
                    if method=='combined_ridge':task['X']=np.column_stack([cl[itr[k]],task['X']]);test=np.column_stack([cl[iva[k]],test]);group=None
                    fits,scale=fit_path(task,group,penalty,grid,args.max_iter)
                    for vi,fit in enumerate(fits):score[method][k,vi,fi]=c_index(tests[k]['time'],tests[k]['event'],test@fit['beta'])
                    if method=='group':cache['group_fits'].append(fits);cache['scales'].append(scale)
            inner_cache.append(cache)
            print('outer',fo,'inner',fi,'complete; p',len(idx),flush=True)
        chosen={m:score[m].mean(axis=2).argmax(axis=1) for m in score}
        mu_scores=np.zeros((2,args.inner))
        for fi,cache in enumerate(inner_cache):
            tasks,tests,gl,idx,_=prepare(cohorts,cache['itr'],cache['iva'],labels,args.background)
            lam=[cache['scales'][k]*FRACTIONS[chosen['group'][k]] for k in range(3)]
            b=[cache['group_fits'][k][chosen['group'][k]]['beta'] for k in range(3)]
            mu_scores[0,fi]=np.mean([c_index(tests[k]['time'],tests[k]['event'],tests[k]['X']@b[k]) for k in range(3)])
            fit=checked_multi(tasks,gl,lam,.1,b,args.max_iter)
            mu_scores[1,fi]=np.mean([c_index(tests[k]['time'],tests[k]['event'],tests[k]['X']@fit['betas'][k]) for k in range(3)])
        mu=[0.,.1][int(mu_scores.mean(axis=1).argmax())]
        tasks,tests,gl,idx,scalers=prepare(cohorts,tr,te,labels,args.background)
        record={'fold':fo,'feature_count':len(idx),'selected_mu':mu,'mu_inner_macro_c':mu_scores.tolist(),'cancers':{}}
        betas=[];lam=[];stored={};fit_metadata=[]
        for k,c in enumerate(CTS):
            stored[c+'_mean']=scalers[k][0];stored[c+'_scale']=scalers[k][1]
            age=np.asarray(cohorts[c]['clinical']['age'],dtype=float)
            med=float(np.nanmedian(age[tr[k]])) if np.isfinite(age[tr[k]]).any() else 0.
            age=np.where(np.isfinite(age),age,med)
            stored[c+'_clinical_age_params']=np.array([med,age[tr[k]].mean(),max(age[tr[k]].std(),1.)])
            record['cancers'][c]={'patient_ids':np.asarray(cohorts[c]['patient_ids'])[te[k]].tolist(),
                'train_patient_ids':np.asarray(cohorts[c]['patient_ids'])[tr[k]].tolist(),
                'time':cohorts[c]['time'][te[k]].tolist(),'event':cohorts[c]['event'][te[k]].astype(int).tolist(),'models':{}}
            cl=clinical_matrix(cohorts[c]['clinical'],tr[k])
            for method in score:
                grid=CLIN_RIDGES if method=='clinical' else RIDGES if 'ridge' in method else FRACTIONS
                penalty='ridge' if method in ['clinical','ridge','combined_ridge'] else method
                task=tasks[k].copy();test=tests[k]['X'];group=gl
                if method=='clinical':task['X']=cl[tr[k]];test=cl[te[k]];group=None
                if method=='combined_ridge':task['X']=np.column_stack([cl[tr[k]],task['X']]);test=np.column_stack([cl[te[k]],test]);group=None
                vi=int(chosen[method][k]);fits,scale=fit_path(task,group,penalty,grid[:vi+1],args.max_iter);fit=fits[-1]
                record['cancers'][c]['models'][method]={'risk':(test@fit['beta']).tolist(),
                    'train_c':c_index(task['time'],task['event'],task['X']@fit['beta']),
                    'test_c':c_index(tests[k]['time'],tests[k]['event'],test@fit['beta']),
                    'parameter':float(grid[vi]*scale),'grid_value':float(grid[vi]),'inner_scores':score[method][k].tolist(),**metadata(fit)}
                record['cancers'][c]['models'][method]['survival_metrics']=evaluate_survival(task['time'],task['event'],task['X']@fit['beta'],tests[k]['time'],tests[k]['event'],test@fit['beta'])
                record['cancers'][c]['models'][method]['survival_1y_3y']=predict_survival(task['time'],task['event'],task['X']@fit['beta'],test@fit['beta'],[365.25,1095.75]).tolist()
                stored[c+'_'+method]=fit['beta']
                if method=='group':betas.append(fit['beta']);lam.append(scale*grid[vi])
        mt=checked_multi(tasks,gl,lam,mu,betas,args.max_iter) if mu else {'betas':betas,'converged':all(record['cancers'][c]['models']['group']['converged'] for c in CTS),'n_iter':0,'seconds':0.,'message':'mu=0: exact reuse of separate fits; see group objective and gradient'}
        for k,c in enumerate(CTS):
            b=mt['betas'][k];risk=tests[k]['X']@b
            record['cancers'][c]['models']['multitask']={'risk':risk.tolist(),'test_c':c_index(tests[k]['time'],tests[k]['event'],risk),
                'train_c':c_index(tasks[k]['time'],tasks[k]['event'],tasks[k]['X']@b),'mu':mu,'lambda':lam[k],**metadata(mt)}
            record['cancers'][c]['models']['multitask']['survival_metrics']=evaluate_survival(tasks[k]['time'],tasks[k]['event'],tasks[k]['X']@b,tests[k]['time'],tests[k]['event'],risk)
            record['cancers'][c]['models']['multitask']['survival_1y_3y']=predict_survival(tasks[k]['time'],tasks[k]['event'],tasks[k]['X']@b,risk,[365.25,1095.75]).tolist()
            stored[c+'_multitask']=b
        np.savez_compressed(base/f'fold_{fo}_coefficients.npz',feature_indices=idx,groups=gl,**stored)
        dump(target,record)
        dump(base/f'fold_{fo}_fit_diagnostics.json',FIT_LOG)
        FIT_LOG.clear()
        print('outer',fo,'mu',mu,[(c,[(m,round(v['test_c'],3)) for m,v in record['cancers'][c]['models'].items()]) for c in CTS],flush=True)
    records=[json.loads((base/f'fold_{i}.json').read_text()) for i in range(args.outer)]
    dump(base/'summary.json',aggregate(records))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--outer',type=int,default=5);p.add_argument('--inner',type=int,default=3)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--max-iter',type=int,default=2000)
    p.add_argument('--name',default='nested_primary');p.add_argument('--background',action='store_true');run(p.parse_args())
