#!/usr/bin/env python3
"""Frozen nested comparison of uncoupled, pairwise and three-task Cox learning.

Each outer fold is an independent, resumable job. All model/graph selection is
completed and written before outer-test predictions are evaluated. The original
submission analysis and its source modules are read-only dependencies.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import sys, json, argparse, hashlib, itertools, pickle, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from sklearn.model_selection import StratifiedKFold

from gastric_mtl.paths import ROOT, COHORT_CACHE, PATHWAYS, REFERENCE_INITIAL, THREE_CANCER

from gastric_mtl.cox import fit_cox, fit_multitask, lambda_max, c_index
from gastric_mtl.data import load_submission_cohorts
from gastric_mtl.solver import fit_stable_newton
from gastric_mtl.warmup import warmup
from gastric_mtl.metrics import predict_survival, evaluate_survival
from workflows.experiments.initial_evaluation import pathway_labels, prepare, dump, digest, metadata
from workflows.experiments.clinical_three_cancer import train as train_clinical

OUT=THREE_CANCER/'runs'
OLD=REFERENCE_INITIAL
CTS=['ESCA','STAD','COAD']
SEEDS=[42,2026]
FRACTIONS=[1.,.3,.1,.03]
MUS=[.0001,.001,.01,.1,1.]
GRAPHS={'pair_ESCA_STAD':[0,1],'pair_ESCA_COAD':[0,2],
        'pair_STAD_COAD':[1,2],'triple':[0,1,2]}

def candidates():
    """40 candidates per graph: 20 equal fractions plus 20 fixed mixed tuples.

    The mixed tuples are sampled without replacement within each strength from
    the remaining Cartesian grid, before any outcome analysis. This is a finite
    joint search, not an exhaustive search or a claimed universal optimum.
    """
    result={}
    for gi,(graph,active) in enumerate(GRAPHS.items()):
        k=len(active); same=[(j,)*k for j in range(4)]
        mixed=[v for v in itertools.product(range(4),repeat=k) if len(set(v))>1]
        rng=np.random.RandomState(190731+gi)
        result[graph]=[]
        for mi,mu in enumerate(MUS):
            choices=same+[mixed[i] for i in rng.choice(len(mixed),size=4,replace=False)]
            for vi,choice in enumerate(choices):
                result[graph].append({'id':f'{graph}_m{mi}_v{vi}','graph':graph,
                    'active':active,'mu':mu,'edge_weight':mu/(k-1),
                    'fraction_indices':list(choice),
                    'fractions':[FRACTIONS[j] for j in choice]})
    return result

def protocol():
    sources=[Path(__file__),ROOT/'src/gastric_mtl/paths.py',ROOT/'workflows/experiments/initial_evaluation.py',
             ROOT/'src/gastric_mtl/cox.py',ROOT/'src/gastric_mtl/data.py',ROOT/'src/gastric_mtl/metrics.py', ROOT/'src/gastric_mtl/second_order.py', ROOT/'src/gastric_mtl/solver.py',ROOT/'src/gastric_mtl/warmup.py',ROOT/'workflows/experiments/clinical_three_cancer.py']
    return {'version':1,'seeds':SEEDS,'outer_folds':5,'inner_folds':3,
        'cancers':CTS,'fractions':FRACTIONS,'positive_mu':MUS,'candidates':candidates(),
        'zero_coupling':'Exact reuse of independently inner-selected single-task fits',
        'edge_normalization':'mu/(K-1) per edge within each complete active component; singleton has no edges',
        'cohort_cache_sha256':digest(COHORT_CACHE),
        'sources':{str(p.relative_to(ROOT)):digest(p) for p in sources},
        'gmt_sha256':{p.name:digest(p) for p in (PATHWAYS).glob('*.gmt')},
        'preprocessing':'Same training-derived feature universe and per-cancer scaling as original nested analysis; original patient partitions retained',
        'solver':{'method':'Task/group-conditioned trust-krylov with analytic Hessian products; raw-gradient early stop; reject nonfinite proposals and resume trust-ncg from best finite training checkpoint; stable risk-set Hessian fallback',
                  'initial_maxiter':600,'retry_maxiter':1200,'raw_gradient_tolerance':1e-5,
                  'warmup':'For joint candidates with any regularisation fraction 0.03, first apply up to 2000 task/group-conditioned L-BFGS-B iterations, in 100-iteration segments, ftol=1e-15, transformed gtol=1e-10, maxcor=20, maxls=40. Then apply the same strict Newton refinement. Other candidates use Newton directly. This numerical speed policy was chosen using a training-only benchmark before any new outer predictions; the partial Newton-only pilot is excluded and every primary fit is restarted.',
                  'acceptance':'Every candidate must satisfy finite objective and raw gradient infinity norm <= 1e-5 before scoring. A residual stops the fold; no candidate is discarded or replaced.',
                  'initialization':'Single-task regularization path from zero; each joint candidate starts from its corresponding converged single-task fits. No historical coefficients or scores reused.'},
        'selection':'Mean inner concordance; graph candidates scored by mean across all three cancers, with unpaired target using its independently selected single-task fit. Ties prefer no sharing, then fewer edges, weaker mu, stronger regularization.',
        'target_selection':'For each target, maximize its own mean inner concordance over single and all candidates whose active component contains the target. This is a target-specific prediction procedure, not one coherent joint fit.',
        'stability':'Report the two repeats separately and fold-wise paired changes. Both repeats reuse patients; conditional patient-bootstrap intervals do not include full refitting uncertainty.',
        'clinical':'Re-tune clinical Cox over ridge [1,.1,.01,0] on the same inner folds. Assess clinical plus a single molecular score from each of single, ESCA-STAD pair and triple procedures. Outer-training scores use inner-held-out coefficients at the inner-selected candidate; candidate selection itself uses all inner outcomes (not fully nested cross-fitting). Standardize the score on outer-training patients and reuse the clinical-selected ridge; all evaluation remains on untouched outer-test patients. No outcome-based score recalibration or extra score-model tuning.',
        'scope':'Full post-hoc rerun on 2026-09-18 of the original two-repeat nested search, using unchanged data, splits, candidate grid and objective with stricter numerical convergence. Internal validation only; prior results have been seen, and this is not prospective or external confirmation.'}

def initialize():
    OUT.mkdir(parents=True,exist_ok=True);p=protocol();path=OUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=p:raise RuntimeError('Frozen protocol/source mismatch; do not mix runs.')
    else:
        if any(f.is_file() for f in OUT.rglob('*')):
            raise RuntimeError('Existing result files have no matching protocol. Use a fresh results workspace; do not mix runs.')
        dump(path,p)
    return p

def solve(tasks,groups,fractions,scales,beta0,mu=0.,outer=False):
    lam=[float(f*s) for f,s in zip(fractions,scales)]
    edge=mu/(len(tasks)-1) if len(tasks)>1 else 0.
    attempts=[];start=beta0
    if len(tasks)>1 and min(fractions)<=.03:
        pre=warmup(tasks,groups,lam,edge,start)
        start=pre.pop('betas');attempts.append(pre)
    for budget in [600,1200]:
        fit=fit_stable_newton(tasks,groups,lam,edge,start,max_iter=budget,gtol=1e-5)
        attempts.append({k:v for k,v in fit.items() if k!='betas'})
        start=fit['betas']
        if fit['gradient_norm']<=1e-5:break
    info={k:v for k,v in fit.items() if k!='betas'}
    info.update(attempts=attempts,total_seconds=sum(a['seconds'] for a in attempts),
                raw_gradient_below_1e_5=bool(fit['gradient_norm']<=1e-5),
                **{'lambda':lam,'fractions':fractions,'mu':mu,'edge_weight':edge})
    if not info['raw_gradient_below_1e_5']:
        raise RuntimeError('Strict convergence gate failed: '+json.dumps({k:info[k] for k in ['gradient_norm','objective','lambda','mu']}))
    print(f"fit tasks={len(tasks)} mu={mu:g} gradient={fit['gradient_norm']:.3g} seconds={info['total_seconds']:.1f}",flush=True)
    return start,info

def cached_single(folder,tasks,groups,scales,outer=False):
    betas=[];infos=[]
    for k,c in enumerate(CTS):
        allb=[];alli=[];warm=np.zeros(tasks[k]['X'].shape[1])
        for vi,f in enumerate(FRACTIONS):
            stem=folder/f'single_{c}_v{vi}';jp=stem.with_suffix('.json');npz=stem.with_suffix('.npz')
            if jp.exists() and npz.exists():
                info=json.loads(jp.read_text());b=np.load(npz)['beta']
            else:
                bs,info=solve([tasks[k]],groups,[f],[scales[k]],[warm],outer=outer);b=bs[0]
                np.savez_compressed(npz,beta=b);dump(jp,info)
            warm=b;allb.append(b);alli.append(info)
        betas.append(allb);infos.append(alli)
    return betas,infos

def candidate_rank(spec):
    # Used only after a numerical score tie, preserving the stronger penalty.
    return (len(spec['active'])*(len(spec['active'])-1)//2,spec['mu'],
            tuple(-v for v in spec['fractions']),spec['id'])

def pick(options,score):
    """A stable rule; candidates contain no outer-test outcomes or scores."""
    return min(options,key=lambda x:(-float(score(x)),candidate_rank(x)))

def outer_job(seed,fo,inner_only=None):
    started=time.monotonic();base=OUT/f'seed_{seed}/fold_{fo}';base.mkdir(parents=True,exist_ok=True)
    if (base/'COMPLETED.json').exists():return {'seed':seed,'fold':fo,'reused':True}
    p=json.loads((OUT/'protocol.json').read_text())
    for path,expected in p['sources'].items():
        if digest(ROOT/path)!=expected:raise RuntimeError('Source changed during experiment: '+path)
    if digest(COHORT_CACHE)!=p['cohort_cache_sha256']:raise RuntimeError('Cache changed')
    cohorts=load_submission_cohorts();labels,names,_=pathway_labels(cohorts[CTS[0]]['genes'])
    folds=[list(StratifiedKFold(5,shuffle=True,random_state=seed+k).split(np.zeros(len(cohorts[c]['event'])),cohorts[c]['event'])) for k,c in enumerate(CTS)]
    tr=[v[fo][0] for v in folds];te=[v[fo][1] for v in folds]
    oldname='nested_primary' if seed==42 else 'nested_repeat_seed2026'
    old=json.loads((OLD/'experiments'/oldname/f'fold_{fo}.json').read_text())
    for k,c in enumerate(CTS):
        ids=np.asarray(cohorts[c]['patient_ids']).astype(str)
        assert ids[te[k]].tolist()==old['cancers'][c]['patient_ids']
        assert not set(ids[tr[k]])&set(ids[te[k]])
    inners=[list(StratifiedKFold(3,shuffle=True,random_state=seed+100+fo*7+k).split(np.zeros(len(tr[k])),cohorts[c]['event'][tr[k]])) for k,c in enumerate(CTS)]
    specs=[s for vals in p['candidates'].values() for s in vals]
    singles=np.zeros((3,4,3));joint={s['id']:np.full((3,3),np.nan) for s in specs}
    split_audit=[]
    for fi in range(3):
        if inner_only is not None and fi!=inner_only:continue
        folder=base/f'inner_{fi}';folder.mkdir(exist_ok=True)
        itr=[tr[k][inners[k][fi][0]] for k in range(3)]
        iva=[tr[k][inners[k][fi][1]] for k in range(3)]
        tasks,tests,gl,idx,_=prepare(cohorts,itr,iva,labels)
        scales=[lambda_max(**t,groups=gl,penalty='group') for t in tasks]
        bs,infos=cached_single(folder,tasks,gl,scales)
        for k in range(3):
            for vi in range(4):singles[k,vi,fi]=c_index(tests[k]['time'],tests[k]['event'],tests[k]['X']@bs[k][vi])
        for si,s in enumerate(specs):
            path=folder/(s['id']+'.json')
            if path.exists():r=json.loads(path.read_text())
            else:
                active=s['active'];initial=[bs[k][vi] for k,vi in zip(active,s['fraction_indices'])]
                b,info=solve([tasks[k] for k in active],gl,s['fractions'],[scales[k] for k in active],initial,s['mu'])
                risks=[(tests[k]['X']@bb).tolist() for k,bb in zip(active,b)]
                scores=[c_index(tests[k]['time'],tests[k]['event'],risk) for k,risk in zip(active,risks)]
                r={'spec':s,'scores':scores,'risks':risks,'fit':info};np.savez_compressed(path.with_suffix('.npz'),**{f'beta_{k}':bb for k,bb in zip(active,b)});dump(path,r)
            for k,val in zip(s['active'],r['scores']):joint[s['id']][k,fi]=val
            if (si+1)%8==0:print(f'seed {seed} outer {fo} inner {fi}: {si+1}/{len(specs)} candidates',flush=True)
        audit={c:{'train_ids':np.asarray(cohorts[c]['patient_ids'])[itr[k]].tolist(),
                  'validation_ids':np.asarray(cohorts[c]['patient_ids'])[iva[k]].tolist()} for k,c in enumerate(CTS)}
        split_audit.append({'inner_fold':fi,'feature_count':len(idx),'feature_indices_sha256':hashlib.sha256(idx.tobytes()).hexdigest(),'cancers':audit})
        if inner_only is not None:
            result={'seed':seed,'fold':fo,'inner_fold':fi,'candidates':len(specs),'phase':'inner complete'}
            dump(folder/'INNER_COMPLETED.json',result)
            return result
        dump(base/'progress.json',{'seed':seed,'fold':fo,'completed_inner_folds':fi+1,'phase':'inner selection'})
    chosen=singles.mean(axis=2).argmax(axis=1)
    single_scores=np.array([singles[k,chosen[k]] for k in range(3)])
    baseline={'id':'single','graph':'single','active':[],'mu':0.,'edge_weight':0.,
              'fraction_indices':chosen.tolist(),'fractions':[FRACTIONS[v] for v in chosen]}
    mean_scores={'single':single_scores.mean(axis=1)}
    for s in specs:
        matrix=joint[s['id']].copy()
        for k in set(range(3))-set(s['active']):matrix[k]=single_scores[k]
        joint[s['id']]=matrix;mean_scores[s['id']]=matrix.mean(axis=1)
    alloptions=[baseline]+specs
    macro=lambda s:mean_scores[s['id']].mean()
    selection={'single':baseline}
    for graph,values in p['candidates'].items():
        selection[graph]=pick([baseline]+values,macro)
        selection['forced_'+graph]=pick(values,macro)
    selection['global_selected']=pick(alloptions,macro)
    targets={c:pick([baseline]+[s for s in specs if k in s['active']],lambda s:mean_scores[s['id']][k]) for k,c in enumerate(CTS)}
    curves={g:{str(mu):pick([s for s in vals if s['mu']==mu],macro) for mu in MUS} for g,vals in p['candidates'].items()}
    # This file is written before any outer-test risk calculation.
    decision={'seed':seed,'fold':fo,'selected_single_fractions':baseline['fractions'],
        'single_inner_scores':singles.tolist(),'candidate_inner_scores':{k:v.tolist() for k,v in joint.items()},
        'candidate_mean_inner_scores':{k:v.tolist() for k,v in mean_scores.items()},
        'selection':selection,'target_selection':targets,'strength_curve_selection':curves,'inner_partitions':split_audit}
    dump(base/'selection_before_test.json',decision)
    tasks,tests,gl,idx,scalers=prepare(cohorts,tr,te,labels)
    scales=[lambda_max(**t,groups=gl,penalty='group') for t in tasks]
    finaldir=base/'final_fits';finaldir.mkdir(exist_ok=True)
    allb,allinfo=cached_single(finaldir,tasks,gl,scales,outer=True)
    baseline_b=[allb[k][v] for k,v in enumerate(chosen)]
    required={s['id']:s for s in list(selection.values())+list(targets.values())+[s for vals in curves.values() for s in vals.values()]}
    fit_b={'single':baseline_b};fit_info={'single':[allinfo[k][v] for k,v in enumerate(chosen)]}
    for name,s in required.items():
        if name=='single':continue
        jp=finaldir/(name+'.json');npz=finaldir/(name+'.npz');active=s['active']
        if jp.exists() and npz.exists():
            info=json.loads(jp.read_text());z=np.load(npz);b=[z[f'beta_{k}'] for k in active]
        else:
            initial=[allb[k][v] for k,v in zip(active,s['fraction_indices'])]
            b,info=solve([tasks[k] for k in active],gl,s['fractions'],[scales[k] for k in active],initial,s['mu'],outer=True)
            np.savez_compressed(npz,**{f'beta_{k}':bb for k,bb in zip(active,b)});dump(jp,info)
        values=list(baseline_b)
        for k,bb in zip(active,b):values[k]=bb
        fit_b[name]=values;fit_info[name]=info
    clinical_models=train_clinical(cohorts,tr,te,inners,tasks,tests,labels,selection,fit_b,chosen,base)
    record={'seed':seed,'fold':fo,'feature_count':len(idx),'selection':selection,'target_selection':targets,'cancers':{},'strength_curves':{}}
    def measure(k,b):
        risk=tests[k]['X']@b;trainrisk=tasks[k]['X']@b
        return {'risk':risk.tolist(),'test_c':c_index(tests[k]['time'],tests[k]['event'],risk),
                'train_c':c_index(tasks[k]['time'],tasks[k]['event'],trainrisk),
                'survival_1y_3y':predict_survival(tasks[k]['time'],tasks[k]['event'],trainrisk,risk,[365.25,1095.75]).tolist(),
                'survival_metrics':evaluate_survival(tasks[k]['time'],tasks[k]['event'],trainrisk,tests[k]['time'],tests[k]['event'],risk)}
    for k,c in enumerate(CTS):
        models={name:{'candidate_id':s['id'],**measure(k,fit_b[s['id']][k])} for name,s in selection.items()}
        s=targets[c];models['target_selected']={'candidate_id':s['id'],**measure(k,fit_b[s['id']][k])}
        models.update(clinical_models[c])
        record['cancers'][c]={'patient_ids':np.asarray(cohorts[c]['patient_ids'])[te[k]].tolist(),
            'train_patient_ids':np.asarray(cohorts[c]['patient_ids'])[tr[k]].tolist(),
            'time':tests[k]['time'].tolist(),'event':tests[k]['event'].astype(int).tolist(),'models':models}
    for graph,vals in curves.items():
        record['strength_curves'][graph]={mu:{'spec':s,'cancers':{c:measure(k,fit_b[s['id']][k]) for k,c in enumerate(CTS)}} for mu,s in vals.items()}
    np.savez_compressed(base/'preprocessing.npz',feature_indices=idx,groups=gl,**{c+'_mean':scalers[k][0] for k,c in enumerate(CTS)},**{c+'_scale':scalers[k][1] for k,c in enumerate(CTS)})
    dump(base/'outer_predictions.json',record)
    dump(base/'COMPLETED.json',{'seed':seed,'fold':fo,'seconds':time.monotonic()-started,
        'selection_sha256':digest(base/'selection_before_test.json'),'prediction_sha256':digest(base/'outer_predictions.json'),
        'protocol_sha256':digest(OUT/'protocol.json'),'final_fit_ids':list(required)})
    print(f'COMPLETE seed {seed} outer {fo}',flush=True)
    return {'seed':seed,'fold':fo,'seconds':time.monotonic()-started}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--init-only',action='store_true')
    ap.add_argument('--seed',type=int);ap.add_argument('--fold',type=int)
    args=ap.parse_args();initialize()
    if args.init_only:print('Protocol frozen',OUT);return
    jobs=[(s,f) for s in SEEDS for f in range(5) if (args.seed is None or s==args.seed) and (args.fold is None or f==args.fold)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        inner_futures={pool.submit(outer_job,s,f,fi):(s,f,fi) for s,f in jobs for fi in range(3)}
        for future in as_completed(inner_futures):print(json.dumps(future.result()),flush=True)
        futures={pool.submit(outer_job,s,f):(s,f) for s,f in jobs}
        for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
    if len(jobs)==10:dump(OUT/'COMPLETED.json',{'completed_jobs':10,'protocol_sha256':digest(OUT/'protocol.json')})

if __name__=='__main__':main()
