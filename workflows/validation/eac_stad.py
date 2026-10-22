#!/usr/bin/env python3
"""Recompute held-out scores, predictions and selection from saved evidence."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,json,collections,hashlib,argparse
from pathlib import Path
import numpy as np
from gastric_mtl.paths import COHORT_CACHE, PATHWAYS, THREE_CANCER, verify_recorded_source
from gastric_mtl.eac import load_eac_cohorts as load_submission_cohorts, prepare
from gastric_mtl.cox import CoxObjective,_group_info,penalty_value_grad,coupling_value_grad
from gastric_mtl.metrics import predict_survival,evaluate_survival
from workflows.experiments.initial_evaluation import pathway_labels,dump,digest
from workflows.experiments.eac_stad import OUT,OLD,CTS,SEEDS,GRAPHS,FRACTIONS,MUS
from workflows.experiments.initial_evaluation import clinical_matrix
from workflows.experiments.clinical_eac import fit_clinical, RIDGES

def concordance(t,e,r):
    # Independent explicit patient-pair loop; excludes event/event time ties.
    a=b=0.
    for i in range(len(t)):
        if not e[i]:continue
        for j in range(len(t)):
            if t[i]<t[j] or (t[i]==t[j] and not e[j]):
                b+=1;a+=1. if r[i]>r[j] else .5 if r[i]==r[j] else 0.
    assert b>0
    return a/b

def rank(s):return (len(s['active'])*(len(s['active'])-1)//2,s['mu'],tuple(-v for v in s['fractions']),s['id'])
def choose(options,scores,k=None):
    return sorted(options,key=lambda s:(-float(np.mean(scores[s['id']]) if k is None else scores[s['id']][k]),rank(s)))[0]

FIT_CHECKS=[]
def check_fit(tasks,gl,active,bs,info,path):
    labels,sizes=_group_info(gl,len(bs[0]));val=0.;grad=[]
    for ai,k in enumerate(active):
        a,g=CoxObjective(**tasks[k]).value_grad(bs[ai]);pv,pg=penalty_value_grad(bs[ai],labels,sizes)
        val+=a+info['lambda'][ai]*pv;grad.append(g+info['lambda'][ai]*pg)
    for ai in range(len(active)):
        for aj in range(ai+1,len(active)):
            a,ga,gb=coupling_value_grad(bs[ai],bs[aj],labels,sizes);val+=info['edge_weight']*a
            grad[ai]+=info['edge_weight']*ga;grad[aj]+=info['edge_weight']*gb
    gn=max(float(np.max(np.abs(g))) for g in grad)
    assert np.isfinite(val) and abs(val-info['objective'])<1e-8
    assert abs(gn-info['gradient_norm'])<1e-8 and gn<=1e-5,(path,gn)
    FIT_CHECKS.append({'path':str(path.relative_to(OUT)),'gradient_norm':gn,'objective':val,'coefficient_sha256':digest(path.with_suffix('.npz'))})


def main():
    validator_sha256=digest(__file__)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-only',action='store_true',help='Audit completed outer folds while training; write only partial audit files')
    args=parser.parse_args()
    if not args.completed_only:assert json.loads((OUT/'COMPLETED.json').read_text())['completed_jobs']==10
    protocol=json.loads((OUT/'protocol.json').read_text());cohorts=load_submission_cohorts()
    manifests=json.loads((OUT.parent/'split_manifest.json').read_text())
    assert digest(COHORT_CACHE)==protocol['cohort_cache_sha256']
    assert json.loads((THREE_CANCER/'solver_validation.json').read_text())['passed']
    assert json.loads((OUT.parent/'preflight.json').read_text())['passed']
    source_modes = {p:verify_recorded_source(p,sha) for p,sha in protocol['sources'].items()}
    for name,sha in protocol['gmt_sha256'].items():assert digest(PATHWAYS/name)==sha
    labels,_,_=pathway_labels(cohorts[CTS[0]]['genes']);specs=[s for v in protocol['candidates'].values() for s in v]
    inner_checks=outer_checks=selection_checks=curve_checks=objective_checks=0
    max_risk_error=max_survival_error=max_objective_error=max_gradient_error=0.;partition_counts={}
    completed_folds=[]
    for seed in SEEDS:
        seen={c:[] for c in CTS}
        for fo in range(5):
            if args.completed_only and not (OUT/f'seed_{seed}/fold_{fo}/COMPLETED.json').exists():continue
            base=OUT/f'seed_{seed}/fold_{fo}';done=json.loads((base/'COMPLETED.json').read_text())
            d=json.loads((base/'selection_before_test.json').read_text());r=json.loads((base/'outer_predictions.json').read_text())
            original=json.loads((THREE_CANCER/'runs'/f'seed_{seed}/fold_{fo}/outer_predictions.json').read_text())
            # The clinical reference does not use the changed expression mask.
            # Identical STAD patients, splits and coding must reproduce it.
            for key in ['risk','survival_1y_3y','test_c']:
                np.testing.assert_allclose(r['cancers']['STAD']['models']['clinical'][key],original['cancers']['STAD']['models']['clinical'][key],rtol=0,atol=1e-10)
            assert r['selection']==d['selection'] and r['target_selection']==d['target_selection']
            assert digest(base/'selection_before_test.json')==done['selection_sha256']
            assert digest(base/'outer_predictions.json')==done['prediction_sha256']
            assert done['protocol_sha256']==digest(OUT/'protocol.json')
            assert (base/'selection_before_test.json').stat().st_mtime_ns<=(base/'outer_predictions.json').stat().st_mtime_ns
            tr=[];te=[]
            manifest=next(m for m in manifests if m['seed']==seed and m['fold']==fo and m['phase']=='outer')
            assert d['selection']['global_selected']==d['selection']['pair_EAC_STAD']
            for c in CTS:
                assert r['cancers'][c]['models']['global_selected']==r['cancers'][c]['models']['pair_EAC_STAD']
                assert r['cancers'][c]['train_patient_ids']==manifest['cancers'][c]['train_ids']
                assert r['cancers'][c]['patient_ids']==manifest['cancers'][c]['test_ids']
                ix={str(v):i for i,v in enumerate(cohorts[c]['patient_ids'])};block=r['cancers'][c]
                a=[ix[v] for v in block['train_patient_ids']];b=[ix[v] for v in block['patient_ids']]
                assert len(set(a))==len(a) and len(set(b))==len(b) and not set(a)&set(b)
                assert set(a)|set(b)==set(range(len(ix)));tr.append(np.array(a));te.append(np.array(b))
                seen[c].extend(block['patient_ids'])
                np.testing.assert_array_equal(block['time'],cohorts[c]['time'][b]);np.testing.assert_array_equal(block['event'],cohorts[c]['event'][b])
            singles=np.zeros((len(CTS),4,3));cs={s['id']:np.full((len(CTS),3),np.nan) for s in specs}
            assert len(d['inner_partitions'])==3
            inner_seen={c:[] for c in CTS}
            clinical_scores=np.zeros((len(CTS),4,3))
            for fi,split in enumerate(d['inner_partitions']):
                assert split['inner_fold']==fi
                itr=[];iva=[]
                for k,c in enumerate(CTS):
                    ix={str(v):i for i,v in enumerate(cohorts[c]['patient_ids'])};b=split['cancers'][c]
                    a=np.array([ix[v] for v in b['train_ids']]);v=np.array([ix[v] for v in b['validation_ids']])
                    assert not set(a)&set(v);assert set(a)|set(v)==set(tr[k]);assert not (set(a)|set(v))&set(te[k])
                    inner_seen[c].extend(v.tolist())
                    itr.append(a);iva.append(v)
                tasks,tests,gl,idx,_=prepare(cohorts,itr,iva,labels)
                for k,c in enumerate(CTS):
                    cl=clinical_matrix(cohorts[c]['clinical'],itr[k])
                    for vi,lam in enumerate(RIDGES):
                        cf=fit_clinical(cl[itr[k]],tasks[k]['time'],tasks[k]['event'],lam)
                        cv,cg=CoxObjective(cl[itr[k]],tasks[k]['time'],tasks[k]['event']).value_grad(cf['beta'])
                        assert np.max(np.abs(cg+lam*cf['beta']))<=1e-5
                        clinical_scores[k,vi,fi]=concordance(tests[k]['time'],tests[k]['event'],cl[iva[k]]@cf['beta'])
                assert split['feature_count']==len(idx)
                assert split['feature_indices_sha256']==hashlib.sha256(idx.tobytes()).hexdigest()
                for k,c in enumerate(CTS):
                    for vi in range(4):
                        beta=np.load(base/f'inner_{fi}/single_{c}_v{vi}.npz')['beta']
                        singles[k,vi,fi]=concordance(tests[k]['time'],tests[k]['event'],tests[k]['X']@beta)
                        inner_checks+=1
                        sp=base/f'inner_{fi}/single_{c}_v{vi}.json'
                        check_fit(tasks,gl,[k],[beta],json.loads(sp.read_text()),sp)
                for s in specs:
                    saved=json.loads((base/f'inner_{fi}'/(s['id']+'.json')).read_text());assert saved['spec']==s
                    sp=base/f'inner_{fi}'/(s['id']+'.json');z=np.load(sp.with_suffix('.npz'));bs=[z[f'beta_{k}'] for k in s['active']]
                    check_fit(tasks,gl,s['active'],bs,saved['fit'],sp)
                    for ai,k in enumerate(s['active']):
                        np.testing.assert_allclose(tests[k]['X']@bs[ai],saved['risks'][ai],rtol=0,atol=1e-10)
                        value=concordance(tests[k]['time'],tests[k]['event'],saved['risks'][ai])
                        assert abs(value-saved['scores'][ai])<1e-12;cs[s['id']][k,fi]=value;inner_checks+=1
            for k,c in enumerate(CTS):
                assert len(inner_seen[c])==len(set(inner_seen[c]))==len(tr[k])
                assert set(inner_seen[c])==set(tr[k])
            np.testing.assert_allclose(singles,d['single_inner_scores'],atol=1e-12,rtol=0)
            cd=json.loads((base/'clinical_selection_before_test.json').read_text())
            np.testing.assert_allclose(clinical_scores,cd['inner_scores'],atol=1e-12,rtol=0)
            assert clinical_scores.mean(axis=2).argmax(axis=1).tolist()==cd['ridge_indices']
            chosen=singles.mean(axis=2).argmax(axis=1);ss=np.array([singles[k,chosen[k]] for k in range(len(CTS))])
            assert [FRACTIONS[v] for v in chosen]==d['selected_single_fractions']
            means={'single':ss.mean(axis=1)}
            for s in specs:
                for k in set(range(len(CTS)))-set(s['active']):cs[s['id']][k]=ss[k]
                means[s['id']]=cs[s['id']].mean(axis=1)
                np.testing.assert_allclose(cs[s['id']],d['candidate_inner_scores'][s['id']],atol=1e-12,rtol=0)
            baseline=d['selection']['single'];assert baseline['mu']==0 and baseline['active']==[]
            for graph,options in protocol['candidates'].items():
                assert choose([baseline]+options,means)==d['selection'][graph]
                assert choose(options,means)==d['selection']['forced_'+graph];selection_checks+=2
                for mu in MUS:assert choose([s for s in options if s['mu']==mu],means)==d['strength_curve_selection'][graph][str(mu)]
            assert choose([baseline]+specs,means)==d['selection']['global_selected'];selection_checks+=1
            for k,c in enumerate(CTS):
                assert choose([baseline]+[s for s in specs if k in s['active']],means,k)==d['target_selection'][c];selection_checks+=1
            tasks,tests,gl,idx,scalers=prepare(cohorts,tr,te,labels);stored=np.load(base/'preprocessing.npz')
            np.testing.assert_array_equal(idx,stored['feature_indices']);np.testing.assert_array_equal(gl,stored['groups'])
            for k,c in enumerate(CTS):
                np.testing.assert_array_equal(scalers[k][0],stored[c+'_mean']);np.testing.assert_array_equal(scalers[k][1],stored[c+'_scale'])
            def coefficients(s):
                bs=[np.load(base/f'final_fits/single_{c}_v{chosen[k]}.npz')['beta'] for k,c in enumerate(CTS)]
                if s['id']!='single':
                    z=np.load(base/'final_fits'/(s['id']+'.npz'))
                    for k in s['active']:bs[k]=z[f'beta_{k}']
                return bs
            for k,c in enumerate(CTS):
                for method,model in r['cancers'][c]['models'].items():
                    if method.startswith('clinical'):
                        cp=base/'clinical_fits'/f'{method}_{c}.npz';z=np.load(cp);b=z['beta']
                        cl=clinical_matrix(cohorts[c]['clinical'],tr[k]);X=z['X_train'];Y=z['X_test']
                        np.testing.assert_array_equal(X[:,:8],cl[tr[k]]);np.testing.assert_array_equal(Y[:,:8],cl[te[k]])
                        info=json.loads(cp.with_suffix('.json').read_text())
                        if method!='clinical':
                            base_method=method[len('clinical_plus_'):];spec=d['selection'][base_method]
                            np.testing.assert_allclose(Y[:,-1],(tests[k]['X']@coefficients(spec)[k]-info['score_mean'])/info['score_scale'],atol=1e-10,rtol=0)
                            ids={v:i for i,v in enumerate(r['cancers'][c]['train_patient_ids'])};raw=np.zeros(len(ids))
                            for fi,split in enumerate(d['inner_partitions']):
                                pids=split['cancers'][c]['validation_ids']
                                if k in spec['active']:
                                    saved=json.loads((base/f'inner_{fi}'/(spec['id']+'.json')).read_text());rr=saved['risks'][spec['active'].index(k)]
                                else:
                                    ii=[np.array([{str(v):i for i,v in enumerate(cohorts[cc]['patient_ids'])}[pid] for pid in split['cancers'][cc]['train_ids']]) for cc in CTS]
                                    vv=[np.array([{str(v):i for i,v in enumerate(cohorts[cc]['patient_ids'])}[pid] for pid in split['cancers'][cc]['validation_ids']]) for cc in CTS]
                                    _,its,_,_,_=prepare(cohorts,ii,vv,labels)
                                    bb=np.load(base/f'inner_{fi}/single_{c}_v{chosen[k]}.npz')['beta'];rr=its[k]['X']@bb
                                raw[[ids[v] for v in pids]]=rr
                            assert abs(raw.mean()-info['score_mean'])<1e-12 and abs(max(raw.std(),1e-8)-info['score_scale'])<1e-12
                            np.testing.assert_allclose(X[:,-1],(raw-info['score_mean'])/info['score_scale'],atol=1e-10,rtol=0)
                        cv,cg=CoxObjective(X,tasks[k]['time'],tasks[k]['event']).value_grad(b)
                        pv,pg=penalty_value_grad(b,np.arange(len(b)),np.ones(len(b)),penalty='ridge')
                        assert abs(cv+info['ridge']*pv-info['objective'])<1e-10
                        assert np.max(np.abs(cg+info['ridge']*pg))<=1e-5
                        risk=Y@b;trainrisk=X@b
                    else:
                        s=d['target_selection'][c] if method=='target_selected' else d['selection'][method]
                        b=coefficients(s)[k];risk=tests[k]['X']@b;trainrisk=tasks[k]['X']@b
                    err=float(np.max(np.abs(risk-model['risk'])));max_risk_error=max(err,max_risk_error);assert err<1e-10
                    assert abs(concordance(tests[k]['time'],tests[k]['event'],risk)-model['test_c'])<1e-12
                    prob=predict_survival(tasks[k]['time'],tasks[k]['event'],trainrisk,risk,[365.25,1095.75])
                    err=float(np.max(np.abs(prob-model['survival_1y_3y'])));max_survival_error=max(err,max_survival_error);assert err<1e-10
                    metric=evaluate_survival(tasks[k]['time'],tasks[k]['event'],trainrisk,tests[k]['time'],tests[k]['event'],risk)
                    for key in ['brier_365.25','brier_1095.75']:
                        assert metric[key]==model['survival_metrics'][key]
                    outer_checks+=1
            for graph,vals in r['strength_curves'].items():
                for mu,block in vals.items():
                    assert block['spec']==d['strength_curve_selection'][graph][mu]
                    bs=coefficients(block['spec'])
                    for k,c in enumerate(CTS):
                        risk=tests[k]['X']@bs[k]
                        np.testing.assert_allclose(risk,block['cancers'][c]['risk'],rtol=0,atol=1e-10)
                        assert abs(concordance(tests[k]['time'],tests[k]['event'],risk)-block['cancers'][c]['test_c'])<1e-12
                        trainrisk=tasks[k]['X']@bs[k]
                        probability=predict_survival(tasks[k]['time'],tasks[k]['event'],trainrisk,risk,[365.25,1095.75])
                        np.testing.assert_allclose(probability,block['cancers'][c]['survival_1y_3y'],rtol=0,atol=1e-10)
                        metric=evaluate_survival(tasks[k]['time'],tasks[k]['event'],trainrisk,tests[k]['time'],tests[k]['event'],risk)
                        for key in ['brier_365.25','brier_1095.75']:
                            assert metric[key]==block['cancers'][c]['survival_metrics'][key]
                        curve_checks+=1
            # Re-evaluate final active-component objectives from retained betas.
            ll,sizes=_group_info(gl,len(idx))
            for path in (base/'final_fits').glob('*.json'):
                info=json.loads(path.read_text());z=np.load(path.with_suffix('.npz'))
                if path.stem.startswith('single_'):
                    c=path.stem.split('_')[1];active=[CTS.index(c)];bs=[z['beta']]
                else:
                    s=next(s for s in specs if s['id']==path.stem);active=s['active'];bs=[z[f'beta_{k}'] for k in active]
                check_fit(tasks,gl,active,bs,info,path)
                val=0.;grad=[]
                for ai,k in enumerate(active):
                    a,g=CoxObjective(**tasks[k]).value_grad(bs[ai]);b,pg=penalty_value_grad(bs[ai],ll,sizes)
                    val+=a+info['lambda'][ai]*b;grad.append(g+info['lambda'][ai]*pg)
                for ai in range(len(active)):
                    for aj in range(ai+1,len(active)):
                        a,ga,gb=coupling_value_grad(bs[ai],bs[aj],ll,sizes);val+=info['edge_weight']*a
                        grad[ai]+=info['edge_weight']*ga;grad[aj]+=info['edge_weight']*gb
                err=abs(val-info['objective']);max_objective_error=max(err,max_objective_error);assert err<1e-8;objective_checks+=1
                err=abs(max(float(np.max(np.abs(g))) for g in grad)-info['gradient_norm'])
                max_gradient_error=max(err,max_gradient_error);assert err<1e-8
            print(f'validated seed {seed} fold {fo}',flush=True)
            completed_folds.append({'seed':seed,'fold':fo})
        for c in CTS:
            assert len(seen[c])==len(set(seen[c]))
            assert set(seen[c]).issubset(set(cohorts[c]['patient_ids']))
            if not args.completed_only:
                assert len(seen[c])==len(cohorts[c]['patient_ids'])
                assert set(seen[c])==set(cohorts[c]['patient_ids'])
        partition_counts[str(seed)]={c:len(seen[c]) for c in CTS}
    result={'passed':True,'patient_partition_counts':partition_counts,'inner_scores_recomputed':inner_checks,
        'outer_models_recomputed':outer_checks,'curve_predictions_recomputed':curve_checks,'selection_rules_recomputed':selection_checks,
        'final_objectives_recomputed':objective_checks,'max_risk_error':max_risk_error,'max_survival_error':max_survival_error,
        'max_objective_error':max_objective_error,'max_gradient_error':max_gradient_error,'cache_and_gmt_hashes_match':True,'source_verification_modes':source_modes,
        'outer_tests_excluded_from_all_inner_partitions':True,'protocol_sha256':digest(OUT/'protocol.json')}
    result['each_outer_training_patient_appears_once_across_inner_validation_partitions']=True
    result['STAD_clinical_reference_reproduces_original_matched_patient_predictions']=True
    result['inner_feature_masks_recomputed']=3*len(completed_folds)
    assert digest(__file__)==validator_sha256,'Validator source changed during the audit'
    result['validator_sha256']=validator_sha256
    result['all_saved_molecular_fits_strict']=True
    result['total_fits_with_objective_and_gradient_recomputed']=len(FIT_CHECKS)
    result['completed_outer_folds_checked']=completed_folds
    result['scope']='partial audit of available completed outer folds' if args.completed_only else 'complete experiment'
    result['complete_experiment_validation']=not args.completed_only
    prefix='partial_' if args.completed_only else ''
    dump(OUT/(prefix+'fit_validation_records.json'),FIT_CHECKS)
    dump(OUT/(prefix+'validation.json'),result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
