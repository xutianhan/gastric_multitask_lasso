#!/usr/bin/env python3
"""Freeze cohort/split manifests and verify horizon support before fitting."""
from workflows.experiments.eac_stad import *


def main():
    initialize()  # Refuse incompatible sources before writing any cohort files.
    co=load_submission_cohorts();labels,names,_=pathway_labels(co['EAC']['genes'])
    audits=[];support=[]
    for seed in SEEDS:
        folds=[list(StratifiedKFold(5,shuffle=True,random_state=seed+k).split(np.zeros(len(co[c]['event'])),co[c]['event'])) for k,c in enumerate(CTS)]
        seen={c:[] for c in CTS}
        for fo in range(5):
            tr=[v[fo][0] for v in folds];te=[v[fo][1] for v in folds]
            inners=[list(StratifiedKFold(3,shuffle=True,random_state=seed+100+fo*7+k).split(np.zeros(len(tr[k])),co[c]['event'][tr[k]])) for k,c in enumerate(CTS)]
            partitions=[('outer',tr,te)]+[(f'inner_{fi}',[tr[k][inners[k][fi][0]] for k in range(len(CTS))],[tr[k][inners[k][fi][1]] for k in range(len(CTS))]) for fi in range(3)]
            for phase,itr,ite in partitions:
                tasks,tests,gl,idx,scalers=prepare(co,itr,ite,labels)
                row={'seed':seed,'fold':fo,'phase':phase,'features':len(idx),'groups':len(np.unique(gl)),'cancers':{}}
                for k,c in enumerate(CTS):
                    assert np.isfinite(tasks[k]['X']).all() and np.isfinite(tests[k]['X']).all()
                    assert np.max(abs(tasks[k]['X'].mean(axis=0)))<1e-9
                    assert np.max(abs(tasks[k]['X'].std(axis=0)-1))<1e-9
                    assert not set(itr[k])&set(ite[k])
                    if phase!='outer':assert not (set(itr[k])|set(ite[k]))&set(te[k])
                    row['cancers'][c]={'train_ids':co[c]['patient_ids'][itr[k]].tolist(),'test_ids':co[c]['patient_ids'][ite[k]].tolist(),'train_n':len(itr[k]),'test_n':len(ite[k]),'train_events':int(co[c]['event'][itr[k]].sum()),'test_events':int(co[c]['event'][ite[k]].sum())}
                    if phase=='outer':
                        seen[c].extend(ite[k].tolist())
                        met=evaluate_survival(tasks[k]['time'],tasks[k]['event'],np.zeros(len(itr[k])),tests[k]['time'],tests[k]['event'],np.zeros(len(ite[k])))
                        for horizon,d in met['details'].items():
                            assert d['supported'],(seed,fo,c,horizon,d)
                            support.append({'seed':seed,'fold':fo,'cancer':c,**d})
                audits.append(row)
            # Same held-out STAD patients as in the original full-cohort study.
            oldname='nested_primary' if seed==42 else 'nested_repeat_seed2026'
            old=json.loads((OLD/'experiments'/oldname/f'fold_{fo}.json').read_text())
            assert co['STAD']['patient_ids'][te[1]].tolist()==old['cancers']['STAD']['patient_ids']
        for c in CTS:assert sorted(seen[c])==list(range(len(co[c]['patient_ids'])))
    # Hold-out expression perturbation must not change the fitted feature/scaler state.
    changed={c:dict(co[c]) for c in CTS}
    for k,c in enumerate(CTS):
        changed[c]['X_log2']=co[c]['X_log2'].copy()
        changed[c]['X_log2'][te[k]]=0.
    first=prepare(co,tr,te,labels);second=prepare(changed,tr,te,labels)
    assert np.array_equal(first[3],second[3])
    for a,b in zip(first[0],second[0]):assert np.array_equal(a['X'],b['X'])
    OUT.parent.mkdir(parents=True,exist_ok=True)
    for c in CTS:co[c]['clinical'].to_csv(OUT.parent/f'cohort_{c}.csv',index_label='patient_id')
    dump(OUT.parent/'split_manifest.json',audits)
    dump(OUT.parent/'preflight.json',{'passed':True,'cohorts':{c:{'n':len(co[c]['event']),'events':int(co[c]['event'].sum())} for c in CTS},'partition_count':len(audits),'holdout_perturbation_invariance':True,'feature_range':[min(a['features'] for a in audits),max(a['features'] for a in audits)],'group_range':[min(a['groups'] for a in audits),max(a['groups'] for a in audits)],'horizon_support':support})
    print(json.dumps({'passed':True,'output':str(OUT),'partitions':len(audits),'min_training_censoring_survival_3y':min(d['g_horizon'] for d in support if d['horizon_days']==1095.75)},indent=2))

if __name__=='__main__':main()
