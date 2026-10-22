"""Clinical reference and training-only two-stage molecular score augmentation."""
import json
import numpy as np
from gastric_mtl.cox import fit_cox, c_index
from gastric_mtl.metrics import predict_survival, evaluate_survival
from workflows.experiments.initial_evaluation import clinical_matrix, dump
from gastric_mtl.eac import prepare

CTS=['EAC','STAD']
RIDGES=[1.,.1,.01,0.]
AUGMENT=['single','pair_EAC_STAD']


def fit_clinical(X,t,e,lam):
    fit=fit_cox(X,t,e,lam=lam,penalty='ridge',max_iter=1000,tol=1e-14,gtol=1e-8)
    if fit['gradient_norm']>1e-5:
        fit=fit_cox(X,t,e,lam=lam,penalty='ridge',beta0=fit['beta'],max_iter=5000,tol=1e-15,gtol=1e-9)
    assert fit['gradient_norm']<=1e-5,('clinical convergence',fit['gradient_norm'])
    return fit


def train(cohorts,tr,te,inners,tasks,tests,labels,selection,fit_b,chosen,base):
    scores=np.zeros((len(CTS),4,3));oof={m:[np.zeros(len(v)) for v in tr] for m in AUGMENT}
    audit=[]
    for fi in range(3):
        itr=[tr[k][inners[k][fi][0]] for k in range(len(CTS))]
        iva=[tr[k][inners[k][fi][1]] for k in range(len(CTS))]
        _,itests,_,_,_=prepare(cohorts,itr,iva,labels)
        for k,c in enumerate(CTS):
            cl=clinical_matrix(cohorts[c]['clinical'],itr[k])
            for vi,lam in enumerate(RIDGES):
                fit=fit_clinical(cl[itr[k]],cohorts[c]['time'][itr[k]],cohorts[c]['event'][itr[k]],lam)
                scores[k,vi,fi]=c_index(cohorts[c]['time'][iva[k]],cohorts[c]['event'][iva[k]],cl[iva[k]]@fit['beta'])
                audit.append({'phase':'inner','fold':fi,'cancer':c,'ridge':lam,'gradient_norm':fit['gradient_norm']})
            for m in AUGMENT:
                s=selection[m]
                if k in s['active']:
                    saved=json.loads((base/f'inner_{fi}'/(s['id']+'.json')).read_text())
                    risk=np.asarray(saved['risks'][s['active'].index(k)])
                else:
                    beta=np.load(base/f'inner_{fi}/single_{c}_v{chosen[k]}.npz')['beta']
                    risk=itests[k]['X']@beta
                oof[m][k][inners[k][fi][1]]=risk
    selected=scores.mean(axis=2).argmax(axis=1)
    dump(base/'clinical_selection_before_test.json',{'inner_scores':scores.tolist(),'ridge_indices':selected.tolist(),'ridges':[RIDGES[i] for i in selected]})
    final={c:{} for c in CTS}
    for k,c in enumerate(CTS):
        cl=clinical_matrix(cohorts[c]['clinical'],tr[k]);lam=RIDGES[selected[k]]
        designs={'clinical':(cl[tr[k]],cl[te[k]],{})}
        for m in AUGMENT:
            raw=oof[m][k];mean=float(raw.mean());sd=max(float(raw.std()),1e-8)
            test_risk=tests[k]['X']@fit_b[selection[m]['id']][k]
            designs['clinical_plus_'+m]=(np.column_stack([cl[tr[k]],(raw-mean)/sd]),np.column_stack([cl[te[k]],(test_risk-mean)/sd]),{'score_mean':mean,'score_scale':sd,'base_candidate':selection[m]['id']})
        for name,(X,Y,meta) in designs.items():
            fit=fit_clinical(X,tasks[k]['time'],tasks[k]['event'],lam);beta=fit.pop('beta')
            path=base/'clinical_fits'/f'{name}_{c}'
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path.with_suffix('.npz'),beta=beta,X_train=X,X_test=Y)
            dump(path.with_suffix('.json'),{**fit,**meta,'ridge':lam})
            final[c][name]=({'X':X,'time':tasks[k]['time'],'event':tasks[k]['event']},Y,beta)
            audit.append({'phase':'outer','cancer':c,'model':name,'ridge':lam,'gradient_norm':fit['gradient_norm']})
    dump(base/'clinical_fit_audit.json',audit)
    # Predictions are evaluated only after all clinical choices/fits are saved.
    result={c:{} for c in CTS}
    for k,c in enumerate(CTS):
        for name,(task,Y,beta) in final[c].items():
            trainrisk=task['X']@beta;risk=Y@beta
            result[c][name]={'risk':risk.tolist(),'test_c':c_index(tests[k]['time'],tests[k]['event'],risk),
                'survival_1y_3y':predict_survival(task['time'],task['event'],trainrisk,risk,[365.25,1095.75]).tolist(),
                'survival_metrics':evaluate_survival(task['time'],task['event'],trainrisk,tests[k]['time'],tests[k]['event'],risk)}
    return result

