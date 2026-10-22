"""Fixed-budget training-only preconditioning before strict Newton refinement."""
import time
import numpy as np
from scipy.optimize import minimize
from gastric_mtl.solver import StableDirectionalObjective
from gastric_mtl.second_order import task_group_transform


def warmup(tasks, groups, lam, edge, beta0, max_iter=2000, gtol=1e-5):
    obj=StableDirectionalObjective(tasks,groups,lam,edge)
    beta=np.asarray(beta0,float).ravel().copy();value,grad=obj.value_grad(beta)
    initial=value;history=[];used=0;started=time.perf_counter()
    class Stop(Exception):pass
    class Invalid(Exception):pass
    while used<max_iter and np.max(np.abs(grad))>gtol:
        base=beta.copy();forward,transpose=task_group_transform(base.reshape(obj.nt,obj.p),obj.labels,obj.sizes,obj.lam,edge,tasks)
        best={'x':base.copy(),'value':value,'grad':grad.copy()};accepted=[0]
        def fg(z):
            x=base+forward(z)
            if not np.isfinite(x).all():raise Invalid()
            v,g=obj.value_grad(x)
            if not np.isfinite(v) or not np.isfinite(g).all():raise Invalid()
            if v<best['value']:best.update(x=x.copy(),value=v,grad=g.copy())
            if np.max(np.abs(g))<=gtol and v<=value+1e-10:
                best.update(x=x.copy(),value=v,grad=g.copy());raise Stop()
            return v,transpose(g)
        def callback(z):accepted[0]+=1
        message=''
        try:
            fit=minimize(fg,np.zeros_like(beta),jac=True,method='L-BFGS-B',callback=callback,
                         options={'maxiter':min(100,max_iter-used),'ftol':1e-15,'gtol':1e-10,'maxcor':20,'maxls':40})
            n=max(1,int(fit.nit));message=str(fit.message)
        except Stop:
            n=max(1,accepted[0]+1);message='Raw gradient criterion reached in preconditioning'
        except (Invalid,FloatingPointError):
            n=max(1,accepted[0]+1);message='Rejected invalid preconditioning trial'
        change=value-best['value'];beta=best['x'];value=best['value'];grad=best['grad'];used+=n
        assert value<=initial+1e-8
        history.append({'n_iter':n,'objective':value,'gradient_norm':float(np.max(np.abs(grad))),'message':message})
        if message.startswith('Rejected') or (len(history)>2 and change<1e-13 and n<2):break
    value,grad=obj.value_grad(beta)
    return {'betas':list(beta.reshape(obj.nt,obj.p)),'objective':value,'initial_objective':initial,
            'gradient_norm':float(np.max(np.abs(grad))),'converged':bool(np.max(np.abs(grad))<=gtol),
            'n_iter':used,'n_eval':obj.nfg,'n_hessp':0,'seconds':time.perf_counter()-started,
            'stage':'fixed-budget task/group-conditioned L-BFGS-B warmup','history':history}
