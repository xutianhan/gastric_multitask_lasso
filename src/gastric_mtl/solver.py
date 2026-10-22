"""Numerically stable Hessian for the unchanged directional Group Lasso model.

Use the original cumulative-sum Hessian for ordinary predictor ranges and exact
log-sum-exp risk-set probabilities if a trust-region trial has a wide range.
The objective, gradient and directional coupling are unchanged.
"""
import numpy as np
import time
from scipy.optimize import minimize
from gastric_mtl.second_order import SmoothSharingObjective
from gastric_mtl.second_order import task_group_transform


class StableDirectionalObjective(SmoothSharingObjective):
    def state(self, x):
        if self.cached_x is not None and np.array_equal(x, self.cached_x):
            return
        beta = np.asarray(x, float).reshape(self.nt, self.p)
        assert np.isfinite(beta).all()
        norms = np.array([np.sqrt(np.bincount(self.labels, weights=b*b,
                         minlength=len(self.sizes)) + self.eps**2) for b in beta])
        risks = []
        for obj, b in zip(self.objs, beta):
            eta = obj.X @ b
            counts = np.bincount(obj.first, weights=obj.event, minlength=obj.n)
            if np.ptp(eta) < 500:
                q = np.exp(eta - eta.max())
                rr = np.cumsum(q[::-1])[::-1]
                risks.append((q, rr, counts, np.cumsum(counts / rr)))
            else:
                log_risk = np.logaddexp.accumulate(eta[::-1])[::-1]
                rows = np.flatnonzero(counts)
                valid = np.arange(obj.n)[None, :] >= rows[:, None]
                probs = np.exp(np.where(valid, eta[None, :] - log_risk[rows, None], -np.inf))
                weights = counts[rows]
                risks.append({'probabilities': probs, 'weights': weights,
                              'marginal': weights @ probs})
        self.cached_x = np.array(x, float, copy=True)
        self.beta, self.norms, self.risks = beta.copy(), norms, risks

    def hessp(self, x, direction):
        self.state(x)
        v = np.asarray(direction).reshape(self.nt, self.p)
        out = np.zeros_like(v)
        labels, sizes, norms, beta = self.labels, self.sizes, self.norms, self.beta
        ng = len(sizes)
        dots = np.array([np.bincount(labels, weights=b*z, minlength=ng) for b,z in zip(beta,v)])
        for k, obj in enumerate(self.objs):
            z = obj.X @ v[k]
            risk = self.risks[k]
            if isinstance(risk, dict):
                probs, weights = risk['probabilities'], risk['weights']
                hv_eta = (2./obj.n)*(risk['marginal']*z - (weights*(probs@z))@probs)
            else:
                q, rr, counts, a = risk
                mean_z = np.cumsum((q*z)[::-1])[::-1] / rr
                hv_eta = (2./obj.n)*q*(z*a - np.cumsum(counts/rr*mean_z))
            out[k] = obj.X.T @ hv_eta
            out[k] += self.lam[k]*(np.sqrt(sizes)/norms[k])[labels]*(v[k]-beta[k]*(dots[k]/norms[k]**2)[labels])
        for k in range(self.nt):
            for j in range(k+1,self.nt):
                if not self.edge:
                    continue
                aa, bb = norms[k], norms[j]
                ab = np.bincount(labels, weights=beta[k]*beta[j], minlength=ng)
                den = np.sqrt(max(0.,2*float(np.dot(sizes,aa*bb-ab)))+self.eps**2)
                ak = sizes[labels]*((bb/aa)[labels]*beta[k]-beta[j])
                aj = sizes[labels]*((aa/bb)[labels]*beta[j]-beta[k])
                dd = float(ak@v[k]+aj@v[j])/den
                dk = (dots[j]/bb/aa-bb*dots[k]/aa**3)[labels]
                dj = (dots[k]/aa/bb-aa*dots[j]/bb**3)[labels]
                dak = sizes[labels]*((bb/aa)[labels]*v[k]+dk*beta[k]-v[j])
                daj = sizes[labels]*((aa/bb)[labels]*v[j]+dj*beta[j]-v[k])
                out[k] += self.edge*(dak/den-ak*dd/den**2)
                out[j] += self.edge*(daj/den-aj*dd/den**2)
        self.nhv += 1
        return out.ravel()


def fit_stable_newton(tasks, groups, lam, edge=0., beta0=None, max_iter=600,
                      gtol=1e-5, chunk=20):
    """Stop on raw stationarity before trlib's roundoff-level trial steps.

    Every accepted checkpoint lowers the training objective. A nonfinite
    Krylov proposal is rejected and the iteration resumes with trust-ncg from
    the best finite training checkpoint, never from held-out performance.
    """
    obj=StableDirectionalObjective(tasks,groups,lam,edge)
    beta=np.zeros(obj.nt*obj.p) if beta0 is None else np.asarray(beta0,float).ravel().copy()
    started=time.perf_counter();history=[];iterations=0
    value,grad=obj.value_grad(beta);initial=value;method='trust-krylov'
    class Stationary(Exception): pass
    class InvalidProposal(Exception): pass
    while iterations<max_iter and np.max(np.abs(grad))>gtol:
        base=beta.copy();transform,transpose=task_group_transform(base.reshape(obj.nt,obj.p),obj.labels,obj.sizes,obj.lam,edge,tasks)
        best={'x':base.copy(),'value':value,'grad':grad.copy()};accepted=[0]
        def fg(z):
            proposal=base+transform(z)
            if not np.isfinite(proposal).all():raise InvalidProposal()
            val,g=obj.value_grad(proposal)
            if not np.isfinite(val) or not np.isfinite(g).all():raise InvalidProposal()
            if val<=best['value']:
                best.update(x=proposal.copy(),value=val,grad=g.copy())
            if np.max(np.abs(g))<=gtol and val<=value+1e-10:
                best.update(x=proposal.copy(),value=val,grad=g.copy())
                raise Stationary()
            return val,transpose(g)
        def hp(z,v):return transpose(obj.hessp(base+transform(z),transform(v)))
        def callback(z):accepted[0]+=1
        budget=min(chunk,max_iter-iterations);message=''
        try:
            fit=minimize(fg,np.zeros_like(beta),jac=True,hessp=hp,method=method,callback=callback,
                         options={'maxiter':budget,'gtol':1e-9})
            used=max(1,int(fit.nit));message=str(fit.message)
        except Stationary:
            used=max(1,accepted[0]+1);message='Raw gradient threshold attained'
        except (InvalidProposal,FloatingPointError) as exc:
            used=max(1,accepted[0]+1);message='Rejected nonfinite numerical trial; trust-ncg fallback'
            method='trust-ncg'
        next_value=best['value'];assert np.isfinite(next_value) and next_value<=value+1e-8
        change=value-next_value;beta=best['x'];value=next_value;grad=best['grad'];iterations+=used
        history.append({'iterations':used,'objective':value,'gradient_norm':float(np.max(np.abs(grad))),
                        'method':method,'message':message,'seconds':time.perf_counter()-started})
        if change<1e-13 and used<2 and np.max(np.abs(grad))>gtol:
            if method=='trust-krylov':method='trust-ncg'
            elif len(history)>2 and history[-2]['objective']-value<1e-13:break
    value,grad=obj.value_grad(beta)
    return {'betas':list(beta.reshape(obj.nt,obj.p)),'objective':value,'initial_objective':initial,
            'gradient_norm':float(np.max(np.abs(grad))),'converged':bool(np.max(np.abs(grad))<=gtol),
            'n_iter':iterations,'n_eval':obj.nfg,'n_hessp':obj.nhv,'seconds':time.perf_counter()-started,'history':history}
