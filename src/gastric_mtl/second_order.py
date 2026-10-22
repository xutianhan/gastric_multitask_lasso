"""Analytic Hessian products for the unchanged smooth Group Lasso Cox objective.

This is an additional numerical solver, not a new prediction model. The original
source is intentionally immutable. Coupling is not assumed to be convex.
"""
import time
import numpy as np
from scipy.optimize import minimize
from gastric_mtl.cox import CoxObjective, _group_info, penalty_value_grad, coupling_value_grad


class SmoothSharingObjective:
    def __init__(self, tasks, groups, lam, edge=0., eps=1e-4):
        self.objs = [CoxObjective(**t) for t in tasks]
        self.nt, self.p = len(tasks), self.objs[0].p
        self.labels, self.sizes = _group_info(groups, self.p)
        self.lam = np.broadcast_to(np.asarray(lam, float), (self.nt,)).copy()
        self.edge, self.eps = float(edge), float(eps)
        self.cached_x = None
        self.nfg = self.nhv = 0

    def state(self, x):
        if self.cached_x is not None and np.array_equal(x, self.cached_x):
            return
        self.cached_x = np.array(x, float, copy=True)
        self.beta = self.cached_x.reshape(self.nt, self.p)
        ng = len(self.sizes)
        self.norms = np.array([np.sqrt(np.bincount(self.labels, weights=b*b, minlength=ng) + self.eps**2) for b in self.beta])
        self.risks = []
        for obj, b in zip(self.objs, self.beta):
            eta = obj.X @ b
            # A single shift suffices at fitted coefficients; explicitly reject
            # unsafe ranges rather than silently returning an invalid Hessian.
            if float(np.ptp(eta)) > 600:
                raise FloatingPointError('Predictor range too wide for Hessian products')
            exp_eta = np.exp(eta - eta.max())
            rr = np.cumsum(exp_eta[::-1])[::-1]
            counts = np.bincount(obj.first, weights=obj.event, minlength=obj.n)
            a = np.cumsum(counts / rr)
            self.risks.append((exp_eta, rr, counts, a))

    def value_grad(self, x):
        self.state(x)
        value, grad = 0., np.zeros_like(self.beta)
        for k, obj in enumerate(self.objs):
            v, g = obj.value_grad(self.beta[k])
            pv, pg = penalty_value_grad(self.beta[k], self.labels, self.sizes, eps=self.eps)
            value += v + self.lam[k]*pv
            grad[k] = g + self.lam[k]*pg
        for k in range(self.nt):
            for j in range(k+1, self.nt):
                if not self.edge: continue
                v, gk, gj = coupling_value_grad(self.beta[k], self.beta[j], self.labels, self.sizes, self.eps)
                value += self.edge*v
                grad[k] += self.edge*gk
                grad[j] += self.edge*gj
        self.nfg += 1
        return float(value), grad.ravel()

    def hessp(self, x, direction):
        self.state(x)
        v = np.asarray(direction).reshape(self.nt, self.p)
        out = np.zeros_like(v)
        labels, sizes, norms, beta = self.labels, self.sizes, self.norms, self.beta
        ng = len(sizes)
        dots = np.array([np.bincount(labels, weights=b*z, minlength=ng) for b,z in zip(beta,v)])
        for k,obj in enumerate(self.objs):
            q, rr, counts, a = self.risks[k]
            z = obj.X @ v[k]
            mean_z = np.cumsum((q*z)[::-1])[::-1] / rr
            hv_eta = (2./obj.n)*q*(z*a - np.cumsum(counts/rr*mean_z))
            out[k] = obj.X.T @ hv_eta
            out[k] += self.lam[k]*(np.sqrt(sizes)/norms[k])[labels]*(v[k]-beta[k]*(dots[k]/norms[k]**2)[labels])
        for k in range(self.nt):
            for j in range(k+1,self.nt):
                if not self.edge:continue
                aa,bb = norms[k],norms[j]
                ab = np.bincount(labels,weights=beta[k]*beta[j],minlength=ng)
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


def fit_second_order(tasks, groups, lam, edge=0., beta0=None, max_iter=500, gtol=1e-5):
    objective = SmoothSharingObjective(tasks, groups, lam, edge)
    x = np.zeros(objective.nt*objective.p) if beta0 is None else np.asarray(beta0,float).ravel()
    initial, gradient = objective.value_grad(x)
    started = time.perf_counter()
    if np.max(np.abs(gradient))<=gtol:
        return {'betas':list(x.reshape(objective.nt,objective.p)), 'objective':initial,
                'gradient_norm':float(np.max(np.abs(gradient))), 'converged':True,
                'n_iter':0,'n_eval':1,'n_hessp':0,'seconds':0.,'status':0,'message':'Initial raw gradient satisfied tolerance'}
    fit = minimize(objective.value_grad,x,jac=True,hessp=objective.hessp,method='trust-krylov',
                   options={'maxiter':int(max_iter),'gtol':float(gtol),'inexact':True})
    value,gradient = objective.value_grad(fit.x)
    assert np.isfinite(value) and value<=initial+1e-8 and np.isfinite(fit.x).all()
    return {'betas':list(fit.x.reshape(objective.nt,objective.p)), 'objective':value,
            'gradient_norm':float(np.max(np.abs(gradient))),
            'converged':bool(np.max(np.abs(gradient))<=gtol),'optimizer_success':bool(fit.success),
            'n_iter':int(fit.nit),'n_eval':objective.nfg,'n_hessp':objective.nhv,
            'seconds':time.perf_counter()-started,'status':int(fit.status),'message':str(fit.message)}


def group_transform(beta, labels, sizes, lam, edge, tasks, eps=1e-4):
    """Symmetric invertible group radial/tangential change of coordinates."""
    nt,p=beta.shape;ng=len(sizes)
    norms=np.array([np.sqrt(np.bincount(labels,weights=b*b,minlength=ng)+eps**2) for b in beta])
    curvature=np.array([np.full(ng,max(.1,2*float(t['event'].mean()))) for t in tasks])
    tang=curvature+np.asarray(lam)[:,None]*np.sqrt(sizes)[None,:]/norms
    radial=curvature+np.asarray(lam)[:,None]*np.sqrt(sizes)[None,:]*eps**2/norms**3
    for k in range(nt):
        for j in range(k+1,nt):
            dot=np.bincount(labels,weights=beta[k]*beta[j],minlength=ng)
            den=np.sqrt(max(0.,2*np.dot(sizes,norms[k]*norms[j]-dot))+eps**2)
            ck=edge*sizes*(norms[j]/norms[k])/den;cj=edge*sizes*(norms[k]/norms[j])/den
            tang[k]+=ck;tang[j]+=cj;radial[k]+=ck;radial[j]+=cj
    ta=1/np.sqrt(tang);ra=1/np.sqrt(radial)
    u=np.array([b/np.sqrt(np.maximum(norms[k]**2-eps**2,1e-30))[labels] for k,b in enumerate(beta)])
    def transform(flat):
        v=np.asarray(flat).reshape(nt,p);out=np.zeros_like(v)
        for k in range(nt):
            dot=np.bincount(labels,weights=u[k]*v[k],minlength=ng)
            out[k]=ta[k][labels]*v[k]+(ra[k]-ta[k])[labels]*u[k]*dot[labels]
        return out.ravel()
    return transform


def fit_conditioned_newton(tasks, groups, lam, edge=0., beta0=None, max_iter=400, gtol=1e-5, chunk=20):
    obj=SmoothSharingObjective(tasks,groups,lam,edge)
    beta=np.zeros(obj.nt*obj.p) if beta0 is None else np.asarray(beta0,float).ravel().copy()
    started=time.perf_counter();history=[];iterations=0
    value,grad=obj.value_grad(beta);initial=value
    while iterations<max_iter and np.max(np.abs(grad))>gtol:
        base=beta.copy();transform=group_transform(base.reshape(obj.nt,obj.p),obj.labels,obj.sizes,obj.lam,edge,tasks)
        def fg(z):
            val,g=obj.value_grad(base+transform(z));return val,transform(g)
        def hp(z,v):return transform(obj.hessp(base+transform(z),transform(v)))
        fit=minimize(fg,np.zeros_like(beta),jac=True,hessp=hp,method='trust-krylov',
                     options={'maxiter':min(chunk,max_iter-iterations),'gtol':1e-9,'inexact':True})
        next_beta=base+transform(fit.x);next_value,grad=obj.value_grad(next_beta)
        assert np.isfinite(next_value) and next_value<=value+1e-8
        change=value-next_value;beta=next_beta;value=next_value;iterations+=max(1,int(fit.nit))
        history.append({'iterations':int(fit.nit),'objective':value,'gradient_norm':float(np.max(np.abs(grad))),
                        'status':int(fit.status),'message':str(fit.message),'seconds':time.perf_counter()-started})
        if len(history)>2 and change<1e-13 and fit.nit<2:break
    return {'betas':list(beta.reshape(obj.nt,obj.p)),'objective':value,'initial_objective':initial,
            'gradient_norm':float(np.max(np.abs(grad))),'converged':bool(np.max(np.abs(grad))<=gtol),
            'n_iter':iterations,'n_eval':obj.nfg,'n_hessp':obj.nhv,'seconds':time.perf_counter()-started,
            'history':history}


def task_group_transform(beta, labels, sizes, lam, edge, tasks, eps=1e-4, kind='directional'):
    """Whiten task/group curvature with small eigensystems and rank-k updates.

    This is an invertible numerical coordinate map. Negative local curvature is
    floored only in the preconditioner; the objective and Hessian are unchanged.
    Returns the forward map and its distinct transpose.
    """
    nt,p=beta.shape;ng=len(sizes)
    norms=np.array([np.sqrt(np.bincount(labels,weights=b*b,minlength=ng)+eps**2) for b in beta])
    dot=np.empty((ng,nt,nt))
    for k in range(nt):
        for j in range(nt):dot[:,k,j]=np.bincount(labels,weights=beta[k]*beta[j],minlength=ng)
    a=np.zeros((ng,nt,nt));d=np.zeros_like(a)
    for k in range(nt):
        a[:,k,k]=max(.1,2*float(tasks[k]['event'].mean()))+lam[k]*np.sqrt(sizes)/norms[k]
        d[:,k,k]=-lam[k]*np.sqrt(sizes)/norms[k]**3
    for k in range(nt):
        for j in range(k+1,nt):
            if kind=='quadratic':
                a[:,k,k]+=edge;a[:,j,j]+=edge;a[:,k,j]-=edge;a[:,j,k]-=edge
                continue
            den=np.sqrt(max(0.,2*np.dot(sizes,norms[k]*norms[j]-dot[:,k,j]))+eps**2)
            c=edge*sizes/den
            a[:,k,k]+=c*norms[j]/norms[k];a[:,j,j]+=c*norms[k]/norms[j]
            a[:,k,j]-=c;a[:,j,k]-=c
            d[:,k,k]-=c*norms[j]/norms[k]**3;d[:,j,j]-=c*norms[k]/norms[j]**3
            d[:,k,j]+=c/(norms[k]*norms[j]);d[:,j,k]+=c/(norms[k]*norms[j])
    ev,u=np.linalg.eigh(a)
    assert ev.min()>0,ev.min()
    half=np.einsum('gik,gk,gjk->gij',u,1/np.sqrt(ev),u)
    inv=np.einsum('gik,gk,gjk->gij',u,1/ev,u)
    gram=inv*dot
    ge,gu=np.linalg.eigh(gram);ge=np.maximum(ge,0.)
    groot=np.einsum('gik,gk,gjk->gij',gu,np.sqrt(ge),gu)
    gi=np.zeros_like(ge);mask=ge>np.maximum(ge.max(axis=1,keepdims=True)*1e-12,1e-30);gi[mask]=1/np.sqrt(ge[mask])
    ginv=np.einsum('gik,gk,gjk->gij',gu,gi,gu)
    small=np.eye(nt)[None,:,:]+np.einsum('gij,gjk,gkl->gil',groot,d,groot)
    me,mu=np.linalg.eigh(small)
    correction=np.einsum('gik,gk,gjk->gij',mu,1/np.sqrt(np.maximum(me,.001))-1,mu)
    q=np.einsum('pij,jp,pjr->ipr',half[labels],beta,ginv[labels])
    def apply_half(v):return np.einsum('pij,jp->ip',half[labels],v)
    def apply_lowrank(v):
        projection=np.array([np.bincount(labels,weights=np.sum(q[:,:,r]*v,axis=0),minlength=ng) for r in range(nt)]).T
        weights=np.einsum('gij,gj->gi',correction,projection)
        return v+np.einsum('ipr,pr->ip',q,weights[labels])
    def forward(v):return apply_half(apply_lowrank(np.asarray(v).reshape(nt,p))).ravel()
    def transpose(v):return apply_lowrank(apply_half(np.asarray(v).reshape(nt,p))).ravel()
    return forward,transpose


class QuadraticSharingObjective(SmoothSharingObjective):
    """Convex ablation retaining the identical Cox loss and group penalty."""
    def __init__(self,tasks,groups,lam,edge=0.,eps=1e-4):
        super().__init__(tasks,groups,lam,0.,eps)
        self.quadratic_edge=float(edge)

    def value_grad(self,x):
        value,flat=super().value_grad(x);grad=flat.reshape(self.nt,self.p)
        for k in range(self.nt):
            for j in range(k+1,self.nt):
                difference=self.beta[k]-self.beta[j]
                value+=.5*self.quadratic_edge*float(difference@difference)
                grad[k]+=self.quadratic_edge*difference;grad[j]-=self.quadratic_edge*difference
        return value,grad.ravel()

    def hessp(self,x,direction):
        out=super().hessp(x,direction).reshape(self.nt,self.p)
        v=np.asarray(direction).reshape(self.nt,self.p)
        for k in range(self.nt):
            for j in range(k+1,self.nt):
                difference=self.quadratic_edge*(v[k]-v[j]);out[k]+=difference;out[j]-=difference
        return out.ravel()


def fit_task_conditioned_newton(tasks,groups,lam,edge=0.,beta0=None,max_iter=400,gtol=1e-5,chunk=20,coupling='directional'):
    assert coupling in ['directional','quadratic']
    cls=QuadraticSharingObjective if coupling=='quadratic' else SmoothSharingObjective
    obj=cls(tasks,groups,lam,edge)
    beta=np.zeros(obj.nt*obj.p) if beta0 is None else np.asarray(beta0,float).ravel().copy()
    started=time.perf_counter();history=[];iterations=0
    value,grad=obj.value_grad(beta);initial=value
    while iterations<max_iter and np.max(np.abs(grad))>gtol:
        base=beta.copy();transform,transpose=task_group_transform(base.reshape(obj.nt,obj.p),obj.labels,obj.sizes,obj.lam,edge,tasks,kind=coupling)
        def fg(z):
            val,g=obj.value_grad(base+transform(z));return val,transpose(g)
        def hp(z,v):return transpose(obj.hessp(base+transform(z),transform(v)))
        fit=minimize(fg,np.zeros_like(beta),jac=True,hessp=hp,method='trust-krylov',
                     options={'maxiter':min(chunk,max_iter-iterations),'gtol':1e-9,'inexact':True})
        next_beta=base+transform(fit.x);next_value,grad=obj.value_grad(next_beta)
        assert np.isfinite(next_value) and next_value<=value+1e-8
        change=value-next_value;beta=next_beta;value=next_value;iterations+=max(1,int(fit.nit))
        history.append({'iterations':int(fit.nit),'objective':value,'gradient_norm':float(np.max(np.abs(grad))),
                        'status':int(fit.status),'message':str(fit.message),'seconds':time.perf_counter()-started})
        if len(history)>2 and change<1e-13 and fit.nit<2:break
    return {'betas':list(beta.reshape(obj.nt,obj.p)),'objective':value,'initial_objective':initial,
            'gradient_norm':float(np.max(np.abs(grad))),'converged':bool(np.max(np.abs(grad))<=gtol),
            'n_iter':iterations,'n_eval':obj.nfg,'n_hessp':obj.nhv,'seconds':time.perf_counter()-started,
            'history':history}
