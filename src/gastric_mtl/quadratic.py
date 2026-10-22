"""Exact quadratic Cox Hessian products stable across predictor ranges.

This recovery backend changes numerical evaluation, not the objective. Risk-set
probabilities are formed from log-sum-exp denominators, avoiding a global
exponential shift that can underflow across very wide predictor ranges.
"""
import numpy as np
from gastric_mtl.second_order import QuadraticSharingObjective
from gastric_mtl.cox import penalty_value_grad


class StableQuadraticSharingObjective(QuadraticSharingObjective):
    def state(self, x):
        if self.cached_x is not None and np.array_equal(x, self.cached_x):
            return
        beta=np.asarray(x,float).reshape(self.nt,self.p)
        assert np.isfinite(beta).all()
        norms=np.array([np.sqrt(np.bincount(self.labels,weights=b*b,minlength=len(self.sizes))+self.eps**2) for b in beta])
        risks=[]
        for obj,b in zip(self.objs,beta):
            eta=obj.X@b
            log_risk=np.logaddexp.accumulate(eta[::-1])[::-1]
            counts=np.bincount(obj.first,weights=obj.event,minlength=obj.n)
            rows=np.flatnonzero(counts)
            valid=np.arange(obj.n)[None,:]>=rows[:,None]
            log_prob=np.where(valid,eta[None,:]-log_risk[rows,None],-np.inf)
            probabilities=np.exp(log_prob)
            assert np.isfinite(probabilities).all()
            weights=counts[rows]
            risks.append((probabilities,weights,weights@probabilities))
        # Commit the cache only after the entire state is valid.
        self.cached_x=np.array(x,float,copy=True)
        self.beta=beta.copy();self.norms=norms;self.risks=risks

    def value_grad(self, x):
        # Objective/gradient evaluation needs no Hessian state. In particular,
        # an oversized trust-region proposal remains evaluable and rejectable.
        beta=np.asarray(x,float).reshape(self.nt,self.p)
        value=0.;grad=np.zeros_like(beta)
        for k,obj in enumerate(self.objs):
            v,g=obj.value_grad(beta[k])
            pv,pg=penalty_value_grad(beta[k],self.labels,self.sizes,eps=self.eps)
            value+=v+self.lam[k]*pv;grad[k]=g+self.lam[k]*pg
        for k in range(self.nt):
            for j in range(k+1,self.nt):
                delta=beta[k]-beta[j]
                value+=.5*self.quadratic_edge*float(delta@delta)
                grad[k]+=self.quadratic_edge*delta;grad[j]-=self.quadratic_edge*delta
        self.nfg+=1
        return float(value),grad.ravel()

    def hessp(self, x, direction):
        self.state(x)
        v=np.asarray(direction,float).reshape(self.nt,self.p);out=np.zeros_like(v)
        dots=np.array([np.bincount(self.labels,weights=b*z,minlength=len(self.sizes)) for b,z in zip(self.beta,v)])
        for k,obj in enumerate(self.objs):
            probabilities,weights,marginal=self.risks[k]
            z=obj.X@v[k]
            hv_eta=(2./obj.n)*(marginal*z-(weights*(probabilities@z))@probabilities)
            out[k]=obj.X.T@hv_eta
            out[k]+=self.lam[k]*(np.sqrt(self.sizes)/self.norms[k])[self.labels]*(v[k]-self.beta[k]*(dots[k]/self.norms[k]**2)[self.labels])
        for k in range(self.nt):
            for j in range(k+1,self.nt):
                delta=self.quadratic_edge*(v[k]-v[j]);out[k]+=delta;out[j]-=delta
        self.nhv+=1
        return out.ravel()
