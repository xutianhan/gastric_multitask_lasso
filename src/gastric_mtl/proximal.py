"""Accelerated proximal solver for the unchanged smooth convex ablation.

The radial proximal map retains epsilon=1e-4, so this does not silently switch
from the frozen smooth group penalty to an exact sparse Group Lasso objective.
"""
import time
import numpy as np
from gastric_mtl.cox import CoxObjective,_group_info,penalty_value_grad

def smooth_group_prox(v,labels,sizes,lam_over_l,eps=1e-4):
    norms=np.sqrt(np.bincount(labels,weights=v*v,minlength=len(sizes)))
    tau=lam_over_l*np.sqrt(sizes);lo=np.zeros_like(norms);hi=norms.copy()
    radius=np.maximum(norms-tau,0.)
    for _ in range(35):
        den=np.sqrt(radius*radius+eps*eps)
        residual=radius+tau*radius/den-norms
        high=residual>0;hi=np.where(high,radius,hi);lo=np.where(high,lo,radius)
        derivative=1+tau*eps*eps/den**3
        proposed=radius-residual/derivative
        invalid=(proposed<lo)|(proposed>hi)|~np.isfinite(proposed)
        proposed=np.where(invalid,(lo+hi)*.5,proposed)
        if np.max(np.abs(proposed-radius))<1e-15:
            radius=proposed;break
        radius=proposed
    ratio=np.divide(radius,norms,out=np.zeros_like(norms),where=norms>0)
    return v*ratio[labels]

def fit_proximal_quadratic(tasks,groups,lam,edge=0.,beta0=None,max_iter=10000,gtol=1e-5):
    objs=[CoxObjective(**t) for t in tasks];nt=len(objs);p=objs[0].p
    labels,sizes=_group_info(groups,p);lam=np.asarray(lam,float)
    x=np.zeros((nt,p)) if beta0 is None else np.asarray(beta0,float).copy()
    def smooth(b):
        value=0.;g=np.zeros_like(b)
        for k,o in enumerate(objs):v,gg=o.value_grad(b[k]);value+=v;g[k]=gg
        for k in range(nt):
            for j in range(k+1,nt):
                dif=b[k]-b[j];value+=.5*edge*float(dif@dif);g[k]+=edge*dif;g[j]-=edge*dif
        return value,g
    def penalty(b):
        value=0.;g=np.zeros_like(b)
        for k in range(nt):v,gg=penalty_value_grad(b[k],labels,sizes);value+=lam[k]*v;g[k]=lam[k]*gg
        return value,g
    started=time.perf_counter();fx,gx=smooth(x);px,pg=penalty(x);value=fx+px;initial=value
    y=x.copy();moment=1.;L=max(1.,edge*nt);history=[];n_eval=1;gradient=float(np.max(np.abs(gx+pg)))
    if gradient<=gtol:return {'betas':list(x),'objective':value,'initial_objective':initial,'gradient_norm':gradient,'converged':True,'n_iter':0,'n_eval':n_eval,'seconds':time.perf_counter()-started,'history':history}
    for iteration in range(1,max_iter+1):
        fy,gy=smooth(y);n_eval+=1
        while True:
            proposal=np.array([smooth_group_prox(y[k]-gy[k]/L,labels,sizes,lam[k]/L) for k in range(nt)])
            delta=proposal-y;fp,gp=smooth(proposal);n_eval+=1
            if fp<=fy+float(np.sum(gy*delta))+.5*L*float(np.sum(delta*delta))+1e-12:break
            L*=2
        pp,pg=penalty(proposal);nextvalue=fp+pp
        if nextvalue>value+1e-12 and moment>1.:
            y=x.copy();moment=1.;continue
        old=x;x=proposal;value=nextvalue;gradient=float(np.max(np.abs(gp+pg)))
        if iteration%100==0 or gradient<=gtol:
            history.append({'iteration':iteration,'objective':value,'gradient_norm':gradient,'L':L,'seconds':time.perf_counter()-started})
        if gradient<=gtol:break
        nextmoment=.5*(1+np.sqrt(1+4*moment*moment))
        if np.sum((y-x)*(x-old))>0:nextmoment=1.
        y=x+(moment-1)/nextmoment*(x-old) if nextmoment>1 else x.copy();moment=nextmoment
    assert np.isfinite(value) and value<=initial+1e-8
    return {'betas':list(x),'objective':value,'initial_objective':initial,'gradient_norm':gradient,'converged':gradient<=gtol,'n_iter':iteration,'n_eval':n_eval,'seconds':time.perf_counter()-started,'history':history}
