"""Candidate-offset calibration from the frozen research implementation."""
import numpy as np
from development_calibration import logit,sigmoid,transform as global_transform
from evaluate_acceptability import LABELS

def design(p,y):
 known=y>=0;counts=known.sum(0);eligible=(y==1).sum(0)>=5;eligible&=(y==0).sum(0)>=5
 jj=np.flatnonzero(eligible);r,c=np.where(known);x=np.zeros((len(r),2+len(jj)));x[:,0]=logit(p)[r,c];x[:,1]=1
 for k,j in enumerate(jj):x[:,k+2]=(c==j)
 w=1/counts[c]/(counts>0).sum()
 return x,y[r,c],w,jj

def objective(theta,x,y,w,lam):
 z=x@theta;v=sigmoid(z);e=max(1,len(theta)-2)
 value=float(np.sum(w*(np.logaddexp(0,z)-y*z))+.5*lam*np.sum(theta[2:]**2)/e)
 grad=x.T@(w*(v-y));grad[2:]+=lam*theta[2:]/e
 hess=x.T@((w*v*(1-v))[:,None]*x);hess[2:,2:]+=np.eye(len(theta)-2)*lam/e
 return value,grad,hess

def fit(p,y,lam):
 x,target,w,jj=design(p,y);theta=np.zeros(x.shape[1]);theta[0]=1
 lo=np.full(len(theta),-2000.);hi=np.full(len(theta),2000.);lo[0]=1e-4;hi[0]=100
 for it in range(200):
  value,grad,hess=objective(theta,x,target,w,lam)
  active=((theta<=lo+1e-10)&(grad>0))|((theta>=hi-1e-10)&(grad<0));pg=grad.copy();pg[active]=0
  if np.max(np.abs(pg))<1e-8:break
  free=~active;direction=np.zeros(len(theta));direction[free]=np.linalg.solve(hess[np.ix_(free,free)]+np.eye(free.sum())*1e-12,-grad[free])
  accepted=False
  for step in 2.**(-np.arange(45)):
   trial=np.clip(theta+step*direction,lo,hi);delta=trial-theta
   if objective(trial,x,target,w,lam)[0]<=value+1e-4*grad@delta+1e-14:
    theta=trial;accepted=True;break
  assert accepted,'Newton line search failed'
 else:raise RuntimeError('Calibration not converged')
 _,grad,_=objective(theta,x,target,w,lam);pg=theta-np.clip(theta-grad,lo,hi)
 assert max(abs(pg))<1e-7
 offsets=np.zeros(p.shape[1]);offsets[jj]=theta[2:]
 return {'kind':'candidate_shrinkage','lambda':lam,'slope':float(theta[0]),'intercept':float(theta[1]),'offsets':offsets.tolist(),'eligible_labels':[LABELS[j] for j in jj],'iterations':it+1,'projected_gradient_max':float(max(abs(pg)))}

def transform(p,param):
 if param['kind']!='candidate_shrinkage':return global_transform(p,param)
 return sigmoid(param['slope']*logit(p)+param['intercept']+np.array(param['offsets']))

def checks():
 rng=np.random.default_rng(42);p=rng.uniform(.05,.95,(30,12));y=rng.integers(0,2,(30,12)).astype(float);y[0]=-1;y[:,11]=-1;y[:4,11]=1
 x,target,w,jj=design(p,y);theta=rng.normal(0,.3,x.shape[1]);theta[0]=.8;lam=.1
 value,g,H=objective(theta,x,target,w,lam);eps=1e-5
 for k in range(len(theta)):
  d=np.eye(len(theta))[k]*eps
  assert abs((objective(theta+d,x,target,w,lam)[0]-objective(theta-d,x,target,w,lam)[0])/(2*eps)-g[k])<1e-7
  assert np.max(abs((objective(theta+d,x,target,w,lam)[1]-objective(theta-d,x,target,w,lam)[1])/(2*eps)-H[:,k]))<1e-7
 a=fit(p,y,lam);p2=p.copy();p2[y<0]=.99;b=fit(p2,y,lam)
 assert a==b and a['offsets'][11]==0
 assert np.isfinite(transform(p,a)).all()
 return {'analytic_gradient_hessian':'finite-difference passed','unknown_mask_invariance':'passed','sparse_label_offset_fallback':'passed'}
