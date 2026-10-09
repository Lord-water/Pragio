"""CRISP equations (2), (7), (8); task-specific sampling lives in the runner."""
import math
import numpy as np
import torch

def estimate_priors(scores, observed, delta=.01, tau=.01):
 scores=np.asarray(scores,dtype=float);observed=np.asarray(observed)
 assert scores.ndim==2 and len(observed)==len(scores) and np.isfinite(scores).all()
 assert ((scores>=0)&(scores<=1)).all() and 0<delta<1 and 0<=tau<=1
 n,c=scores.shape;result=[];details=[]
 for j in range(c):
  pos=observed==j;np_=int(pos.sum())
  if np_==0:raise ValueError('No observed positive for class '+str(j))
  # Ties must be included together because the paper uses score >= threshold.
  order=np.argsort(-scores[:,j],kind='stable');v=scores[order,j]
  ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),n-1]
  q=(ends+1)/n;qp=np.cumsum(pos[order])[ends]/np_;keep=qp>0
  q,qp,ends=q[keep],qp[keep],ends[keep]
  penalty=(1+tau)*(math.sqrt(math.log(4/delta)/(2*n))+math.sqrt(math.log(4/delta)/(2*np_)))
  ucb=(q+penalty)/qp;best=int(np.argmin(ucb));raw=float(q[best]/qp[best]);pi=min(1.,max(0.,raw))
  result.append(pi);details.append({'label_index':j,'n':n,'positive_n':np_,'threshold':float(v[ends[best]]),
   'q':float(q[best]),'q_positive':float(qp[best]),'ucb':float(ucb[best]),'raw_prior':raw,'prior':pi})
 return np.array(result),details

def crisp_risk(natural_logits, positive_logits, priors, shift=0.):
 """Each column in positive_logits contains uniform positive samples of that class.
 Natural rows are sampled from the population, not the positive-stratified pool.
 Divide the paper's sum by C to match per-label human BCE scaling.
 """
 assert natural_logits.ndim==2 and positive_logits.ndim==2
 assert natural_logits.shape[1]==positive_logits.shape[1]==len(priors)
 p=priors.to(device=natural_logits.device,dtype=natural_logits.dtype)
 positive=(1-torch.sigmoid(positive_logits+shift*p)).mean(0)
 marginal=torch.sigmoid(natural_logits).mean(0)
 return (2*p*positive+(marginal-p).abs()).mean()
