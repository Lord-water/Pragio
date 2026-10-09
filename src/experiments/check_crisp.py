import json,math
from pathlib import Path
import numpy as np
import torch
from crisp_core import estimate_priors,crisp_risk
rng=np.random.default_rng(31415)
for n in [24,101,301]:
 scores=np.round(rng.random((n,4)),1);obs=np.arange(n)%4
 pi,detail=estimate_priors(scores,obs)
 for j in range(4):
  p=scores[obs==j,j];pen=1.01*(math.sqrt(math.log(400)/(2*n))+math.sqrt(math.log(400)/(2*len(p))))
  candidates=[]
  for z in sorted(set(scores[:,j]),reverse=True):
   q=(scores[:,j]>=z).mean();qp=(p>=z).mean()
   if qp:candidates.append(((q+pen)/qp,q/qp,z))
  best=min(candidates,key=lambda a:a[0]);assert abs(pi[j]-best[1])<1e-12
allone,_=estimate_priors(np.ones((24,4))*.5,np.arange(24)%4);assert np.array_equal(allone,np.ones(4))
try:estimate_priors(np.ones((24,4))*.5,np.zeros(24));raise AssertionError('Missing class accepted')
except ValueError:pass
for shift in [0.,1.]:
 a=torch.tensor(rng.normal(size=(7,4)),dtype=torch.float64,requires_grad=True)
 b=torch.tensor(rng.normal(size=(3,4)),dtype=torch.float64,requires_grad=True);p=torch.tensor([.12,.4,.8,.95],dtype=torch.float64)
 ours=crisp_risk(a,b,p,shift)
 ref=sum(2*p[j]*sum(1-torch.sigmoid(b[i,j]+shift*p[j]) for i in range(3))/3+abs(sum(torch.sigmoid(a[i,j]) for i in range(7))/7-p[j]) for j in range(4))/4
 assert abs(float(ours-ref))<1e-12
 go=torch.autograd.grad(ours,(a,b),retain_graph=True);gr=torch.autograd.grad(ref,(a,b));assert all(torch.allclose(x,y,atol=1e-12,rtol=1e-12) for x,y in zip(go,gr))
 assert torch.autograd.gradcheck(lambda x,y:crisp_risk(x,y,p,shift),(a,b))
out={'status':'passed','checks':['Eq2 agrees with independent exhaustive thresholds including ties','constant scores imply prior one','missing-positive class rejected','Eq7/Eq8 loss and gradients match scalar reference','finite-difference gradient check'], 'test_labels_read':False}
print(json.dumps(out))
