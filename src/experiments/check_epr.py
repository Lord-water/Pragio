"""Synthetic value, gradient and masking checks against a scalar EPR formula."""
import torch
from epr_loss import epr_loss, expected_count_from_annotations
for n in [1,4,16]:
    torch.manual_seed(71+n)
    z=torch.randn(n,12,dtype=torch.float64,requires_grad=True); obs=torch.arange(n)%12
    for k in [1.,6.,12.]:
        actual=epr_loss(z,obs,k)
        positive=sum(-torch.log(torch.sigmoid(z[i,obs[i]])+1e-5) for i in range(n))/(n*12)
        count=sum(torch.sigmoid(z[i,j]) for i in range(n) for j in range(12))/n
        expected=positive+((count-k)/12)**2
        ga=torch.autograd.grad(actual,z,retain_graph=True)[0]
        ge=torch.autograd.grad(expected,z,retain_graph=True)[0]
        assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
        assert torch.allclose(ga,ge,atol=1e-12,rtol=1e-12)
extra=torch.randn(2,12,dtype=torch.float64,requires_grad=True)
zz=torch.cat([z.detach(),extra]); obs2=torch.cat([obs,torch.tensor([-1,-1])])
assert torch.allclose(epr_loss(zz,obs2,k),epr_loss(z,obs,k))
assert torch.count_nonzero(torch.autograd.grad(epr_loss(zz,obs2,k),extra)[0])==0
assert expected_count_from_annotations({'x':{'a':'A','b':'U'},'y':{'a':'N','b':'N'}},['a','b'])==.5+1/3
print('EPR scalar value/gradient, U-mask and training-prior checks passed')
