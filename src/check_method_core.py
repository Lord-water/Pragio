"""Small CPU checks for scientific implementation errors; no research results."""
import json
import torch
from torch.nn import functional as F
from method_core import ParticleHeads, joint_loss


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261001)
    checks = []
    head = ParticleHeads(5, 12).double()
    x = torch.randn(3, 5, dtype=torch.double)
    o = head(x)
    assert torch.allclose(o['choice_log_probs'].exp().sum(-1), torch.ones(3, dtype=torch.double))
    checks.append('normalized choice probabilities')
    before = o['choice_log_probs'].detach().clone()
    with torch.no_grad():
        head.preference.add_(100)
    assert torch.allclose(head(x)['choice_log_probs'], before)
    checks.append('preference common-shift invariance')

    a = torch.tensor([[0.1, 0.3, -0.5]], dtype=torch.double, requires_grad=True)
    output = {'acceptability_logits': a, 'choice_log_probs': F.log_softmax(F.logsigmoid(a), -1)}
    result = joint_loss(output, human=torch.tensor([[1., -1., 0.]]))
    result['loss'].backward()
    assert a.grad[0, 0] < 0 and a.grad[0, 1] == 0 and a.grad[0, 2] > 0
    assert result['human_count'] == 2
    checks.append('unknown human label excluded; positive and negative gradient directions')

    output = head(x)
    empty = joint_loss(output, observed=torch.full((3,), -1), human=torch.full((3, 12), -1))
    assert empty['loss'].item() == 0 and torch.isfinite(empty['loss'])
    checks.append('all-unknown supervision finite zero')

    a = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    b = torch.randn(3, dtype=torch.double, requires_grad=True)
    def objective(a, b):
        output = {'acceptability_logits': a,
                  'choice_log_probs': F.log_softmax(F.logsigmoid(a) + b - b.mean(), -1)}
        return joint_loss(output, torch.tensor([0, 2]), torch.tensor([[1., -1., 0.], [-1., 0., 1.]]))['loss']
    assert torch.autograd.gradcheck(objective, (a, b))
    checks.append('finite-difference joint gradient check')
    for mode in ['coupled', 'no_preference', 'independent']:
        model = ParticleHeads(5, 12, mode).double()
        output = model(x)
        human = torch.full((3, 12), -1.)
        human[:, 1] = 0
        human[:, 2] = 1
        loss = joint_loss(output, torch.tensor([2, 2, 2]), human)['loss']
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    checks.append('all comparison heads receive finite gradients')
    print(json.dumps({'status': 'passed', 'device': 'cpu', 'torch': torch.__version__, 'checks': checks,
                      'research_claim': 'none; no human labels or actual training used'}, indent=2))


if __name__ == '__main__':
    main()
