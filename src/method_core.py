"""Pragio heads and losses for separate usage and acceptability supervision."""
import torch
from torch import nn
from torch.nn import functional as F


class ParticleHeads(nn.Module):
    """Identical feature inputs for coupled and independent two-task controls."""

    def __init__(self, hidden_size, num_labels, mode='coupled'):
        super().__init__()
        if mode not in {'coupled', 'no_preference', 'independent', 'single', 'an_ls'}:
            raise ValueError(mode)
        self.mode = mode
        self.acceptability = nn.Linear(hidden_size, num_labels)
        if mode == 'coupled':
            self.preference = nn.Parameter(torch.zeros(num_labels))
        elif mode == 'independent':
            self.choice = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        a_logits = self.acceptability(features)
        if self.mode == 'independent':
            choice_logits = self.choice(features)
        elif self.mode in {'single', 'an_ls'}:
            choice_logits = a_logits
        else:
            choice_logits = F.logsigmoid(a_logits)
            if self.mode == 'coupled':
                choice_logits = choice_logits + self.preference - self.preference.mean()
        return {'acceptability_logits': a_logits,
                'choice_log_probs': F.log_softmax(choice_logits, dim=-1)}


def joint_loss(output, observed=None, human=None, human_weight=1.0):
    """observed: [B] index or -1; human: [B,L], 1=A, 0=N, -1=U/unrated.

    Unknown labels never become negatives. Supervision-free batches should not
    be used as training steps. Dataset weights are development choices.
    """
    logits = output['acceptability_logits']
    zero = logits.sum() * 0
    natural_loss, human_loss = zero, zero
    natural_count = human_count = 0
    if human_weight < 0:
        raise ValueError('human_weight must be nonnegative')
    if observed is not None:
        if observed.shape != logits.shape[:1]:
            raise ValueError('observed shape mismatch')
        if not torch.all((observed >= -1) & (observed < logits.shape[1])):
            raise ValueError('invalid observed index')
        known = observed >= 0
        natural_count = int(known.sum())
        if natural_count:
            natural_loss = F.nll_loss(output['choice_log_probs'][known], observed[known])
    if human is not None:
        if human.shape != logits.shape:
            raise ValueError('human shape mismatch')
        if not torch.all((human == -1) | (human == 0) | (human == 1)):
            raise ValueError('human labels must be -1, 0, 1')
        known = human >= 0
        human_count = int(known.sum())
        if human_count:
            human_loss = F.binary_cross_entropy_with_logits(logits[known], human[known].to(logits.dtype))
    return {'loss': natural_loss + human_weight * human_loss,
            'natural_loss': natural_loss, 'human_loss': human_loss,
            'natural_count': natural_count, 'human_count': human_count}


def an_ls_natural_loss(logits, observed, smoothing=0.1):
    """AN-LS natural-label comparator, Cole et al. CVPR 2021.

    Uses smoothed targets 1-smoothing and smoothing, matching the authors'
    loss_an_ls definition (not epsilon/2). Stable logits implementation;
    additional human BCE is our matched-supervision extension, not original AN-LS.
    https://github.com/elijahcole/single-positive-multi-label/blob/main/losses.py
    """
    if not 0 <= smoothing < 0.5:
        raise ValueError('smoothing must be in [0, 0.5)')
    keep = observed >= 0
    if not keep.any(): return logits.sum() * 0
    targets = torch.full_like(logits[keep], smoothing)
    targets.scatter_(1, observed[keep, None], 1.0 - smoothing)
    return F.binary_cross_entropy_with_logits(logits[keep], targets)
