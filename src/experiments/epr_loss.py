"""EPR adaptation of Cole et al. CVPR 2021, with the author's reduction convention.

Reference: https://github.com/elijahcole/single-positive-multi-label/blob/main/losses.py
Human-label BCE is added by our runner; this is not a reproduction of image results.
"""
import torch
def epr_loss(logits,observed,expected_num_pos):
 keep=observed>=0
 if not keep.any():return logits.sum()*0
 p=logits[keep].sigmoid();nlabels=p.shape[1]
 if not 0<=expected_num_pos<=nlabels:raise ValueError('Invalid expected label count')
 positive=p.gather(1,observed[keep,None]).squeeze(1)
 positive_loss=-torch.log(positive+1e-5).mean()/nlabels
 regularizer=((p.sum(1).mean()-expected_num_pos)/nlabels)**2
 return positive_loss+regularizer


def expected_count_from_annotations(annotations, labels):
    """Sum training-only Laplace-smoothed A rates; U does not enter denominators."""
    if not annotations:
        raise ValueError('Training annotations are required')
    for row in annotations.values():
        if set(row) != set(labels) or any(v not in ('A', 'N', 'U') for v in row.values()):
            raise ValueError('Expected complete A/N/U candidate judgments')
    return sum((1 + sum(row[label] == 'A' for row in annotations.values())) /
               (2 + sum(row[label] != 'U' for row in annotations.values()))
               for label in labels)
