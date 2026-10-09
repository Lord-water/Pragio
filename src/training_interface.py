"""Shared input/encoder interface; no training is launched by importing this file."""
import json
from pathlib import Path
import torch
from torch import nn
from method_core import ParticleHeads, joint_loss, an_ls_natural_loss

LABELS = ['吗', '呢', '吧', '嘛', '呗', '罢了', '啊', '哦', '呀', '啦', '么', '咯']


def load_records(path, allowed_split):
    records = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records or any(r['split'] != allowed_split for r in records):
        raise ValueError('Wrong or empty split: ' + str(path))
    if len({r['record_id'] for r in records}) != len(records):
        raise ValueError('Duplicate record ids')
    return records


def attach_human_labels(records, annotations):
    """annotations is a reviewed record_id -> label -> A/N/U mapping.

    Raw independent annotator responses need adjudication/prespecified treatment
    before this function; it must not silently resolve disagreements.
    """
    ids = {r['record_id'] for r in records}
    if set(annotations) - ids:
        raise ValueError('Human labels contain records outside this split')
    values = {'A': 1.0, 'N': 0.0, 'U': -1.0}
    rows = []
    for record in records:
        labels = annotations.get(record['record_id'], {})
        if set(labels) - set(LABELS) or any(v not in values for v in labels.values()):
            raise ValueError('Invalid annotation labels')
        row = dict(record)
        row['human_labels'] = [values[labels[p]] if p in labels else -1.0 for p in LABELS]
        row['observed_index'] = LABELS.index(record['observed_particle'])
        if labels.get(record['observed_particle']) == 'N':
            row['observed_index'] = -1
            row['observed_human_conflict'] = True
        rows.append(row)
    return rows


def collate_records(records, tokenizer, max_length=256):
    if not tokenizer.mask_token or not tokenizer.sep_token:
        raise ValueError('This interface requires a BERT-style tokenizer')
    histories = [(' ' + tokenizer.sep_token + ' ').join(r['history']) for r in records]
    targets = [r['target_body'] + tokenizer.mask_token + r['target_punctuation'] for r in records]
    # Keeping the target sequence is intentional. Only earlier context may be cut.
    encoded = tokenizer(histories, targets, padding=True, truncation='only_first',
                        max_length=max_length, return_tensors='pt')
    observed = torch.tensor([r.get('observed_index', LABELS.index(r['observed_particle'])) for r in records], dtype=torch.long)
    human = torch.tensor([r.get('human_labels', [-1.] * len(LABELS)) for r in records], dtype=torch.float32)
    return encoded, observed, human


class ContextParticleModel(nn.Module):
    def __init__(self, encoder, mode='coupled'):
        super().__init__()
        self.encoder = encoder
        self.heads = ParticleHeads(encoder.config.hidden_size, len(LABELS), mode)
        self.mode = mode

    def forward(self, **encoded):
        output = self.encoder(**encoded)
        return self.heads(output.last_hidden_state[:, 0])

    def loss(self, output, observed, human, human_weight=1.0, smoothing=0.1):
        if self.mode == 'an_ls':
            human_result = joint_loss(output, human=human, human_weight=human_weight)
            natural = an_ls_natural_loss(output['acceptability_logits'], observed, smoothing)
            return natural + human_result['loss']
        if self.mode == 'single':
            # Reference-only original-label CE; not the matched-human-budget control.
            return joint_loss(output, observed=observed)['loss']
        return joint_loss(output, observed, human, human_weight)['loss']
