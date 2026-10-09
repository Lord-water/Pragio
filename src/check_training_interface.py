"""CPU path/gradient checks on explicitly synthetic tiny-BERT examples."""
import json
import tempfile
from pathlib import Path
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast
from training_interface import ContextParticleModel, collate_records, attach_human_labels, LABELS
from method_core import an_ls_natural_loss


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261002)
    records = [
        {'record_id': 'synthetic-a', 'history': ['现在出发吗'], 'target_body': '我们一起去', 'target_punctuation': '。', 'observed_particle': '吧'},
        {'record_id': 'synthetic-b', 'history': ['这件事已经完成了'], 'target_body': '你已经知道了', 'target_punctuation': '？', 'observed_particle': '吗'},
    ]
    annotations = {'synthetic-a': {'吧': 'A', '吗': 'N'}, 'synthetic-b': {'吗': 'N', '呢': 'A'}}
    rows = attach_human_labels(records, annotations)
    assert rows[1]['observed_index'] == -1
    checks = ['human/original conflict disables that weak label']
    try:
        attach_human_labels(records, {'out-of-split': {'吧': 'A'}})
        raise AssertionError('did not reject cross-split label')
    except ValueError:
        checks.append('cross-split annotation rejected')
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + sorted(set('现在出发吗我们一起去。这件事已经完成了你知道？吧呢'))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'vocab.txt'
        path.write_text('\n'.join(vocab), encoding='utf-8')
        tok = BertTokenizerFast(vocab_file=str(path), do_lower_case=False)
        encoded, observed, human = collate_records(rows, tok, max_length=64)
        assert (encoded['input_ids'] == tok.mask_token_id).sum().item() == 2
        # Actual input strings never contain the hidden observed target particle.
        checks.append('paired context/target tokenization and one target mask per example')
        for mode in ['coupled', 'no_preference', 'independent', 'single', 'an_ls']:
            cfg = BertConfig(vocab_size=len(vocab), hidden_size=24, num_hidden_layers=1, num_attention_heads=4, intermediate_size=32)
            model = ContextParticleModel(BertModel(cfg, add_pooling_layer=False), mode)
            out = model(**encoded)
            assert out['acceptability_logits'].shape == (2, len(LABELS))
            loss = model.loss(out, observed, human)
            loss.backward()
            assert torch.isfinite(loss) and model.encoder.embeddings.word_embeddings.weight.grad.abs().sum() > 0
            checks.append(mode + ': finite CPU forward/backward through encoder')
    z = torch.tensor([[0.2, -0.8, 1.1]], dtype=torch.double)
    p = z.sigmoid()
    y = torch.tensor([[0.9, 0.1, 0.1]], dtype=torch.double)
    ref = -(y * p.log() + (1-y) * (1-p).log()).mean()
    assert torch.allclose(an_ls_natural_loss(z, torch.tensor([0]), .1), ref)
    checks.append('AN-LS stable formula matches smoothed Bernoulli reference')
    print(json.dumps({'status': 'passed', 'scope': 'synthetic CPU interface test, not research evidence', 'checks': checks}, indent=2))


if __name__ == '__main__': main()
