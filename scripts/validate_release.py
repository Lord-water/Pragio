"""Validate released labels, partition isolation and optional restored text hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def check(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    labels = json.loads((ROOT/'data/labels/label_order.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT/'data/source_manifest.json').read_text(encoding='utf-8'))
    indices = {}
    for split, count in manifest['counts'].items():
        rows = [json.loads(line) for line in (ROOT/f'data/index/{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        check(len(rows) == count, 'Corpus count mismatch: '+split)
        check(len({r['record_id'] for r in rows}) == count, 'Duplicate corpus IDs')
        check(all(r['split'] == split and r['source_member'] == manifest['member'] for r in rows), 'Source/split mismatch')
        indices[split] = {r['record_id']: r for r in rows}
    for i, a in enumerate(indices):
        for b in list(indices)[i+1:]:
            for key in ['record_id','dialogue_id','body_key','group_id']:
                check(not ({r[key] for r in indices[a].values()} & {r[key] for r in indices[b].values()}), 'Split overlap: '+key)
    all_ids = set()
    for split, expected in manifest['labels'].items():
        ann = json.loads((ROOT/f'data/labels/{split}.json').read_text(encoding='utf-8'))
        check(len(ann) == expected['contexts'], 'Annotation count mismatch: '+split)
        check(not (set(ann) & all_ids), 'Annotation partitions overlap')
        all_ids.update(ann)
        corpus_split = split if split in ('train','dev') else 'heldout'
        check(set(ann) <= set(indices[corpus_split]), 'Labels outside assigned corpus partition')
        counts = {'A':0,'N':0,'U':0}
        for row in ann.values():
            check(set(row) == set(labels), 'Candidate mismatch')
            for value in row.values():
                check(value in counts, 'Invalid judgment')
                counts[value] += 1
        check(all(counts[k] == expected[k] for k in counts), 'Judgment count mismatch')
    folds = json.loads((ROOT/'protocols/development_folds.json').read_text(encoding='utf-8'))
    dev = json.loads((ROOT/'data/labels/dev.json').read_text(encoding='utf-8'))
    check(set(folds) == set(dev) and set(folds.values()) == set(range(5)), 'Development folds mismatch')
    restored = []
    for split, index in indices.items():
        p = ROOT/f'data/local/{split}_ready.jsonl'
        if not p.exists():
            continue
        seen = []
        for line in p.read_text(encoding='utf-8').splitlines():
            r = json.loads(line); seen.append(r['record_id'])
            check(r['record_id'] in index, 'Unexpected restored ID')
            ref = index[r['record_id']]
            check(all(r.get(k) == v for k,v in ref.items() if k != 'text_fields_sha256'), 'Restored index mismatch')
            text = {k:r[k] for k in ['history','raw_target','target_body','target_punctuation','target_original']}
            digest = hashlib.sha256(json.dumps(text,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
            check(digest == ref['text_fields_sha256'], 'Restored text mismatch')
        check(seen == list(index), 'Restored record order/count mismatch')
        restored.append(split)
    print(json.dumps({'status':'passed','corpus_records':sum(map(len,indices.values())),
                      'annotated_contexts':len(all_ids),'restored_text_checked':restored}))

if __name__ == '__main__':
    main()
