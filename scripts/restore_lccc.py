"""Rehydrate fixed study inputs from a lawfully obtained LCCC-large archive.

The released index contains no dialogue text. Each reconstructed record is checked
against its text-field hash. This restores selected rows, not a new random split.
"""
import argparse,hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from corpus_io import stream_array,detokenize,extract

def rehydrate(index,dialogue):
    r=dict(index);expected=r.pop('text_fields_sha256');t=r['turn_index']
    original=dialogue[t];fields,reason=extract(original)
    if reason or fields['observed_particle']!=r['observed_particle']:raise ValueError('Source version/extraction mismatch: '+r['record_id'])
    fields['history']=[detokenize(v) for v in dialogue[max(0,t-3):t]]
    fields['raw_target']=original
    text={k:fields[k] for k in ['history','raw_target','target_body','target_punctuation','target_original']}
    actual=hashlib.sha256(json.dumps(text,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    if actual!=expected:raise ValueError('Reconstructed text mismatch: '+r['record_id'])
    r.update(fields);return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True,help='LCCC-large ZIP containing LCCD.json, or the extracted LCCD.json')
    ap.add_argument('--output',type=Path,default=ROOT/'data/local');args=ap.parse_args()
    for split in ['train','dev','heldout']:
        if (args.output/(split+'_ready.jsonl')).exists():
            raise FileExistsError('Refusing to overwrite restored inputs')
    indices={s:[json.loads(l) for l in (ROOT/'data/index'/(s+'.jsonl')).read_text(encoding='utf-8').splitlines()] for s in ['train','dev','heldout']}
    wanted={}
    for rows in indices.values():
        for r in rows:wanted.setdefault(r['dialogue_index'],[]).append(r)
    restored={}
    def consume(stream):
        for n,d in enumerate(stream_array(stream)):
            if n in wanted:
                for r in wanted.pop(n):restored[r['record_id']]=rehydrate(r,d)
            if not wanted:break
        if wanted:raise ValueError('Source lacks requested dialogue indices')
    if zipfile.is_zipfile(args.source):
        with zipfile.ZipFile(args.source) as z,z.open('LCCD.json') as raw,io.TextIOWrapper(raw,encoding='utf-8') as stream:consume(stream)
    else:
        with args.source.open(encoding='utf-8') as stream:consume(stream)
    args.output.mkdir(parents=True,exist_ok=True)
    for split,rows in indices.items():
        dest=args.output/(split+'_ready.jsonl')
        with dest.open('x',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(restored[r['record_id']],ensure_ascii=False)+'\n')
    print(json.dumps({s:len(v) for s,v in indices.items()}))
if __name__=='__main__':main()
