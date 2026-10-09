"""Fit on user-supplied development data, or apply an already fixed calibrator."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluate_acceptability import LABELS
from development_calibration import fit as shared_fit
from candidate_calibration import fit as candidate_fit, transform

def read_scores(path):
    raw = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(raw, list):
        result = {r['record_id']: r['acceptability'] for r in raw}
        if len(result) != len(raw): raise ValueError('Duplicate record IDs')
    else: result = raw
    if not isinstance(result, dict) or not result: raise ValueError('Expected nonempty score mapping')
    values = np.asarray(list(result.values()), dtype=float)
    if values.shape != (len(result), len(LABELS)) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Expected 12 finite probabilities per record')
    return result

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('--dev-predictions', type=Path, required=True)
    fit.add_argument('--dev-annotations', type=Path, required=True)
    fit.add_argument('--dev-records', type=Path, required=True)
    fit.add_argument('--kind', choices=['raw','temperature','affine_sigmoid','candidate_shrinkage'], required=True)
    fit.add_argument('--lambda', dest='regularization', type=float, default=1.)
    fit.add_argument('--output', type=Path, required=True)
    apply = sub.add_parser('apply')
    apply.add_argument('--predictions', type=Path, required=True)
    apply.add_argument('--parameters', type=Path, required=True)
    apply.add_argument('--family', choices=['shared','candidate'], help='Required for selection bundles')
    apply.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.command == 'fit':
        scores = read_scores(args.dev_predictions)
        annotations = json.loads(args.dev_annotations.read_text(encoding='utf-8'))
        records = [json.loads(line) for line in args.dev_records.read_text(encoding='utf-8').splitlines() if line.strip()]
        ids = {r['record_id'] for r in records}
        if not records or len(ids) != len(records) or any(r['split'] != 'dev' for r in records):
            raise ValueError('Unique development records are required')
        if not annotations or set(annotations)-ids or set(scores)-ids or set(annotations)-set(scores):
            raise ValueError('Annotations/predictions must match the development partition')
        rows = list(annotations)
        for row in annotations.values():
            if set(row) != set(LABELS) or any(v not in ['A','N','U'] for v in row.values()):
                raise ValueError('Explicit A/N/U labels for all 12 candidates are required')
        p = np.array([scores[r] for r in rows], dtype=float)
        mapping = {'A':1.,'N':0.,'U':-1.}
        y = np.array([[mapping[annotations[r][label]] for label in LABELS] for r in rows])
        if not (y == 1).any() or not (y == 0).any(): raise ValueError('Both positive and negative judgments are required')
        if args.regularization <= 0: raise ValueError('Shrinkage regularization must be positive')
        params = candidate_fit(p,y,args.regularization) if args.kind == 'candidate_shrinkage' else shared_fit(p,y,args.kind)
        result = {'labels':LABELS,'parameters':params,'fit_split':'dev'}
    else:
        scores = read_scores(args.predictions)
        saved = json.loads(args.parameters.read_text(encoding='utf-8'))
        if saved['labels'] != LABELS: raise ValueError('Candidate order mismatch')
        if 'parameters' in saved:
            params = saved['parameters']
        else:
            if not args.family: ap.error('--family is required for a selection bundle')
            params = saved[args.family]
        values = transform(np.asarray(list(scores.values()),dtype=float),params)
        result = {r:v for r,v in zip(scores,values.tolist())}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print('Saved', args.output)

if __name__ == '__main__': main()
