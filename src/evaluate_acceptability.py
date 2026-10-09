"""Evaluate fixed human labels with identical unknown masks; pure stdlib."""
import argparse
import json
import math
from pathlib import Path

LABELS=['吗','呢','吧','嘛','呗','罢了','啊','哦','呀','啦','么','咯']


def auc(positive,negative):
    if not positive or not negative:return None
    return sum(float(p>n)+0.5*float(p==n) for p in positive for n in negative)/(len(positive)*len(negative))


def metrics(predictions,annotations):
    if not annotations:raise ValueError('No human annotations')
    if set(annotations)-set(predictions):raise ValueError('Missing predictions')
    per_label={};bces=[];briers=[]
    within=[];top_A=top_N=top_U=0
    for rid,labels in annotations.items():
        if set(labels)!=set(LABELS) or any(v not in {'A','N','U'} for v in labels.values()):
            raise ValueError('Complete explicit A/N/U mapping required')
        scores=predictions[rid]
        if len(scores)!=12 or any(not math.isfinite(s) or s<0 or s>1 for s in scores):raise ValueError('Invalid probabilities')
        status=labels[LABELS[max(range(12),key=lambda i:scores[i])]]
        top_A+=status=='A';top_N+=status=='N';top_U+=status=='U'
        value=auc([s for p,s in zip(LABELS,scores) if labels[p]=='A'],
                  [s for p,s in zip(LABELS,scores) if labels[p]=='N'])
        if value is not None:within.append(value)
    for i,label in enumerate(LABELS):
        pairs=[(predictions[rid][i],int(row[label]=='A')) for rid,row in annotations.items() if row[label]!='U']
        pos=[p for p,y in pairs if y];neg=[p for p,y in pairs if not y]
        brier=sum((p-y)**2 for p,y in pairs)/len(pairs) if pairs else None
        bce=sum(-y*math.log(max(p,1e-7))-(1-y)*math.log(max(1-p,1e-7)) for p,y in pairs)/len(pairs) if pairs else None
        per_label[label]={'A':len(pos),'N':len(neg),'U':len(annotations)-len(pairs),
                          'auc':auc(pos,neg),'brier':brier,'bce':bce}
        if brier is not None:briers.append(brier);bces.append(bce)
    aucs=[r['auc'] for r in per_label.values() if r['auc'] is not None]
    n=len(annotations)
    return {'cases':n,'macro_within_label_auc':sum(aucs)/len(aucs) if aucs else None,
            'auc_labels':len(aucs),'auc_excluded_labels':[p for p,r in per_label.items() if r['auc'] is None],
            'macro_brier':sum(briers)/len(briers) if briers else None,
            'macro_bce':sum(bces)/len(bces) if bces else None,
            'within_sentence_pairwise_accuracy':sum(within)/len(within) if within else None,
            'within_sentence_eligible_cases':len(within),'per_label':per_label,
            'top1':{'A':top_A,'N':top_N,'U':top_U,'known_coverage':(top_A+top_N)/n,
                    'accepted_among_known':top_A/(top_A+top_N) if top_A+top_N else None,
                    'acceptability_lower_bound':top_A/n,'acceptability_upper_bound':(top_A+top_U)/n},
            'selection_rule':'higher macro_within_label_auc; exact ties lower macro_brier; no statistical significance inferred from development metrics'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--predictions',required=True)
    ap.add_argument('--annotations',required=True)
    ap.add_argument('--dev-records',required=True)
    ap.add_argument('--train-annotations',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    annotations=json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    train=json.loads(Path(args.train_annotations).read_text(encoding='utf-8'))
    dev=[json.loads(line) for line in Path(args.dev_records).read_text(encoding='utf-8').splitlines()]
    if any(r['split']!='dev' for r in dev):raise ValueError('Only development evaluation is supported by this script')
    ids={r['record_id'] for r in dev}
    if set(annotations)-ids or set(train)&set(annotations):raise ValueError('Wrong split or training labels reused as dev')
    raw=json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    predictions={r['record_id']:r['acceptability'] for r in raw}
    if len(raw)!=len(predictions) or set(predictions)-ids:raise ValueError('Duplicate/wrong-split predictions')
    prior=[]
    for p in LABELS:
        known=[r[p] for r in train.values() if r[p]!='U']
        prior.append((known.count('A')+1)/(len(known)+2))
    result={'model':metrics(predictions,annotations),
            'constant_train_prior':metrics({rid:prior for rid in annotations},annotations),
            'constant_prior_probabilities':dict(zip(LABELS,prior)),
            'scope':'independent development labels for selection; not final test results'}
    with Path(args.output).open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result['model'].items() if k!='per_label'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
