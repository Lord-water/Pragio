"""Select a checkpoint and calibrators using only the released 50-context dev set."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evaluate_acceptability import LABELS,metrics
import development_calibration as shared
import candidate_calibration as candidate
from calibrate import read_scores

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path,nargs='+',required=True,help='Raw development predictions in checkpoint order; JSON dict or list of records')
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    ann=json.loads((ROOT/'data/labels/dev.json').read_text(encoding='utf-8'));ids=sorted(ann)
    foldmap=json.loads((ROOT/'protocols/development_folds.json').read_text(encoding='utf-8'))
    folds=np.array([foldmap[i] for i in ids]);y=np.array([[{'A':1,'N':0,'U':-1}[ann[i][l]] for l in LABELS] for i in ids])
    allowed_ids={json.loads(line)['record_id'] for line in (ROOT/'data/index/dev.jsonl').read_text(encoding='utf-8').splitlines()}
    options=[]
    for f in a.predictions:
        pred=read_scores(f)
        if set(pred)-allowed_ids:raise ValueError('Predictions outside the development partition')
        if set(ids)-set(pred):raise ValueError('Missing development predictions')
        pred={i:pred[i] for i in ids};m=metrics(pred,ann)
        options.append((f,pred,m))
    chosen=min(range(len(options)),key=lambda i:(-options[i][2]['macro_within_label_auc'],options[i][2]['macro_brier'],i))
    f,pred,m=options[chosen];p=np.array([pred[i] for i in ids]);scores={};params={}
    for kind in shared.KINDS:
        out=np.empty_like(p)
        for k in range(5):
            v=folds==k;out[v]=shared.transform(p[v],shared.fit(p[~v],y[~v],kind))
        scores[kind]=shared.loss_metrics(out,y)['macro_brier'];params[kind]=shared.fit(p,y,kind)
    selected_shared=min(scores,key=scores.get)
    candidate_scores={'shared':scores[selected_shared]};candidate_params={'shared':params[selected_shared]}
    for lam in [.1,1.]:
        key='shrinkage_'+str(lam);out=np.empty_like(p)
        for k in range(5):
            v=folds==k;out[v]=candidate.transform(p[v],candidate.fit(p[~v],y[~v],lam))
        candidate_scores[key]=shared.loss_metrics(out,y)['macro_brier'];candidate_params[key]=candidate.fit(p,y,lam)
    selected_candidate=min(candidate_scores,key=candidate_scores.get)
    result={'labels':LABELS,'fit_split':'dev','selected_checkpoint':str(f),'development_metrics':m,'shared':params[selected_shared],'candidate':candidate_params[selected_candidate],'shared_oof_brier':scores,'candidate_oof_brier':candidate_scores,'scope':'Only development labels used. Repeated training is a replication, not a new prospective confirmation.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as out:json.dump(result,out,ensure_ascii=False,indent=2)
    print('Selected',f.name,'shared:',selected_shared,'candidate:',selected_candidate)
if __name__=='__main__':main()
