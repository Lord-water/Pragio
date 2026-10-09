"""Score supplied predictions against explicit A/N/U annotations."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluate_acceptability import metrics
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--annotations',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    scores=json.loads(a.predictions.read_text(encoding='utf-8'))
    if isinstance(scores,list):
        raw=scores;scores={r['record_id']:r['acceptability'] for r in raw}
        if len(raw)!=len(scores):raise ValueError('Duplicate record IDs')
    result=metrics(scores,json.loads(a.annotations.read_text(encoding='utf-8')))
    result.pop('selection_rule',None)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('Saved',a.output)
if __name__=='__main__':main()
