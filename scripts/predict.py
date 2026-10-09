"""CPU float32 prediction from an explicitly supplied, trusted local checkpoint."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--batch-size',type=int,default=16)
    a=ap.parse_args()
    import torch
    from transformers import AutoModel,AutoTokenizer
    from training_interface import ContextParticleModel
    torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);cfg=ck['config'];mode=ck['mode']
    if mode in ['human_only','human_matched']:mode='independent'
    if mode in ['crisp_adapted','epr_adapted']:mode='an_ls'
    backbone=ROOT/cfg['backbone'];tok=AutoTokenizer.from_pretrained(backbone,local_files_only=True)
    model=ContextParticleModel(AutoModel.from_pretrained(backbone,local_files_only=True,add_pooling_layer=False),mode)
    model.load_state_dict(ck['model'],strict=True);model.float().eval()
    rows=[json.loads(l) for l in a.inputs.read_text(encoding='utf-8').splitlines() if l.strip()]
    if cfg.get('remove_history'):rows=[dict(r,history=[]) for r in rows]
    result={}
    with torch.inference_mode():
        for i in range(0,len(rows),a.batch_size):
            batch=rows[i:i+a.batch_size]
            enc=tok([(' '+tok.sep_token+' ').join(r['history']) for r in batch],[r['target_body']+tok.mask_token+r['target_punctuation'] for r in batch],padding=True,truncation='only_first',max_length=cfg['max_length'],return_tensors='pt')
            output=model(**enc);values=output['acceptability_logits'].sigmoid().tolist()
            for r,v in zip(batch,values):
                if r['record_id'] in result:raise ValueError('Duplicate record ID')
                result[r['record_id']]=v
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False)
    print('Saved raw acceptability scores for',len(result),'contexts; use development-only calibration before evaluation.')
if __name__=='__main__':main()
