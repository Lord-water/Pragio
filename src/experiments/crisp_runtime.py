import hashlib,json
from pathlib import Path
import numpy as np
import torch
from crisp_core import estimate_priors
from training_interface import collate_records

@torch.no_grad()
def update_priors(model,population,tok,device,cfg,run,step,smoke=False):
 model.eval();rows=population
 if smoke:
  rows=[]
  for j in range(12):rows.extend([r for r in population if r['observed_index']==j][:2])
 scores=[];obs=[]
 for start in range(0,len(rows),cfg['eval_batch_size']):
  batch=rows[start:start+cfg['eval_batch_size']];enc,observed,_=collate_records(batch,tok,cfg['max_length'])
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
   o=model(**{k:v.to(device) for k,v in enc.items()})
  scores.append(o['acceptability_logits'].float().sigmoid().cpu().numpy());obs.extend(observed.tolist())
 scores=np.concatenate(scores);priors,detail=estimate_priors(scores,np.array(obs),cfg['crisp_delta'],cfg['crisp_tau'])
 raw=run/f'prior_step{step}.npz';np.savez_compressed(raw,scores=scores,observed=np.array(obs))
 report={'step':step,'n':len(rows),'priors':priors.tolist(),'details':detail,'predictions_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'scope':'smoke' if smoke else 'natural training partition only','test_used':False}
 (run/f'prior_step{step}.json').write_text(json.dumps(report,indent=2));print(json.dumps({'event':'priors_updated',**report}),flush=True)
 return torch.tensor(priors,dtype=torch.float32,device=device)
