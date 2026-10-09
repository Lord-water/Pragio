"""Full-data development training; human training fit is never validation evidence."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import sys
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
from training_interface import ContextParticleModel, load_records, attach_human_labels, collate_records, LABELS
from method_core import joint_loss, an_ls_natural_loss


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, value):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)


@torch.no_grad()
def evaluate(model, records, tokenizer, device, batch_size, max_length):
    model.eval()
    natural_sum=correct=n=human_count=0
    human_sum=0.0
    predictions=[]
    for start in range(0,len(records),batch_size):
        batch=records[start:start+batch_size]
        encoded,observed,human=collate_records(batch,tokenizer,max_length)
        encoded={k:v.to(device) for k,v in encoded.items()}
        observed,human=observed.to(device),human.to(device)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            output=model(**encoded)
        output={k:v.float() for k,v in output.items()}
        parts=joint_loss(output,observed,human)
        natural_sum+=float(parts['natural_loss'])*parts['natural_count']
        human_sum+=float(parts['human_loss'])*parts['human_count']
        human_count+=parts['human_count']
        valid=observed>=0
        correct+=int((output['choice_log_probs'].argmax(-1)[valid]==observed[valid]).sum())
        n+=int(valid.sum())
        scores=output['acceptability_logits'].sigmoid().cpu().tolist()
        choice=output['choice_log_probs'].exp().cpu().tolist()
        predictions.extend({'record_id':r['record_id'],'acceptability':a,'choice':q} for r,a,q in zip(batch,scores,choice))
    return {'observed_accuracy':correct/n if n else None,'observed_nll':natural_sum/n if n else None,
            'observed_count':n,'human_bce':human_sum/human_count if human_count else None,
            'known_human_judgments':human_count},predictions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=['coupled','independent','an_ls','no_preference','single'],required=True)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--max-steps',type=int,default=0,help='Debug only; positive values mark run as smoke, not full-data evidence')
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    run=Path(args.run_dir);run.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(cfg.get('cpu_threads',2))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not args.max_steps:raise RuntimeError('Full training requires CUDA')
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
        free,total=torch.cuda.mem_get_info()
        if free<cfg['min_free_gpu_gib']*1024**3:raise RuntimeError('Insufficient free GPU memory; no existing jobs modified')
    seed=cfg['seed'];random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    annotations={} if args.mode=='single' else json.loads(Path(cfg['human_annotations']).read_text(encoding='utf-8'))
    train=attach_human_labels(load_records(cfg['train'],'train'),annotations)
    dev=load_records(cfg['dev'],'dev')
    if cfg.get('remove_history',False):
        train=[dict(r,history=[]) for r in train];dev=[dict(r,history=[]) for r in dev]
    assert not ({r['group_id'] for r in train}&{r['group_id'] for r in dev})
    assert not ({r['body_key'] for r in train}&{r['body_key'] for r in dev})
    human_rows=[r for r in train if any(v>=0 for v in r['human_labels'])]
    if not human_rows and args.mode!='single':raise ValueError('No human supervision')
    tok=AutoTokenizer.from_pretrained(cfg['backbone'],local_files_only=True)
    encoder=AutoModel.from_pretrained(cfg['backbone'],local_files_only=True,add_pooling_layer=False)
    model=ContextParticleModel(encoder,args.mode).to(device)
    groups=[]
    for name,param in model.named_parameters():
        rate=cfg['encoder_lr'] if name.startswith('encoder.') else cfg['head_lr']
        if name=='heads.preference':rate=cfg['preference_lr']
        decay=0.0 if name.endswith('bias') or 'LayerNorm' in name or name=='heads.preference' else cfg['weight_decay']
        groups.append({'params':[param],'lr':rate,'weight_decay':decay})
    optimizer=torch.optim.AdamW(groups)
    steps_per_epoch=math.ceil(len(train)/cfg['natural_batch_size'])
    planned_steps=steps_per_epoch*cfg['epochs']
    total_steps=min(planned_steps,args.max_steps) if args.max_steps else planned_steps
    scheduler=get_linear_schedule_with_warmup(optimizer,int(total_steps*cfg['warmup_fraction']),total_steps)
    environment={'torch':torch.__version__,'transformers':__import__('transformers').__version__,
                 'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
                 'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                 'pid':os.getpid(),'scope':'debug smoke only' if args.max_steps else 'full-training-partition development run; human dev pending'}
    metadata={'config':cfg,'mode':args.mode,'environment':environment,'labels':LABELS,
              'data_sha256':{key:sha(cfg[key]) for key in ['train','dev','human_annotations'] if key!='human_annotations' or args.mode!='single'},
              'code_sha256':{name:sha(BASE/name) for name in ['train_context_model.py','training_interface.py','method_core.py']},
              'counts':{'train':len(train),'dev':len(dev),'human_cases':len(human_rows),
                        'known_human_judgments':sum(v>=0 for r in human_rows for v in r['human_labels'])},
              'planned_steps':planned_steps,'executed_step_limit':total_steps,
              'checkpoint_selection':'save every completed epoch; no selection on training fit or observed-word accuracy',
              'test_used':False,'runner_sha256':sha(__file__),'history_removed':cfg.get('remove_history',False),'human_train_labels_read':args.mode!='single'}
    (run/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    emit(run/'events.jsonl',{'event':'started',**metadata['counts'],'mode':args.mode,'total_steps':total_steps})
    # Separate generators keep the human and natural example schedules identical across modes.
    hrng=random.Random(seed+200);human_queue=[]
    def next_human():
        nonlocal human_queue
        if not human_queue:
            human_queue=list(range(len(human_rows)));hrng.shuffle(human_queue)
        return human_rows[human_queue.pop()]
    torch.manual_seed(seed+1)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed+1)
    step=0;started=time.time()
    for epoch in range(1,cfg['epochs']+1):
        order=list(range(len(train)));random.Random(seed+epoch).shuffle(order)
        for start in range(0,len(order),cfg['natural_batch_size']):
            model.train()
            natural=[train[i] for i in order[start:start+cfg['natural_batch_size']]]
            humans=[] if args.mode=='single' else [next_human() for _ in range(cfg['human_batch_size'])]
            encoded,observed,human=collate_records(natural+humans,tok,cfg['max_length'])
            # Human samples add BCE only; natural samples add weak original-label loss only.
            observed[len(natural):]=-1;human[:len(natural)]=-1
            encoded={k:v.to(device) for k,v in encoded.items()}
            observed,human=observed.to(device),human.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                output=model(**encoded)
                output={k:v.float() for k,v in output.items()}
                parts=joint_loss(output,observed,human,human_weight=cfg['human_weight'])
                if args.mode=='an_ls':
                    natural_loss=an_ls_natural_loss(output['acceptability_logits'],observed,cfg['an_ls_smoothing'])
                    loss=natural_loss+cfg['human_weight']*parts['human_loss']
                elif args.mode=='single':
                    natural_loss=parts['natural_loss'];loss=natural_loss
                else:
                    natural_loss=parts['natural_loss'];loss=parts['loss']
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['max_grad_norm'])
            if not torch.isfinite(grad_norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();scheduler.step();step+=1
            if step==1 or step%cfg['log_every']==0:
                emit(run/'events.jsonl',{'event':'train','epoch':epoch,'step':step,'loss':float(loss.detach()),
                     'natural_loss':float(natural_loss.detach()),'human_loss':float(parts['human_loss'].detach()),
                     'elapsed_s':round(time.time()-started,1),'max_gpu_allocated_gib':round(torch.cuda.max_memory_allocated()/1024**3,3) if device.type=='cuda' else 0})
            if step>=total_steps:break
        complete_epoch=start+len(natural)>=len(order)
        tag=f'epoch{epoch}' if complete_epoch else f'debug_step{step}'
        dev_metrics,dev_predictions=evaluate(model,dev if not args.max_steps else dev[:16],tok,device,cfg['eval_batch_size'],cfg['max_length'])
        fit_metrics,_=evaluate(model,human_rows,tok,device,cfg['eval_batch_size'],cfg['max_length'])
        for filename,value in [(tag+'_diagnostics.json',{'weak_label_dev':dev_metrics,'human_train_fit_only':fit_metrics,
                   'scope':'These do not establish human acceptability generalization or a winning method'}),
                               (tag+'_dev_predictions.json',dev_predictions)]:
            (run/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        torch.save({'model':model.state_dict(),'mode':args.mode,'epoch':epoch,'step':step,'config':cfg},run/(tag+'.pt'))
        torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                    'human_queue':human_queue,'human_rng':hrng.getstate(),'epoch':epoch,'step':step,
                    'natural_cursor':start+len(natural),'checkpoint':tag+'.pt'},run/'latest_training_state.pt')
        emit(run/'events.jsonl',{'event':'epoch_saved','tag':tag,'step':step,'weak_label_dev':dev_metrics,'human_train_fit_only':fit_metrics})
        if step>=total_steps:break
    result={'status':'completed','mode':args.mode,'steps':step,'scope':environment['scope'],
            'elapsed_s':round(time.time()-started,1),'human_dev_evaluation':'pending independent labels','test_used':False}
    (run/'completion.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    emit(run/'events.jsonl',result)


if __name__=='__main__':main()
