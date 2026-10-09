"""Human-only control with the joint run's human schedule and optimizer updates."""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
import torch
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
from training_interface import ContextParticleModel,load_records,attach_human_labels,collate_records
from method_core import joint_loss
from train_context_model import evaluate,emit,sha

def human_schedule(rows,seed):
    rng=random.Random(seed+200);queue=[]
    while True:
        if not queue:
            queue=list(range(len(rows)));rng.shuffle(queue)
        yield rows[queue.pop()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True);ap.add_argument('--run-dir',required=True)
    ap.add_argument('--max-steps',type=int,default=0)
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    run=Path(args.run_dir);run.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(cfg.get('cpu_threads',2));seed=cfg['seed']
    random.seed(seed);torch.manual_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu' and not args.max_steps:raise RuntimeError('Full training requires CUDA')
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
        if torch.cuda.mem_get_info()[0]<cfg['min_free_gpu_gib']*1024**3:raise RuntimeError('Insufficient GPU memory')
    ann=json.loads(Path(cfg['human_annotations']).read_text(encoding='utf-8'))
    train=attach_human_labels(load_records(cfg['train'],'train'),ann)
    rows=[r for r in train if any(v>=0 for v in r['human_labels'])]
    dev=load_records(cfg['dev'],'dev')
    assert len(rows)==48 and sum(v>=0 for r in rows for v in r['human_labels'])==549
    assert not {r['group_id'] for r in train}&{r['group_id'] for r in dev}
    assert not {r['body_key'] for r in train}&{r['body_key'] for r in dev}
    tok=AutoTokenizer.from_pretrained(cfg['backbone'],local_files_only=True)
    enc=AutoModel.from_pretrained(cfg['backbone'],local_files_only=True,add_pooling_layer=False)
    model=ContextParticleModel(enc,'independent').to(device)
    for p in model.heads.choice.parameters():p.requires_grad_(False)
    if cfg.get('freeze_encoder',False):
        for p in model.encoder.parameters():p.requires_grad_(False)
    groups=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            lr=cfg['encoder_lr'] if name.startswith('encoder.') else cfg['head_lr']
            decay=0.0 if name.endswith('bias') or 'LayerNorm' in name else cfg['weight_decay']
            groups.append({'params':[p],'lr':lr,'weight_decay':decay})
    optimizer=torch.optim.AdamW(groups)
    planned=cfg['total_steps'];limit=min(planned,args.max_steps) if args.max_steps else planned
    scheduler=get_linear_schedule_with_warmup(optimizer,int(limit*cfg['warmup_fraction']),limit)
    assert planned==6006 and cfg['checkpoint_steps']==[2002,4004,6006]
    scope='CPU/debug smoke only' if args.max_steps else 'Matched optimizer updates and human exposure, not matched FLOPs; retrospective supplement'
    metadata={'mode':'human_only','config':cfg,'scope':scope,'pid':os.getpid(),'counts':{'human_contexts':48,'known':549,'natural_loss_examples':0},
       'human_schedule':'Same train-order 48 rows; Random(seed+200), shuffled queue popped once per human item',
       'data_sha256':{n:sha(cfg[n]) for n in ['train','dev','human_annotations']},
       'code_sha256':{'runner':sha(__file__),**{n:sha(BASE/n) for n in ['method_core.py','training_interface.py','train_context_model.py']}},
       'planned_steps':planned,'step_limit':limit,'selection_steps':cfg['checkpoint_steps'],'test_used':False}
    (run/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    stream=human_schedule(rows,seed);torch.manual_seed(seed+1)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed+1)
    exposure=hashlib.sha256();started=time.time()
    for step in range(1,limit+1):
        model.train()
        if cfg.get('freeze_encoder',False):model.encoder.eval()
        batch=[next(stream) for _ in range(cfg['human_batch_size'])]
        for row in batch:exposure.update((row['record_id']+'\n').encode())
        encoded,observed,human=collate_records(batch,tok,cfg['max_length'])
        encoded={k:v.to(device) for k,v in encoded.items()};human=human.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            output=model(**encoded)
            loss=joint_loss({k:v.float() for k,v in output.items()},human=human,human_weight=cfg['human_weight'])['loss']
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        assert model.heads.choice.weight.grad is None
        if cfg.get('freeze_encoder',False):assert all(p.grad is None for p in model.encoder.parameters())
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['max_grad_norm'])
        if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
        optimizer.step();scheduler.step()
        if step==1 or step%cfg['log_every']==0:
            emit(run/'events.jsonl',{'event':'train','step':step,'human_loss':float(loss.detach()),'elapsed_s':round(time.time()-started,1)})
        if step in cfg['checkpoint_steps'] or step==limit:
            tag=f'step{step}' if not args.max_steps else f'debug_step{step}'
            _,pred=evaluate(model,dev if not args.max_steps else dev[:4],tok,device,cfg['eval_batch_size'],cfg['max_length'])
            fit,_=evaluate(model,rows if not args.max_steps else rows[:4],tok,device,cfg['eval_batch_size'],cfg['max_length'])
            (run/(tag+'_dev_predictions.json')).write_text(json.dumps(pred,ensure_ascii=False),encoding='utf-8')
            (run/(tag+'_diagnostics.json')).write_text(json.dumps({'human_train_fit_only':fit,'human_exposures':step*cfg['human_batch_size'],'exposure_sha256':exposure.hexdigest()}),encoding='utf-8')
            torch.save({'model':model.state_dict(),'mode':'human_only','step':step,'config':cfg},run/(tag+'.pt'))
            emit(run/'events.jsonl',{'event':'checkpoint_saved','step':step,'tag':tag,'exposure_sha256':exposure.hexdigest()})
    result={'status':'completed','mode':'human_only','steps':limit,'scope':scope,'elapsed_s':round(time.time()-started,1),'human_exposures':limit*cfg['human_batch_size'],'test_used':False}
    (run/'completion.json').write_text(json.dumps(result,indent=2),encoding='utf-8');emit(run/'events.jsonl',result)
if __name__=='__main__':main()
