"""Download the recorded pretrained revision for one training configuration."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text(encoding='utf-8'))
    from huggingface_hub import snapshot_download
    dest=ROOT/cfg['backbone']
    if ROOT not in dest.resolve().parents:raise ValueError('Backbone directory must be inside the project')
    snapshot_download(repo_id=cfg['backbone_repo'],revision=cfg['backbone_revision'],local_dir=str(dest),allow_patterns=['config.json','model.safetensors','pytorch_model.bin','tokenizer*','vocab.txt','special_tokens_map.json'])
    print(dest)
if __name__=='__main__':main()
