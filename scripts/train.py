"""Launch one recorded configuration; no training starts without an explicit run."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    registry=json.loads((ROOT/'protocols/training_runs.json').read_text(encoding='utf-8'))
    ap=argparse.ArgumentParser();ap.add_argument('--run',choices=sorted(registry),required=True)
    ap.add_argument('--output',type=Path);ap.add_argument('--max-steps',type=int,default=0)
    ap.add_argument('--dry-run',action='store_true');args=ap.parse_args();entry=registry[args.run]
    out=args.output.resolve() if args.output else ROOT/'outputs/runs'/args.run
    command=[sys.executable,str(ROOT/entry['runner']),'--config',str(ROOT/entry['config']),'--run-dir',str(out),*entry['arguments']]
    if args.max_steps:command+=['--max-steps',str(args.max_steps)]
    if args.dry_run:print(json.dumps(command,indent=2));return
    subprocess.run(command,cwd=ROOT,check=True)
if __name__=='__main__':main()
