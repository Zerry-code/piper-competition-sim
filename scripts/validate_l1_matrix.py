"""Nine explicit source/destination tasks, independent randomized episodes."""
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,sys
from run_harvest import ROOT


def run(item):
    seed,source,destination=item
    with (ROOT/'artifacts'/f'l1-matrix-{seed}.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/run_l1.py'),'--source',source,'--destination',destination,'--seed',str(seed),'--time-limit','120'],stdout=log,stderr=log,check=False)
    p=ROOT/'artifacts/l1'/str(seed)/'result.json'
    if not p.exists():return dict(seed=seed,source=source,destination=destination,status='process_failed')
    r=json.loads(p.read_text(encoding='utf-8'))
    return dict(seed=seed,source=source,destination=destination,status=r['status'],evaluation=r.get('requested_evaluation'),sim_time_s=r['evaluation']['time_s'],error=r.get('error'))


if __name__=='__main__':
    pairs=[(s,d) for s in ['red','green','yellow'] for d in ['red','green','yellow']]
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,[(740+i,s,d) for i,(s,d) in enumerate(pairs)]))
    (ROOT/'artifacts/l1-matrix.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=True),flush=True)
