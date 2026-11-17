"""Replay reported banana layout; perception supplies all targets/landing reservations."""
import json,argparse,numpy as np
from web_console import Console,ROOT,WebFruitSkills,Harvest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--force-detour',action='store_true');parser.add_argument('--category',default='banana');parser.add_argument('--seed',type=int,default=718823998);args=parser.parse_args()
    output=ROOT/('artifacts/persimmon-route-validation' if args.category=='persimmon' else 'artifacts/banana-detour-validation' if args.force_detour else 'artifacts/banana-route-validation')
    app=Console(output_root=output);app.reset('l2',args.seed)
    task=dict(category='all',destination='mapping',assignments=dict(banana='yellow',persimmon='yellow',cucumber='green',pear='green',bell_pepper='red',potato='red'))
    skills=WebFruitSkills(app.world,app.output,task);skills.record_stage_images=False
    if args.force_detour:
        original=skills.skill.cartesian_path
        def segment(q,destination,yaw,current=None):
            if current is not None and current[2]<.23 and destination[2]<.23 and np.linalg.norm(np.asarray(destination)[:2]-np.asarray(current)[:2])>.45:
                raise RuntimeError('Injected straight transfer rejection to exercise detour')
            return original(q,destination,yaw,current)
        skills.skill.cartesian_path=segment
    harvest=Harvest(skills,max_attempts=12,max_deferrals=2,max_preparations_per_target=8)
    try:
        targets=skills.overview();banana=next(t for t in targets if t.category==args.category)
        skills.overview=lambda:[banana]
        try:report=harvest.run()
        except Exception as exc:report=dict(status='error',error=str(exc))
        report.update(events=harvest.events,navigation=skills.navigation_events,evaluation=app.world.evaluation())
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('navigation','evaluation')},ensure_ascii=True),flush=True)
    finally:app.renderer.close();app.world.close()
if __name__=='__main__':main()
