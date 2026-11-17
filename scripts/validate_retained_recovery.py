"""Actual reported pear failure plus optional lowering-fault injection."""
import json,argparse
import run_harvest as runtime
from web_console import Console,ROOT,WebFruitSkills,Harvest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inject-lower-timeout',action='store_true');args=parser.parse_args()
    output=ROOT/('artifacts/retained-pear-recovery-fault' if args.inject_lower_timeout else 'artifacts/retained-pear-recovery')
    app=Console(output_root=output);app.reset('l2',4221772094)
    skills=WebFruitSkills(app.world,app.output,dict(category='all',destination='mapping',assignments=dict(banana='yellow',persimmon='yellow',cucumber='green',pear='green',bell_pepper='red',potato='red')));skills.record_stage_images=False;skills.record_sensor_frames=False
    fault={'injected':False,'rejected_check':False,'reject_pending':False}
    original_quality=runtime.contact_quality;original_pick=skills.skill.pick
    def pick(plan):
        result=original_pick(plan)
        if args.inject_lower_timeout and not fault['rejected_check']:fault['reject_pending']=True
        return result
    def quality(backend):
        result=original_quality(backend)
        if fault['reject_pending'] and result['loaded']:
            fault.update(reject_pending=False,rejected_check=True)
            return dict(result,accepted=False,reason='Injected post-lift opposing-contact rejection')
        return result
    skills.skill.pick=pick;runtime.contact_quality=quality
    original=skills.skill.vertical_release_move
    def vertical(distance,width):
        if args.inject_lower_timeout and distance<0 and not fault['injected']:
            fault['injected']=True;raise TimeoutError('Injected transient lower tracking failure')
        return original(distance,width)
    skills.skill.vertical_release_move=vertical
    harvest=Harvest(skills,max_attempts=20,max_deferrals=3,max_preparations_per_target=8)
    try:
        targets=skills.overview();pear=next(t for t in targets if t.category=='pear');skills.overview=lambda:[pear]
        try:report=harvest.run()
        except Exception as exc:report=dict(status='error',error=str(exc))
        report.update(fault=fault,events=harvest.events,navigation=skills.navigation_events,evaluation=app.world.evaluation())
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('navigation','evaluation')},ensure_ascii=True),flush=True)
    finally:
        runtime.contact_quality=original_quality
        app.renderer.close();app.world.close()
if __name__=='__main__':main()
