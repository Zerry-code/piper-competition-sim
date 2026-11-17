"""User pear/cucumber incident: observed one-leg recovery, no predicted second-leg veto."""
import json,time
from web_console import Console,ROOT,WebFruitSkills
from piper_sim.harvest import Harvest


def main():
    output=ROOT/'artifacts/progressive-recovery-validation';app=Console(output_root=output);app.reset('l2',3221152482)
    app.world.cfg['tasks']['l2']['time_limit_s']=420
    task=json.loads((ROOT/'artifacts/web-console/scene-1-3221152482/task-1.json').read_text(encoding='utf-8'))['parsed']
    skills=WebFruitSkills(app.world,app.output,task);skills.record_stage_images=False;skills.record_sensor_frames=False
    original=skills.skill.callback;last=[None]
    def callback(stage):
        original(stage)
        if stage!=last[0]:print('STAGE',stage,'SIM',round(app.world.elapsed_s,2),flush=True);last[0]=stage
    skills.skill.callback=callback
    start=time.monotonic();harvest=Harvest(skills,max_attempts=None,max_preparations_per_target=8,max_deferrals=8)
    try:result=harvest.run()
    except Exception as exc:result=dict(status='error',error=repr(exc),completed=list(harvest.completed))
    finally:
        closures={c:sum(e.get('target')==c and e.get('gripper_closed') and e.get('event')!='preparation_failed' for e in harvest.events if 'held' in e or e.get('event')=='object_dropped') for c in ['pear','cucumber']}
        report=dict(result=result,events=harvest.events,navigation=skills.navigation_events,evaluation=app.world.evaluation(),wall_s=time.monotonic()-start,seed=3221152482,physical_closures=closures,second_leg_gate=False)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=True),flush=True)
        app.renderer.close();app.world.close()

if __name__=='__main__':main()
