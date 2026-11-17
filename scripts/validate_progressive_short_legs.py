"""Physical fallback exercise: long legs rejected by an injected planning constraint."""
import json,time
import numpy as np
from web_console import Console,ROOT
from run_l1 import L1Skills
from piper_sim.harvest import Harvest,RetryableFailure


def main():
    output=ROOT/'artifacts/progressive-short-leg-validation';app=Console(output_root=output);app.reset('l1',2329111945)
    app.world.cfg['tasks']['l2']['time_limit_s']=240
    skills=L1Skills(app.world,app.output,None,source='red',destination='yellow');skills.record_stage_images=False;skills.record_sensor_frames=False
    original_observe=skills.observe_target
    def observe(t,retry):
        if not getattr(t,'staging_used',False):raise RetryableFailure('VALIDATION: direct route intentionally unavailable',phase='planning')
        return original_observe(t,retry)
    skills.observe_target=observe
    original_plan=skills.skill.plan
    def plan(e,point,floor):
        if '前方灵活区单段' in skills.skill.cfg.get('planning_target',''):
            raise RuntimeError('VALIDATION: sweet-region leg intentionally unavailable')
        return original_plan(e,point,floor)
    skills.skill.plan=plan
    original_one=skills.skill._plan_one
    def one(e,destination,floor,*args,**kwargs):
        source=np.asarray(getattr(e,'object_center',e.center))[:2]
        if np.linalg.norm(np.asarray(destination)-source)>.135:
            raise RuntimeError('VALIDATION: only short physical transport legs allowed')
        return original_one(e,destination,floor,*args,**kwargs)
    skills.skill._plan_one=one
    harvest=Harvest(skills,max_attempts=None,max_preparations_per_target=8);start=time.monotonic()
    try:result=harvest.run()
    except Exception as exc:result=dict(status='error',error=repr(exc))
    finally:
        report=dict(result=result,events=harvest.events,navigation=skills.navigation_events,evaluation=app.world.evaluation(),wall_s=time.monotonic()-start,
            injected_long_leg_rejection_m=.135,real_contacts=True,no_teleport=True)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=True),flush=True)
        app.renderer.close();app.world.close()

if __name__=='__main__':main()
