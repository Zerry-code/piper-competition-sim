"""Physical two-leg fallback check. The initial direct failure is injected explicitly."""
import argparse,json,time
from web_console import Console,ROOT,WebFruitSkills
from run_l1 import L1Skills
from piper_sim.harvest import Harvest,RetryableFailure


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--banana',action='store_true');args=parser.parse_args()
    output=ROOT/('artifacts/staging-banana-validation' if args.banana else 'artifacts/staging-physical-validation')
    app=Console(output_root=output);app.reset('l2' if args.banana else 'l1',1464315472 if args.banana else 2329111945)
    skills=WebFruitSkills(app.world,app.output,dict(category='banana',destination='yellow',assignments={'banana':'yellow'})) if args.banana else L1Skills(app.world,app.output,None,source='red',destination='yellow')
    skills.record_stage_images=False;skills.record_sensor_frames=False
    original=skills.observe_target
    def observe(target,retry):
        if not getattr(target,'staging_used',False):
            raise RetryableFailure('VALIDATION injection: defer direct route to exercise two real pick/place legs',phase='planning')
        return original(target,retry)
    skills.observe_target=observe
    old_callback=skills.skill.callback;last=[None]
    def callback(stage):
        old_callback(stage)
        if stage!=last[0]:print('STAGE',stage,'SIM',round(app.world.elapsed_s,2),flush=True);last[0]=stage
    skills.skill.callback=callback
    start=time.monotonic();harvest=Harvest(skills,max_attempts=12,max_preparations_per_target=3,max_deferrals=4)
    try:
        result=harvest.run()
    except Exception as exc:result=dict(status='error',error=str(exc))
    finally:
        report=dict(result=result,events=harvest.events,navigation=skills.navigation_events,
            evaluation=app.world.evaluation(),wall_s=time.monotonic()-start,
            direct_failure_injected=True,simulation_physics=True,teleport_or_weld=False)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=True),flush=True)
        app.renderer.close();app.world.close()

if __name__=='__main__':main()
