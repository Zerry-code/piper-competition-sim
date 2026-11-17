"""Reported six-fruit layout with middle-first grasps and real staged recovery."""
import json,time
from web_console import Console,ROOT,WebFruitSkills
from piper_sim.harvest import Harvest


def main():
    output=ROOT/'artifacts/staging-fruit-validation';app=Console(output_root=output);app.reset('l2',1464315472)
    task=json.loads((ROOT/'artifacts/web-console/scene-1-1464315472/task-1.json').read_text(encoding='utf-8'))['parsed']
    skills=WebFruitSkills(app.world,app.output,task);skills.record_stage_images=False;skills.record_sensor_frames=False
    original=skills.skill.callback;last=[None]
    def callback(stage):
        original(stage)
        if stage!=last[0]:print('STAGE',stage,'SIM',round(app.world.elapsed_s,2),flush=True);last[0]=stage
    skills.skill.callback=callback
    start=time.monotonic();harvest=Harvest(skills,max_attempts=25,max_preparations_per_target=4,max_deferrals=5)
    try:result=harvest.run()
    except Exception as exc:result=dict(status='error',error=str(exc))
    finally:
        report=dict(result=result,navigation=skills.navigation_events,events=harvest.events,evaluation=app.world.evaluation(),wall_s=time.monotonic()-start,direct_failure_injected=False)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=True),flush=True)
        app.renderer.close();app.world.close()

if __name__=='__main__':main()
