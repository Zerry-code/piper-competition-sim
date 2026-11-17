"""Replay the reported banana scene, with strict progress states and bridge lookahead."""
import json,time
from web_console import Console,ROOT,WebFruitSkills
from piper_sim.harvest import Harvest


def main():
    output=ROOT/'artifacts/banana-bridge-incident-validation';app=Console(output_root=output);app.reset('l2',615397612)
    task=json.loads((ROOT/'artifacts/web-console/scene-1-615397612/task-1.json').read_text(encoding='utf-8'))['parsed']
    skills=WebFruitSkills(app.world,app.output,task);skills.record_stage_images=False;skills.record_sensor_frames=False
    original=skills.skill.callback;last=[None]
    def callback(stage):
        original(stage)
        if stage!=last[0]:print('STAGE',stage,'SIM',round(app.world.elapsed_s,2),flush=True);last[0]=stage
    skills.skill.callback=callback
    start=time.monotonic();harvest=Harvest(skills,max_attempts=30,max_preparations_per_target=8,max_deferrals=8)
    try:result=harvest.run()
    except Exception as exc:result=dict(status='error',error=repr(exc))
    finally:
        report=dict(result=result,events=harvest.events,navigation=skills.navigation_events,evaluation=app.world.evaluation(),wall_s=time.monotonic()-start,seed=615397612,official_rules_profile=False)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=True),flush=True)
        app.renderer.close();app.world.close()

if __name__=='__main__':main()
