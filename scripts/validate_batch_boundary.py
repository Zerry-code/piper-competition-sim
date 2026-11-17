"""Batch-only visual checks; optional physical placement-offset fault injection."""
import argparse,json,time,copy
import numpy as np
from web_console import Console,ROOT,WebFruitSkills
from run_l1 import L1Skills
from piper_sim.harvest import Harvest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--offset',action='store_true');args=parser.parse_args()
    output=ROOT/('artifacts/batch-boundary-offset-validation' if args.offset else 'artifacts/batch-boundary-validation')
    app=Console(output_root=output);app.reset('l1' if args.offset else 'l2',2329111945 if args.offset else 1464315472)
    if args.offset:
        skills=L1Skills(app.world,app.output,None,source='red',destination='yellow')
        original_place=skills.skill.place;injected=[False]
        def place(plan):
            if not injected[0]:
                injected[0]=True;plan=copy.deepcopy(plan)
                tray=app.world.cfg['trays']['yellow']
                y=tray['center_xy_m'][1]-tray['half_size_xy_m'][1]-.005
                plan['place'][1]=y;plan['above'][1]=y
                print('INJECT physical placement Y offset',y,flush=True)
            return original_place(plan)
        skills.skill.place=place
    else:
        task=json.loads((ROOT/'artifacts/web-console/scene-1-1464315472/task-1.json').read_text(encoding='utf-8'))['parsed']
        skills=WebFruitSkills(app.world,app.output,task)
    skills.record_stage_images=False;skills.record_sensor_frames=False
    original=skills.skill.callback;last=[None]
    def callback(stage):
        original(stage)
        if stage!=last[0]:print('STAGE',stage,'SIM',round(app.world.elapsed_s,2),flush=True);last[0]=stage
    skills.skill.callback=callback
    start=time.monotonic();harvest=Harvest(skills,max_attempts=30,max_preparations_per_target=6,max_deferrals=5)
    try:result=harvest.run()
    except Exception as exc:result=dict(status='error',error=str(exc))
    finally:
        report=dict(result=result,navigation=skills.navigation_events,events=harvest.events,evaluation=app.world.evaluation(),
            wall_s=time.monotonic()-start,placement_offset_injected=args.offset,no_teleport_or_weld=True)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=True),flush=True)
        app.renderer.close();app.world.close()

if __name__=='__main__':main()
