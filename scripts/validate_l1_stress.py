"""Reproduce the reported L1 stress seed through the Chinese web-task pipeline."""
import argparse,json,time
from web_console import Console,ROOT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['baseline','stress'],default='stress');parser.add_argument('--seed',type=int,default=3102957277);args=parser.parse_args()
    output=ROOT/('artifacts/l1-'+args.profile+'-validation')
    app=Console(output_root=output);app.gap_profile=args.profile;app.reset('l1',args.seed)
    # Keep task callbacks and sensor processing; omit browser JPEG encoding/pacing.
    app.render=lambda *a,**k:None
    def bind(skills):
        skills.record_stage_images=False;skills.record_sensor_frames=False
        skills.skill.cfg['planning_observer']=app.planning_update
        skills.skill.cfg['planning_report_observer']=app.planning_report
        original=skills.skill.callback
        def callback(stage):
            app.stage(stage);original(stage)
        skills.skill.callback=callback
    app.bind_callback=bind
    start=time.monotonic()
    try:
        app.execute('把三个方块放到各自对应颜色的卡纸上。')
        result=app.data['result']
    except Exception as exc:
        result=dict(status='error',error=repr(exc))
    finally:
        report=dict(result=result,evaluation=app.world.evaluation(),wall_s=time.monotonic()-start,seed=args.seed,profile=args.profile,sim_s=app.world.elapsed_s)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True),flush=True)
        app.renderer.close();app.world.close()

if __name__=='__main__':main()
