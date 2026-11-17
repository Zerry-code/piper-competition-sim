"""Same-seed task timing with normal console rendering and pacing."""
import json,time,argparse
from web_console import Console,ROOT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fast',action='store_true');args=parser.parse_args()
    output=ROOT/('artifacts/task-timing-review/fast' if args.fast else 'artifacts/task-timing-review/after');app=Console(output_root=output);app.reset('l2',1017941739)
    original=app.callback;last=[None]
    def callback(skills,hook,stage):
        if stage!=last[0]:print(stage,round(app.world.elapsed_s,2),flush=True);last[0]=stage
        if args.fast:
            app.stage(stage);return hook(stage)
        return original(skills,hook,stage)
    app.callback=callback
    try:
        try:app.execute('香蕉和柿子全放到黄色卡纸上，黄瓜和梨放到绿色区域，剩下的水果放到红色区域')
        except Exception as exc:app.update(result=dict(status='error',error=str(exc)))
        assignments=app.data.get('parsed',{}).get('assignments',{})
        physical={o['category']:assignments.get(o['category']) in app.world.events[o['id']]['placed'] for o in app.world.objects}
        from piper_sim.timing import summarize
        current_stable={o['category']:app.world.events[o['id']]['stable_s'].get(assignments.get(o['category']),0)>=.4 for o in app.world.objects}
        report=dict(result=app.data['result'],physical=physical,physical_current_stable_0_4_s=current_stable,judge_window_s=app.world.cfg['provisional_judging']['stable_hold_s'],timing=summarize(app.data['logs'],app.world.elapsed_s,time.monotonic()-app.started),record=app.current_record)
        (ROOT/('artifacts/task-timing-review/fast.json' if args.fast else 'artifacts/task-timing-review/after.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='record'},ensure_ascii=True),flush=True)
    finally:app.renderer.close();app.world.close()
if __name__=='__main__':main()
