"""Real LLM + sensor perception + contact physics on the user's two-command sequence."""
import copy,json
import numpy as np
from web_console import Console,ROOT


def main():
    app=Console(output_root=ROOT/'artifacts/compound-l1')
    app.reset('l1',seed=2351137725)
    # Remove presentation pacing only; retain the exact skills, sensor inputs,
    # language service and physics. No object state is edited.
    app.callback=lambda skills,original,stage:original(stage)
    results=[]
    try:
        before=float(app.world.backend.data.time);q=app.world.backend.measured().copy()
        app.execute('把绿色方块放到彩色卡纸上，然后把剩下的方块放到空置位置上')
        results.append(dict(kind='ambiguous',result=copy.deepcopy(app.data['result']),no_motion=bool(float(app.world.backend.data.time)==before and np.allclose(q,app.world.backend.measured()))))
        for instruction in ['把红色方块放到黄色卡纸上','把绿色方块放到绿色卡纸上，然后把剩下的方块放到完全空着的卡纸上']:
            app.execute(instruction)
            results.append(dict(instruction=instruction,result=copy.deepcopy(app.data['result']),plan=copy.deepcopy(app.data['parsed'])))
        expected={'cube_red':'yellow','cube_green':'green','cube_yellow':'red'};physical=[]
        for obj in app.world.objects:
            low,high=app.world.bounds(obj['id']);tray=app.world.cfg['trays'][expected[obj['category']]]
            center=np.array(tray['center_xy_m']);half=np.array(tray['half_size_xy_m'])
            physical.append(dict(category=obj['category'],destination=expected[obj['category']],inside=bool(np.all(low[:2]>=center-half-.001) and np.all(high[:2]<=center+half+.001)),not_stacked=bool(low[2]<.065)))
        report=dict(results=results,physical=physical,passed=all(x['inside'] and x['not_stacked'] for x in physical) and results[0]['no_motion'])
    except Exception as exc:report=dict(results=results,error=f'{type(exc).__name__}: {exc}',passed=False)
    finally:
        (ROOT/'artifacts/compound-l1-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True),flush=True)
        if app.renderer:app.renderer.close()
        app.world.close()


if __name__=='__main__':main()
