"""Exact user sentence, real local LLM and contact-physics validation."""
import json
import numpy as np
from web_console import Console,ROOT


def main():
    app=Console(output_root=ROOT/'artifacts/same-color-l1')
    app.reset('l1',seed=2351137725)
    app.callback=lambda skills,original,stage:original(stage)
    instruction='把三个颜色的方块分别放入对应颜色的区域。'
    try:
        app.execute(instruction)
        physical=[]
        for obj in app.world.objects:
            color=obj['category'].removeprefix('cube_');tray=app.world.cfg['trays'][color]
            low,high=app.world.bounds(obj['id']);center=np.array(tray['center_xy_m']);half=np.array(tray['half_size_xy_m'])
            physical.append(dict(category=obj['category'],destination=color,inside=bool(np.all(low[:2]>=center-half-.001) and np.all(high[:2]<=center+half+.001)),not_stacked=bool(low[2]<.065)))
        report=dict(instruction=instruction,parsed=app.data['parsed'],result=app.data['result'],physical=physical,
                    passed=app.data['result']['status']=='complete' and all(p['inside'] and p['not_stacked'] for p in physical))
    except Exception as exc:report=dict(passed=False,error=f'{type(exc).__name__}: {exc}')
    finally:
        (ROOT/'artifacts/same-color-l1-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=True),flush=True)
        if app.renderer:app.renderer.close()
        app.world.close()


if __name__=='__main__':main()
